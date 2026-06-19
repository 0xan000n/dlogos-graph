"""Recurring-guest resolution (spec §7.3 / §7.4a).

Guests — not just hosts — are belief-tracking subjects: "what does *[guest]*
believe about X, and has it changed?" only works if the same guest resolves to
one stable id across the several episodes they appear in. We combine three
cheap signals, none of which alone is reliable:

1. **Episode metadata** — guest/author fields and the episode title from the
   RSS feed / show notes.
2. **The spoken intro** — the host's *"my guest today is …"* utterance in the
   diarized transcript, parsed by regex into a spoken name and tied to the
   guest's diarization label for that episode.
3. **A Wikidata match** — canonicalize the parsed name to a stable Wikidata
   **QID**, disambiguated by domain context, so the same person merges across
   shows. The Wikidata client uses ``httpx`` *lazily* (imported inside the
   method) and is *injectable*, so unit tests pass a fake and hit no network.

Policy (spec §7.3): a guest that recurs across ``min_appearances`` episodes
*and* canonicalizes to a Wikidata QID is promoted to a stable
:class:`~dlogos.speakers.identity.CanonicalSpeaker`. The one-off long tail is
*left per-episode* — that is acceptable for the PoC and avoids the worst
failure of confidently merging two different people.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from dlogos.schema import SpeakerRef, Transcript
from dlogos.speakers.identity import CanonicalSpeaker, SpeakerResolution

# --------------------------------------------------------------------------- #
# Intro-pattern extraction
# --------------------------------------------------------------------------- #
# A capitalized name: 1–3 tokens, each Titlecased, allowing internal
# apostrophes/hyphens (e.g. "O'Brien", "Jean-Luc"). Deliberately tight so we do
# not scoop up arbitrary capitalized phrases. The ``(?-i:...)`` scope forces
# case-sensitivity for the *name* even when the surrounding cue phrase is
# matched case-insensitively — otherwise IGNORECASE lets a leading "[A-Z]" also
# match a lowercase token (e.g. trailing "on") and the name over-captures.
_NAME = r"(?-i:[A-Z][a-zA-Z'’.-]+(?:\s+[A-Z][a-zA-Z'’.-]+){0,2})"

# Host intro phrasings. Each must capture the name in group "name".
_INTRO_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        rf"\bmy guest today is\s+(?P<name>{_NAME})",
        rf"\bjoining me (?:today\s+)?is\s+(?P<name>{_NAME})",
        rf"\bI'?m here with\s+(?P<name>{_NAME})",
        rf"\b(?:please\s+)?welcom(?:e|ing)\s+(?P<name>{_NAME})",
        rf"\btoday I'?m (?:joined|talking) (?:by|with)\s+(?P<name>{_NAME})",
        rf"\bour guest (?:today )?is\s+(?P<name>{_NAME})",
    )
)

# Trailing filler that can ride along after the captured name.
_TRAILING_FILLER = re.compile(
    r"\b(today|here|with me|on the show|back)\b.*$", re.IGNORECASE
)


def _clean_name(raw: str) -> str:
    name = raw.strip().strip(".,").strip()
    name = _TRAILING_FILLER.sub("", name).strip()
    return name


def extract_intro_names(transcript: Transcript) -> list[str]:
    """Pull candidate guest names from host *intro* utterances.

    Scans every segment for the known intro phrasings and returns the cleaned
    spoken names in order of appearance, de-duplicated (first occurrence wins).
    Returns an empty list when no intro pattern fires.
    """

    found: list[str] = []
    seen: set[str] = set()
    for seg in transcript.segments:
        for pat in _INTRO_PATTERNS:
            for m in pat.finditer(seg.text):
                name = _clean_name(m.group("name"))
                key = name.lower()
                if name and key not in seen:
                    seen.add(key)
                    found.append(name)
    return found


# --------------------------------------------------------------------------- #
# Wikidata canonicalization (lazy httpx, injectable)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WikidataMatch:
    """A canonicalization result for a guest name."""

    qid: str
    label: str
    description: str | None = None


@runtime_checkable
class WikidataClient(Protocol):
    """Maps a person name (+ optional domain context) to a Wikidata QID.

    Production hits the Wikidata SPARQL/entity-search endpoint; tests inject a
    fake. ``context`` carries domain hints (e.g. the show's domain tags) for
    disambiguation. Returns ``None`` when no confident match exists.
    """

    def lookup(
        self, name: str, context: list[str] | None = None
    ) -> WikidataMatch | None:  # pragma: no cover - protocol
        ...


class HttpxWikidataClient:
    """Default :class:`WikidataClient` over the Wikidata entity-search API.

    ``httpx`` is imported **lazily** inside :meth:`lookup` so importing this
    module (and the whole speakers package) never requires it. This client is
    *not* exercised by unit tests — tests inject a fake — so the network path
    stays out of the deterministic suite.
    """

    def __init__(
        self,
        endpoint: str = "https://www.wikidata.org/w/api.php",
        *,
        timeout: float = 10.0,
        language: str = "en",
    ) -> None:
        self.endpoint = endpoint
        self.timeout = timeout
        self.language = language

    def lookup(
        self, name: str, context: list[str] | None = None
    ) -> WikidataMatch | None:
        import httpx  # lazy: keep core import-light

        params = {
            "action": "wbsearchentities",
            "search": name,
            "language": self.language,
            "format": "json",
            "type": "item",
            "limit": 5,
        }
        try:
            resp = httpx.get(self.endpoint, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception:  # network/parse failure → no confident match
            return None

        hits = data.get("search") or []
        if not hits:
            return None

        # Prefer a hit whose description overlaps the domain context; else the
        # top-ranked hit (the API already returns best-match-first).
        chosen = hits[0]
        if context:
            ctx = {c.lower() for c in context}
            for hit in hits:
                desc = (hit.get("description") or "").lower()
                if any(token in desc for token in ctx):
                    chosen = hit
                    break

        return WikidataMatch(
            qid=chosen["id"],
            label=chosen.get("label", name),
            description=chosen.get("description"),
        )


# --------------------------------------------------------------------------- #
# Candidate aggregation across episodes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GuestCandidate:
    """A guest *name* observed in one episode, with its evidence + label.

    ``diarization_label`` ties the spoken/metadata name to a per-episode
    speaker label so the resolved id can be written back onto that episode's
    claims. ``from_metadata`` / ``from_intro`` record which signals fired
    (both is strongest).
    """

    name: str
    episode_id: str
    show_id: str
    diarization_label: str | None = None
    from_metadata: bool = False
    from_intro: bool = False
    context: tuple[str, ...] = ()


@dataclass(frozen=True)
class GuestResolution:
    """The outcome of resolving one guest name across the corpus.

    ``resolved`` is a stable :class:`CanonicalSpeaker` when the guest recurred
    enough *and* canonicalized to a Wikidata QID; otherwise ``None`` and the
    appearances stay per-episode (the long tail).
    """

    name: str
    appearances: tuple[GuestCandidate, ...]
    resolved: CanonicalSpeaker | None
    wikidata: WikidataMatch | None

    @property
    def is_resolved(self) -> bool:
        return self.resolved is not None

    @property
    def episode_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(a.episode_id for a in self.appearances))

    def resolution_for(self, episode_id: str) -> SpeakerResolution | None:
        """A per-episode :class:`SpeakerResolution` for this guest, if resolved.

        Returns ``None`` when the guest is unresolved (long tail) or did not
        appear in ``episode_id``. Score is ``1.0`` — name resolution is exact,
        not a similarity match.
        """

        if self.resolved is None:
            return None
        for appearance in self.appearances:
            if appearance.episode_id == episode_id:
                return SpeakerResolution(
                    label=appearance.diarization_label or "",
                    resolved=self.resolved,
                    score=1.0,
                )
        return None


def _normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().lower()


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


# --------------------------------------------------------------------------- #
# The resolver
# --------------------------------------------------------------------------- #
@dataclass
class GuestResolver:
    """Resolves recurring guests; leaves the long tail per-episode.

    Inject a :class:`WikidataClient` (a fake in tests). Feed per-episode
    candidates via :meth:`add_episode` (which combines the metadata names and
    the parsed intro names for that episode), then call :meth:`resolve`.

    A guest is promoted to a stable id only when it appears in at least
    ``min_appearances`` distinct episodes **and** Wikidata returns a QID — both
    gates, because a recurring but unidentifiable name, or a one-off
    identifiable name, is not safe to merge across shows at PoC scale.
    """

    wikidata: WikidataClient
    min_appearances: int = 2
    require_wikidata: bool = True
    _candidates: dict[str, list[GuestCandidate]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def add_candidate(self, candidate: GuestCandidate) -> None:
        self._candidates[_normalize_name(candidate.name)].append(candidate)

    def add_episode(
        self,
        transcript: Transcript,
        *,
        show_id: str,
        metadata_names: list[str] | None = None,
        guest_label: str | None = None,
        context: list[str] | None = None,
    ) -> list[GuestCandidate]:
        """Register one episode's guest candidates from all signals.

        Combines ``metadata_names`` (from RSS/show-notes) with names parsed
        from the diarized intro. A name seen via *both* signals is recorded
        once with both flags set. ``guest_label`` is the diarization label of
        the guest's turns in this episode (best-effort; may be ``None``).
        Returns the candidates added for this episode.
        """

        intro_names = extract_intro_names(transcript)
        meta = metadata_names or []

        meta_norm = {_normalize_name(n) for n in meta}
        intro_norm = {_normalize_name(n) for n in intro_names}

        # Preserve the nicest surface form (prefer metadata casing, else intro).
        display: dict[str, str] = {}
        for n in intro_names:
            display.setdefault(_normalize_name(n), n)
        for n in meta:  # metadata wins the display form
            display[_normalize_name(n)] = n

        ctx = tuple(context or [])
        added: list[GuestCandidate] = []
        for norm in dict.fromkeys([*meta_norm, *intro_norm]):
            cand = GuestCandidate(
                name=display[norm],
                episode_id=transcript.episode_id,
                show_id=show_id,
                diarization_label=guest_label,
                from_metadata=norm in meta_norm,
                from_intro=norm in intro_norm,
                context=ctx,
            )
            self.add_candidate(cand)
            added.append(cand)
        return added

    def resolve(self) -> list[GuestResolution]:
        """Resolve all accumulated candidates into per-name decisions."""

        results: list[GuestResolution] = []
        for norm, appearances in self._candidates.items():
            episodes = {a.episode_id for a in appearances}
            recurring = len(episodes) >= self.min_appearances

            display = self._display_name(appearances)
            match: WikidataMatch | None = None
            if recurring:
                # Merge domain context across appearances for disambiguation.
                ctx = list(
                    dict.fromkeys(
                        c for a in appearances for c in a.context
                    )
                )
                match = self.wikidata.lookup(display, ctx or None)

            resolved: CanonicalSpeaker | None = None
            if recurring and (match is not None or not self.require_wikidata):
                speaker_id = (
                    f"guest-{match.qid}" if match else f"guest-{_slug(display)}"
                )
                show_ids = tuple(
                    dict.fromkeys(a.show_id for a in appearances)
                )
                resolved = CanonicalSpeaker(
                    speaker_id=speaker_id,
                    name=match.label if match else display,
                    is_host=False,
                    show_ids=show_ids,
                    wikidata_qid=match.qid if match else None,
                )

            results.append(
                GuestResolution(
                    name=display,
                    appearances=tuple(appearances),
                    resolved=resolved,
                    wikidata=match,
                )
            )
        return results

    @staticmethod
    def _display_name(appearances: list[GuestCandidate]) -> str:
        """Choose the display name: prefer one seen in metadata, else first."""

        for a in appearances:
            if a.from_metadata:
                return a.name
        return appearances[0].name


def apply_guest_resolution(
    claims_speaker_refs: list[SpeakerRef],
    resolution: GuestResolution,
    episode_id: str,
) -> None:
    """Write a resolved guest's id/name onto matching per-episode SpeakerRefs.

    Mutates in place every :class:`~dlogos.schema.SpeakerRef` whose ``label``
    matches the guest's diarization label in ``episode_id`` and that is not yet
    resolved. A no-op when the guest is unresolved or absent from the episode.
    """

    per_ep = resolution.resolution_for(episode_id)
    if per_ep is None or per_ep.resolved is None or not per_ep.label:
        return
    for ref in claims_speaker_refs:
        if ref.label == per_ep.label and ref.resolved_id is None:
            ref.resolved_id = per_ep.resolved.speaker_id
            ref.name = per_ep.resolved.name
