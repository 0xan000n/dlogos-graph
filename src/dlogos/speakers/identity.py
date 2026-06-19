"""Host-anchored cross-episode speaker identity (spec §7.3).

Diarization labels (``SPEAKER_00`` …) are *per-episode*: ``SPEAKER_00`` in one
episode has no relation to ``SPEAKER_00`` in the next. To answer "who said
what across the corpus" we must map each per-episode label to a **canonical
speaker** that is stable across episodes.

The cheap, high-value anchor is the *host*: hosts recur in every episode of
their show and the corpus manifest already names them (``known_hosts``,
sometimes with reference audio). We seed a **voiceprint gallery** from those
known hosts, then match each episode's diarization labels to a gallery entry by
**cosine similarity** of voice embeddings. A label that matches a host above a
threshold resolves to that host; everything else is left unresolved (and is a
candidate for guest resolution — :mod:`dlogos.speakers.guests` — or stays a
per-episode speaker).

The voice embedder is *injected* via the :class:`VoiceEmbedder` protocol, so
unit tests pass a deterministic fake and never load ``pyannote``/``torch``.
Diarization → confident misattribution is the corpus's top risk (spec §11), so
matching is deliberately conservative: ambiguous cases (two gallery entries
within ``margin`` of each other) are left unresolved rather than guessed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from dlogos.schema import SpeakerRef, Transcript


# --------------------------------------------------------------------------- #
# Injected dependency: a voice embedder
# --------------------------------------------------------------------------- #
@runtime_checkable
class VoiceEmbedder(Protocol):
    """Turns a *voice-sample reference* into a fixed-length vector.

    The argument is an opaque sample key (a path to reference audio, an episode
    diarization-segment id, etc.). Production uses a speaker-embedding model
    (e.g. pyannote / SpeechBrain), imported *lazily* in a concrete adapter so
    importing this module needs only core deps. Tests inject a fake that maps
    known keys to fixed unit vectors.
    """

    def embed(self, sample_ref: str) -> list[float]:  # pragma: no cover - protocol
        ...


# --------------------------------------------------------------------------- #
# Gallery seed + canonical speaker
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HostSeed:
    """One known host from the corpus manifest, used to seed the gallery.

    ``sample_refs`` are voice-sample keys (reference-audio ids) the injected
    :class:`VoiceEmbedder` knows how to turn into vectors. A host with no
    sample refs cannot be voiceprint-matched (it still defines the canonical
    speaker, and may be resolvable by other signals).

    This is intentionally a *local* lightweight input type so the speakers
    package does not depend on the (separately owned) ingestion manifest
    module; the ingestion layer can adapt its manifest rows into ``HostSeed``s.
    """

    speaker_id: str
    name: str
    show_id: str
    sample_refs: tuple[str, ...] = ()


@runtime_checkable
class ManifestRowLike(Protocol):
    """The slice of a corpus-manifest row this module's adapter consumes.

    Declared structurally (duck-typed) so the speakers package keeps **no**
    import dependency on the separately-owned ingestion manifest module — any
    object exposing these three attributes (the real
    :class:`dlogos.ingestion.manifest.ManifestRow` does) can be adapted.
    """

    show_id: str
    known_hosts: list[str]
    reference_audio: dict[str, str]


def host_speaker_id(name: str) -> str:
    """Derive a stable canonical-host id from a host display name.

    Deterministic and collision-resistant enough for the gallery: the same
    host name always yields the same ``host-<slug>`` id, so a host recurring
    across shows merges into one gallery entry (see
    :meth:`HostGallery.add_host`).
    """

    slug = "-".join(name.lower().split())
    return f"host-{slug}" if slug else "host-unknown"


def seeds_from_manifest_rows(rows: list[ManifestRowLike]) -> list[HostSeed]:
    """Adapt corpus-manifest rows into :class:`HostSeed`\\s.

    Each named host in a row's ``known_hosts`` becomes one seed; the seed's
    ``sample_refs`` are populated from the row's per-host ``reference_audio``
    (empty when the host has no reference, which is allowed — the host still
    defines a canonical speaker but cannot be voiceprint-matched). A host that
    appears across multiple shows produces multiple seeds with the same
    ``speaker_id``; :meth:`HostGallery.add_host` merges them.
    """

    seeds: list[HostSeed] = []
    for row in rows:
        reference_audio = getattr(row, "reference_audio", {}) or {}
        for name in row.known_hosts:
            ref = reference_audio.get(name)
            seeds.append(
                HostSeed(
                    speaker_id=host_speaker_id(name),
                    name=name,
                    show_id=row.show_id,
                    sample_refs=(ref,) if ref else (),
                )
            )
    return seeds


@dataclass(frozen=True)
class CanonicalSpeaker:
    """A resolved, cross-episode speaker (host or recurring guest)."""

    speaker_id: str
    name: str
    is_host: bool = False
    show_ids: tuple[str, ...] = ()
    wikidata_qid: str | None = None


@dataclass(frozen=True)
class SpeakerResolution:
    """The outcome of resolving one per-episode diarization label.

    ``resolved`` is ``None`` when no gallery entry cleared the threshold (the
    label stays a per-episode speaker). ``score`` is the cosine similarity to
    the matched entry (or to the best non-matching entry, for diagnostics).
    """

    label: str
    resolved: CanonicalSpeaker | None
    score: float

    @property
    def is_resolved(self) -> bool:
        return self.resolved is not None

    def to_speaker_ref(self) -> SpeakerRef:
        """Project to the shared :class:`~dlogos.schema.SpeakerRef`.

        An unresolved label yields a bare ``SpeakerRef`` carrying only the
        per-episode diarization label — exactly what the rest of the pipeline
        treats as "per-episode speaker".
        """

        if self.resolved is None:
            return SpeakerRef(label=self.label)
        return SpeakerRef(
            label=self.label,
            resolved_id=self.resolved.speaker_id,
            name=self.resolved.name,
        )


# --------------------------------------------------------------------------- #
# Cosine helper
# --------------------------------------------------------------------------- #
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# --------------------------------------------------------------------------- #
# The gallery
# --------------------------------------------------------------------------- #
@dataclass
class HostGallery:
    """A voiceprint gallery seeded from manifest ``known_hosts``.

    Construct with :meth:`from_hosts` (which embeds every seed's sample refs
    via the injected embedder and averages them into one centroid per host),
    then call :meth:`resolve_label` / :meth:`resolve_transcript` to map
    per-episode diarization labels to canonical hosts.

    Matching rule (conservative — top risk is *misattribution*):

    - compute cosine similarity of the label's voiceprint to every host
      centroid;
    - the best host must clear ``threshold`` **and** beat the runner-up by at
      least ``margin`` (so two similar-sounding hosts don't get a coin-flip
      assignment);
    - otherwise the label is left unresolved.
    """

    embedder: VoiceEmbedder
    threshold: float = 0.75
    margin: float = 0.05
    _centroids: dict[str, np.ndarray] = field(default_factory=dict)
    _speakers: dict[str, CanonicalSpeaker] = field(default_factory=dict)

    # -- construction ----------------------------------------------------- #
    @classmethod
    def from_hosts(
        cls,
        hosts: list[HostSeed],
        embedder: VoiceEmbedder,
        *,
        threshold: float = 0.75,
        margin: float = 0.05,
    ) -> "HostGallery":
        gallery = cls(embedder=embedder, threshold=threshold, margin=margin)
        for host in hosts:
            gallery.add_host(host)
        return gallery

    @classmethod
    def from_manifest_rows(
        cls,
        rows: list[ManifestRowLike],
        embedder: VoiceEmbedder,
        *,
        threshold: float = 0.75,
        margin: float = 0.05,
    ) -> "HostGallery":
        """Seed a gallery directly from corpus-manifest rows.

        Adapts ``rows`` into :class:`HostSeed`\\s (via
        :func:`seeds_from_manifest_rows`, which pulls each host's
        ``sample_refs`` from the row's ``reference_audio``) and builds the
        gallery — the manifest→gallery bridge.
        """

        return cls.from_hosts(
            seeds_from_manifest_rows(rows),
            embedder,
            threshold=threshold,
            margin=margin,
        )

    def add_host(self, host: HostSeed) -> None:
        """Add (or merge into) a host's gallery entry.

        Multiple seeds for the same ``speaker_id`` (e.g. the same host across
        two shows) accumulate their sample refs into one centroid and union
        their show ids.
        """

        vectors = [
            np.asarray(self.embedder.embed(ref), dtype=float)
            for ref in host.sample_refs
        ]

        existing = self._speakers.get(host.speaker_id)
        if existing is not None:
            show_ids = tuple(dict.fromkeys((*existing.show_ids, host.show_id)))
            self._speakers[host.speaker_id] = CanonicalSpeaker(
                speaker_id=existing.speaker_id,
                name=existing.name,
                is_host=True,
                show_ids=show_ids,
                wikidata_qid=existing.wikidata_qid,
            )
            if vectors:
                prior = self._centroids.get(host.speaker_id)
                stacked = vectors if prior is None else [prior, *vectors]
                self._centroids[host.speaker_id] = np.mean(
                    np.stack(stacked), axis=0
                )
        else:
            self._speakers[host.speaker_id] = CanonicalSpeaker(
                speaker_id=host.speaker_id,
                name=host.name,
                is_host=True,
                show_ids=(host.show_id,),
            )
            if vectors:
                self._centroids[host.speaker_id] = np.mean(
                    np.stack(vectors), axis=0
                )

    # -- introspection ---------------------------------------------------- #
    @property
    def speakers(self) -> dict[str, CanonicalSpeaker]:
        return dict(self._speakers)

    def speaker(self, speaker_id: str) -> CanonicalSpeaker | None:
        return self._speakers.get(speaker_id)

    # -- matching --------------------------------------------------------- #
    def resolve_voiceprint(
        self, vector: list[float] | np.ndarray, *, label: str = ""
    ) -> SpeakerResolution:
        """Match an already-computed voiceprint vector against the gallery."""

        vec = np.asarray(vector, dtype=float)
        ranked = sorted(
            ((sid, _cosine(vec, cen)) for sid, cen in self._centroids.items()),
            key=lambda kv: kv[1],
            reverse=True,
        )
        if not ranked:
            return SpeakerResolution(label=label, resolved=None, score=0.0)

        best_id, best_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else float("-inf")

        clears_threshold = best_score >= self.threshold
        clears_margin = (best_score - runner_up) >= self.margin
        if clears_threshold and clears_margin:
            return SpeakerResolution(
                label=label, resolved=self._speakers[best_id], score=best_score
            )
        # Ambiguous or below threshold: leave unresolved, but report the score.
        return SpeakerResolution(label=label, resolved=None, score=best_score)

    def resolve_label(self, label: str, sample_ref: str) -> SpeakerResolution:
        """Embed a label's voice sample, then match it against the gallery."""

        vec = self.embedder.embed(sample_ref)
        return self.resolve_voiceprint(vec, label=label)

    def resolve_transcript(
        self, transcript: Transcript, sample_refs: dict[str, str]
    ) -> dict[str, SpeakerResolution]:
        """Resolve every distinct diarization label in an episode.

        ``sample_refs`` maps each per-episode label (``SPEAKER_00`` …) to the
        voice-sample key for that label in this episode. Labels absent from the
        map are reported unresolved with a zero score (no voiceprint to match).
        """

        labels = list(dict.fromkeys(seg.speaker for seg in transcript.segments))
        out: dict[str, SpeakerResolution] = {}
        for label in labels:
            ref = sample_refs.get(label)
            if ref is None:
                out[label] = SpeakerResolution(
                    label=label, resolved=None, score=0.0
                )
            else:
                out[label] = self.resolve_label(label, ref)
        return out
