"""Persistent cross-episode speaker identity — name-driven (plan Phase 2).

Diarization labels (``A``/``B`` …) are *per-episode*: they reset every episode
and carry no cross-episode meaning. The hosted AssemblyAI path gives us those
labels, **not** voice embeddings, so cross-episode speaker identity here keys on
**names** — spoken intros, manifest hosts, and guest metadata — canonicalized to
a stable ``speaker_id`` (to a Wikidata QID where one is supplied). Voiceprint
identity (pyannote/WhisperX, GPU) is the documented scale-path and out of scope.

Three collaborators:

- :class:`SqliteSpeakerStore` (Task 2.1) — a stdlib ``sqlite3`` table mapping a
  speaker (by QID, then normalized name) to a stable ``speaker_id``, persistent
  across runs so the same host/guest in episode N+1 reuses episode N's id.
- :class:`NameSpeakerResolver` (Task 2.2) — turns a per-episode ``label->name``
  map into per-label :class:`~dlogos.speakers.identity.SpeakerResolution`s
  against the store; a label with no name stays unresolved (the pipeline's
  per-episode fallback then handles it).
- :func:`extract_label_names` (Task 2.3) — mines the diarized transcript for
  self-intros (*"I'm X"*) and host guest-intros (*"my guest today is X"*),
  reusing the guest-intro regexes from :mod:`dlogos.speakers.guests` (DRY), and
  ties each detected name to a diarization label.

``sqlite3`` is stdlib (NO new dependency); nothing heavy is imported at module
top. Reuses :class:`CanonicalSpeaker`/:class:`SpeakerResolution` from
:mod:`dlogos.speakers.identity` and the intro patterns from
:mod:`dlogos.speakers.guests`.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Protocol, runtime_checkable

from dlogos.schema import Transcript
from dlogos.speakers.guests import (
    _INTRO_PATTERNS,
    _NAME,
    _clean_name,
)
from dlogos.speakers.identity import CanonicalSpeaker, SpeakerResolution


# --------------------------------------------------------------------------- #
# Name normalization + id minting (shared by store + extraction)
# --------------------------------------------------------------------------- #
def _normalize_name(name: str) -> str:
    """Casefold + collapse whitespace so ``Darian  Woods`` == ``darian woods``."""

    return re.sub(r"\s+", " ", name).strip().lower()


def _mint_speaker_id(*, norm_name: str | None, qid: str | None) -> str:
    """QID-first id: ``spk-wd-<QID>`` when anchored, else ``spk-<sha1[:10]>``."""

    if qid:
        return f"spk-wd-{qid}"
    digest = hashlib.sha1(norm_name.encode("utf-8")).hexdigest()[:10]
    return f"spk-{digest}"


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #
@runtime_checkable
class CanonicalSpeakerStore(Protocol):
    """The persistent canonical-speaker index contract.

    :class:`NameSpeakerResolver` depends only on this Protocol, never on a
    concrete backend. ``canonical_for`` resolves a speaker by QID first, then by
    normalized name, minting and persisting a stable ``speaker_id`` on first
    sight so the same real-world speaker recurs to one id across episodes.
    """

    def canonical_for(
        self, *, name: str | None = None, qid: str | None = None
    ) -> CanonicalSpeaker: ...

    def get(self, speaker_id: str) -> CanonicalSpeaker | None: ...

    def all(self) -> list[CanonicalSpeaker]: ...


# --------------------------------------------------------------------------- #
# SQLite backend (Task 2.1)
# --------------------------------------------------------------------------- #
_SCHEMA = """
CREATE TABLE IF NOT EXISTS speakers (
    speaker_id   TEXT PRIMARY KEY,
    qid          TEXT,
    norm_name    TEXT,
    display_name TEXT NOT NULL
)
"""


class SqliteSpeakerStore:
    """Persistent :class:`CanonicalSpeakerStore` over stdlib ``sqlite3``.

    One table ``speakers(speaker_id PK, qid, norm_name, display_name)``.
    ``canonical_for`` resolves by QID first (``spk-wd-<QID>``), then by
    normalized name (``spk-<sha1(norm_name)[:10]>``), minting on first sight and
    persisting — so re-opening the same DB path returns the same id for the same
    speaker. This is the cross-episode/cross-run stability the name-driven
    resolver relies on.
    """

    def __init__(self, path: str | Path) -> None:
        import sqlite3

        self._path = str(path)
        self._conn = sqlite3.connect(self._path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    # -- row <-> record ----------------------------------------------------- #
    @staticmethod
    def _to_record(row) -> CanonicalSpeaker:
        return CanonicalSpeaker(
            speaker_id=row["speaker_id"],
            name=row["display_name"],
            wikidata_qid=row["qid"],
        )

    # -- reads -------------------------------------------------------------- #
    def get(self, speaker_id: str) -> CanonicalSpeaker | None:
        row = self._conn.execute(
            "SELECT * FROM speakers WHERE speaker_id = ?", (speaker_id,)
        ).fetchone()
        return self._to_record(row) if row is not None else None

    def all(self) -> list[CanonicalSpeaker]:
        rows = self._conn.execute("SELECT * FROM speakers").fetchall()
        return [self._to_record(r) for r in rows]

    # -- resolve-or-mint ---------------------------------------------------- #
    def canonical_for(
        self, *, name: str | None = None, qid: str | None = None
    ) -> CanonicalSpeaker:
        """Resolve a speaker to a stable id, minting + persisting on first sight.

        QID wins: a ``qid`` keys on ``spk-wd-<QID>`` regardless of the spoken
        surface name. Otherwise the normalized name keys on
        ``spk-<sha1(norm_name)[:10]>``. The display name is kept from the first
        sighting (and re-asserted on every lookup so the row always exists).
        """

        if not name and not qid:
            raise ValueError("canonical_for requires a name or a qid")

        norm = _normalize_name(name) if name else None
        speaker_id = _mint_speaker_id(norm_name=norm or "", qid=qid)
        display = name or (qid or speaker_id)

        existing = self.get(speaker_id)
        if existing is not None:
            return existing

        self._conn.execute(
            """
            INSERT INTO speakers (speaker_id, qid, norm_name, display_name)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(speaker_id) DO NOTHING
            """,
            (speaker_id, qid, norm, display),
        )
        self._conn.commit()
        return CanonicalSpeaker(
            speaker_id=speaker_id, name=display, wikidata_qid=qid
        )

    def close(self) -> None:
        self._conn.close()


# --------------------------------------------------------------------------- #
# Name-driven resolver (Task 2.2)
# --------------------------------------------------------------------------- #
class NameSpeakerResolver:
    """Resolves per-episode diarization labels to stable ids by *name*.

    Inject a :class:`CanonicalSpeakerStore` (the sqlite store in practice). Given
    a ``label->name`` map (from :func:`extract_label_names` + manifest), each
    named label is canonicalized via ``store.canonical_for`` to a stable
    ``speaker_id`` (cross-episode); a label with no name is reported *unresolved*
    so the pipeline's host-gallery/guest/fallback chain handles it per-episode.
    Pure over the injected store — no network, no heavy deps.
    """

    def __init__(self, store: CanonicalSpeakerStore) -> None:
        self._store = store

    def resolve(
        self,
        transcript: Transcript,
        label_names: dict[str, str],
        *,
        qids: dict[str, str] | None = None,
    ) -> dict[str, SpeakerResolution]:
        """Resolve every distinct label in the transcript.

        ``label_names`` maps a per-episode diarization label to a detected name;
        ``qids`` optionally maps a *name* to its Wikidata QID (so a recurring
        guest keys on QID across shows). A named label yields a resolved
        :class:`SpeakerResolution` (score ``1.0`` — name resolution is exact); a
        label absent from ``label_names`` yields an unresolved one.
        """

        qids = qids or {}
        labels = list(dict.fromkeys(seg.speaker for seg in transcript.segments))
        # Include any named labels even if they never own a segment (defensive).
        for lbl in label_names:
            if lbl not in labels:
                labels.append(lbl)

        out: dict[str, SpeakerResolution] = {}
        for label in labels:
            name = label_names.get(label)
            if not name:
                out[label] = SpeakerResolution(
                    label=label, resolved=None, score=0.0
                )
                continue
            speaker = self._store.canonical_for(name=name, qid=qids.get(name))
            out[label] = SpeakerResolution(
                label=label, resolved=speaker, score=1.0
            )
        return out


# --------------------------------------------------------------------------- #
# Name extraction from intros + manifest (Task 2.3)
# --------------------------------------------------------------------------- #
# Self-intro phrasings ("I'm X" / "I am X" / "this is X"). Reuses the same
# tight ``_NAME`` capture as the guest-intro patterns in guests.py (DRY) so a
# self-intro name is bounded identically to a guest-intro name.
_SELF_INTRO_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        rf"\bI'?m\s+(?P<name>{_NAME})",
        rf"\bI am\s+(?P<name>{_NAME})",
        rf"\bthis is\s+(?P<name>{_NAME})",
    )
)


def _first_other_label(labels: list[str], host_label: str) -> str | None:
    """The first distinct diarization label that is not ``host_label``."""

    for lbl in labels:
        if lbl != host_label:
            return lbl
    return None


def extract_label_names(
    transcript: Transcript, *, known_hosts: list[str]
) -> dict[str, str]:
    """Map diarization labels to detected speaker names from intros + manifest.

    Two signals, both tied to the *segment* (and thus the diarization label)
    where they occur:

    - **Self-intro** (*"I'm X"* / *"this is X"*) → the speaking label *is* ``X``.
    - **Host guest-intro** (*"my guest today is X"*, reusing the guests.py
      patterns — DRY) → the host utters it, so ``X`` is attributed to the first
      *other* diarization label (the inferred guest).

    A detected name that matches a manifest ``known_hosts`` entry (case-folded)
    is normalized to the known-host surface form. Returns ``{label: name}``;
    empty when nothing fired.
    """

    labels = list(dict.fromkeys(seg.speaker for seg in transcript.segments))
    known_by_norm = {_normalize_name(h): h for h in known_hosts}

    def _canonicalize(name: str) -> str:
        return known_by_norm.get(_normalize_name(name), name)

    out: dict[str, str] = {}

    # Pass 1: self-intros bind a name to the speaking label directly.
    for seg in transcript.segments:
        if seg.speaker in out:
            continue
        for pat in _SELF_INTRO_PATTERNS:
            m = pat.search(seg.text)
            if m:
                name = _clean_name(m.group("name"))
                if name:
                    out[seg.speaker] = _canonicalize(name)
                    break

    # Pass 2: host guest-intros attribute the guest name to the *other* label.
    for seg in transcript.segments:
        for pat in _INTRO_PATTERNS:
            for m in pat.finditer(seg.text):
                name = _clean_name(m.group("name"))
                if not name:
                    continue
                guest_label = _first_other_label(labels, seg.speaker)
                if guest_label is not None and guest_label not in out:
                    out[guest_label] = _canonicalize(name)

    return out
