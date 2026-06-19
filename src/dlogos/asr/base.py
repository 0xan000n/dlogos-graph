"""ASR backend contract + the talk-time-threshold helper.

This module is deliberately dependency-light: it defines the
:class:`ASRBackend` ``Protocol`` (the seam every transcription backend plugs
into) and the talk-time helpers that prune low-signal diarization speakers
*before* cross-episode speaker identity runs. Importing it never touches a
heavy/optional dep, so unit tests load it with the core group only.

Talk-time pruning rationale (spec §7.2 / §11): diarization often emits a few
spurious labels — a stray cough, a clipped ad bumper, a moment of crosstalk
mis-segmented into its own "speaker". Carrying those forward inflates the
speaker gallery and is one path to *confident misattribution* (the top
correctness risk). Corpus-scale precedent (SPoRC) drops speakers under ~5% of
total talk time; we expose that threshold rather than hard-coding it.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from dlogos.schema import Transcript, TranscriptSegment


@runtime_checkable
class ASRBackend(Protocol):
    """The contract every ASR backend satisfies.

    A backend takes a path to an audio file and returns a fully populated,
    speaker-labeled, time-stamped :class:`~dlogos.schema.Transcript`. Backends
    may be real (WhisperX + pyannote, lazily importing torch) or fake (the
    deterministic mock used in tests / offline pipeline runs).

    Kept a ``Protocol`` (not an ABC) so callers depend on the *shape*, not an
    inheritance tree — the mock and the real backend are interchangeable and
    injectable, which is what keeps unit tests free of heavy deps.
    """

    def transcribe(self, audio_path: str) -> Transcript:
        """Transcribe + diarize ``audio_path`` into a :class:`Transcript`."""
        ...


@dataclass(frozen=True)
class TalkTimeStats:
    """Per-speaker talk-time summary for a transcript.

    ``by_speaker`` maps each diarization label to its total spoken seconds;
    ``fractions`` maps each label to its share of total talk time in [0, 1];
    ``total_s`` is the sum of all segment durations (not the wall-clock episode
    duration — silence/music gaps are excluded).
    """

    by_speaker: dict[str, float] = field(default_factory=dict)
    fractions: dict[str, float] = field(default_factory=dict)
    total_s: float = 0.0


def _segment_duration(segment: TranscriptSegment) -> float:
    """Clamp a segment's duration to be non-negative.

    Real diarizers occasionally emit ``t_end < t_start`` on boundary glitches;
    treating those as zero-length keeps the talk-time math robust rather than
    letting a negative value silently cancel real talk time.
    """

    return max(0.0, segment.t_end - segment.t_start)


def talk_time_by_speaker(transcript: Transcript) -> TalkTimeStats:
    """Compute per-speaker talk time and each speaker's fraction of the total.

    Deterministic and side-effect free. When the transcript has no segments
    (or all zero-length), ``total_s`` is ``0.0`` and ``fractions`` is empty —
    no division by zero.
    """

    by_speaker: dict[str, float] = defaultdict(float)
    for segment in transcript.segments:
        by_speaker[segment.speaker] += _segment_duration(segment)

    total = sum(by_speaker.values())
    fractions: dict[str, float] = {}
    if total > 0.0:
        fractions = {spk: secs / total for spk, secs in by_speaker.items()}

    return TalkTimeStats(
        by_speaker=dict(by_speaker),
        fractions=fractions,
        total_s=total,
    )


def drop_low_talk_time_speakers(
    transcript: Transcript,
    min_fraction: float = 0.05,
) -> Transcript:
    """Drop speakers below ``min_fraction`` of total talk time.

    Returns a *new* :class:`~dlogos.schema.Transcript` with every segment
    belonging to a dropped speaker removed; the input is left untouched.

    The default ``0.05`` (5%) follows the SPoRC corpus-scale precedent
    (spec §7.2). Pruning these labels here — before cross-episode speaker
    identity — keeps the host-anchored gallery clean and removes a path to
    confident misattribution (spec §11).

    Edge behaviour, chosen to be safe and predictable:

    - ``min_fraction <= 0`` keeps everyone (a no-op filter), but still returns a
      fresh copy.
    - ``min_fraction`` outside ``[0, 1]`` is a programming error and raises
      :class:`ValueError`.
    - The threshold is *strict-less-than*: a speaker at exactly ``min_fraction``
      is **kept** (drop only those strictly below the bar).
    - A transcript with no talk time (empty / all zero-length) is returned as a
      copy unchanged — there is nothing to rank.
    """

    if not (0.0 <= min_fraction <= 1.0):
        raise ValueError(
            f"min_fraction must be in [0, 1]; got {min_fraction!r}"
        )

    stats = talk_time_by_speaker(transcript)

    # Nothing to prune: no talk time, or threshold disabled.
    if stats.total_s <= 0.0 or min_fraction <= 0.0:
        return transcript.model_copy(deep=True)

    keep = {
        spk
        for spk, frac in stats.fractions.items()
        if frac >= min_fraction
    }
    kept_segments = [
        segment.model_copy(deep=True)
        for segment in transcript.segments
        if segment.speaker in keep
    ]

    return transcript.model_copy(update={"segments": kept_segments}, deep=True)
