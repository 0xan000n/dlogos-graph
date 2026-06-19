"""Tests for the talk-time threshold helper (``dlogos.asr.base``).

These exercise the SPoRC-style pruning that drops diarization speakers under a
small share of total talk time — a guard against carrying spurious labels into
speaker identity (one path to confident misattribution). Core deps only; no
torch / whisperx / pyannote is imported.
"""

from __future__ import annotations

import pytest

from dlogos.asr.base import (
    ASRBackend,
    drop_low_talk_time_speakers,
    talk_time_by_speaker,
)
from dlogos.asr.mock_backend import MockASRBackend
from dlogos.schema import Transcript, TranscriptSegment


def _transcript(*segments: TranscriptSegment, episode_id: str = "ep-x") -> Transcript:
    duration = max((s.t_end for s in segments), default=0.0)
    return Transcript(
        episode_id=episode_id,
        language="en",
        segments=list(segments),
        duration_s=duration,
    )


def _seg(speaker: str, t_start: float, t_end: float) -> TranscriptSegment:
    return TranscriptSegment(speaker=speaker, text=f"{speaker} talk", t_start=t_start, t_end=t_end)


# --------------------------------------------------------------------------- #
# talk_time_by_speaker
# --------------------------------------------------------------------------- #
def test_talk_time_sums_and_fractions() -> None:
    t = _transcript(
        _seg("A", 0.0, 9.0),   # 9s
        _seg("B", 9.0, 10.0),  # 1s
    )
    stats = talk_time_by_speaker(t)
    assert stats.by_speaker == {"A": 9.0, "B": 1.0}
    assert stats.total_s == 10.0
    assert stats.fractions["A"] == pytest.approx(0.9)
    assert stats.fractions["B"] == pytest.approx(0.1)
    # Fractions sum to 1.
    assert sum(stats.fractions.values()) == pytest.approx(1.0)


def test_talk_time_accumulates_multiple_segments_per_speaker() -> None:
    t = _transcript(
        _seg("A", 0.0, 4.0),
        _seg("B", 4.0, 5.0),
        _seg("A", 5.0, 6.0),  # A again
    )
    stats = talk_time_by_speaker(t)
    assert stats.by_speaker == {"A": 5.0, "B": 1.0}
    assert stats.total_s == 6.0


def test_talk_time_empty_transcript_is_zero_no_division() -> None:
    stats = talk_time_by_speaker(_transcript())
    assert stats.total_s == 0.0
    assert stats.by_speaker == {}
    assert stats.fractions == {}


def test_talk_time_ignores_negative_duration_segments() -> None:
    # A boundary-glitch segment (t_end < t_start) contributes zero, not negative.
    t = _transcript(
        _seg("A", 0.0, 10.0),
        TranscriptSegment(speaker="B", text="glitch", t_start=10.0, t_end=10.0),
    )
    stats = talk_time_by_speaker(t)
    assert stats.by_speaker["A"] == 10.0
    assert stats.by_speaker["B"] == 0.0
    assert stats.total_s == 10.0


# --------------------------------------------------------------------------- #
# drop_low_talk_time_speakers
# --------------------------------------------------------------------------- #
def test_drops_speaker_below_threshold() -> None:
    # B holds 1/100 = 1% < 5% default → dropped; A (99%) kept.
    t = _transcript(
        _seg("A", 0.0, 99.0),
        _seg("B", 99.0, 100.0),
    )
    pruned = drop_low_talk_time_speakers(t)
    speakers = {s.speaker for s in pruned.segments}
    assert speakers == {"A"}


def test_keeps_speaker_above_threshold() -> None:
    # B holds 10/100 = 10% > 5% → kept.
    t = _transcript(
        _seg("A", 0.0, 90.0),
        _seg("B", 90.0, 100.0),
    )
    pruned = drop_low_talk_time_speakers(t)
    assert {s.speaker for s in pruned.segments} == {"A", "B"}


def test_threshold_is_inclusive_at_boundary() -> None:
    # B holds exactly 5% → strict-less-than means it is KEPT.
    t = _transcript(
        _seg("A", 0.0, 95.0),
        _seg("B", 95.0, 100.0),
    )
    pruned = drop_low_talk_time_speakers(t, min_fraction=0.05)
    assert {s.speaker for s in pruned.segments} == {"A", "B"}


def test_drops_all_segments_for_a_dropped_speaker() -> None:
    # B speaks twice but still totals 2% < 5% → both B segments removed.
    t = _transcript(
        _seg("A", 0.0, 49.0),
        _seg("B", 49.0, 50.0),
        _seg("A", 50.0, 99.0),
        _seg("B", 99.0, 100.0),
    )
    pruned = drop_low_talk_time_speakers(t)
    assert all(s.speaker == "A" for s in pruned.segments)
    assert len(pruned.segments) == 2


def test_zero_threshold_keeps_everyone() -> None:
    t = _transcript(
        _seg("A", 0.0, 99.0),
        _seg("B", 99.0, 100.0),
    )
    pruned = drop_low_talk_time_speakers(t, min_fraction=0.0)
    assert {s.speaker for s in pruned.segments} == {"A", "B"}


def test_does_not_mutate_input() -> None:
    t = _transcript(
        _seg("A", 0.0, 99.0),
        _seg("B", 99.0, 100.0),
    )
    before = len(t.segments)
    _ = drop_low_talk_time_speakers(t)
    assert len(t.segments) == before  # original untouched
    assert {s.speaker for s in t.segments} == {"A", "B"}


def test_empty_transcript_returns_copy_unchanged() -> None:
    t = _transcript()
    pruned = drop_low_talk_time_speakers(t)
    assert pruned is not t
    assert pruned.segments == []


@pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0, -1.0])
def test_out_of_range_threshold_rejected(bad: float) -> None:
    t = _transcript(_seg("A", 0.0, 10.0))
    with pytest.raises(ValueError):
        drop_low_talk_time_speakers(t, min_fraction=bad)


def test_returns_transcript_type_with_metadata_preserved() -> None:
    t = _transcript(_seg("A", 0.0, 100.0), episode_id="ep-keep")
    pruned = drop_low_talk_time_speakers(t)
    assert isinstance(pruned, Transcript)
    assert pruned.episode_id == "ep-keep"
    assert pruned.language == "en"
    assert pruned.duration_s == t.duration_s


def test_mock_backend_satisfies_protocol() -> None:
    # Runtime-checkable Protocol: the mock is a valid ASRBackend.
    assert isinstance(MockASRBackend(), ASRBackend)
