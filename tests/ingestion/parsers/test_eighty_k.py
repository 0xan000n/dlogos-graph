"""Offline tests for the 80,000 Hours transcript parser.

Reads a real (but trimmed) fixture extracted from an 80000hours.org episode
page — no network. Asserts the parser recovers ordered, name-attributed,
real-timestamp-anchored segments from this source's ``Name:`` + chapter-header
shape.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dlogos.ingestion.parsers.eighty_k import parse
from dlogos.schema import TranscriptSegment

FIXTURE = Path(__file__).parents[2] / "fixtures" / "transcripts" / "eighty_k.txt"


@pytest.fixture(scope="module")
def segments() -> list[TranscriptSegment]:
    text = FIXTURE.read_text(encoding="utf-8")
    return parse(text)


def test_parses_multiple_segments(segments: list[TranscriptSegment]) -> None:
    assert len(segments) >= 3


def test_speakers_are_real_human_names(segments: list[TranscriptSegment]) -> None:
    speakers = {s.speaker for s in segments}
    # The real human names from this episode — not empty, not a diarization
    # label like "A"/"SPEAKER_00".
    assert "Rob Wiblin" in speakers
    assert "Ryan Greenblatt" in speakers
    for spk in speakers:
        assert spk.strip()
        assert spk not in {"A", "B"}
        assert not spk.startswith("SPEAKER_")
        # A real "First Last" name, not a stray label.
        assert " " in spk


def test_captures_a_speaker_change(segments: list[TranscriptSegment]) -> None:
    speakers_in_order = [s.speaker for s in segments]
    assert len(set(speakers_in_order)) >= 2
    # An actual adjacent change, not just two distinct speakers far apart.
    assert any(a != b for a, b in zip(speakers_in_order, speakers_in_order[1:]))


def test_spans_are_monotonic_and_non_overlapping(
    segments: list[TranscriptSegment],
) -> None:
    prev_end = 0.0
    for seg in segments:
        assert 0.0 <= seg.t_start <= seg.t_end
        assert seg.t_start >= prev_end - 1e-9  # non-decreasing, no overlap
        prev_end = seg.t_end


def test_real_chapter_timestamps_are_parsed(
    segments: list[TranscriptSegment],
) -> None:
    # The fixture's chapter headers carry real offsets: ``[00:00:00]`` anchors
    # the first turn at 0.0; ``[00:01:10]`` and ``[00:05:15]`` (= 70s and 315s)
    # anchor later turns. These exact offsets must appear as segment starts,
    # proving real timestamps were parsed rather than purely synthesized.
    assert segments[0].t_start == pytest.approx(0.0)
    starts = {round(s.t_start, 2) for s in segments}
    assert 70.0 in starts  # [00:01:10]
    assert 315.0 in starts  # [00:05:15]


def test_text_is_clean_and_nonempty(segments: list[TranscriptSegment]) -> None:
    for seg in segments:
        assert seg.text.strip()
        # Speaker labels and chapter headers must not leak into the spoken text.
        assert "[00:" not in seg.text
        assert not seg.text.endswith(":")
