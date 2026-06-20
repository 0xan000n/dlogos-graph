"""Offline tests for the Dwarkesh web-transcript parser.

Reads a real (trimmed) fixture of ``www.dwarkesh.com`` extracted text — no
network — and asserts the parser recovers ordered, name-attributed,
real-timestamped speaker turns from the ``name`` / ``HH:MM:SS`` / ``utterance``
block layout, while dropping the page's nav/TOC/section-header boilerplate.
Stdlib + the shared schema only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dlogos.ingestion.parsers.dwarkesh import parse
from dlogos.schema import TranscriptSegment

_FIXTURE = Path(__file__).parents[2] / "fixtures" / "transcripts" / "dwarkesh.txt"


@pytest.fixture(scope="module")
def segments() -> list[TranscriptSegment]:
    text = _FIXTURE.read_text(encoding="utf-8")
    return parse(text)


def test_yields_several_segments(segments: list[TranscriptSegment]) -> None:
    assert len(segments) >= 3


def test_speakers_are_real_human_names(segments: list[TranscriptSegment]) -> None:
    names = {s.speaker for s in segments}
    # The real human names, exactly as written — not diarization labels/empties.
    assert "Dwarkesh Patel" in names
    assert "Andrej Karpathy" in names
    for s in segments:
        assert s.speaker.strip(), "every segment carries a non-empty speaker"
        assert s.speaker not in {"A", "B", "Speaker", "SPEAKER_00"}


def test_first_turn_is_the_host_opening(segments: list[TranscriptSegment]) -> None:
    # Pre-transcript boilerplate (title, byline, "Timestamps" TOC, section
    # header) is dropped: the first real turn is the host's open at 00:00:00.
    first = segments[0]
    assert first.speaker == "Dwarkesh Patel"
    assert first.t_start == 0.0
    assert "speaking with" in first.text


def test_captures_a_speaker_change(segments: list[TranscriptSegment]) -> None:
    speakers = [s.speaker for s in segments]
    assert any(
        a != b for a, b in zip(speakers, speakers[1:], strict=False)
    ), "at least one adjacent speaker change is captured"


def test_real_timestamps_are_parsed(segments: list[TranscriptSegment]) -> None:
    # HH:MM:SS offsets are parsed, not synthesized: the second turn starts at
    # 00:00:07 = 7s exactly (Karpathy's first reply in the source).
    assert segments[0].t_start == 0.0
    assert segments[1].t_start == 7.0
    assert segments[1].speaker == "Andrej Karpathy"
    # A later turn carries a minutes-scale real offset (00:12:15 = 735s),
    # which monotonic word-count synthesis could not have reached on this
    # short slice — proof the source timestamps drive the spans.
    assert any(s.t_start == 735.0 for s in segments)


def test_spans_are_monotonic_and_well_formed(
    segments: list[TranscriptSegment],
) -> None:
    prev_start = -1.0
    for s in segments:
        assert 0.0 <= s.t_start <= s.t_end
        assert s.t_start >= prev_start, "starts are non-decreasing"
        prev_start = s.t_start


def test_boilerplate_is_excluded(segments: list[TranscriptSegment]) -> None:
    # The TOC label text and section-header label never become their own turns.
    joined_speakers = " ".join(s.speaker for s in segments)
    assert "Timestamps" not in joined_speakers
    assert "Transcript" not in joined_speakers
