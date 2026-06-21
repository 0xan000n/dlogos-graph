"""Offline tests for the TGS (The Great Simplification) PDF-transcript parser.

Reads a trimmed but *real* extraction of TGS-214 (Tristan Harris) from
``tests/fixtures/transcripts/tgs.txt`` — no network, no PDF deps, stdlib only.
The fixture preserves the source's quirks (``[HH:MM:SS] Name:`` turn labels, bare
continuation timestamps mid-turn, the spliced ``"N The Great Simplification"``
page footer, ``fi``-ligature artifacts) so the parser is exercised on the shape
the live fetch layer actually produces.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dlogos.ingestion.parsers.tgs import parse
from dlogos.schema import TranscriptSegment

_FIXTURE = Path(__file__).parent.parent.parent / "fixtures" / "transcripts" / "tgs.txt"


@pytest.fixture(scope="module")
def segments() -> list[TranscriptSegment]:
    return parse(_FIXTURE.read_text(encoding="utf-8"))


def test_parses_multiple_segments(segments: list[TranscriptSegment]) -> None:
    assert len(segments) >= 3
    assert all(isinstance(s, TranscriptSegment) for s in segments)


def test_speakers_are_real_human_names(segments: list[TranscriptSegment]) -> None:
    speakers = {s.speaker for s in segments}
    # The real human names appear exactly as in the source — not diarization
    # labels, not empty, not single letters.
    assert "Tristan Harris" in speakers
    assert "Nate Hagens" in speakers
    assert all(s.speaker.strip() for s in segments)
    assert all(len(s.speaker) > 2 for s in segments)


def test_speaker_change_is_captured(segments: list[TranscriptSegment]) -> None:
    assert any(a.speaker != b.speaker for a, b in zip(segments, segments[1:]))


def test_real_timestamps_are_parsed(segments: list[TranscriptSegment]) -> None:
    # TGS carries native [HH:MM:SS] turn-starts; the first turn opens at 0:00 and
    # later turns at their real offsets (e.g. [00:00:47] Nate Hagens -> 47s).
    assert segments[0].t_start == pytest.approx(0.0)
    assert any(s.t_start == pytest.approx(47.0) for s in segments)
    # Not every turn collapsed to the synthesized pace: a real >1s jump exists.
    assert any(b.t_start - a.t_start > 1.0 for a, b in zip(segments, segments[1:]))


def test_spans_are_monotonic_and_non_overlapping(
    segments: list[TranscriptSegment],
) -> None:
    for a, b in zip(segments, segments[1:]):
        assert a.t_start <= b.t_start
        assert a.t_end <= b.t_start  # non-overlapping
    assert all(0.0 <= s.t_start <= s.t_end for s in segments)


def test_continuation_timestamps_folded_not_split(
    segments: list[TranscriptSegment],
) -> None:
    # Bare [HH:MM:SS] continuation markers stay inside their turn and are stripped
    # from the body — they must not appear as text or spawn extra turns.
    assert not any("[00:" in s.text for s in segments)
    # The opening Tristan turn absorbed its [00:00:18] continuation, so it runs up
    # to Nate's real next start rather than ending early.
    assert segments[0].speaker == "Tristan Harris"
    assert segments[0].t_end == pytest.approx(47.0)


def test_page_footer_boilerplate_is_scrubbed(
    segments: list[TranscriptSegment],
) -> None:
    assert not any("Great Simplification" in s.text for s in segments)
    assert not any("Great Simpliﬁcation" in s.text for s in segments)


def test_leading_header_dropped(segments: list[TranscriptSegment]) -> None:
    # The auto-generated "PLEASE NOTE ..." header precedes the first turn and must
    # not leak into the first segment.
    assert "PLEASE NOTE" not in segments[0].text
    assert "auto-generated" not in segments[0].text


def test_empty_and_boilerplate_only_input() -> None:
    assert parse("") == []
    assert parse("Just a page of nav text with no speaker turns.") == []
