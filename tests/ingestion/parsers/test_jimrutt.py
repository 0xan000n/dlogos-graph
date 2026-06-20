"""Offline tests for the Jim Rutt Show transcript parser.

Reads a trimmed *real* fixture (``tests/fixtures/transcripts/jimrutt.txt``,
extracted once from the live EP 327 page) and asserts the parser recovers
ordered, name-labelled speaker turns. No network: the fixture is committed text
and the parser is pure + stdlib-only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dlogos.ingestion.parsers.jimrutt import parse
from dlogos.schema import TranscriptSegment

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "transcripts"
    / "jimrutt.txt"
)


@pytest.fixture(scope="module")
def segments() -> list[TranscriptSegment]:
    return parse(_FIXTURE.read_text(encoding="utf-8"))


def test_recovers_several_turns(segments: list[TranscriptSegment]) -> None:
    assert len(segments) >= 3


def test_speakers_are_real_human_names(
    segments: list[TranscriptSegment],
) -> None:
    speakers = {s.speaker for s in segments}
    # Real first-name labels from the EP 327 page, not empty / "A" / "B".
    assert {"Jim", "Nate"} <= speakers
    assert all(s.speaker.strip() for s in segments)
    assert "" not in speakers and "A" not in speakers and "B" not in speakers


def test_captures_a_speaker_change(
    segments: list[TranscriptSegment],
) -> None:
    speakers = [s.speaker for s in segments]
    # The opening exchange is Jim -> Nate -> Jim: an actual turn change exists.
    assert any(a != b for a, b in zip(speakers, speakers[1:]))
    assert speakers[0] == "Jim"
    assert "Nate" in speakers


def test_spans_are_monotonic_and_well_formed(
    segments: list[TranscriptSegment],
) -> None:
    prev_end = 0.0
    for s in segments:
        assert 0.0 <= s.t_start <= s.t_end
        assert s.t_start >= prev_end - 1e-9  # non-decreasing across the list
        prev_end = s.t_end


def test_boilerplate_is_dropped(
    segments: list[TranscriptSegment],
) -> None:
    blob = " ".join(s.text for s in segments).lower()
    # Leading disclaimer / header nav and trailing subscribe nav are excluded.
    assert "rough transcript which has not been revised" not in blob
    assert "more subscribe options" not in blob
    assert "skip to content" not in blob
    # The first real turn's content survived intact.
    assert segments[0].text.startswith("Quick reminder to folks")
