"""Offline tests for the Singju Post transcript parser.

Reads a trimmed *real* fixture (``tests/fixtures/transcripts/singjupost.txt``,
extracted once from the live Making Sense #469 w/ Tristan Harris page) and
asserts the parser recovers ordered, name-labelled speaker turns, drops the
interleaved section sub-heads and the surrounding site chrome, and lays down
monotonic spans. No network: the fixture is committed text and the parser is
pure + stdlib-only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dlogos.ingestion.parsers.singjupost import parse
from dlogos.schema import TranscriptSegment

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "transcripts"
    / "singjupost.txt"
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
    # The two real ALL-CAPS name labels from the Making Sense #469 page,
    # not empty / "A" / "B" / site-chrome labels like "TRANSCRIPT".
    assert {"SAM HARRIS", "TRISTAN HARRIS"} <= speakers
    assert all(s.speaker.strip() for s in segments)
    assert "" not in speakers and "A" not in speakers and "B" not in speakers
    assert "TRANSCRIPT" not in speakers


def test_captures_a_speaker_change(
    segments: list[TranscriptSegment],
) -> None:
    speakers = [s.speaker for s in segments]
    # The opening exchange alternates Sam <-> Tristan: a real turn change exists.
    assert any(a != b for a, b in zip(speakers, speakers[1:]))
    assert speakers[0] == "SAM HARRIS"
    assert "TRISTAN HARRIS" in speakers


def test_spans_are_monotonic_and_well_formed(
    segments: list[TranscriptSegment],
) -> None:
    prev_end = 0.0
    for s in segments:
        assert 0.0 <= s.t_start <= s.t_end
        assert s.t_start >= prev_end - 1e-9  # non-decreasing across the list
        prev_end = s.t_end


def test_section_headings_and_boilerplate_dropped(
    segments: list[TranscriptSegment],
) -> None:
    blob = " ".join(s.text for s in segments)
    # Editorial section sub-heads interleaved between turns are not speech.
    assert "From Social Media to AI" not in blob
    assert "The AI Dilemma: Two Choices" not in blob
    assert "Predicting the Future: Incentives and Outcomes" not in blob
    # Leading page chrome (the "TRANSCRIPT:" marker line) is excluded, and the
    # first real turn's content survived intact.
    assert "TRANSCRIPT" not in blob
    assert segments[0].text.startswith("I’m here with Tristan Harris")


def test_continuation_paragraphs_join_their_turn(
    segments: list[TranscriptSegment],
) -> None:
    # A turn whose later paragraphs arrived as separate label-less lines should
    # be a single segment carrying all of them — not split, not heading-polluted.
    joined = next(
        s for s in segments if "But to get to your question" in s.text
    )
    assert joined.speaker == "TRISTAN HARRIS"
    assert "Aza Raskin" in joined.text  # a later paragraph of the same turn
    assert "Predicting the Future" not in joined.text  # heading not glued in
