"""Offline tests for the lexfridman.com web-transcript parser.

Reads a real (trimmed) fixture of ``lexfridman.com`` extracted text — no
network — and asserts the parser recovers ordered, name-attributed,
real-timestamped speaker turns from the page's *multi-line* block layout
(``Name`` / ``(HH:MM:SS)`` / text each on their own line), inherits the speaker
across nameless continuation blocks of a monologue, and drops the chapter
headings / pre-transcript chrome. Stdlib + the shared schema only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dlogos.ingestion.parsers.lexfridman import parse
from dlogos.schema import TranscriptSegment

_FIXTURE = Path(__file__).parents[2] / "fixtures" / "transcripts" / "lexfridman.txt"


@pytest.fixture(scope="module")
def segments() -> list[TranscriptSegment]:
    text = _FIXTURE.read_text(encoding="utf-8")
    return parse(text)


def test_yields_several_segments(segments: list[TranscriptSegment]) -> None:
    # The multi-line block layout must yield many turns, not zero (the prior
    # single-line assumption produced 0 on the real page).
    assert len(segments) >= 5


def test_speakers_are_real_human_names(segments: list[TranscriptSegment]) -> None:
    names = {s.speaker for s in segments}
    # The real human names, exactly as written — not diarization labels/empties.
    assert "Lex Fridman" in names
    assert "Demis Hassabis" in names
    for s in segments:
        assert s.speaker.strip(), "every segment carries a non-empty speaker"
        assert s.speaker not in {"A", "B", "Speaker", "SPEAKER_00"}


def test_first_turn_is_the_host_opening(segments: list[TranscriptSegment]) -> None:
    # Pre-transcript chrome (page title, bio, "Menu", nav) is dropped: the first
    # real turn is Lex at 00:00:00.
    first = segments[0]
    assert first.speaker == "Lex Fridman"
    assert first.t_start == 0.0
    assert "hard for us humans" in first.text


def test_captures_a_speaker_change(segments: list[TranscriptSegment]) -> None:
    speakers = [s.speaker for s in segments]
    assert any(
        a != b for a, b in zip(speakers, speakers[1:], strict=False)
    ), "at least one adjacent speaker change is captured"
    # Specifically the Lex -> Demis handoff at the top.
    assert speakers[0] == "Lex Fridman"
    assert speakers[1] == "Demis Hassabis"


def test_continuation_block_inherits_speaker(
    segments: list[TranscriptSegment],
) -> None:
    # The third block has NO name line in the source (its timestamp line
    # "(00:00:27)" is preceded directly by the previous turn's text) — it is a
    # continuation of Demis's monologue and must inherit his name rather than
    # become an empty/None speaker or steal a heading.
    third = segments[2]
    assert third.speaker == "Demis Hassabis"
    assert third.t_start == 27.0


def test_real_timestamps_are_parsed(segments: list[TranscriptSegment]) -> None:
    # HH:MM:SS offsets are parsed, not synthesized: the second turn starts at
    # 00:00:12 = 12s exactly (Demis's first reply in the source).
    assert segments[0].t_start == 0.0
    assert segments[1].t_start == 12.0
    assert segments[1].speaker == "Demis Hassabis"
    # A later turn carries a minutes-scale real offset (00:08:53 = 533s), which
    # monotonic word-count synthesis could not have reached on this slice —
    # proof the source timestamps drive the spans.
    assert any(s.t_start == 533.0 for s in segments)


def test_spans_are_monotonic_and_well_formed(
    segments: list[TranscriptSegment],
) -> None:
    prev_start = -1.0
    for s in segments:
        assert 0.0 <= s.t_start <= s.t_end
        assert s.t_start >= prev_start, "starts are non-decreasing"
        prev_start = s.t_start


def test_headings_and_chrome_are_excluded(
    segments: list[TranscriptSegment],
) -> None:
    # Chapter headings ("Episode highlight", "Introduction", "Learnable patterns
    # in nature") and the page title never become their own turns or speakers.
    joined_speakers = " ".join(s.speaker for s in segments)
    for noise in ("Transcript", "Episode highlight", "Introduction"):
        assert noise not in joined_speakers
    for s in segments:
        assert s.speaker in {"Lex Fridman", "Demis Hassabis"}
