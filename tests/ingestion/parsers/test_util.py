"""Tests for the shared, stdlib-only parser helpers (parsers/_util.py).

These are the contract the per-source parser agents build on, so each helper is
exercised directly: timestamp parsing across formats, monotonic span synthesis,
explicit-timestamp finalization, ordering invariants, and stage-direction
cleanup. Stdlib + the shared schema only — no network, no fixtures, offline.
"""

from __future__ import annotations

import pytest

from dlogos.ingestion.parsers._util import (
    SEC_PER_WORD,
    clean_text,
    finalize_segments,
    parse_hms,
    synthesize_spans,
)
from dlogos.schema import TranscriptSegment


# --------------------------------------------------------------------------- #
# parse_hms
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0:00", 0.0),
        ("00:30", 30.0),
        ("1:30", 90.0),  # MM:SS
        ("3:04", 184.0),  # MM:SS, not H:MM
        ("1:03:04", 3784.0),  # H:MM:SS
        ("01:02:03", 3723.0),
        ("[00:01:23]", 83.0),  # bracket wrapper
        ("(2:05)", 125.0),  # paren wrapper
        ("12:34.5", 754.5),  # fractional seconds
        ("Tristan Harris [00:10:00]: hello", 600.0),  # embedded in a line
    ],
)
def test_parse_hms_formats(raw: str, expected: float) -> None:
    assert parse_hms(raw) == pytest.approx(expected)


@pytest.mark.parametrize(
    "raw",
    ["", "no timestamp here", "Jim Rutt:", "abc:def", "1234", "year 2024"],
)
def test_parse_hms_no_match_returns_none(raw: str) -> None:
    assert parse_hms(raw) is None


def test_parse_hms_first_match_wins() -> None:
    assert parse_hms("00:10 then 00:20") == pytest.approx(10.0)


def test_parse_hms_does_not_raise_on_garbage() -> None:
    # Total over arbitrary input — must never raise.
    for s in ["::", ":", "99:99:99:99", "\n\t", ":::abc"]:
        assert parse_hms(s) is None or isinstance(parse_hms(s), float)


# --------------------------------------------------------------------------- #
# synthesize_spans
# --------------------------------------------------------------------------- #
def test_synthesize_spans_monotonic_and_typed() -> None:
    turns = [
        ("Jim Rutt", "one two three four"),  # 4 words
        ("Tristan Harris", "five six"),  # 2 words
        ("Jim Rutt", "seven"),  # 1 word
    ]
    segs = synthesize_spans(turns)

    assert all(isinstance(s, TranscriptSegment) for s in segs)
    assert [s.speaker for s in segs] == ["Jim Rutt", "Tristan Harris", "Jim Rutt"]

    # First span starts at 0 and is laid end-to-end (non-overlapping).
    assert segs[0].t_start == 0.0
    assert segs[0].t_end == pytest.approx(4 * SEC_PER_WORD)
    assert segs[1].t_start == segs[0].t_end
    assert segs[1].t_end == pytest.approx(segs[1].t_start + 2 * SEC_PER_WORD)
    assert segs[2].t_start == segs[1].t_end


def test_synthesize_spans_ordering_invariant() -> None:
    turns = [("S", "word " * (i + 1)) for i in range(10)]
    segs = synthesize_spans(turns)
    for prev, nxt in zip(segs, segs[1:]):
        assert prev.t_end <= nxt.t_start  # non-decreasing, non-overlapping
        assert prev.t_start <= prev.t_end


def test_synthesize_spans_custom_pace() -> None:
    segs = synthesize_spans([("A", "two words")], sec_per_word=1.0)
    assert segs[0].t_end == pytest.approx(2.0)


def test_synthesize_spans_drops_empty_turns() -> None:
    segs = synthesize_spans(
        [("A", "hello"), ("B", "   "), ("C", "[music]"), ("D", "world")]
    )
    assert [s.speaker for s in segs] == ["A", "D"]


# --------------------------------------------------------------------------- #
# finalize_segments
# --------------------------------------------------------------------------- #
def test_finalize_honors_explicit_starts() -> None:
    rows = [
        ("Lex", "intro words here", 0.0),
        ("Guest", "a reply", 10.0),
        ("Lex", "closing", 25.0),
    ]
    segs = finalize_segments(rows)
    assert [s.t_start for s in segs] == [0.0, 10.0, 25.0]
    # A timestamped turn runs up to the next explicit start.
    assert segs[0].t_end == 10.0
    assert segs[1].t_end == 25.0
    # The last (no following timestamp) synthesizes from its word count.
    assert segs[2].t_end == pytest.approx(25.0 + 1 * SEC_PER_WORD)


def test_finalize_fills_missing_starts_monotonically() -> None:
    rows = [
        ("A", "one two", None),  # synth start 0
        ("B", "three", None),  # synth start at A's end; stretches up to C=10
        ("C", "four five six", 10.0),  # explicit jump to 10, 3 words -> end 11.2
        ("D", "seven", None),  # synth, continues from C's end (11.2)
    ]
    segs = finalize_segments(rows)
    assert segs[0].t_start == 0.0
    assert segs[1].t_start == pytest.approx(2 * SEC_PER_WORD)
    # B is the turn immediately before the explicit timestamp, so it fills the
    # gap up to it (no hole before the timestamped turn).
    assert segs[1].t_end == 10.0
    assert segs[2].t_start == 10.0
    assert segs[2].t_end == pytest.approx(10.0 + 3 * SEC_PER_WORD)
    # D resumes from the running cursor (C's end), not the explicit start.
    assert segs[3].t_start == pytest.approx(10.0 + 3 * SEC_PER_WORD)
    # Global ordering invariant.
    for prev, nxt in zip(segs, segs[1:]):
        assert prev.t_end <= nxt.t_start


def test_finalize_clamps_backwards_timestamps() -> None:
    # A stray backwards timestamp must not produce a backwards/negative span.
    rows = [
        ("A", "first", 100.0),
        ("B", "second goes backwards", 5.0),  # earlier than prior cursor
        ("C", "third", None),
    ]
    segs = finalize_segments(rows)
    for prev, nxt in zip(segs, segs[1:]):
        assert prev.t_start <= nxt.t_start  # non-decreasing despite the stray
    assert all(s.t_start <= s.t_end for s in segs)
    assert all(s.t_start >= 0.0 for s in segs)


def test_finalize_drops_empty_after_clean() -> None:
    rows = [("A", "real", 0.0), ("B", "[laughs]", 1.0), ("C", "more", 2.0)]
    segs = finalize_segments(rows)
    assert [s.speaker for s in segs] == ["A", "C"]


def test_finalize_empty_input() -> None:
    assert finalize_segments([]) == []


def test_finalize_equal_timestamps_never_collapse() -> None:
    rows = [("A", "x", 5.0), ("B", "y", 5.0)]
    segs = finalize_segments(rows)
    assert segs[0].t_end > segs[0].t_start
    assert segs[1].t_start >= segs[0].t_end


# --------------------------------------------------------------------------- #
# clean_text
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("hello   world", "hello world"),
        ("line one\nline two", "line one line two"),
        ("  padded  ", "padded"),
        ("speech [laughs] more", "speech more"),
        ("[music] starts now", "starts now"),
        ("text [inaudible 00:12] resumes", "text resumes"),
        ("ha (laughs) ha", "ha ha"),
        ("clap (applause) clap", "clap clap"),
        ("[laughs]", ""),
        ("   ", ""),
    ],
)
def test_clean_text(raw: str, expected: str) -> None:
    assert clean_text(raw) == expected


def test_clean_text_preserves_legit_parentheticals() -> None:
    # A real aside is NOT a stage direction — keep it.
    assert clean_text("the model (more on that later) is large") == (
        "the model (more on that later) is large"
    )


def test_clean_text_collapses_tabs_and_newlines() -> None:
    assert clean_text("a\t\tb\n\nc") == "a b c"
