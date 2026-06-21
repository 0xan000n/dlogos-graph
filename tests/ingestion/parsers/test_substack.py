"""Offline tests for the Substack transcript parser (parsers/substack.py).

One parser handles the two layouts Substack renders, so both are exercised
against real (trimmed) fixtures extracted from the assigned source pages:

- ``substack.txt`` — the Doom Debates / ``lironshapira.substack.com`` *block*
  layout (speaker name, blank line, ``HH:MM:SS`` line, then text), which carries
  real per-turn timestamps the parser must read.
- ``substack_inline.txt`` — the *Your Undivided Attention* /
  ``centerforhumanetechnology.substack.com`` *inline* ``Name: text`` layout, with
  no per-turn timing, so spans are synthesized monotonically.

Stdlib + the shared schema only — the fixtures are local text files, nothing
hits the network. Fixtures were trimmed from a one-time real fetch; the committed
tests never fetch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dlogos.ingestion.parsers.substack import parse
from dlogos.schema import TranscriptSegment

_FIXTURES = Path(__file__).parent.parent.parent / "fixtures" / "transcripts"
_BLOCK_FIXTURE = _FIXTURES / "substack.txt"
_INLINE_FIXTURE = _FIXTURES / "substack_inline.txt"


def _load(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _assert_monotonic(segs: list[TranscriptSegment]) -> None:
    """Spans are well-formed and laid out non-decreasing / non-overlapping."""

    for s in segs:
        assert s.t_start >= 0.0
        assert s.t_start <= s.t_end
    for prev, nxt in zip(segs, segs[1:]):
        assert prev.t_end <= nxt.t_start


# --------------------------------------------------------------------------- #
# Block layout (Doom Debates / Liron Shapira) — real timestamps
# --------------------------------------------------------------------------- #
def test_block_parses_segments() -> None:
    segs = parse(_load(_BLOCK_FIXTURE))
    assert len(segs) >= 3
    assert all(isinstance(s, TranscriptSegment) for s in segs)


def test_block_speakers_are_real_human_names() -> None:
    segs = parse(_load(_BLOCK_FIXTURE))
    speakers = {s.speaker for s in segs}
    # The real debate participants, taken verbatim from the page (full names on
    # first reference, short names thereafter — the resolver collapses them).
    assert "Ben Goertzel" in speakers
    assert "Liron Shapira" in speakers
    assert {"Ben", "Liron"} <= speakers
    # No empty/placeholder labels.
    assert all(s.speaker.strip() for s in segs)
    assert not ({"A", "B", ""} & speakers)


def test_block_captures_speaker_change() -> None:
    segs = parse(_load(_BLOCK_FIXTURE))
    assert len({s.speaker for s in segs}) >= 2
    # Adjacent turns alternate speakers at least once (a real back-and-forth).
    assert any(a.speaker != b.speaker for a, b in zip(segs, segs[1:]))


def test_block_spans_monotonic() -> None:
    _assert_monotonic(parse(_load(_BLOCK_FIXTURE)))


def test_block_parses_real_timestamps() -> None:
    # The block layout carries native ``HH:MM:SS`` offsets; the cold open starts
    # at 00:00:00 and the captured "Headroom" turn at 00:15:17 = 917s. These are
    # parsed, not synthesized, so they land on the real values (not a word-count
    # cursor).
    segs = parse(_load(_BLOCK_FIXTURE))
    assert segs[0].t_start == pytest.approx(0.0)
    starts = {round(s.t_start) for s in segs}
    assert 917 in starts  # 00:15:17 — only reachable from a parsed timestamp
    assert max(s.t_start for s in segs) > 900  # spans the real ~16-minute range


def test_block_drops_header_and_section_headers() -> None:
    # Nav/boilerplate ("Subscribe", "Transcript", Links) before the first turn and
    # standalone section headers ("Cold Open", "Introducing Dr. Ben Goertzel") are
    # never emitted as speakers.
    segs = parse(_load(_BLOCK_FIXTURE))
    speakers = {s.speaker for s in segs}
    for junk in ("Subscribe", "Transcript", "Cold Open", "Sign in"):
        assert junk not in speakers


# --------------------------------------------------------------------------- #
# Inline layout (Your Undivided Attention / Center for Humane Technology)
# --------------------------------------------------------------------------- #
def test_inline_parses_segments() -> None:
    segs = parse(_load(_INLINE_FIXTURE))
    assert len(segs) >= 3
    assert all(isinstance(s, TranscriptSegment) for s in segs)


def test_inline_speakers_are_real_human_names() -> None:
    segs = parse(_load(_INLINE_FIXTURE))
    speakers = {s.speaker for s in segs}
    assert "Tristan Harris" in speakers
    assert {"Tim Fist", "Janet Egan"} <= speakers
    assert all(s.speaker.strip() for s in segs)
    assert not ({"A", "B", ""} & speakers)


def test_inline_captures_speaker_change() -> None:
    segs = parse(_load(_INLINE_FIXTURE))
    assert len({s.speaker for s in segs}) >= 2
    assert any(a.speaker != b.speaker for a, b in zip(segs, segs[1:]))


def test_inline_spans_monotonic() -> None:
    _assert_monotonic(parse(_load(_INLINE_FIXTURE)))


def test_inline_drops_pre_roll_and_footer() -> None:
    # The host's unlabeled cold-open narration (no "Name:" prefix) before the
    # first labeled turn is dropped, and footer nav after the last turn never
    # becomes a turn.
    segs = parse(_load(_INLINE_FIXTURE))
    speakers = {s.speaker for s in segs}
    for junk in ("Subscribe", "RECOMMENDED MEDIA", "Privacy", "Terms"):
        assert junk not in speakers


# --------------------------------------------------------------------------- #
# Cross-layout robustness
# --------------------------------------------------------------------------- #
def test_empty_text_yields_no_segments() -> None:
    assert parse("") == []
    assert parse("\n\n   \n") == []


def test_text_with_no_turns_yields_no_segments() -> None:
    # Pure boilerplate, no speaker turns at all.
    assert parse("Subscribe\nSign in\n© 2026 Center for Humane Technology\n") == []
