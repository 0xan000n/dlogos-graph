"""Tests for transcript chunking: overlap, bounds, and segment atomicity."""

from __future__ import annotations

import pytest

from dlogos.extraction.chunking import (
    DEFAULT_MAX_CHARS,
    Chunk,
    chunk_transcript,
)
from dlogos.schema import Transcript, TranscriptSegment


def _transcript(segments: list[TranscriptSegment], episode_id: str = "ep-x") -> Transcript:
    duration = max((s.t_end for s in segments), default=0.0)
    return Transcript(
        episode_id=episode_id,
        language="en",
        segments=segments,
        duration_s=duration,
    )


def _seg(speaker: str, text: str, t_start: float, t_end: float) -> TranscriptSegment:
    return TranscriptSegment(speaker=speaker, text=text, t_start=t_start, t_end=t_end)


def test_empty_transcript_yields_no_chunks(synthetic_transcript: Transcript) -> None:
    empty = _transcript([])
    assert chunk_transcript(empty) == []


def test_short_transcript_is_one_chunk(synthetic_transcript: Transcript) -> None:
    chunks = chunk_transcript(synthetic_transcript)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.episode_id == synthetic_transcript.episode_id
    # All segments present, in order, indices preserved.
    assert [cs.index for cs in chunk.segments] == list(
        range(len(synthetic_transcript.segments))
    )
    assert chunk.t_start == synthetic_transcript.segments[0].t_start
    assert chunk.t_end == synthetic_transcript.segments[-1].t_end


def test_render_carries_speaker_labels_and_timestamps(
    synthetic_transcript: Transcript,
) -> None:
    chunk = chunk_transcript(synthetic_transcript)[0]
    rendered = chunk.render()
    # Every speaker label present in the prompt body.
    for seg in synthetic_transcript.segments:
        assert seg.speaker in rendered
    # Timestamp markers present.
    assert "[0.00-4.50]" in rendered
    assert "SPEAKER_01:" in rendered


def test_splits_into_multiple_chunks_when_over_budget() -> None:
    # 10 segments, each ~ "SPEAKER_00: " + 40 'x' -> force a tiny max_chars.
    segs = [
        _seg("SPEAKER_00", "x" * 40, float(i), float(i) + 1.0) for i in range(10)
    ]
    chunks = chunk_transcript(_transcript(segs), max_chars=120, overlap_segments=0)
    assert len(chunks) > 1
    # Concatenated (dedup of overlap=0) covers all original indices once.
    seen = [cs.index for ch in chunks for cs in ch.segments]
    assert sorted(set(seen)) == list(range(10))
    assert len(seen) == 10  # no overlap duplication


def test_overlap_repeats_tail_segments() -> None:
    # Lines render to ~64 chars each; budget fits ~3 segments per chunk so
    # overlap=1 has room to repeat a whole tail segment.
    segs = [
        _seg("SPEAKER_00", "y" * 40, float(i), float(i) + 1.0) for i in range(8)
    ]
    chunks = chunk_transcript(_transcript(segs), max_chars=200, overlap_segments=1)
    assert len(chunks) >= 2
    # Every chunk holds at least 2 segments, so overlap is meaningful.
    assert all(len(c.segments) >= 2 for c in chunks)
    # The last segment index of chunk k reappears as the first of chunk k+1.
    for a, b in zip(chunks, chunks[1:]):
        assert a.segments[-1].index == b.segments[0].index


def test_overlap_two_segments() -> None:
    segs = [
        _seg("SPEAKER_00", "z" * 30, float(i), float(i) + 1.0) for i in range(12)
    ]
    chunks = chunk_transcript(_transcript(segs), max_chars=160, overlap_segments=2)
    assert len(chunks) >= 2
    for a, b in zip(chunks, chunks[1:]):
        # Last two of chunk a are the first two of chunk b (when chunk a long enough).
        tail = [cs.index for cs in a.segments[-2:]]
        head = [cs.index for cs in b.segments[: len(tail)]]
        # Overlap may be clamped to chunk_span-1, so head starts at tail's start.
        assert head[0] == tail[-1] or head[0] == tail[0]


def test_never_splits_mid_segment() -> None:
    segs = [
        _seg("SPEAKER_00", "w" * 50, float(i), float(i) + 1.0) for i in range(6)
    ]
    chunks = chunk_transcript(_transcript(segs), max_chars=130, overlap_segments=0)
    # Each chunk's segments echo whole original segment text (never truncated).
    originals = {i: segs[i].text for i in range(len(segs))}
    for ch in chunks:
        for cs in ch.segments:
            assert cs.text == originals[cs.index]


def test_oversized_single_segment_becomes_its_own_chunk() -> None:
    big_text = "q" * 500
    segs = [
        _seg("SPEAKER_00", "short", 0.0, 1.0),
        _seg("SPEAKER_01", big_text, 1.0, 30.0),
        _seg("SPEAKER_00", "short again", 30.0, 31.0),
    ]
    chunks = chunk_transcript(_transcript(segs), max_chars=100, overlap_segments=0)
    # The oversized segment is not dropped; it appears somewhere intact.
    all_texts = [cs.text for ch in chunks for cs in ch.segments]
    assert big_text in all_texts
    # And it is alone in its chunk (over budget but un-split).
    big_chunk = next(
        ch for ch in chunks if any(cs.text == big_text for cs in ch.segments)
    )
    assert len(big_chunk.segments) == 1


def test_chunk_bounds_match_segment_extents() -> None:
    segs = [
        _seg("SPEAKER_00", "a" * 30, 0.0, 5.0),
        _seg("SPEAKER_01", "b" * 30, 5.0, 11.0),
        _seg("SPEAKER_00", "c" * 30, 11.0, 18.0),
        _seg("SPEAKER_01", "d" * 30, 18.0, 25.0),
    ]
    chunks = chunk_transcript(_transcript(segs), max_chars=90, overlap_segments=0)
    for ch in chunks:
        assert ch.t_start == min(cs.t_start for cs in ch.segments)
        assert ch.t_end == max(cs.t_end for cs in ch.segments)
        assert ch.t_start <= ch.t_end


def test_chunk_index_is_monotonic_from_zero() -> None:
    segs = [
        _seg("SPEAKER_00", "m" * 40, float(i), float(i) + 1.0) for i in range(9)
    ]
    chunks = chunk_transcript(_transcript(segs), max_chars=120, overlap_segments=1)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_default_max_chars_keeps_typical_transcript_intact(
    synthetic_transcript: Transcript,
) -> None:
    chunks = chunk_transcript(synthetic_transcript, max_chars=DEFAULT_MAX_CHARS)
    assert len(chunks) == 1


def test_invalid_params_raise() -> None:
    t = _transcript([_seg("SPEAKER_00", "x", 0.0, 1.0)])
    with pytest.raises(ValueError):
        chunk_transcript(t, max_chars=0)
    with pytest.raises(ValueError):
        chunk_transcript(t, overlap_segments=-1)


def test_chunking_terminates_with_large_overlap() -> None:
    # overlap >= chunk length must still make forward progress (no infinite loop).
    segs = [
        _seg("SPEAKER_00", "p" * 40, float(i), float(i) + 1.0) for i in range(6)
    ]
    chunks = chunk_transcript(_transcript(segs), max_chars=110, overlap_segments=99)
    assert len(chunks) >= 1
    # Last chunk reaches the final segment.
    assert chunks[-1].segments[-1].index == 5
