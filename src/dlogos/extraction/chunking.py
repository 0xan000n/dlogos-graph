"""Overlapping, speaker-labelled chunking of a transcript (§7.4).

The extractor has a bounded context, so a long episode transcript must be
split into windows. Two invariants matter for correctness:

- **Never split mid-segment.** A diarized segment is one utterance by one
  speaker; cutting it in half would strip the speaker label off the tail and
  invite misattribution. Segments are therefore the atomic unit — a chunk is
  always a contiguous run of *whole* segments.
- **Overlap by whole segments.** Claims often span a turn boundary (a question
  in one segment, the answer in the next). Carrying the last ``overlap_segments``
  segments of one chunk into the head of the next means a boundary-straddling
  claim is visible to at least one chunk in full. Overlap is expressed in
  segments, not characters, so it too respects segment atomicity.

Each :class:`Chunk` carries the speaker-labelled text (what goes into the
prompt) plus the ``[t_start, t_end]`` audio span it covers, which the extractor
uses to bound every emitted ``source_span`` (§6).

A single oversized segment (longer than ``max_chars`` on its own) is **not**
dropped or split: it becomes its own chunk. Losing the claims in a long
monologue would be worse than handing the model a slightly over-budget window,
and downstream the span still points at the real segment timestamps.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from dlogos.schema import Transcript, TranscriptSegment

# Default window budget, in characters of speaker-labelled text. Conservative
# relative to a typical open-weight context so the prompt scaffold + response
# also fit. Callers may override per model.
DEFAULT_MAX_CHARS = 6000
# Default number of whole segments to carry from the tail of one chunk into the
# head of the next, so a claim spanning a turn boundary is seen intact.
DEFAULT_OVERLAP_SEGMENTS = 1


class ChunkSegment(BaseModel):
    """One segment as it appears inside a chunk.

    A thin echo of :class:`~dlogos.schema.TranscriptSegment` carrying the
    fields the prompt and span-bounding need. ``index`` is the segment's
    position in the *original* transcript, so overlap regions can be identified
    and a chunk's provenance is unambiguous.
    """

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0, description="Position in the original transcript.")
    speaker: str = Field(description="Per-episode diarization label.")
    text: str
    t_start: float = Field(ge=0.0)
    t_end: float = Field(ge=0.0)


class Chunk(BaseModel):
    """A contiguous run of whole segments fed to the extractor as one prompt."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    chunk_index: int = Field(ge=0)
    segments: list[ChunkSegment] = Field(min_length=1)
    t_start: float = Field(ge=0.0, description="Earliest segment start in the chunk.")
    t_end: float = Field(ge=0.0, description="Latest segment end in the chunk.")

    @property
    def char_len(self) -> int:
        """Length of the rendered, speaker-labelled prompt body."""

        return len(self.render())

    def render(self) -> str:
        """Speaker-labelled, timestamped text block for the prompt.

        One line per segment::

            [12.50-19.50] SPEAKER_01: OpenAI is moving fast...

        Carrying the per-segment timestamps lets the model anchor each claim's
        ``source_span`` to a real segment rather than guessing.
        """

        lines = [
            f"[{s.t_start:.2f}-{s.t_end:.2f}] {s.speaker}: {s.text}"
            for s in self.segments
        ]
        return "\n".join(lines)


def _segment_render_len(seg: TranscriptSegment, index: int) -> int:
    """Rendered length of a single segment line (matches :meth:`Chunk.render`)."""

    return len(f"[{seg.t_start:.2f}-{seg.t_end:.2f}] {seg.speaker}: {seg.text}")


def _to_chunk_segment(seg: TranscriptSegment, index: int) -> ChunkSegment:
    return ChunkSegment(
        index=index,
        speaker=seg.speaker,
        text=seg.text,
        t_start=seg.t_start,
        t_end=seg.t_end,
    )


def chunk_transcript(
    transcript: Transcript,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_segments: int = DEFAULT_OVERLAP_SEGMENTS,
) -> list[Chunk]:
    """Split ``transcript`` into overlapping, speaker-labelled chunks.

    Parameters
    ----------
    transcript:
        The diarized episode transcript to window.
    max_chars:
        Soft upper bound on the rendered character length of a chunk. A chunk
        accumulates whole segments until adding the next would exceed this; the
        next segment then opens a fresh chunk. The bound is *soft*: a single
        segment longer than ``max_chars`` becomes its own (over-budget) chunk
        rather than being split or dropped.
    overlap_segments:
        How many whole segments from the tail of a chunk are repeated at the
        head of the following chunk. ``0`` disables overlap. Clamped to be
        non-negative and strictly less than the producing chunk's length, so
        overlap can never cause non-advancing / infinite chunking.

    Returns
    -------
    list[Chunk]
        Ordered chunks. Empty if the transcript has no segments.
    """

    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_segments < 0:
        raise ValueError("overlap_segments must be non-negative")

    segments = transcript.segments
    n = len(segments)
    if n == 0:
        return []

    chunks: list[Chunk] = []
    start = 0  # index of the first segment in the current chunk
    chunk_index = 0

    while start < n:
        # Greedily accumulate whole segments until the next would overflow.
        cur_len = 0
        end = start  # exclusive end index
        while end < n:
            seg_len = _segment_render_len(segments[end], end)
            sep = 1 if end > start else 0  # newline join cost
            prospective = cur_len + sep + seg_len
            if end > start and prospective > max_chars:
                break
            cur_len = prospective
            end += 1
            # An oversized lone segment fills exactly one chunk on its own.
            if cur_len > max_chars:
                break

        # Guarantee forward progress: at least one segment per chunk.
        if end == start:
            end = start + 1

        window = segments[start:end]
        chunk_segs = [_to_chunk_segment(seg, start + i) for i, seg in enumerate(window)]
        chunks.append(
            Chunk(
                episode_id=transcript.episode_id,
                chunk_index=chunk_index,
                segments=chunk_segs,
                t_start=min(s.t_start for s in chunk_segs),
                t_end=max(s.t_end for s in chunk_segs),
            )
        )
        chunk_index += 1

        if end >= n:
            break

        # Advance, carrying back up to overlap_segments whole segments — but
        # never so far that the next chunk fails to advance past this one.
        chunk_span = end - start
        effective_overlap = min(overlap_segments, max(0, chunk_span - 1))
        start = end - effective_overlap

    return chunks
