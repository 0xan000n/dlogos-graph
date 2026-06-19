"""The deliberate weak baseline: dumb top-k vector RAG over RAW transcripts.

This is the *naive-RAG arm's* retriever (spec §9 arm 3), and it exists to make
the headline comparison honest. The dLogos arm and the naive arm must run over
the **same source data** (the ~200-pod transcripts) so that "graph structure
beats dumb retrieval" is isolatable: if the structure didn't help, the two arms
score the same and dLogos *loses* the credibility argument.

To keep that isolation real, this module shares **nothing** with the graph
retrieval path:

- It indexes **raw transcript chunks** (windows of diarized segments), not
  reified Claims. There is no extraction, no controlled-vocabulary predicate, no
  subject resolution feeding it.
- There is **no graph**: no store, no edges, no traversal, no neighborhood
  expansion.
- There is **no temporal model**: no bitemporal validity windows, no
  event-time filter, no "as-of" query.
- There is **no stance / sentiment / consensus** synthesis: a chunk is just
  text plus where it came from.

Retrieval is exactly what a junior would build in an afternoon: embed every
chunk once with an injected embedder, embed the query, return the cosine top-k.
The provenance carried on each chunk (episode + ``[t_start, t_end]`` + the
*dominant diarization speaker* of the window) is projected into a
:class:`~dlogos.eval.arms.Citation` so the eval's speaker-verified check can
still bite — naive retrieval will happily surface a topically-relevant window
whose dominant speaker is the wrong attribution, which is precisely the failure
mode the check is meant to catch.

Import-light: numpy + the shared pydantic schema only. The embedder is an
injected protocol (the test ``FakeEmbedder`` satisfies it), so unit tests run
with the core deps and no network. No heavy/optional dependency is imported,
even lazily — this baseline is meant to be trivial.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from dlogos.eval.arms import Citation
from dlogos.schema import Transcript, TranscriptSegment


# --------------------------------------------------------------------------- #
# Injected embedder (structural — the conftest FakeEmbedder satisfies it).
# --------------------------------------------------------------------------- #
@runtime_checkable
class Embedder(Protocol):
    """Minimal embedder surface: text in, vector out.

    Identical in shape to :class:`dlogos.retrieval.hybrid.Embedder` but declared
    here so this module does not import the graph-retrieval path (the whole
    point of the baseline is independence). ``embed_batch`` is optional; the
    index falls back to per-item ``embed`` when it is absent.
    """

    def embed(self, text: str) -> list[float]: ...


# --------------------------------------------------------------------------- #
# The unit of the dumb index: a raw transcript chunk.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TranscriptChunk:
    """One window of raw transcript text, with the provenance to cite it.

    Deliberately flat and structure-free: it is the *verbatim* concatenation of
    one or more diarized segments plus where in the episode they came from. No
    claim, no stance, no subject — that is the whole point of the naive baseline.

    ``speaker_label`` is the dominant diarization label across the window (the
    label that speaks for the most wall-clock time in it). The naive arm
    attributes the window to whatever speaker id that label resolves to — which
    is exactly how a dumb pipeline misattributes a multi-speaker window.
    """

    episode_id: str
    t_start: float
    t_end: float
    text: str
    speaker_label: str
    embedding: list[float] | None = None


def _dominant_label(segments: list[TranscriptSegment]) -> str:
    """The diarization label that speaks for the most time across ``segments``.

    Ties (and an empty window) break deterministically by label so chunking is
    reproducible. This mirrors how a naive RAG pipeline would pick a single
    "speaker" for a multi-speaker window — and how it gets attribution wrong.
    """

    durations: dict[str, float] = {}
    for seg in segments:
        durations[seg.speaker] = durations.get(seg.speaker, 0.0) + max(
            0.0, seg.t_end - seg.t_start
        )
    if not durations:
        return ""
    # Most wall-clock time first; ties broken by label for determinism.
    return max(sorted(durations), key=lambda label: durations[label])


def chunk_transcript(
    transcript: Transcript, *, segments_per_chunk: int = 3
) -> list[TranscriptChunk]:
    """Chunk a diarized transcript into fixed-size windows of raw segments.

    Windows are non-overlapping runs of ``segments_per_chunk`` consecutive
    diarized segments (the last window may be shorter). Each chunk's ``text`` is
    the verbatim segment texts joined with spaces — no normalization, no
    extraction. ``t_start``/``t_end`` span the window; ``speaker_label`` is the
    window's dominant diarization label.

    This is the most naive sensible chunking: by transcript order, fixed count,
    no semantic boundaries. That naivety is the point.
    """

    if segments_per_chunk < 1:
        raise ValueError("segments_per_chunk must be >= 1")

    chunks: list[TranscriptChunk] = []
    segs = transcript.segments
    for i in range(0, len(segs), segments_per_chunk):
        window = segs[i : i + segments_per_chunk]
        if not window:
            continue
        text = " ".join(s.text for s in window).strip()
        chunks.append(
            TranscriptChunk(
                episode_id=transcript.episode_id,
                t_start=window[0].t_start,
                t_end=window[-1].t_end,
                text=text,
                speaker_label=_dominant_label(window),
            )
        )
    return chunks


# --------------------------------------------------------------------------- #
# Cosine helper (local — no shared retrieval code).
# --------------------------------------------------------------------------- #
def _cosine(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(va @ vb) / (na * nb)


# --------------------------------------------------------------------------- #
# The dumb index.
# --------------------------------------------------------------------------- #
class NaiveVectorIndex:
    """A flat cosine top-k index over raw transcript chunks. Nothing more.

    Build it from transcripts (or pre-chunked windows), embedding every chunk
    once with the injected embedder. ``search`` embeds the query and returns the
    cosine-nearest chunks, highest similarity first, ties broken by
    ``(episode_id, t_start)`` for determinism.

    No graph, no temporal filter, no stance — the deliberate weak control.
    """

    def __init__(self, chunks: list[TranscriptChunk], embedder: Embedder) -> None:
        self._embedder = embedder
        # Embed every chunk that does not already carry a vector. Use embed_batch
        # if the embedder offers it (still pure-local, just fewer calls).
        to_embed = [c for c in chunks if c.embedding is None]
        vectors: list[list[float]] = []
        if to_embed:
            batch = getattr(embedder, "embed_batch", None)
            if callable(batch):
                vectors = list(batch([c.text for c in to_embed]))
            else:
                vectors = [embedder.embed(c.text) for c in to_embed]
        vec_iter = iter(vectors)
        self._chunks: list[TranscriptChunk] = [
            c
            if c.embedding is not None
            else TranscriptChunk(
                episode_id=c.episode_id,
                t_start=c.t_start,
                t_end=c.t_end,
                text=c.text,
                speaker_label=c.speaker_label,
                embedding=next(vec_iter),
            )
            for c in chunks
        ]

    @classmethod
    def from_transcripts(
        cls,
        transcripts: list[Transcript],
        embedder: Embedder,
        *,
        segments_per_chunk: int = 3,
    ) -> NaiveVectorIndex:
        """Build the index straight from raw transcripts.

        Chunks every transcript with :func:`chunk_transcript` and embeds the
        windows. This is the only entry point the naive arm needs — hand it the
        same transcripts the graph was built from and it indexes them dumbly.
        """

        chunks: list[TranscriptChunk] = []
        for t in transcripts:
            chunks.extend(
                chunk_transcript(t, segments_per_chunk=segments_per_chunk)
            )
        return cls(chunks, embedder)

    @property
    def size(self) -> int:
        return len(self._chunks)

    def search(self, query: str, *, k: int = 8) -> list[TranscriptChunk]:
        """Cosine top-k over the indexed chunks. Highest similarity first."""

        if k <= 0 or not self._chunks:
            return []
        qvec = self._embedder.embed(query)
        scored: list[tuple[float, str, float, TranscriptChunk]] = []
        for c in self._chunks:
            vec = c.embedding if c.embedding is not None else self._embedder.embed(c.text)
            sim = _cosine(qvec, vec)
            scored.append((sim, c.episode_id, c.t_start, c))
        # Sort by similarity desc, then (episode_id, t_start) asc for determinism.
        scored.sort(key=lambda row: (-row[0], row[1], row[2]))
        return [row[3] for row in scored[:k]]


# --------------------------------------------------------------------------- #
# The arm-3 retriever: projects chunks to citations (the VectorRetriever shape).
# --------------------------------------------------------------------------- #
class TranscriptRagRetriever:
    """Arm-3 retriever (spec §9): dumb top-k over raw transcripts -> Citations.

    Satisfies the :class:`dlogos.eval.arms.VectorRetriever` protocol so
    :class:`~dlogos.eval.arms.ModelNaiveRagArm` consumes it directly. It wraps a
    :class:`NaiveVectorIndex` and projects each retrieved chunk to a
    :class:`~dlogos.eval.arms.Citation`, attributing the span to whatever the
    chunk's dominant diarization label resolves to.

    ``label_to_speaker`` maps a per-episode diarization label to a resolved
    speaker id; supply the *same* resolution the rest of the system uses so the
    speaker-verified check is a fair test. When a label is unmapped, the chunk's
    raw label is carried as the attribution — a naive pipeline does not know any
    better, and the check will (correctly) reject it.

    Crucially, this retriever NEVER touches a graph, a consensus trend, or a
    temporal window. It is the structure-free control.
    """

    def __init__(
        self,
        index: NaiveVectorIndex,
        *,
        label_to_speaker: dict[str, str] | None = None,
        default_k: int = 8,
    ) -> None:
        self._index = index
        self._label_to_speaker = dict(label_to_speaker or {})
        self._default_k = default_k

    @classmethod
    def from_transcripts(
        cls,
        transcripts: list[Transcript],
        embedder: Embedder,
        *,
        label_to_speaker: dict[str, str] | None = None,
        segments_per_chunk: int = 3,
        default_k: int = 8,
    ) -> TranscriptRagRetriever:
        """Convenience builder: chunk + index raw transcripts, then wrap them."""

        index = NaiveVectorIndex.from_transcripts(
            transcripts, embedder, segments_per_chunk=segments_per_chunk
        )
        return cls(
            index,
            label_to_speaker=label_to_speaker,
            default_k=default_k,
        )

    def _attribute(self, chunk: TranscriptChunk) -> str:
        """Resolve the chunk's dominant label to a speaker id (or carry it raw)."""

        return self._label_to_speaker.get(chunk.speaker_label, chunk.speaker_label)

    async def retrieve(self, query: str, *, k: int = 8) -> list[Citation]:
        """Return the cosine top-k raw-transcript spans as citations."""

        top = self._index.search(query, k=k or self._default_k)
        out: list[Citation] = []
        for chunk in top:
            out.append(
                Citation(
                    episode_id=chunk.episode_id,
                    t_start=chunk.t_start,
                    t_end=chunk.t_end,
                    speaker_id=self._attribute(chunk),
                    snippet=chunk.text,
                )
            )
        return out
