"""Tests for the independent naive-RAG baseline (spec §9 arm 3).

The whole point of this module is *isolation*: the naive arm must retrieve over
RAW transcript chunks with no graph, no temporal model and no stance, so that
"graph structure beats dumb retrieval on identical data" is genuinely testable.
These tests pin that independence and the dumb cosine top-k behaviour.

All collaborators are the deterministic conftest fakes -- no network, no graph
store, no heavy deps.
"""

from __future__ import annotations

import inspect

from dlogos.eval.arms import Citation
from dlogos.retrieval.naive_rag import (
    NaiveVectorIndex,
    TranscriptChunk,
    TranscriptRagRetriever,
    chunk_transcript,
)
from dlogos.schema import Transcript, TranscriptSegment


def _two_topic_transcript() -> Transcript:
    """A transcript whose windows cleanly split Apple vs. OpenAI topics.

    Segment texts are chosen to match the conftest FakeEmbedder lookup table so
    cosine ordering is deterministic and meaningful (not just hash noise).
    """

    segments = [
        TranscriptSegment(speaker="SPEAKER_00", text="Apple", t_start=0.0, t_end=4.0),
        TranscriptSegment(speaker="SPEAKER_01", text="Apple", t_start=4.0, t_end=8.0),
        TranscriptSegment(speaker="SPEAKER_00", text="Apple", t_start=8.0, t_end=12.0),
        TranscriptSegment(speaker="SPEAKER_01", text="OpenAI", t_start=12.0, t_end=16.0),
        TranscriptSegment(speaker="SPEAKER_01", text="OpenAI", t_start=16.0, t_end=20.0),
        TranscriptSegment(speaker="SPEAKER_00", text="OpenAI", t_start=20.0, t_end=24.0),
    ]
    return Transcript(
        episode_id="ep-0001", language="en", segments=segments, duration_s=24.0
    )


# --------------------------------------------------------------------------- #
# Chunking — raw, order-based, structure-free.
# --------------------------------------------------------------------------- #
def test_chunk_transcript_windows_raw_segments_with_provenance() -> None:
    transcript = _two_topic_transcript()
    chunks = chunk_transcript(transcript, segments_per_chunk=3)

    assert len(chunks) == 2
    first, second = chunks
    # Verbatim concatenation of the raw segment texts -- no extraction.
    assert first.text == "Apple Apple Apple"
    assert second.text == "OpenAI OpenAI OpenAI"
    # Provenance spans the window.
    assert first.episode_id == "ep-0001"
    assert (first.t_start, first.t_end) == (0.0, 12.0)
    assert (second.t_start, second.t_end) == (12.0, 24.0)


def test_chunk_dominant_speaker_label_is_by_wall_clock_time() -> None:
    # SPEAKER_01 holds the floor longest in this window -> dominant label.
    segments = [
        TranscriptSegment(speaker="SPEAKER_00", text="a", t_start=0.0, t_end=1.0),
        TranscriptSegment(speaker="SPEAKER_01", text="b", t_start=1.0, t_end=9.0),
    ]
    transcript = Transcript(
        episode_id="ep-x", language="en", segments=segments, duration_s=9.0
    )
    [chunk] = chunk_transcript(transcript, segments_per_chunk=5)
    assert chunk.speaker_label == "SPEAKER_01"


def test_last_window_may_be_short() -> None:
    transcript = _two_topic_transcript()
    chunks = chunk_transcript(transcript, segments_per_chunk=4)
    # 6 segments, window=4 -> [4, 2].
    assert [len(c.text.split()) for c in chunks] == [4, 2]


# --------------------------------------------------------------------------- #
# The dumb index — cosine top-k, deterministic.
# --------------------------------------------------------------------------- #
def test_index_returns_cosine_topk_over_raw_chunks(fake_embedder) -> None:
    # One segment per chunk so each chunk's text is a single table word the
    # FakeEmbedder maps to a fixed unit vector -- cosine ordering is then real
    # and seed-independent (multi-word chunks would hash, which is not what this
    # test is about).
    transcript = _two_topic_transcript()
    index = NaiveVectorIndex.from_transcripts(
        [transcript], fake_embedder, segments_per_chunk=1
    )
    assert index.size == 6

    # The "Apple" chunks are nearest the Apple query; OpenAI chunks are far.
    top = index.search("Apple", k=1)
    assert len(top) == 1
    assert top[0].text == "Apple"
    # The bottom of the ranking is the off-topic OpenAI chunk.
    assert index.search("Apple", k=6)[-1].text == "OpenAI"


def test_index_search_is_deterministic_and_respects_k(fake_embedder) -> None:
    transcript = _two_topic_transcript()
    index = NaiveVectorIndex.from_transcripts([transcript], fake_embedder)

    a = [c.text for c in index.search("OpenAI", k=2)]
    b = [c.text for c in index.search("OpenAI", k=2)]
    assert a == b  # deterministic
    assert len(index.search("OpenAI", k=1)) == 1
    assert index.search("OpenAI", k=0) == []


def test_precomputed_chunk_embeddings_are_not_re_embedded(fake_embedder) -> None:
    # A chunk carrying its own vector must be indexed as-is (no embedder call
    # needed for it). We give it a vector pointing exactly at "OpenAI".
    chunk = TranscriptChunk(
        episode_id="ep-0001",
        t_start=0.0,
        t_end=4.0,
        text="totally unrelated words here",
        speaker_label="SPEAKER_00",
        embedding=fake_embedder.embed("OpenAI"),
    )
    index = NaiveVectorIndex([chunk], fake_embedder)
    top = index.search("OpenAI", k=1)
    assert top[0] is chunk  # matched via its precomputed vector, not its text


# --------------------------------------------------------------------------- #
# The arm-3 retriever — projects chunks to citations, INDEPENDENT of any graph.
# --------------------------------------------------------------------------- #
async def test_retriever_yields_citations_over_raw_transcripts(fake_embedder) -> None:
    transcript = _two_topic_transcript()
    retriever = TranscriptRagRetriever.from_transcripts(
        [transcript],
        fake_embedder,
        label_to_speaker={"SPEAKER_00": "spk-host", "SPEAKER_01": "spk-analyst"},
        # Single-segment chunks -> table-word embeddings -> deterministic cosine.
        segments_per_chunk=1,
    )
    cites = await retriever.retrieve("Apple", k=1)

    assert cites and all(isinstance(c, Citation) for c in cites)
    top = cites[0]
    assert top.episode_id == "ep-0001"
    assert top.snippet == "Apple"
    # The first Apple segment is SPEAKER_00 -> resolved to the host speaker id.
    assert top.speaker_id == "spk-host"


async def test_retriever_is_structurally_a_vector_retriever(fake_embedder) -> None:
    """It satisfies the arm's VectorRetriever surface (async retrieve/k)."""

    from dlogos.eval.arms import VectorRetriever

    retriever = TranscriptRagRetriever.from_transcripts(
        [_two_topic_transcript()], fake_embedder
    )
    assert isinstance(retriever, VectorRetriever)
    sig = inspect.signature(retriever.retrieve)
    assert "k" in sig.parameters


async def test_retriever_takes_no_graph_store_and_imports_no_graph() -> None:
    """Independence guard: the naive baseline must not touch the graph path.

    Constructing/operating the retriever requires only transcripts + an
    embedder. We assert (a) its constructor signature mentions no store/graph,
    and (b) the module's source references neither the graph subpackage nor the
    graph-coupled hybrid retriever -- so it cannot be quietly riding the
    structure under test.
    """

    import dlogos.retrieval.naive_rag as mod

    src = inspect.getsource(mod)
    # The module must not import or reference the graph retrieval path. (The
    # docstring may *mention* the words "graph"/"consensus" to say it does NOT
    # do them; what matters is no import/usage of those code paths.)
    assert "import" in src  # sanity: we are reading real source
    assert "dlogos.graph" not in src
    assert "from dlogos.retrieval.hybrid" not in src
    assert "from dlogos.retrieval.consensus" not in src
    assert "HybridRetriever" not in src
    assert "GraphStore" not in src
    assert "consensus_over_time" not in src

    params = set(inspect.signature(TranscriptRagRetriever.__init__).parameters)
    assert not {"store", "graph", "surface"} & params


async def test_retriever_unmapped_label_carried_raw_for_check_to_reject(
    fake_embedder,
) -> None:
    # No label_to_speaker -> the raw diarization label is the attribution. A
    # naive pipeline doesn't know better; the speaker-verified check will reject
    # it, which is the failure mode arm 3 is meant to expose.
    retriever = TranscriptRagRetriever.from_transcripts(
        [_two_topic_transcript()], fake_embedder, segments_per_chunk=3
    )
    [cite] = await retriever.retrieve("Apple", k=1)
    assert cite.speaker_id in {"SPEAKER_00", "SPEAKER_01"}
