"""Tests for the concrete GraphRetrievalSurface + the arm retriever adapters.

These exercise the integration wiring that makes the MCP tools and the dLogos /
naive-RAG eval arms delegate to the REAL retrieval objects over a real graph
store — built here by loading the conftest synthetic claims through the loader
into a :class:`~dlogos.graph.fake_store.FakeGraphStore`. No network, no heavy
deps; the conftest ``FakeEmbedder`` supplies deterministic vectors.
"""

from __future__ import annotations

from dlogos.eval.arms import DLogosGraphRetriever, GraphVectorRetriever
from dlogos.graph.fake_store import FakeGraphStore
from dlogos.graph.loader import ClaimLoader
from dlogos.mcp.server import (
    GraphRetrievalSurface,
    RetrievalSurface,
    consensus_trend_handler,
    search_dialogue_handler,
)
from dlogos.resolution.subjects import resolve_subjects


def _loaded_surface(synthetic_claims, claim_event_times, fake_embedder):
    """Resolve + load the synthetic claims, then build a real surface over them."""

    resolution = resolve_subjects(synthetic_claims, fake_embedder)
    store = FakeGraphStore()
    loader = ClaimLoader(event_times=claim_event_times)
    loader.bulk_load(store, resolution.claims, bypass_llm_dedup=True)
    surface = GraphRetrievalSurface.from_graph_store(
        store,
        fake_embedder,
        consensus_claims=resolution.claims,
        event_times=claim_event_times,
    )
    return surface, resolution


def test_surface_satisfies_retrieval_surface_protocol(
    synthetic_claims, claim_event_times, fake_embedder
) -> None:
    surface, _ = _loaded_surface(
        synthetic_claims, claim_event_times, fake_embedder
    )
    assert isinstance(surface, RetrievalSurface)


def test_search_handler_over_real_surface_returns_attributed_hits(
    synthetic_claims, claim_event_times, fake_embedder
) -> None:
    surface, _ = _loaded_surface(
        synthetic_claims, claim_event_times, fake_embedder
    )
    res = search_dialogue_handler(surface, "Apple", top_k=10)
    assert res.hits
    # Every hit carries an attributed speaker + a real span (provenance).
    for h in res.hits:
        assert h.speaker_id
        assert h.episode_id
        assert h.t_start is not None


def test_consensus_handler_over_real_surface_buckets_by_canonical_id(
    synthetic_claims, claim_event_times, fake_embedder
) -> None:
    surface, resolution = _loaded_surface(
        synthetic_claims, claim_event_times, fake_embedder
    )
    canonical_id = resolution.claims[0].subject_entity.canonical_id
    res = consensus_trend_handler(surface, canonical_id, window_days=30)
    # All four synthetic Apple claims aggregate under one canonical subject.
    assert res.subject == canonical_id
    assert sum(p.claim_count for p in res.points) == len(synthetic_claims)
    # Several attributed speakers across the timeline.
    assert "spk-analyst" in res.all_speakers
    assert "spk-host" in res.all_speakers


async def test_dlogos_graph_retriever_yields_context_and_citations(
    synthetic_claims, claim_event_times, fake_embedder
) -> None:
    surface, _ = _loaded_surface(
        synthetic_claims, claim_event_times, fake_embedder
    )
    retriever = DLogosGraphRetriever(surface, top_k=10)
    context, citations = await retriever.query("How has consensus on Apple moved?")

    # Structured, attributed evidence + speaker-verified-ready citations.
    assert "Consensus on" in context or "Attributed spans" in context
    assert citations
    for c in citations:
        assert c.speaker_id
        assert c.episode_id


async def test_naive_rag_retriever_yields_only_spans_no_structure(
    synthetic_claims, claim_event_times, fake_embedder
) -> None:
    surface, _ = _loaded_surface(
        synthetic_claims, claim_event_times, fake_embedder
    )
    retriever = GraphVectorRetriever(surface, k=5)
    citations = await retriever.retrieve("Apple hardware", k=5)
    # Returns citations (spans) but does not synthesize a consensus trend.
    assert citations
    assert all(c.episode_id for c in citations)
