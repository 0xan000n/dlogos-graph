"""Tests for hybrid retrieval (spec §8).

Deterministic: the injected ``fake_embedder`` (conftest) maps known surface
forms to fixed unit vectors; BM25 runs on the pure-python fallback (rank_bm25
is not a core dep); the in-memory store is hand-built. No network, no infra.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from dlogos.retrieval.hybrid import (
    GraphStore,
    HybridRetriever,
    RetrievableClaim,
    TemporalMode,
    claims_from_graph_store,
    in_memory_store,
    reciprocal_rank_fusion,
)
from dlogos.schema import SourceSpan, Stance


def _dt(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Reciprocal Rank Fusion — the required ordering test.
# --------------------------------------------------------------------------- #
def test_rrf_rewards_items_ranked_well_across_lists() -> None:
    """An item near the top of *both* lists beats items top of only one.

    semantic: [A, B, C]   lexical: [B, A, D]
    A: 1/(60+1) + 1/(60+2)   B: 1/(60+2) + 1/(60+1)  -> A and B tie high
    C: 1/(60+3)              D: 1/(60+3)              -> tie low
    Tie between A and B is broken by id ascending, so A first.
    """

    fused = reciprocal_rank_fusion([["A", "B", "C"], ["B", "A", "D"]])
    order = [cid for cid, _ in fused]
    assert order[:2] == ["A", "B"]  # both appear in both lists, A wins the tie
    assert set(order[2:]) == {"C", "D"}
    # C and D each appear once at rank 3 — equal score, id order C before D.
    assert order[2:] == ["C", "D"]


def test_rrf_consensus_item_beats_single_list_leader() -> None:
    """An item that is #2 in both lists beats an item that is #1 in only one."""

    # X is rank-1 in list A only. Y is rank-2 in both lists.
    a = ["X", "Y", "Z"]
    b = ["W", "Y", "Z"]
    fused = dict(reciprocal_rank_fusion([a, b]))
    # Y: 1/62 + 1/62 = 2/62 ≈ 0.03226 ; X: 1/61 ≈ 0.01639
    assert fused["Y"] > fused["X"]
    assert fused["Y"] > fused["W"]


def test_rrf_weights_bias_a_signal() -> None:
    # With lexical weighted 10x, the lexical-only leader should win over a
    # semantic-only leader. (Items shared across lists accumulate from both,
    # so we keep the lists disjoint to isolate the weight effect.)
    semantic = ["S1", "S2"]
    lexical = ["L1", "L2"]
    fused = dict(reciprocal_rank_fusion([semantic, lexical], weights=[1.0, 10.0]))
    assert fused["L1"] > fused["S1"]
    # Equal-weighting instead would put the two leaders level.
    even = dict(reciprocal_rank_fusion([semantic, lexical], weights=[1.0, 1.0]))
    assert even["L1"] == even["S1"]


def test_rrf_is_deterministic_and_total() -> None:
    fused1 = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])
    fused2 = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])
    assert fused1 == fused2
    assert {cid for cid, _ in fused1} == {"a", "b", "c"}


def test_rrf_rejects_mismatched_weights() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])


# --------------------------------------------------------------------------- #
# Fixtures for retriever tests.
# --------------------------------------------------------------------------- #
def _claim(
    claim_id: str,
    text: str,
    *,
    neighbors: tuple[str, ...] = (),
    event_time: datetime | None = None,
    valid_to: datetime | None = None,
    stance: Stance = Stance.asserts,
) -> RetrievableClaim:
    return RetrievableClaim(
        claim_id=claim_id,
        text=text,
        stance=stance,
        source_span=SourceSpan(episode_id=f"ep-{claim_id}", t_start=0.0, t_end=1.0),
        neighbor_ids=neighbors,
        event_time=event_time,
        valid_from=event_time,
        valid_to=valid_to,
    )


@pytest.fixture
def claims() -> list[RetrievableClaim]:
    return [
        _claim("c-apple", "Apple", neighbors=("c-iphone",), event_time=_dt(2026, 1, 1)),
        _claim("c-iphone", "the iPhone", neighbors=("c-apple",), event_time=_dt(2026, 2, 1)),
        _claim("c-openai", "OpenAI", event_time=_dt(2026, 3, 1)),
        _claim("c-misc", "a wholly unrelated topic", event_time=_dt(2026, 4, 1)),
    ]


# --------------------------------------------------------------------------- #
# in_memory_store satisfies the GraphStore protocol.
# --------------------------------------------------------------------------- #
def test_in_memory_store_is_a_graph_store(claims) -> None:
    store = in_memory_store(claims)
    assert isinstance(store, GraphStore)  # runtime_checkable structural check
    assert len(store.all_claims()) == 4
    assert store.get_claim("c-apple").text == "Apple"
    assert store.get_claim("missing") is None
    assert store.neighbors("c-apple") == ["c-iphone"]


# --------------------------------------------------------------------------- #
# Semantic signal: the embedder places 'Apple' nearest the 'Apple' query.
# --------------------------------------------------------------------------- #
def test_semantic_retrieval_ranks_query_match_first(claims, fake_embedder) -> None:
    store = in_memory_store(claims)
    retriever = HybridRetriever(store, fake_embedder, traversal_depth=0)
    results = retriever.retrieve("Apple", top_k=4, traverse=False)
    ids = [r.claim.claim_id for r in results]
    # 'Apple' exactly matches; 'the iPhone' is the near neighbor (see FakeEmbedder).
    assert ids[0] == "c-apple"
    assert ids.index("c-iphone") < ids.index("c-openai")


# --------------------------------------------------------------------------- #
# Lexical signal: exact term match surfaces even when not embedded near.
# --------------------------------------------------------------------------- #
def test_lexical_retrieval_catches_exact_term(claims, fake_embedder) -> None:
    store = in_memory_store(claims)
    retriever = HybridRetriever(
        store,
        fake_embedder,
        traversal_depth=0,
        semantic_weight=0.0,  # lean entirely on BM25
        lexical_weight=1.0,
    )
    results = retriever.retrieve("OpenAI", top_k=4, traverse=False)
    assert results[0].claim.claim_id == "c-openai"
    assert results[0].lexical_rank == 1


# --------------------------------------------------------------------------- #
# Graph traversal: pulls a neighbor of a strong seed even if off-query.
# --------------------------------------------------------------------------- #
def test_graph_traversal_pulls_connected_neighborhood(claims, fake_embedder) -> None:
    store = in_memory_store(claims)
    retriever = HybridRetriever(store, fake_embedder, traversal_depth=1)
    # Query matches Apple; iPhone is its graph neighbor and should rank highly
    # thanks to the traversal signal even though the query text is just "Apple".
    results = retriever.retrieve("Apple", top_k=4, traverse=True, seed_k=2)
    ids = [r.claim.claim_id for r in results]
    iphone = next(r for r in results if r.claim.claim_id == "c-iphone")
    assert iphone.graph_rank is not None  # surfaced by traversal
    assert ids[0] == "c-apple"
    # The connected Apple/iPhone pair outranks the unrelated 'misc' claim.
    assert ids.index("c-iphone") < ids.index("c-misc")


def test_traversal_can_be_disabled(claims, fake_embedder) -> None:
    store = in_memory_store(claims)
    retriever = HybridRetriever(store, fake_embedder, traversal_depth=1)
    results = retriever.retrieve("Apple", traverse=False)
    assert all(r.graph_rank is None for r in results)


# --------------------------------------------------------------------------- #
# Temporal filter on validity windows.
# --------------------------------------------------------------------------- #
def test_temporal_filter_restricts_to_overlapping_window(claims, fake_embedder) -> None:
    store = in_memory_store(claims)
    retriever = HybridRetriever(store, fake_embedder, traversal_depth=0)
    # Only Feb–Mar 2026: should exclude the Jan 'Apple' and Apr 'misc' claims.
    results = retriever.retrieve(
        "Apple",
        top_k=10,
        as_of_start=_dt(2026, 2, 1),
        as_of_end=_dt(2026, 3, 15),
        traverse=False,
    )
    ids = {r.claim.claim_id for r in results}
    assert ids == {"c-iphone", "c-openai"}


def test_invalidated_claim_excluded_in_validity_overlap_mode(fake_embedder) -> None:
    # Under validity_overlap mode, a claim invalidated end-of-Jan (valid_to set)
    # should not match a Feb-onward as-of window, while a still-live claim does.
    expired = _claim(
        "c-expired",
        "Apple",
        event_time=_dt(2026, 1, 1),
        valid_to=_dt(2026, 1, 31),
    )
    live = _claim("c-live", "Apple", event_time=_dt(2026, 1, 1))  # never invalidated
    store = in_memory_store([expired, live])
    retriever = HybridRetriever(store, fake_embedder, traversal_depth=0)
    results = retriever.retrieve(
        "Apple",
        as_of_start=_dt(2026, 2, 1),
        as_of_end=None,
        temporal_mode=TemporalMode.validity_overlap,
        traverse=False,
    )
    ids = {r.claim.claim_id for r in results}
    # The un-invalidated Jan claim is still valid in Feb (open-ended window);
    # the invalidated one is not.
    assert ids == {"c-live"}


def test_event_time_mode_is_the_default(fake_embedder) -> None:
    # A Jan claim is *made* before a Feb-onward window, so event_time mode
    # (the default) excludes it even though its validity is open-ended.
    jan = _claim("c-jan", "Apple", event_time=_dt(2026, 1, 1))
    feb = _claim("c-feb", "Apple", event_time=_dt(2026, 2, 10))
    store = in_memory_store([jan, feb])
    retriever = HybridRetriever(store, fake_embedder, traversal_depth=0)
    results = retriever.retrieve(
        "Apple", as_of_start=_dt(2026, 2, 1), as_of_end=None, traverse=False
    )
    assert {r.claim.claim_id for r in results} == {"c-feb"}


def test_claim_without_temporal_info_survives_filter(fake_embedder) -> None:
    # No event/validity info -> treated as always-valid (not excluded).
    timeless = RetrievableClaim(claim_id="c-timeless", text="Apple")
    store = in_memory_store([timeless])
    retriever = HybridRetriever(store, fake_embedder, traversal_depth=0)
    results = retriever.retrieve(
        "Apple", as_of_start=_dt(2026, 1, 1), as_of_end=_dt(2026, 1, 2), traverse=False
    )
    assert [r.claim.claim_id for r in results] == ["c-timeless"]


def test_empty_after_temporal_filter_returns_empty(claims, fake_embedder) -> None:
    store = in_memory_store(claims)
    retriever = HybridRetriever(store, fake_embedder, traversal_depth=0)
    results = retriever.retrieve(
        "Apple",
        as_of_start=_dt(2030, 1, 1),
        as_of_end=_dt(2030, 2, 1),
        traverse=False,
    )
    assert results == []


# --------------------------------------------------------------------------- #
# top_k and provenance.
# --------------------------------------------------------------------------- #
def test_top_k_limits_results_and_records_ranks(claims, fake_embedder) -> None:
    store = in_memory_store(claims)
    retriever = HybridRetriever(store, fake_embedder, traversal_depth=0)
    results = retriever.retrieve("Apple", top_k=2, traverse=False)
    assert len(results) == 2
    # Scores are monotonically non-increasing.
    assert results[0].score >= results[1].score
    # Each surfaced result carries at least one signal rank.
    for r in results:
        assert r.semantic_rank is not None or r.lexical_rank is not None


# --------------------------------------------------------------------------- #
# Adapter from the graph module's store contract (lazy, structural).
# --------------------------------------------------------------------------- #
class _FakeRow:
    """Mimics dlogos.graph QueryResult: .claim/.subject/.event_time."""

    def __init__(self, claim, subject, event_time):
        self.claim = claim
        self.subject = subject
        self.event_time = event_time


class _FakeClaim:
    def __init__(self, claim_id, obj, stance, subject_id):
        self.claim_id = claim_id
        self.object = obj
        self.stance = stance
        self.speaker_id = "spk"
        self.subject_canonical_id = subject_id
        self.source_span = SourceSpan(episode_id="ep", t_start=0.0, t_end=1.0)


class _FakeSubject:
    def __init__(self, name):
        self.name = name


class _FakeEdge:
    def __init__(self, src, dst, invalidated=False):
        self.src_id = src
        self.dst_id = dst
        self.invalidated = invalidated


class _FakeGraphModuleStore:
    """Minimal stand-in for dlogos.graph.fake_store.FakeGraphStore."""

    def __init__(self, rows, edges):
        self._rows = rows
        self.edges = {f"e{i}": e for i, e in enumerate(edges)}

    def query(self, *, as_of=None):
        return self._rows


def test_claims_from_graph_store_adapter(fake_embedder) -> None:
    rows = [
        _FakeRow(
            _FakeClaim("c1", "is a strong company", Stance.asserts, "ent-apple"),
            _FakeSubject("Apple"),
            _dt(2026, 1, 1),
        ),
        _FakeRow(
            _FakeClaim("c2", "has plateaued", Stance.disputes, "ent-apple"),
            _FakeSubject("iPhone"),
            _dt(2026, 2, 1),
        ),
    ]
    edges = [_FakeEdge("c1", "c2"), _FakeEdge("c1", "c3", invalidated=True)]
    store = _FakeGraphModuleStore(rows, edges)

    retrievable = claims_from_graph_store(store, embedder=fake_embedder)
    by_id = {c.claim_id: c for c in retrievable}

    assert by_id["c1"].text == "Apple is a strong company"
    assert by_id["c1"].subject_id == "ent-apple"
    assert by_id["c1"].stance is Stance.asserts
    assert by_id["c1"].embedding is not None
    # Live edge c1<->c2 becomes mutual adjacency; invalidated c1->c3 dropped.
    assert by_id["c1"].neighbor_ids == ("c2",)
    assert by_id["c2"].neighbor_ids == ("c1",)

    # The adapter output drops straight into a retriever.
    retriever = HybridRetriever(in_memory_store(retrievable), fake_embedder)
    results = retriever.retrieve("Apple", top_k=2)
    assert {r.claim.claim_id for r in results} == {"c1", "c2"}
