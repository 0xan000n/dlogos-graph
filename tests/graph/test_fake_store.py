"""Tests for the in-memory FakeGraphStore implementing the GraphStore contract."""

from __future__ import annotations

from datetime import datetime, timezone

from dlogos.graph.fake_store import FakeGraphStore
from dlogos.graph.loader import ClaimLoader
from dlogos.graph.store import (
    ClaimNode,
    EdgeType,
    EntityNode,
    GraphEdge,
    GraphStore,
    SpeakerNode,
)
from dlogos.schema import Predicate, SourceSpan, Stance


def _dt(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)


def _claim_node(claim_id: str, *, speaker="spk-1", subject="ent-apple") -> ClaimNode:
    return ClaimNode(
        claim_id=claim_id,
        predicate=Predicate.rates_negative,
        stance=Stance.asserts,
        object="apple hardware plateaued",
        sentiment=-0.5,
        confidence=0.8,
        source_span=SourceSpan(episode_id="ep-1", t_start=1.0, t_end=2.0),
        speaker_id=speaker,
        subject_canonical_id=subject,
    )


def _asserts_edge(
    edge_id: str, claim_id: str, *, event: datetime, speaker="spk-1"
) -> GraphEdge:
    return GraphEdge(
        edge_id=edge_id,
        type=EdgeType.asserts,
        src_id=speaker,
        dst_id=claim_id,
        event_time=event,
        ingestion_time=_dt(2026, 6, 18),
        valid_from=event,
    )


def test_fake_store_satisfies_protocol() -> None:
    # @runtime_checkable structural match against the contract.
    assert isinstance(FakeGraphStore(), GraphStore)


def test_bulk_load_returns_count_and_loads_nodes() -> None:
    store = FakeGraphStore()
    speakers = [SpeakerNode(speaker_id="spk-1"), SpeakerNode(speaker_id="spk-2")]
    entities = [EntityNode(canonical_id="ent-apple", name="Apple", type="organization")]
    claims = [_claim_node("c1"), _claim_node("c2", speaker="spk-2")]
    edges = [
        _asserts_edge("e1", "c1", event=_dt(2026, 1, 10)),
        _asserts_edge("e2", "c2", event=_dt(2026, 2, 1), speaker="spk-2"),
    ]
    count = store.bulk_load(
        speakers=speakers, entities=entities, claims=claims, edges=edges
    )
    assert count == 2
    assert store.claim_count() == 2
    assert set(store.speakers) == {"spk-1", "spk-2"}
    # Default bulk path bypasses LLM dedup.
    assert store.llm_dedup_invocations == 0


def test_invalidation_preserves_history() -> None:
    store = FakeGraphStore()
    store.bulk_load(
        speakers=[SpeakerNode(speaker_id="spk-1")],
        entities=[EntityNode(canonical_id="ent-apple", name="Apple", type="organization")],
        claims=[_claim_node("c1")],
        edges=[_asserts_edge("e1", "c1", event=_dt(2026, 1, 10))],
    )
    # Invalidate as of March.
    assert store.invalidate("e1", at=_dt(2026, 3, 1)) is True

    # The edge is NOT deleted — still present, just flagged + windowed.
    edge = store.edges["e1"]
    assert edge.invalidated is True
    assert edge.valid_to == _dt(2026, 3, 1)
    assert len(store.edges) == 1
    # The claim node itself is also preserved (reified, contradictable).
    assert store.claim_count() == 1

    # Re-invalidating an already-dead edge is a no-op (returns False).
    assert store.invalidate("e1", at=_dt(2026, 5, 1)) is False
    assert store.edges["e1"].valid_to == _dt(2026, 3, 1)
    # Unknown edge id -> False.
    assert store.invalidate("nope", at=_dt(2026, 3, 1)) is False


def test_current_state_query_returns_only_live_edges() -> None:
    store = FakeGraphStore()
    store.bulk_load(
        speakers=[SpeakerNode(speaker_id="spk-1")],
        entities=[EntityNode(canonical_id="ent-apple", name="Apple", type="organization")],
        claims=[_claim_node("c1"), _claim_node("c2")],
        edges=[
            _asserts_edge("e1", "c1", event=_dt(2026, 1, 10)),
            _asserts_edge("e2", "c2", event=_dt(2026, 1, 10)),
        ],
    )
    # Both claims live initially.
    assert {r.claim.claim_id for r in store.query()} == {"c1", "c2"}

    # Invalidate the edge for c2 — current-state must drop it.
    store.invalidate("e2", at=_dt(2026, 3, 1))
    live = store.query()
    assert {r.claim.claim_id for r in live} == {"c1"}

    # ... but a point-in-time query before March still surfaces c2 (history).
    history = store.query(as_of=_dt(2026, 2, 1))
    assert {r.claim.claim_id for r in history} == {"c1", "c2"}

    # ... and include_invalidated also surfaces it regardless of time.
    forced = store.query(include_invalidated=True)
    assert {r.claim.claim_id for r in forced} == {"c1", "c2"}


def test_query_filters_combine_with_and() -> None:
    store = FakeGraphStore()
    store.bulk_load(
        speakers=[SpeakerNode(speaker_id="spk-1"), SpeakerNode(speaker_id="spk-2")],
        entities=[
            EntityNode(canonical_id="ent-apple", name="Apple", type="organization"),
            EntityNode(canonical_id="ent-openai", name="OpenAI", type="organization"),
        ],
        claims=[
            _claim_node("c1", speaker="spk-1", subject="ent-apple"),
            _claim_node("c2", speaker="spk-2", subject="ent-apple"),
            _claim_node("c3", speaker="spk-1", subject="ent-openai"),
        ],
        edges=[
            _asserts_edge("e1", "c1", event=_dt(2026, 1, 10), speaker="spk-1"),
            _asserts_edge("e2", "c2", event=_dt(2026, 1, 10), speaker="spk-2"),
            _asserts_edge("e3", "c3", event=_dt(2026, 1, 10), speaker="spk-1"),
        ],
    )
    by_subject = store.query(subject_canonical_id="ent-apple")
    assert {r.claim.claim_id for r in by_subject} == {"c1", "c2"}

    by_speaker_and_subject = store.query(
        subject_canonical_id="ent-apple", speaker_id="spk-1"
    )
    assert {r.claim.claim_id for r in by_speaker_and_subject} == {"c1"}

    by_predicate = store.query(predicate=Predicate.rates_negative)
    assert {r.claim.claim_id for r in by_predicate} == {"c1", "c2", "c3"}


def test_query_result_joins_speaker_subject_and_event_time() -> None:
    store = FakeGraphStore()
    store.bulk_load(
        speakers=[SpeakerNode(speaker_id="spk-1", name="Jane")],
        entities=[EntityNode(canonical_id="ent-apple", name="Apple", type="organization")],
        claims=[_claim_node("c1")],
        edges=[_asserts_edge("e1", "c1", event=_dt(2026, 1, 10))],
    )
    (row,) = store.query()
    assert row.speaker is not None and row.speaker.name == "Jane"
    assert row.subject is not None and row.subject.canonical_id == "ent-apple"
    assert row.event_time == _dt(2026, 1, 10)


def test_loader_into_fake_store_end_to_end(synthetic_claims, claim_event_times) -> None:
    """The shared synthetic claims load through the real loader into the store.

    The fixture claims are speaker-resolved but NOT entity-resolved
    (canonical_id is None off the extractor), so we assign canonical ids the way
    the resolution module would before bulk loading — all 'Apple' surface forms
    collapse to one canonical entity.
    """
    resolved = [
        c.model_copy(
            update={
                "subject_entity": c.subject_entity.model_copy(
                    update={"canonical_id": "ent-apple"}
                )
            }
        )
        for c in synthetic_claims
    ]
    store = FakeGraphStore()
    loader = ClaimLoader(event_times=claim_event_times, ingestion_time=_dt(2026, 6, 18))
    loaded = loader.bulk_load(store, resolved)

    assert loaded == len(synthetic_claims) == 4
    assert store.claim_count() == 4
    # Consensus does not fragment: one canonical Apple entity (spec §7.4a).
    assert set(store.entities) == {"ent-apple"}
    # Two resolved speakers across the four claims (analyst, host, guest-b).
    assert set(store.speakers) == {"spk-analyst", "spk-host", "spk-guest-b"}
    # No LLM dedup ran on the bulk fast path.
    assert store.llm_dedup_invocations == 0

    # Event-time flows from the per-episode publish dates.
    rows = store.query(subject_canonical_id="ent-apple")
    assert len(rows) == 4
    assert all(r.event_time is not None for r in rows)
