"""Tests for the direct :class:`Neo4jStore` — pure Cypher/param building.

No live database is touched here. We exercise the pure helpers the store is
factored into:

- node/edge param builders (pydantic record -> Cypher param map),
- read-side rehydration (Neo4j record dict -> pydantic),
- the ``_rows_to_results`` filter/temporal transform (raw rows -> QueryResults),
- the constant Cypher text + the constraints DDL,
- structural conformance to the ``GraphStore`` Protocol (no driver needed),
- the invalidate param map.

The store's *I/O* methods (``bulk_load`` / ``query`` / ``invalidate`` against a
live driver) cannot run offline; their first real execution is the one-episode
smoke run. A single live-DB integration test guards itself behind a
``NEO4J_TEST_URI`` env var so the offline suite stays green.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from dlogos.graph.neo4j_store import (
    ALL_EDGES_CYPHER,
    CLAIM_LABEL,
    EDGE_REL,
    ENTITY_LABEL,
    INVALIDATE_EDGE_CYPHER,
    QUERY_CLAIMS_CYPHER,
    SHARED_ID,
    SHARED_LABEL,
    SPEAKER_LABEL,
    UPSERT_EDGES_CYPHER,
    UPSERT_NODES_CYPHER,
    Neo4jStore,
    build_invalidate_params,
    claim_from_record,
    claim_param,
    constraint_statements,
    edge_from_record,
    edge_param,
    entity_from_record,
    entity_param,
    speaker_from_record,
    speaker_param,
)
from dlogos.graph.store import (
    ClaimNode,
    EdgeType,
    EntityNode,
    GraphEdge,
    GraphStore,
    SpeakerNode,
)
from dlogos.schema import EntityType, Predicate, SourceSpan, Stance


# --------------------------------------------------------------------------- #
# fixtures / builders
# --------------------------------------------------------------------------- #
def _dt(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)


def _claim_node(claim_id: str = "claim-1", *, speaker="spk-1", subject="ent-apple") -> ClaimNode:
    return ClaimNode(
        claim_id=claim_id,
        predicate=Predicate.rates_negative,
        stance=Stance.asserts,
        object="apple hardware plateaued",
        sentiment=-0.5,
        confidence=0.8,
        source_span=SourceSpan(
            episode_id="ep-1", t_start=1.0, t_end=2.0, transcript_offset=42
        ),
        speaker_id=speaker,
        subject_canonical_id=subject,
    )


def _asserts_edge(
    edge_id="e1", claim_id="claim-1", *, speaker="spk-1", event=None
) -> GraphEdge:
    event = event or _dt(2026, 1, 10)
    return GraphEdge(
        edge_id=edge_id,
        type=EdgeType.asserts,
        src_id=speaker,
        dst_id=claim_id,
        event_time=event,
        ingestion_time=_dt(2026, 6, 18),
        valid_from=event,
    )


# --------------------------------------------------------------------------- #
# Protocol conformance — no driver needed.
# --------------------------------------------------------------------------- #
def test_neo4j_store_satisfies_protocol() -> None:
    # Structural @runtime_checkable match: a None driver is fine for isinstance.
    store = Neo4jStore(driver=None)
    assert isinstance(store, GraphStore)


def test_store_exposes_every_protocol_method() -> None:
    for name in ("add_claim_triplet", "bulk_load", "query", "invalidate"):
        assert callable(getattr(Neo4jStore, name))


# --------------------------------------------------------------------------- #
# Param builders project a shared __id__ and flatten nested types.
# --------------------------------------------------------------------------- #
def test_speaker_param_projects_shared_id() -> None:
    spk = SpeakerNode(speaker_id="spk-1", name="Jane", is_host=True)
    p = speaker_param(spk)
    assert p[SHARED_ID] == "spk-1"
    assert p["speaker_id"] == "spk-1"
    assert p["name"] == "Jane"
    assert p["is_host"] is True


def test_entity_param_projects_shared_id() -> None:
    ent = EntityNode(
        canonical_id="ent-apple", name="Apple", type=EntityType.organization,
        aliases=["Apple", "AAPL"],
    )
    p = entity_param(ent)
    assert p[SHARED_ID] == "ent-apple"
    assert p["canonical_id"] == "ent-apple"
    assert p["type"] == "organization"  # enum -> wire value
    assert p["aliases"] == ["Apple", "AAPL"]


def test_claim_param_flattens_source_span_to_scalars() -> None:
    p = claim_param(_claim_node())
    # Neo4j has no nested-map property; span must be flattened to scalars.
    assert "source_span" not in p
    assert p["span_episode_id"] == "ep-1"
    assert p["span_t_start"] == 1.0
    assert p["span_t_end"] == 2.0
    assert p["span_transcript_offset"] == 42
    assert p[SHARED_ID] == "claim-1"
    assert p["predicate"] == "rates_negative"
    assert p["stance"] == "asserts"
    assert p["speaker_id"] == "spk-1"
    assert p["subject_canonical_id"] == "ent-apple"


def test_edge_param_carries_wire_type_and_bitemporal_stamps() -> None:
    e = _asserts_edge()
    p = edge_param(e)
    assert p["type"] == "asserts"
    assert p["edge_id"] == "e1"
    assert p["src_id"] == "spk-1"
    assert p["dst_id"] == "claim-1"
    # Bitemporal fields are all present (ISO strings via mode="json").
    for field in ("event_time", "ingestion_time", "valid_from", "valid_to", "invalidated"):
        assert field in p
    assert p["invalidated"] is False
    assert p["valid_to"] is None


# --------------------------------------------------------------------------- #
# Round-trip: param-out then record-in reproduces the pydantic record.
# --------------------------------------------------------------------------- #
def test_claim_param_round_trips_through_claim_from_record() -> None:
    original = _claim_node()
    rebuilt = claim_from_record(claim_param(original))
    assert rebuilt == original


def test_speaker_record_round_trip() -> None:
    spk = SpeakerNode(speaker_id="spk-1", name="Jane", is_host=True, wikidata_qid="Q42")
    assert speaker_from_record(speaker_param(spk)) == spk


def test_entity_record_round_trip() -> None:
    ent = EntityNode(
        canonical_id="ent-apple", name="Apple", type=EntityType.organization,
        aliases=["Apple"],
    )
    assert entity_from_record(entity_param(ent)) == ent


def test_edge_param_round_trips_through_edge_from_record() -> None:
    e = _asserts_edge()
    rebuilt = edge_from_record(edge_param(e))
    assert rebuilt == e


def test_speaker_and_entity_from_record_handle_none() -> None:
    assert speaker_from_record(None) is None
    assert entity_from_record(None) is None


# --------------------------------------------------------------------------- #
# _rows_to_results: the pure filter + bitemporal-liveness transform.
# --------------------------------------------------------------------------- #
def _row(claim: ClaimNode, edge: GraphEdge | None, *, speaker=None, subject=None) -> dict:
    """Build a raw Cypher-style row of flat property dicts."""
    return {
        "claim": claim_param(claim),
        "asserts_edge": edge_param(edge) if edge is not None else None,
        "speaker": speaker_param(speaker) if speaker is not None else None,
        "subject": entity_param(subject) if subject is not None else None,
    }


def test_rows_to_results_returns_only_claims_with_live_asserts_edge() -> None:
    live = _row(_claim_node("c1"), _asserts_edge("e1", "c1"))
    # A claim with no asserting edge is excluded from the current-state view.
    orphan = _row(_claim_node("c2", speaker="spk-2"), None)
    out = Neo4jStore._rows_to_results(
        [live, orphan],
        subject_canonical_id=None,
        speaker_id=None,
        predicate=None,
        as_of=None,
        include_invalidated=False,
    )
    assert [r.claim.claim_id for r in out] == ["c1"]


def test_rows_to_results_drops_invalidated_in_current_state_but_keeps_in_history() -> None:
    dead_edge = _asserts_edge("e2", "c2").model_copy(
        update={"invalidated": True, "valid_to": _dt(2026, 3, 1)}
    )
    rows = [
        _row(_claim_node("c1"), _asserts_edge("e1", "c1")),
        _row(_claim_node("c2"), dead_edge),
    ]
    # Current state: invalidated c2 is dropped.
    current = Neo4jStore._rows_to_results(
        rows, subject_canonical_id=None, speaker_id=None, predicate=None,
        as_of=None, include_invalidated=False,
    )
    assert {r.claim.claim_id for r in current} == {"c1"}

    # Point-in-time before the March invalidation: c2 reappears (history).
    history = Neo4jStore._rows_to_results(
        rows, subject_canonical_id=None, speaker_id=None, predicate=None,
        as_of=_dt(2026, 2, 1), include_invalidated=False,
    )
    assert {r.claim.claim_id for r in history} == {"c1", "c2"}

    # include_invalidated surfaces it regardless of time.
    forced = Neo4jStore._rows_to_results(
        rows, subject_canonical_id=None, speaker_id=None, predicate=None,
        as_of=None, include_invalidated=True,
    )
    assert {r.claim.claim_id for r in forced} == {"c1", "c2"}


def test_rows_to_results_filters_combine_with_and() -> None:
    rows = [
        _row(_claim_node("c1", speaker="spk-1", subject="ent-apple"),
             _asserts_edge("e1", "c1", speaker="spk-1")),
        _row(_claim_node("c2", speaker="spk-2", subject="ent-apple"),
             _asserts_edge("e2", "c2", speaker="spk-2")),
        _row(_claim_node("c3", speaker="spk-1", subject="ent-openai"),
             _asserts_edge("e3", "c3", speaker="spk-1")),
    ]
    by_subject = Neo4jStore._rows_to_results(
        rows, subject_canonical_id="ent-apple", speaker_id=None, predicate=None,
        as_of=None, include_invalidated=False,
    )
    assert {r.claim.claim_id for r in by_subject} == {"c1", "c2"}

    by_both = Neo4jStore._rows_to_results(
        rows, subject_canonical_id="ent-apple", speaker_id="spk-1", predicate=None,
        as_of=None, include_invalidated=False,
    )
    assert {r.claim.claim_id for r in by_both} == {"c1"}

    by_pred = Neo4jStore._rows_to_results(
        rows, subject_canonical_id=None, speaker_id=None,
        predicate=Predicate.rates_negative, as_of=None, include_invalidated=False,
    )
    assert {r.claim.claim_id for r in by_pred} == {"c1", "c2", "c3"}

    none_match = Neo4jStore._rows_to_results(
        rows, subject_canonical_id=None, speaker_id=None,
        predicate=Predicate.endorses, as_of=None, include_invalidated=False,
    )
    assert none_match == []


def test_rows_to_results_joins_speaker_subject_and_event_time() -> None:
    spk = SpeakerNode(speaker_id="spk-1", name="Jane")
    sub = EntityNode(canonical_id="ent-apple", name="Apple", type=EntityType.organization)
    edge = _asserts_edge("e1", "c1", event=_dt(2026, 1, 10))
    row = _row(_claim_node("c1"), edge, speaker=spk, subject=sub)
    (result,) = Neo4jStore._rows_to_results(
        [row], subject_canonical_id=None, speaker_id=None, predicate=None,
        as_of=None, include_invalidated=False,
    )
    assert result.speaker is not None and result.speaker.name == "Jane"
    assert result.subject is not None and result.subject.canonical_id == "ent-apple"
    assert result.event_time == _dt(2026, 1, 10)


def test_rows_to_results_is_sorted_by_claim_id() -> None:
    rows = [
        _row(_claim_node("c3"), _asserts_edge("e3", "c3")),
        _row(_claim_node("c1"), _asserts_edge("e1", "c1")),
        _row(_claim_node("c2"), _asserts_edge("e2", "c2")),
    ]
    out = Neo4jStore._rows_to_results(
        rows, subject_canonical_id=None, speaker_id=None, predicate=None,
        as_of=None, include_invalidated=False,
    )
    assert [r.claim.claim_id for r in out] == ["c1", "c2", "c3"]


def test_rows_to_results_skips_rows_with_no_claim() -> None:
    out = Neo4jStore._rows_to_results(
        [{"claim": None, "asserts_edge": None, "speaker": None, "subject": None}],
        subject_canonical_id=None, speaker_id=None, predicate=None,
        as_of=None, include_invalidated=False,
    )
    assert out == []


# --------------------------------------------------------------------------- #
# Constraints / index DDL.
# --------------------------------------------------------------------------- #
def test_constraint_statements_are_idempotent_and_cover_all_keys() -> None:
    stmts = constraint_statements()
    joined = "\n".join(stmts)
    # Every statement is re-runnable.
    assert all("IF NOT EXISTS" in s for s in stmts)
    # Uniqueness on each node's natural key.
    assert "REQUIRE n.speaker_id IS UNIQUE" in joined
    assert "REQUIRE n.canonical_id IS UNIQUE" in joined
    assert "REQUIRE n.claim_id IS UNIQUE" in joined
    # The shared-id index backing edge-endpoint MATCH and the edge_id index.
    assert f"ON (n.{SHARED_ID})" in joined
    assert "r.edge_id" in joined or "(r.edge_id)" in joined


# --------------------------------------------------------------------------- #
# Cypher constants: assert the structural shape the store relies on.
# --------------------------------------------------------------------------- #
def test_upsert_nodes_cypher_merges_by_natural_key_and_stamps_shared_label() -> None:
    c = UPSERT_NODES_CYPHER
    assert f"MERGE (n:{SPEAKER_LABEL} {{speaker_id: s.speaker_id}})" in c
    assert f"MERGE (n:{ENTITY_LABEL} {{canonical_id: e.canonical_id}})" in c
    assert f"MERGE (n:{CLAIM_LABEL} {{claim_id: c.claim_id}})" in c
    # Every node also gets the shared :GraphNode super-label so edges can MATCH
    # endpoints by __id__.
    assert c.count(f"n:{SHARED_LABEL}") == 3
    assert "UNWIND $speakers AS s" in c
    assert "UNWIND $entities AS e" in c
    assert "UNWIND $claims AS c" in c


def test_upsert_edges_cypher_matches_by_shared_id_and_merges_on_edge_id() -> None:
    c = UPSERT_EDGES_CYPHER
    assert f"MATCH (src:{SHARED_LABEL} {{{SHARED_ID}: e.src_id}})" in c
    assert f"MATCH (dst:{SHARED_LABEL} {{{SHARED_ID}: e.dst_id}})" in c
    assert f"MERGE (src)-[r:{EDGE_REL} {{edge_id: e.edge_id}}]->(dst)" in c
    # Re-loading a live edge must not resurrect an already-invalidated one.
    assert "coalesce(r.invalidated, false)" in c


def test_invalidate_edge_cypher_only_flips_live_edges() -> None:
    c = INVALIDATE_EDGE_CYPHER
    assert f"MATCH ()-[r:{EDGE_REL} {{edge_id: $edge_id}}]->()" in c
    assert "coalesce(r.invalidated, false) = false" in c
    assert "SET r.invalidated = true, r.valid_to = $at" in c
    assert "RETURN r.edge_id AS edge_id" in c  # so callers learn whether it changed


def test_query_and_all_edges_cypher_shapes() -> None:
    assert f"MATCH (c:{CLAIM_LABEL})" in QUERY_CLAIMS_CYPHER
    assert "type: 'asserts'" in QUERY_CLAIMS_CYPHER
    assert "AS asserts_edge" in QUERY_CLAIMS_CYPHER
    assert f"MATCH ()-[r:{EDGE_REL}]->()" in ALL_EDGES_CYPHER
    assert "properties(r) AS rel" in ALL_EDGES_CYPHER


# --------------------------------------------------------------------------- #
# Invalidate param map (datetime -> ISO).
# --------------------------------------------------------------------------- #
def test_build_invalidate_params_serializes_datetime() -> None:
    params = build_invalidate_params("e1", at=_dt(2026, 3, 1))
    assert params == {"edge_id": "e1", "at": "2026-03-01T00:00:00+00:00"}


# --------------------------------------------------------------------------- #
# bulk_load rejects the (unsupported) per-add LLM dedup path — no driver needed
# because the guard fires before any session is opened.
# --------------------------------------------------------------------------- #
def test_bulk_load_rejects_dedup_path_without_touching_driver() -> None:
    store = Neo4jStore(driver=None)  # a None driver proves no I/O happened
    with pytest.raises(NotImplementedError):
        store.bulk_load(
            speakers=[], entities=[], claims=[], edges=[], bypass_llm_dedup=False
        )


# --------------------------------------------------------------------------- #
# Live-DB integration — skipped unless a real Neo4j is configured. This is the
# only path that exercises the store against actual infra; its true first run
# is the one-episode smoke.
# --------------------------------------------------------------------------- #
_LIVE_URI = os.environ.get("NEO4J_TEST_URI")


@pytest.mark.skipif(
    not _LIVE_URI,
    reason="set NEO4J_TEST_URI (+ NEO4J_TEST_USER/PASSWORD) to run the live-DB test",
)
def test_live_neo4j_round_trip() -> None:  # pragma: no cover - requires a live Neo4j
    """Bulk-load one claim, query it back, invalidate, confirm current-state drop.

    Mirrors the FakeGraphStore contract tests against a real database. Runs only
    when NEO4J_TEST_URI is set; uses a throwaway database/labels are namespaced
    by the deterministic test ids so re-runs are idempotent.
    """
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        _LIVE_URI,
        auth=(
            os.environ.get("NEO4J_TEST_USER", "neo4j"),
            os.environ.get("NEO4J_TEST_PASSWORD", "neo4j"),
        ),
    )
    store = Neo4jStore(driver)
    try:
        store.bootstrap_constraints()
        speaker = SpeakerNode(speaker_id="spk-live", name="Jane")
        entity = EntityNode(
            canonical_id="ent-live", name="Apple", type=EntityType.organization
        )
        claim = _claim_node("claim-live", speaker="spk-live", subject="ent-live")
        edge = _asserts_edge("edge-live", "claim-live", speaker="spk-live")

        loaded = store.bulk_load(
            speakers=[speaker], entities=[entity], claims=[claim], edges=[edge]
        )
        assert loaded == 1

        rows = store.query(subject_canonical_id="ent-live")
        assert [r.claim.claim_id for r in rows] == ["claim-live"]
        assert rows[0].speaker is not None and rows[0].speaker.name == "Jane"
        assert rows[0].event_time == _dt(2026, 1, 10)

        # Idempotent re-load: still exactly one claim.
        assert store.bulk_load(
            speakers=[speaker], entities=[entity], claims=[claim], edges=[edge]
        ) == 1
        assert len(store.query(subject_canonical_id="ent-live")) == 1

        # Invalidate -> current-state drops it; history (as_of) still sees it.
        assert store.invalidate("edge-live", at=_dt(2026, 3, 1)) is True
        assert store.query(subject_canonical_id="ent-live") == []
        history = store.query(
            subject_canonical_id="ent-live", as_of=_dt(2026, 2, 1)
        )
        assert [r.claim.claim_id for r in history] == ["claim-live"]
        # Re-invalidating an already-dead edge is a no-op.
        assert store.invalidate("edge-live", at=_dt(2026, 5, 1)) is False
    finally:
        # Clean up the throwaway nodes so the DB is reusable.
        with driver.session() as session:
            session.run(
                f"MATCH (n:{SHARED_LABEL}) "
                "WHERE n.__id__ IN ['spk-live','ent-live','claim-live'] "
                "DETACH DELETE n"
            )
        driver.close()
