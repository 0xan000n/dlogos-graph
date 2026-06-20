"""Tests for the vis-network graph export (src/dlogos/graph/export.py).

These build a real :class:`FakeGraphStore`, ``bulk_load`` a small but realistic
slice of the dialogue ontology (two speakers, two entities, two claims, and the
asserts/about/mentions/disputes/supersedes/appears_in edges between them), and
assert the export carries the right node groups, edge labels, claim snippets,
and grounded spans. A round-trip test confirms ``write_graph_json`` emits valid
stdlib JSON to a tmp file.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from dlogos.graph.export import export_graph, write_graph_json
from dlogos.graph.fake_store import FakeGraphStore
from dlogos.graph.store import (
    ClaimNode,
    EdgeType,
    EntityNode,
    GraphEdge,
    SpeakerNode,
)
from dlogos.schema import EntityType, Predicate, SourceSpan, Stance


def _dt(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Fixtures-as-builders
# --------------------------------------------------------------------------- #
def _speakers() -> list[SpeakerNode]:
    return [
        SpeakerNode(speaker_id="spk-host", name="Alice Host", is_host=True),
        SpeakerNode(speaker_id="spk-guest", name="Bob Guest", wikidata_qid="Q42"),
    ]


def _entities() -> list[EntityNode]:
    return [
        EntityNode(
            canonical_id="ent-apple",
            name="Apple",
            type=EntityType.organization,
            aliases=["AAPL", "Apple Inc."],
        ),
        EntityNode(canonical_id="ent-ai", name="AI", type=EntityType.concept),
    ]


def _claims() -> list[ClaimNode]:
    return [
        ClaimNode(
            claim_id="c1",
            predicate=Predicate.rates_negative,
            stance=Stance.asserts,
            object="Apple hardware has plateaued and innovation stalled",
            sentiment=-0.6,
            confidence=0.82,
            source_span=SourceSpan(
                episode_id="ep-1", t_start=190.0, t_end=200.0
            ),
            speaker_id="spk-host",
            subject_canonical_id="ent-apple",
        ),
        ClaimNode(
            claim_id="c2",
            predicate=Predicate.rates_positive,
            stance=Stance.disputes,
            object="Apple silicon is the most exciting hardware in a decade",
            sentiment=0.7,
            confidence=0.75,
            source_span=SourceSpan(
                episode_id="ep-1", t_start=205.5, t_end=219.2
            ),
            speaker_id="spk-guest",
            subject_canonical_id="ent-apple",
        ),
    ]


def _edges() -> list[GraphEdge]:
    e = _dt(2026, 1, 10)
    common = dict(event_time=e, ingestion_time=_dt(2026, 6, 18), valid_from=e)
    return [
        GraphEdge(edge_id="e1", type=EdgeType.asserts, src_id="spk-host", dst_id="c1", **common),
        GraphEdge(edge_id="e2", type=EdgeType.about, src_id="c1", dst_id="ent-apple", **common),
        GraphEdge(edge_id="e3", type=EdgeType.mentions, src_id="c1", dst_id="ent-ai", **common),
        GraphEdge(edge_id="e4", type=EdgeType.asserts, src_id="spk-guest", dst_id="c2", **common),
        GraphEdge(edge_id="e5", type=EdgeType.about, src_id="c2", dst_id="ent-apple", **common),
        GraphEdge(edge_id="e6", type=EdgeType.disputes, src_id="c2", dst_id="c1", **common),
        GraphEdge(edge_id="e7", type=EdgeType.supersedes, src_id="c2", dst_id="c1", **common),
        GraphEdge(edge_id="e8", type=EdgeType.appears_in, src_id="spk-host", dst_id="ep-1", **common),
    ]


def _loaded_store() -> FakeGraphStore:
    store = FakeGraphStore()
    store.bulk_load(
        speakers=_speakers(),
        entities=_entities(),
        claims=_claims(),
        edges=_edges(),
    )
    return store


# --------------------------------------------------------------------------- #
# export_graph
# --------------------------------------------------------------------------- #
def test_export_shape_and_keys() -> None:
    graph = export_graph(_loaded_store())
    assert set(graph) == {"nodes", "edges"}
    for node in graph["nodes"]:
        assert set(node) == {"id", "label", "group", "title"}
    # Edges carry the vis-network keys (plus an internal id used for sorting).
    for edge in graph["edges"]:
        assert {"from", "to", "label", "title"} <= set(edge)


def test_node_groups_cover_all_types() -> None:
    graph = export_graph(_loaded_store())
    by_id = {n["id"]: n for n in graph["nodes"]}

    assert by_id["spk-host"]["group"] == "speaker"
    assert by_id["spk-guest"]["group"] == "speaker"
    assert by_id["ent-apple"]["group"] == "entity"
    assert by_id["ent-ai"]["group"] == "entity"
    assert by_id["c1"]["group"] == "claim"
    assert by_id["c2"]["group"] == "claim"

    groups = {n["group"] for n in graph["nodes"]}
    assert groups == {"speaker", "entity", "claim"}


def test_node_count_matches_store() -> None:
    store = _loaded_store()
    graph = export_graph(store)
    # 2 speakers + 2 entities + 2 claims = 6 nodes (one per stored node, no dups).
    expected = len(store.speakers) + len(store.entities) + len(store.claims)
    assert len(graph["nodes"]) == expected == 6


def test_speaker_and_entity_labels() -> None:
    graph = export_graph(_loaded_store())
    by_id = {n["id"]: n for n in graph["nodes"]}
    # Speaker label uses the resolved name when present.
    assert by_id["spk-host"]["label"] == "Alice Host"
    assert "host" in by_id["spk-host"]["title"]
    # Entity label is the canonical name; aliases ride in the title.
    assert by_id["ent-apple"]["label"] == "Apple"
    assert "AAPL" in by_id["ent-apple"]["title"]


def test_claim_label_is_short_snippet() -> None:
    graph = export_graph(_loaded_store())
    c1 = next(n for n in graph["nodes"] if n["id"] == "c1")
    # The on-node label is a short snippet, not the whole claim.
    assert len(c1["label"]) <= 61  # snippet budget + ellipsis
    assert c1["label"]  # non-empty
    # It is derived from the claim content (predicate/object), not the raw id.
    assert c1["label"] != "c1"


def test_claim_title_carries_span_and_speaker() -> None:
    graph = export_graph(_loaded_store())
    by_id = {n["id"]: n for n in graph["nodes"]}

    c1_title = by_id["c1"]["title"]
    # Full claim text present (not truncated in the title).
    assert "plateaued" in c1_title
    # Grounded span carried verbatim.
    assert "ep-1" in c1_title
    assert "190.0" in c1_title
    assert "200.0" in c1_title
    # Attributed speaker carried.
    assert "spk-host" in c1_title

    c2_title = by_id["c2"]["title"]
    assert "205.5" in c2_title
    assert "219.2" in c2_title
    assert "spk-guest" in c2_title


def test_edge_labels_are_edge_types() -> None:
    graph = export_graph(_loaded_store())
    labels = {e["label"] for e in graph["edges"]}
    assert labels == {
        "asserts",
        "about",
        "mentions",
        "disputes",
        "supersedes",
        "appears_in",
    }


def test_edges_connect_expected_endpoints() -> None:
    graph = export_graph(_loaded_store())
    by_eid = {e["id"]: e for e in graph["edges"]}
    # asserts: speaker -> claim
    assert (by_eid["e1"]["from"], by_eid["e1"]["to"]) == ("spk-host", "c1")
    # about: claim -> entity
    assert (by_eid["e2"]["from"], by_eid["e2"]["to"]) == ("c1", "ent-apple")
    # disputes: claim -> claim
    assert (by_eid["e6"]["from"], by_eid["e6"]["to"]) == ("c2", "c1")
    assert by_eid["e6"]["label"] == "disputes"


def test_export_is_deterministic_and_sorted() -> None:
    store = _loaded_store()
    first = export_graph(store)
    second = export_graph(store)
    assert first == second
    # Sorted by id for stable output.
    node_ids = [n["id"] for n in first["nodes"]]
    assert node_ids == sorted(node_ids)
    edge_ids = [e["id"] for e in first["edges"]]
    assert edge_ids == sorted(edge_ids)


def test_empty_store_exports_empty_lists() -> None:
    graph = export_graph(FakeGraphStore())
    assert graph == {"nodes": [], "edges": []}


# --------------------------------------------------------------------------- #
# write_graph_json
# --------------------------------------------------------------------------- #
def test_write_graph_json_round_trips(tmp_path) -> None:
    store = _loaded_store()
    out = tmp_path / "graph.json"
    write_graph_json(store, out)

    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))

    # The file content equals the in-memory export exactly.
    assert loaded == export_graph(store)
    assert {n["id"] for n in loaded["nodes"]} == {
        "spk-host",
        "spk-guest",
        "ent-apple",
        "ent-ai",
        "c1",
        "c2",
    }
    assert len(loaded["edges"]) == 8


def test_write_graph_json_is_stable_bytes(tmp_path) -> None:
    store = _loaded_store()
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    write_graph_json(store, a)
    write_graph_json(store, b)
    # Deterministic serialization: byte-identical across writes.
    assert a.read_bytes() == b.read_bytes()
