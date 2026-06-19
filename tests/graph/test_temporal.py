"""Tests for the bitemporal helpers (validity, invalidate-not-delete, current-state)."""

from __future__ import annotations

from datetime import datetime, timezone

from dlogos.graph.store import EdgeType, GraphEdge
from dlogos.graph.temporal import (
    apply_contradiction,
    current_state,
    invalidate_edge,
    is_live_at,
    set_validity_window,
)


def _dt(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)


def _edge(edge_id: str, event: datetime) -> GraphEdge:
    return GraphEdge(
        edge_id=edge_id,
        type=EdgeType.asserts,
        src_id="spk-1",
        dst_id="claim-1",
        event_time=event,
        ingestion_time=_dt(2026, 6, 18),
        valid_from=event,
    )


def test_set_validity_window_is_non_mutating() -> None:
    e = _edge("e1", _dt(2026, 1, 10))
    updated = set_validity_window(e, valid_to=_dt(2026, 3, 1))
    assert e.valid_to is None  # original untouched
    assert updated.valid_to == _dt(2026, 3, 1)
    # Setting a window does NOT mark the edge invalidated.
    assert updated.invalidated is False


def test_invalidate_sets_window_and_flag_without_delete() -> None:
    e = _edge("e1", _dt(2026, 1, 10))
    inv = invalidate_edge(e, at=_dt(2026, 3, 1))
    assert inv.invalidated is True
    assert inv.valid_to == _dt(2026, 3, 1)
    # Non-mutating: the original is preserved (history not rewritten in place).
    assert e.invalidated is False
    assert e.valid_to is None


def test_invalidate_is_idempotent_keeps_earliest() -> None:
    e = _edge("e1", _dt(2026, 1, 10))
    once = invalidate_edge(e, at=_dt(2026, 3, 1))
    twice = invalidate_edge(once, at=_dt(2026, 5, 1))
    # Re-invalidation does not move the original invalidation time forward.
    assert twice.valid_to == _dt(2026, 3, 1)


def test_is_live_at_current_state_excludes_invalidated() -> None:
    live = _edge("e1", _dt(2026, 1, 10))
    dead = invalidate_edge(_edge("e2", _dt(2026, 1, 10)), at=_dt(2026, 3, 1))
    # as_of=None means "current state": only non-invalidated edges are live.
    assert is_live_at(live, None) is True
    assert is_live_at(dead, None) is False


def test_is_live_at_point_in_time_surfaces_history() -> None:
    dead = invalidate_edge(_edge("e2", _dt(2026, 1, 10)), at=_dt(2026, 3, 1))
    # Before the invalidation, the fact was live (history is queryable).
    assert is_live_at(dead, _dt(2026, 2, 1)) is True
    # At/after the invalidation, it is no longer valid.
    assert is_live_at(dead, _dt(2026, 3, 1)) is False
    assert is_live_at(dead, _dt(2026, 4, 1)) is False


def test_is_live_at_respects_valid_from() -> None:
    future = _edge("e3", _dt(2026, 5, 1))
    # A point-in-time query before the fact started sees nothing.
    assert is_live_at(future, _dt(2026, 4, 1)) is False
    assert is_live_at(future, _dt(2026, 5, 1)) is True


def test_current_state_filters_to_live_set() -> None:
    e1 = _edge("e1", _dt(2026, 1, 10))
    e2 = invalidate_edge(_edge("e2", _dt(2026, 1, 10)), at=_dt(2026, 3, 1))
    e3 = _edge("e3", _dt(2026, 2, 1))
    live_now = current_state([e1, e2, e3], None)
    assert {e.edge_id for e in live_now} == {"e1", "e3"}
    # Order is preserved.
    assert [e.edge_id for e in live_now] == ["e1", "e3"]


def test_current_state_point_in_time_includes_then_live_edge() -> None:
    e1 = _edge("e1", _dt(2026, 1, 10))
    e2 = invalidate_edge(_edge("e2", _dt(2026, 1, 10)), at=_dt(2026, 3, 1))
    # As of Feb, the soon-to-be-invalidated edge was still live.
    as_of_feb = current_state([e1, e2], _dt(2026, 2, 1))
    assert {e.edge_id for e in as_of_feb} == {"e1", "e2"}


def test_apply_contradiction_invalidates_only_target_and_keeps_history() -> None:
    e1 = _edge("e1", _dt(2026, 1, 10))
    e2 = _edge("e2", _dt(2026, 2, 1))
    edges = [e1, e2]
    updated = apply_contradiction(
        edges, superseded_edge_id="e1", at=_dt(2026, 3, 1)
    )
    by_id = {e.edge_id: e for e in updated}
    assert by_id["e1"].invalidated is True
    assert by_id["e1"].valid_to == _dt(2026, 3, 1)
    assert by_id["e2"].invalidated is False
    # The invalidated edge is retained in the list (invalidate-not-delete).
    assert len(updated) == 2
    # Source list untouched.
    assert e1.invalidated is False


def test_apply_contradiction_no_match_is_noop() -> None:
    edges = [_edge("e1", _dt(2026, 1, 10))]
    updated = apply_contradiction(
        edges, superseded_edge_id="missing", at=_dt(2026, 3, 1)
    )
    assert [e.edge_id for e in updated] == ["e1"]
    assert updated[0].invalidated is False
