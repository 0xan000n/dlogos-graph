"""Bitemporal helpers: validity windows, invalidate-not-delete, current-state.

These are pure functions over :class:`~dlogos.graph.store.GraphEdge` lists so
they unit-test without any backend and so the fake store and the real Graphiti
store can share identical semantics. They encode the core temporal rules from
spec §6:

- A fact carries two independent time axes — *event-time* (when it was said)
  and *ingestion-time* (when we processed it) — plus a validity interval.
- A fact that stops being true is **invalidated** (``valid_to`` set,
  ``invalidated=True``), never deleted: history is preserved, no snapshots.
- Contradiction handling is **invalidation, not recompute**: new conflicting
  knowledge invalidates the prior edge.
- The **current-state filter** returns the latest non-invalidated edges valid
  at a point in time.

The anti-pattern the spec calls out — snapshotting graph state at intervals —
is deliberately *not* implemented; validity intervals replace snapshots.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from dlogos.graph.store import GraphEdge


def set_validity_window(
    edge: "GraphEdge",
    *,
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
) -> "GraphEdge":
    """Return a copy of ``edge`` with its validity window set.

    Non-mutating (returns a new edge) so callers never accidentally alias graph
    state. ``valid_from`` defaults to the edge's existing ``valid_from`` (or its
    ``event_time`` if unset); passing ``valid_to`` does NOT mark the edge
    invalidated — use :func:`invalidate_edge` for that, which is the
    semantically distinct "this fact stopped being true" operation.
    """
    new_from = valid_from if valid_from is not None else edge.valid_from
    update: dict[str, object] = {"valid_from": new_from}
    if valid_to is not None:
        update["valid_to"] = valid_to
    return edge.model_copy(update=update)


def invalidate_edge(edge: "GraphEdge", *, at: datetime) -> "GraphEdge":
    """Return a copy of ``edge`` invalidated as of ``at`` (never delete).

    Sets ``valid_to = at`` and ``invalidated = True``. Idempotent in effect: an
    already-invalidated edge is returned unchanged (its original ``valid_to`` is
    preserved so the *earliest* invalidation time wins — history is not
    rewritten by a later re-invalidation).
    """
    if edge.invalidated:
        return edge
    return edge.model_copy(update={"valid_to": at, "invalidated": True})


def is_live_at(edge: "GraphEdge", as_of: datetime | None = None) -> bool:
    """Whether ``edge`` is a live (non-invalidated) fact valid at ``as_of``.

    - An invalidated edge is live only *before* its ``valid_to`` (its history is
      visible to a point-in-time query that predates the invalidation).
    - With ``as_of=None`` the check is "live now": invalidated edges are not
      live; non-invalidated edges are live iff they have started
      (``valid_from <= now``-agnostic: we treat ``as_of=None`` as "latest",
      ignoring future ``valid_from`` only when an explicit ``as_of`` is given).
    """
    if as_of is None:
        # "Current state": only non-invalidated edges are live.
        return not edge.invalidated
    # Point-in-time: the edge must have started by as_of ...
    if edge.valid_from > as_of:
        return False
    # ... and not yet stopped being valid by as_of.
    if edge.valid_to is not None and edge.valid_to <= as_of:
        return False
    return True


def current_state(
    edges: Iterable["GraphEdge"], as_of: datetime | None = None
) -> list["GraphEdge"]:
    """Filter ``edges`` to the live set (latest non-invalidated).

    With ``as_of=None`` this returns every non-invalidated edge — the
    current-state fast path (spec §6). With an explicit ``as_of`` it returns the
    edges valid at that instant, so an invalidated edge still shows up for a
    query that predates its invalidation. Order is preserved.
    """
    return [e for e in edges if is_live_at(e, as_of)]


def apply_contradiction(
    edges: list["GraphEdge"],
    *,
    superseded_edge_id: str,
    at: datetime,
) -> list["GraphEdge"]:
    """Invalidate the edge identified by ``superseded_edge_id`` as of ``at``.

    Contradiction = invalidation, not recompute (spec §6). Returns a NEW list
    with the matching edge replaced by its invalidated copy; all other edges
    pass through unchanged, and history (the invalidated edge) is retained.
    If no edge matches the id, the list is returned with identical contents.
    """
    result: list["GraphEdge"] = []
    for edge in edges:
        if edge.edge_id == superseded_edge_id:
            result.append(invalidate_edge(edge, at=at))
        else:
            result.append(edge)
    return result
