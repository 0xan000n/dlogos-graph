"""The dLogos temporal graph subpackage.

This package owns the bitemporal store seam (spec §6, §7.5, §7.6):

- :mod:`dlogos.graph.store` — the :class:`GraphStore` ``Protocol`` plus the
  real Graphiti/Neo4j-backed implementation (heavy deps imported *lazily*).
- :mod:`dlogos.graph.loader` — Approach-B loader mapping a *resolved*
  :class:`~dlogos.schema.ExtractedClaim` to a reified ``Claim`` node plus
  ``Speaker``/canonical ``Entity`` nodes and bitemporal edges, with an
  explicit **bulk** fast path that bypasses per-add LLM node-dedup.
- :mod:`dlogos.graph.temporal` — bitemporal helpers: validity windows,
  invalidate-not-delete on contradiction, and the current-state filter.
- :mod:`dlogos.graph.fake_store` — an in-memory :class:`GraphStore` for tests.

Only :mod:`dlogos.graph.store`'s real backend touches ``graphiti-core`` /
``neo4j``, and only inside functions. Importing this package (or any of its
modules) requires the *core* dependency group alone, so unit tests run without
the optional ``graph`` extra installed.
"""

from __future__ import annotations

from dlogos.graph.loader import ClaimLoader, GraphTriplet
from dlogos.graph.store import GraphStore, QueryResult
from dlogos.graph.temporal import (
    apply_contradiction,
    current_state,
    invalidate_edge,
    set_validity_window,
)

__all__ = [
    "GraphStore",
    "QueryResult",
    "GraphTriplet",
    "ClaimLoader",
    "set_validity_window",
    "invalidate_edge",
    "apply_contradiction",
    "current_state",
]
