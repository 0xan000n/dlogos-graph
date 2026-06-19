"""Retrieval over the dLogos temporal dialogue graph (spec §8).

Two surfaces:

- :mod:`dlogos.retrieval.hybrid` — hybrid retrieval over a ``GraphStore``:
  semantic similarity (an injected embedder) + BM25/TF-IDF lexical matching +
  breadth-first graph traversal, fused with **Reciprocal Rank Fusion**, with an
  optional temporal filter on claim validity windows.

- :mod:`dlogos.retrieval.consensus` — the headline primitive: a *pure*
  function that buckets resolved claims about a subject over time windows and
  computes the net stance/sentiment trend plus the attributed speakers per
  bucket. This is what powers "how has consensus on X moved".

Nothing here imports a heavy/optional dependency at module load time; the only
hard dependencies are the core group (numpy + the shared pydantic schema).
"""

from __future__ import annotations

from dlogos.retrieval.consensus import (
    ConsensusBucket,
    ConsensusTrend,
    TrendDirection,
    consensus_over_time,
)
from dlogos.retrieval.hybrid import (
    GraphStore,
    HybridRetriever,
    RetrievableClaim,
    RetrievalResult,
    TemporalMode,
    claims_from_graph_store,
    in_memory_store,
    reciprocal_rank_fusion,
)

__all__ = [
    # consensus
    "ConsensusBucket",
    "ConsensusTrend",
    "TrendDirection",
    "consensus_over_time",
    # hybrid
    "GraphStore",
    "HybridRetriever",
    "RetrievableClaim",
    "RetrievalResult",
    "TemporalMode",
    "claims_from_graph_store",
    "in_memory_store",
    "reciprocal_rank_fusion",
]
