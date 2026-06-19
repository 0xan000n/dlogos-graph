"""dLogos MCP server surface (spec §8, §9).

The corpus is queryable from Claude on day one through a small set of MCP
tools: ``search_dialogue``, ``who_discussed``, ``consensus_trend``,
``belief_history``, ``provenance_lookup``.

Design split (so the tools are unit-testable without the ``mcp`` package):

- The MCP protocol object (built by :func:`dlogos.mcp.server.build_server`)
  imports ``mcp`` **lazily**, inside the function, and registers each tool as a
  thin wrapper that delegates to a plain *handler* function.
- The handler functions (:mod:`dlogos.mcp.server`) take an injected
  :class:`~dlogos.mcp.server.RetrievalSurface` and return plain Pydantic result
  models. They never import ``mcp``, so tests exercise them directly against a
  fake retrieval surface with the core dependency group only.

Nothing here imports a heavy/optional dependency at module load time.
"""

from __future__ import annotations

from dlogos.mcp.server import (
    BeliefHistoryResult,
    ConsensusTrendResult,
    DialogueHit,
    ProvenanceResult,
    RetrievalSurface,
    SearchDialogueResult,
    TrendPoint,
    WhoDiscussedResult,
    belief_history_handler,
    build_server,
    consensus_trend_handler,
    provenance_lookup_handler,
    search_dialogue_handler,
    who_discussed_handler,
)

__all__ = [
    "BeliefHistoryResult",
    "ConsensusTrendResult",
    "DialogueHit",
    "ProvenanceResult",
    "RetrievalSurface",
    "SearchDialogueResult",
    "TrendPoint",
    "WhoDiscussedResult",
    "belief_history_handler",
    "build_server",
    "consensus_trend_handler",
    "provenance_lookup_handler",
    "search_dialogue_handler",
    "who_discussed_handler",
]
