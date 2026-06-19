"""Direct Neo4j implementation of the :class:`GraphStore` contract (smoke path).

A driver-level Neo4j store we control end-to-end — deliberately **not** Graphiti
(:class:`~dlogos.graph.store.GraphitiStore`). The smoke run isolates dLogos's own
graph logic from the separate Graphiti-integration spike, so this backend speaks
Cypher we write directly against the official ``neo4j`` Python driver. It mirrors
:class:`~dlogos.graph.fake_store.FakeGraphStore`'s *semantics* exactly — idempotent
upsert by id, bitemporal invalidate-not-delete, current-state-vs-point-in-time
querying, dedup-bypass bulk load — so retrieval, the loader, and the eval arms run
against it unchanged.

Heavy imports are lazy:

- ``neo4j`` is imported **inside** :meth:`connect` / the session helpers only, so
  importing this module needs the *core* dependency group alone. Construction via
  :meth:`connect`; tests drive the pure Cypher/param builders (the ``build_*``
  functions and ``_*_cypher`` constants) with no driver at all, and an optional
  live-DB integration test guards itself behind a ``NEO4J_*`` env check.

Reification & bitemporality (spec §6) are preserved on the wire:

- ``Claim`` is a *node* (not an edge) carrying stance/sentiment/confidence/span.
- ``Speaker`` / ``Entity`` nodes ``MERGE`` by their canonical id (idempotent).
- Every relationship (``asserts`` / ``about`` / ``appears_in`` / ``mentions`` /
  ``agrees_with`` / ``disputes`` / ``supersedes``) carries the bitemporal stamps
  ``event_time`` / ``ingestion_time`` / ``valid_from`` / ``valid_to`` / ``invalidated``.
- Invalidation sets ``valid_to`` + ``invalidated`` and never deletes — history is
  preserved so a point-in-time query that predates the invalidation still sees it.

The Cypher uses a single ``:GraphNode`` super-label carrying a ``__id__`` property
(the node's canonical id, whichever flavour it is) so edges can ``MATCH`` endpoints
by one indexed property regardless of node type, exactly the pattern the
relationship upsert needs.

These real-infra paths CANNOT be exercised offline (no live Neo4j here): their
unit tests cover the pure Cypher/param *building*; their first real execution is
the one-episode smoke run itself.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Iterable

from dlogos.graph.store import (
    ClaimNode,
    EntityNode,
    GraphEdge,
    QueryResult,
    SpeakerNode,
)
from dlogos.graph.temporal import is_live_at
from dlogos.schema import Predicate, SourceSpan

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from dlogos.config import Settings


# --------------------------------------------------------------------------- #
# Node labels & the shared id property.
#
# Every node also carries the `:GraphNode {__id__}` super-label so an edge can
# MATCH its endpoints by one indexed property, whatever the endpoint's type.
# --------------------------------------------------------------------------- #
SHARED_LABEL = "GraphNode"
SHARED_ID = "__id__"
SPEAKER_LABEL = "Speaker"
ENTITY_LABEL = "Entity"
CLAIM_LABEL = "Claim"
EDGE_REL = "REL"


# --------------------------------------------------------------------------- #
# Constraints / indexes bootstrap (idempotent).
# --------------------------------------------------------------------------- #
def constraint_statements() -> list[str]:
    """Cypher DDL to bootstrap uniqueness constraints + lookup indexes.

    All ``IF NOT EXISTS`` so re-running is a no-op. A uniqueness constraint on
    each node's natural key makes ``MERGE`` cheap and idempotent; the shared
    ``__id__`` index backs the edge-endpoint ``MATCH``; the ``edge_id`` index
    backs invalidation and idempotent edge upsert.
    """
    return [
        f"CREATE CONSTRAINT speaker_id IF NOT EXISTS "
        f"FOR (n:{SPEAKER_LABEL}) REQUIRE n.speaker_id IS UNIQUE",
        f"CREATE CONSTRAINT entity_id IF NOT EXISTS "
        f"FOR (n:{ENTITY_LABEL}) REQUIRE n.canonical_id IS UNIQUE",
        f"CREATE CONSTRAINT claim_id IF NOT EXISTS "
        f"FOR (n:{CLAIM_LABEL}) REQUIRE n.claim_id IS UNIQUE",
        f"CREATE INDEX shared_id IF NOT EXISTS "
        f"FOR (n:{SHARED_LABEL}) ON (n.{SHARED_ID})",
        f"CREATE INDEX rel_edge_id IF NOT EXISTS "
        f"FOR ()-[r:{EDGE_REL}]-() ON (r.edge_id)",
    ]


# --------------------------------------------------------------------------- #
# Pure node/edge param builders (no driver; unit-tested directly).
#
# Each builder flattens a pydantic record into a Cypher-friendly param map. We
# project a `__id__` onto every node param so the shared-label MERGE/MATCH and
# the edge endpoint MATCH all key off one property.
# --------------------------------------------------------------------------- #
def speaker_param(speaker: SpeakerNode) -> dict[str, Any]:
    """Flatten a :class:`SpeakerNode` to a MERGE param map (with ``__id__``)."""
    data = speaker.model_dump(mode="json")
    data[SHARED_ID] = speaker.speaker_id
    return data


def entity_param(entity: EntityNode) -> dict[str, Any]:
    """Flatten an :class:`EntityNode` to a MERGE param map (with ``__id__``)."""
    data = entity.model_dump(mode="json")
    data[SHARED_ID] = entity.canonical_id
    return data


def claim_param(claim: ClaimNode) -> dict[str, Any]:
    """Flatten a :class:`ClaimNode` to a MERGE param map.

    Neo4j has no nested-map property type, so the reified claim's
    ``source_span`` is flattened to scalar ``span_*`` columns (and rehydrated by
    :func:`claim_from_record` on read). ``__id__`` mirrors ``claim_id``.
    """
    data = claim.model_dump(mode="json")
    span = data.pop("source_span")
    data["span_episode_id"] = span["episode_id"]
    data["span_t_start"] = span["t_start"]
    data["span_t_end"] = span["t_end"]
    data["span_transcript_offset"] = span["transcript_offset"]
    data[SHARED_ID] = claim.claim_id
    # Enum -> wire value (predicate/stance already serialized by mode="json").
    return data


def edge_param(edge: GraphEdge) -> dict[str, Any]:
    """Flatten a :class:`GraphEdge` to a relationship-upsert param map."""
    data = edge.model_dump(mode="json")
    data["type"] = edge.type.value
    return data


# --------------------------------------------------------------------------- #
# Pure read-side rehydration (record dict -> pydantic). Unit-tested directly.
# --------------------------------------------------------------------------- #
def claim_from_record(node: dict[str, Any]) -> ClaimNode:
    """Rebuild a :class:`ClaimNode` from a flat Neo4j node-property dict."""
    return ClaimNode(
        claim_id=node["claim_id"],
        predicate=Predicate(node["predicate"]),
        stance=node["stance"],
        object=node["object"],
        sentiment=node["sentiment"],
        confidence=node["confidence"],
        source_span=SourceSpan(
            episode_id=node["span_episode_id"],
            t_start=node["span_t_start"],
            t_end=node["span_t_end"],
            transcript_offset=node.get("span_transcript_offset"),
        ),
        speaker_id=node["speaker_id"],
        subject_canonical_id=node["subject_canonical_id"],
    )


def speaker_from_record(node: dict[str, Any] | None) -> SpeakerNode | None:
    """Rebuild a :class:`SpeakerNode` from a node dict, or ``None``."""
    if not node:
        return None
    return SpeakerNode(
        speaker_id=node["speaker_id"],
        name=node.get("name"),
        is_host=node.get("is_host", False),
        wikidata_qid=node.get("wikidata_qid"),
    )


def entity_from_record(node: dict[str, Any] | None) -> EntityNode | None:
    """Rebuild an :class:`EntityNode` from a node dict, or ``None``."""
    if not node:
        return None
    return EntityNode(
        canonical_id=node["canonical_id"],
        name=node["name"],
        type=node["type"],
        aliases=list(node.get("aliases") or []),
    )


def edge_from_record(rel: dict[str, Any]) -> GraphEdge:
    """Rebuild a :class:`GraphEdge` from a flat relationship-property dict."""
    return GraphEdge(
        edge_id=rel["edge_id"],
        type=rel["type"],
        src_id=rel["src_id"],
        dst_id=rel["dst_id"],
        event_time=rel["event_time"],
        ingestion_time=rel["ingestion_time"],
        valid_from=rel["valid_from"],
        valid_to=rel.get("valid_to"),
        invalidated=rel.get("invalidated", False),
    )


# --------------------------------------------------------------------------- #
# Parameterized, dedup-free Cypher (the bulk / per-add write path).
#
# Kept as module constants so a test can assert the exact text and so the write
# path is reviewable in one place. Node upsert MERGEs by natural key AND stamps
# the shared `:GraphNode {__id__}` super-label so edges can MATCH by `__id__`.
# --------------------------------------------------------------------------- #
UPSERT_NODES_CYPHER = f"""
UNWIND $speakers AS s
  MERGE (n:{SPEAKER_LABEL} {{speaker_id: s.speaker_id}})
  SET n += s, n:{SHARED_LABEL}
WITH count(*) AS _s
UNWIND $entities AS e
  MERGE (n:{ENTITY_LABEL} {{canonical_id: e.canonical_id}})
  SET n += e, n:{SHARED_LABEL}
WITH count(*) AS _e
UNWIND $claims AS c
  MERGE (n:{CLAIM_LABEL} {{claim_id: c.claim_id}})
  SET n += c, n:{SHARED_LABEL}
"""

# Idempotent edge upsert. MERGE keys only on `edge_id` so re-loads do not
# duplicate; `coalesce` preserves a prior invalidation so a re-load of a live
# edge never resurrects (clobbers) an already-invalidated one — mirroring
# FakeGraphStore._upsert_edge.
UPSERT_EDGES_CYPHER = f"""
UNWIND $edges AS e
  MATCH (src:{SHARED_LABEL} {{{SHARED_ID}: e.src_id}})
  MATCH (dst:{SHARED_LABEL} {{{SHARED_ID}: e.dst_id}})
  MERGE (src)-[r:{EDGE_REL} {{edge_id: e.edge_id}}]->(dst)
  SET r.type = e.type,
      r.src_id = e.src_id,
      r.dst_id = e.dst_id,
      r.event_time = e.event_time,
      r.ingestion_time = e.ingestion_time,
      r.valid_from = e.valid_from,
      r.valid_to = CASE WHEN coalesce(r.invalidated, false)
                        THEN r.valid_to ELSE e.valid_to END,
      r.invalidated = coalesce(r.invalidated, false) OR e.invalidated
"""

# Invalidate-not-delete. Only flips a still-live edge; returns the id so the
# caller can tell whether anything changed (False when absent/already dead).
INVALIDATE_EDGE_CYPHER = f"""
MATCH ()-[r:{EDGE_REL} {{edge_id: $edge_id}}]->()
WHERE coalesce(r.invalidated, false) = false
SET r.invalidated = true, r.valid_to = $at
RETURN r.edge_id AS edge_id
"""

# Read every claim with its asserting speaker, subject entity, and the
# asserting edge (so the temporal filter runs in Python over the bitemporal
# stamps — identical logic to FakeGraphStore.query, one source of truth via
# is_live_at). LEFT joins keep a claim even if a node is momentarily missing.
QUERY_CLAIMS_CYPHER = f"""
MATCH (c:{CLAIM_LABEL})
OPTIONAL MATCH (spk:{SPEAKER_LABEL})-[a:{EDGE_REL} {{type: 'asserts'}}]->(c)
OPTIONAL MATCH (sub:{ENTITY_LABEL} {{canonical_id: c.subject_canonical_id}})
RETURN c AS claim, spk AS speaker, sub AS subject, a AS asserts_edge
"""

# All relationships flattened to property maps — backs the `edges` accessor the
# retrieval adapter (claims_from_graph_store) reads to build claim adjacency.
ALL_EDGES_CYPHER = f"""
MATCH ()-[r:{EDGE_REL}]->()
RETURN properties(r) AS rel
"""


def build_invalidate_params(edge_id: str, *, at: datetime) -> dict[str, Any]:
    """Param map for :data:`INVALIDATE_EDGE_CYPHER` (datetime -> ISO string)."""
    return {"edge_id": edge_id, "at": at.isoformat()}


def _node_props(value: Any) -> dict[str, Any] | None:
    """Coerce a driver node/record value to a plain property dict, or ``None``.

    The ``neo4j`` driver returns ``Node`` objects that are ``Mapping``-like
    (``dict(node)`` yields its properties); a ``None`` OPTIONAL MATCH stays
    ``None``. Keeping this tolerant lets tests feed plain dicts.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    # neo4j.graph.Node and Relationship are Mapping-like.
    try:
        return dict(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


# --------------------------------------------------------------------------- #
# The store.
# --------------------------------------------------------------------------- #
class Neo4jStore:
    """Driver-level Neo4j implementation of :class:`GraphStore`.

    Structurally a ``GraphStore`` (``isinstance(store, GraphStore)`` is ``True``
    via ``@runtime_checkable``). Construct with :meth:`connect`; close with
    :meth:`close` (or use as a context manager). The ``neo4j`` driver symbol is
    imported lazily so importing this module needs only the core deps.
    """

    def __init__(self, driver: Any, *, database: str | None = None) -> None:
        self._driver = driver
        self._database = database

    # -- construction / lifecycle ------------------------------------------- #
    @classmethod
    def connect(
        cls, settings: "Settings | None" = None, *, database: str | None = None
    ) -> "Neo4jStore":
        """Open a Neo4j driver from settings (lazy ``neo4j`` import).

        Raises a clear :class:`ImportError` if the optional ``graph`` extra is
        not installed. ``NEO4J_URI`` / ``NEO4J_USER`` / ``NEO4J_PASSWORD`` come
        from :class:`~dlogos.config.Settings`.
        """
        if settings is None:
            from dlogos.config import settings as default_settings

            settings = default_settings
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "Neo4jStore requires the optional 'graph' extra (neo4j). "
                "Install with: uv sync --extra graph"
            ) from exc

        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        return cls(driver, database=database)

    def close(self) -> None:  # pragma: no cover - requires a live driver
        """Close the underlying driver."""
        self._driver.close()

    def __enter__(self) -> "Neo4jStore":  # pragma: no cover - convenience
        return self

    def __exit__(self, *exc: object) -> None:  # pragma: no cover - convenience
        self.close()

    # -- internal session helper -------------------------------------------- #
    def _session(self) -> Any:  # pragma: no cover - requires a live driver
        if self._database:
            return self._driver.session(database=self._database)
        return self._driver.session()

    def _run(self, cypher: str, /, **params: Any) -> list[dict[str, Any]]:  # pragma: no cover - requires a live driver
        """Run one statement and return its rows as plain dicts."""
        with self._session() as session:
            result = session.run(cypher, **params)
            return [record.data() for record in result]

    # -- bootstrap ---------------------------------------------------------- #
    def bootstrap_constraints(self) -> None:  # pragma: no cover - requires a live driver
        """Create the uniqueness constraints + indexes (idempotent).

        Run once before the first load. Each statement is ``IF NOT EXISTS`` so
        repeated calls are no-ops.
        """
        with self._session() as session:
            for stmt in constraint_statements():
                session.run(stmt)

    # -- contract: per-add path --------------------------------------------- #
    def add_claim_triplet(
        self,
        claim: ClaimNode,
        *,
        speaker: SpeakerNode,
        subject: EntityNode,
        edges: list[GraphEdge],
        mentions: list[EntityNode] | None = None,
    ) -> None:  # pragma: no cover - requires a live Neo4j; first run is the smoke
        """Per-add path: upsert one resolved claim and its nodes/edges."""
        entities = [subject, *(mentions or [])]
        with self._session() as session:
            session.run(
                UPSERT_NODES_CYPHER,
                speakers=[speaker_param(speaker)],
                entities=[entity_param(e) for e in entities],
                claims=[claim_param(claim)],
            )
            if edges:
                session.run(
                    UPSERT_EDGES_CYPHER,
                    edges=[edge_param(e) for e in edges],
                )

    # -- contract: bulk fast path ------------------------------------------- #
    def bulk_load(
        self,
        *,
        speakers: list[SpeakerNode],
        entities: list[EntityNode],
        claims: list[ClaimNode],
        edges: list[GraphEdge],
        bypass_llm_dedup: bool = True,
    ) -> int:  # pragma: no cover - requires a live Neo4j; first run is the smoke
        """Load a PRE-RESOLVED batch in one shot; return claims loaded.

        ``bypass_llm_dedup=True`` is the only supported path: resolution already
        ran upstream, so the store performs a plain idempotent upsert with NO
        per-add LLM node-dedup (spec §7.5/§7.6). Passing ``False`` is rejected
        — there is no per-add LLM resolution in this direct backend (use
        :meth:`add_claim_triplet` for the incremental path). Mirrors
        :class:`~dlogos.graph.store.GraphitiStore.bulk_load`.
        """
        if not bypass_llm_dedup:
            raise NotImplementedError(
                "Neo4jStore.bulk_load only supports the dedup-bypass fast path; "
                "per-add LLM dedup is intentionally disabled (spec §7.5/§7.6). "
                "Resolution runs upstream in our resolution module."
            )
        with self._session() as session:
            session.run(
                UPSERT_NODES_CYPHER,
                speakers=[speaker_param(s) for s in speakers],
                entities=[entity_param(e) for e in entities],
                claims=[claim_param(c) for c in claims],
            )
            if edges:
                session.run(
                    UPSERT_EDGES_CYPHER,
                    edges=[edge_param(e) for e in edges],
                )
        return len(claims)

    # -- contract: query ---------------------------------------------------- #
    def query(
        self,
        *,
        subject_canonical_id: str | None = None,
        speaker_id: str | None = None,
        predicate: Predicate | None = None,
        as_of: datetime | None = None,
        include_invalidated: bool = False,
    ) -> list[QueryResult]:  # pragma: no cover - requires a live Neo4j; first run is the smoke
        """Return claim rows matching the filters, current-state by default.

        A claim is included only if it has at least one live ``asserts`` edge
        under the requested temporal view, unless ``include_invalidated`` is
        set. Temporal liveness is decided in Python by
        :func:`~dlogos.graph.temporal.is_live_at` over the bitemporal stamps,
        so it is byte-for-byte the same rule the fake store uses. Filters
        combine with AND; rows are sorted by ``claim_id`` for stable output.
        """
        rows = self._run(QUERY_CLAIMS_CYPHER)
        return self._rows_to_results(
            rows,
            subject_canonical_id=subject_canonical_id,
            speaker_id=speaker_id,
            predicate=predicate,
            as_of=as_of,
            include_invalidated=include_invalidated,
        )

    @staticmethod
    def _rows_to_results(
        rows: Iterable[dict[str, Any]],
        *,
        subject_canonical_id: str | None,
        speaker_id: str | None,
        predicate: Predicate | None,
        as_of: datetime | None,
        include_invalidated: bool,
    ) -> list[QueryResult]:
        """Pure transform: raw Cypher rows -> filtered, sorted QueryResults.

        Factored out (no driver) so the filter/temporal logic is unit-tested
        directly against synthetic rows. Each row is
        ``{claim, speaker, subject, asserts_edge}`` of node/relationship dicts.
        """
        results: list[QueryResult] = []
        for row in rows:
            claim_props = _node_props(row.get("claim"))
            if claim_props is None:
                continue
            edge_props = _node_props(row.get("asserts_edge"))
            edge = edge_from_record(edge_props) if edge_props is not None else None

            # Temporal liveness gate (same rule as FakeGraphStore.query): the
            # claim needs a live asserting edge under the requested view.
            if not include_invalidated:
                if edge is None or not is_live_at(edge, as_of):
                    continue

            claim = claim_from_record(claim_props)
            if (
                subject_canonical_id is not None
                and claim.subject_canonical_id != subject_canonical_id
            ):
                continue
            if speaker_id is not None and claim.speaker_id != speaker_id:
                continue
            if predicate is not None and claim.predicate != predicate:
                continue

            results.append(
                QueryResult(
                    claim=claim,
                    speaker=speaker_from_record(_node_props(row.get("speaker"))),
                    subject=entity_from_record(_node_props(row.get("subject"))),
                    event_time=edge.event_time if edge is not None else None,
                )
            )
        results.sort(key=lambda r: r.claim.claim_id)
        return results

    # -- contract: invalidate ----------------------------------------------- #
    def invalidate(self, edge_id: str, *, at: datetime) -> bool:  # pragma: no cover - requires a live Neo4j
        """Invalidate an edge in place (preserve history); return success.

        ``True`` when a still-live edge was flipped, ``False`` if it was absent
        or already invalidated — matching :class:`FakeGraphStore.invalidate`.
        """
        rows = self._run(INVALIDATE_EDGE_CYPHER, **build_invalidate_params(edge_id, at=at))
        return len(rows) > 0

    # -- retrieval-adapter accessor (not part of the Protocol) -------------- #
    @property
    def edges(self) -> dict[str, GraphEdge]:  # pragma: no cover - requires a live Neo4j
        """All edges as ``{edge_id: GraphEdge}``.

        Mirrors :class:`FakeGraphStore`'s ``edges`` mapping so the retrieval
        adapter (:func:`dlogos.retrieval.hybrid.claims_from_graph_store`) builds
        the same claim adjacency for graph traversal. Invalidated edges are kept
        (the adapter filters them by their ``invalidated`` flag), so history is
        not hidden from callers that ask for it.
        """
        rows = self._run(ALL_EDGES_CYPHER)
        out: dict[str, GraphEdge] = {}
        for row in rows:
            rel = _node_props(row.get("rel"))
            if rel is None:
                continue
            edge = edge_from_record(rel)
            out[edge.edge_id] = edge
        return out
