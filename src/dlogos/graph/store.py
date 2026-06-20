"""Graph store seam: the :class:`GraphStore` ``Protocol`` and graph records.

This module defines the *contract* every backend obeys (the fake in-memory
store for tests, and the real Graphiti/Neo4j store) plus the small set of graph
record types that flow across it. The records are reified per spec §6: a
``ClaimNode`` is a node (not an edge) so it can carry stance/sentiment/
confidence and later be contradicted or superseded; edges carry the bitemporal
validity metadata via :class:`~dlogos.schema.BitemporalFact`.

Heavy dependencies (``graphiti-core``, ``neo4j``) are imported **lazily**,
inside :class:`GraphitiStore` methods only — importing this module needs the
core dependency group alone, so unit tests never require the ``graph`` extra.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from dlogos.schema import BitemporalFact, EntityType, Predicate, SourceSpan, Stance

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from dlogos.config import Settings


# --------------------------------------------------------------------------- #
# Graph node records (reified; spec §6)
# --------------------------------------------------------------------------- #
class SpeakerNode(BaseModel):
    """A resolved human — host or recurring guest (spec §6).

    ``speaker_id`` is the canonical id produced by cross-episode speaker
    identity; ``wikidata_qid`` is filled for recurring guests where resolved.
    """

    model_config = ConfigDict(extra="forbid")

    speaker_id: str
    name: str | None = None
    is_host: bool = False
    wikidata_qid: str | None = None


class EntityNode(BaseModel):
    """A canonical entity a claim is about or mentions (spec §6).

    ``canonical_id`` is the resolution-assigned cluster id so claims about
    *Apple* / *iPhone* / *Apple hardware* collapse onto one node. ``name`` is a
    representative surface form; ``aliases`` keeps the merged surface forms.
    """

    model_config = ConfigDict(extra="forbid")

    canonical_id: str
    name: str
    type: EntityType
    aliases: list[str] = Field(default_factory=list)
    wikidata_qid: str | None = None


class ClaimNode(BaseModel):
    """A reified claim node (spec §6).

    Reified — not an edge — so it carries stance/sentiment/confidence and a
    source span, and can be contradicted or superseded later. ``claim_id`` is
    stable and deterministic (assigned by the loader) so re-loads are
    idempotent.
    """

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    predicate: Predicate
    stance: Stance
    object: str
    sentiment: float
    confidence: float
    source_span: SourceSpan
    # Denormalized refs for cheap traversal-free filtering.
    speaker_id: str
    subject_canonical_id: str


# --------------------------------------------------------------------------- #
# Graph edge records (bitemporal; spec §6)
# --------------------------------------------------------------------------- #
class EdgeType(str, Enum):
    """The dialogue ontology edge types (spec §6)."""

    asserts = "asserts"  # Speaker -> Claim
    about = "about"  # Claim -> Entity (subject)
    mentions = "mentions"  # Claim -> Entity
    agrees_with = "agrees_with"  # Claim -> Claim
    disputes = "disputes"  # Claim -> Claim
    supersedes = "supersedes"  # Claim -> Claim (same speaker, updated position)
    appears_in = "appears_in"  # Speaker -> Episode


class GraphEdge(BitemporalFact):
    """A bitemporal edge between two graph nodes.

    Extends :class:`~dlogos.schema.BitemporalFact` so every edge carries the
    two independent time axes (event-time / ingestion-time) and a validity
    interval. Invalidation sets ``valid_to`` + ``invalidated`` rather than
    deleting (spec §6: invalidate-not-delete).
    """

    model_config = ConfigDict(extra="forbid")

    edge_id: str
    type: EdgeType
    src_id: str
    dst_id: str


# --------------------------------------------------------------------------- #
# Query result
# --------------------------------------------------------------------------- #
class QueryResult(BaseModel):
    """A flat result row returned by :meth:`GraphStore.query`.

    Deliberately backend-agnostic: the fake store and the Graphiti store both
    return these so retrieval code is written once against the contract.
    """

    model_config = ConfigDict(extra="forbid")

    claim: ClaimNode
    speaker: SpeakerNode | None = None
    subject: EntityNode | None = None
    event_time: datetime | None = None


# --------------------------------------------------------------------------- #
# The store contract
# --------------------------------------------------------------------------- #
@runtime_checkable
class GraphStore(Protocol):
    """The bitemporal graph store contract.

    Both the in-memory :class:`~dlogos.graph.fake_store.FakeGraphStore` (tests)
    and :class:`GraphitiStore` (real Graphiti/Neo4j) implement this. Retrieval,
    the loader, and the eval arms depend only on this Protocol — never on a
    concrete backend.
    """

    def add_claim_triplet(
        self,
        claim: ClaimNode,
        *,
        speaker: SpeakerNode,
        subject: EntityNode,
        edges: list[GraphEdge],
        mentions: list[EntityNode] | None = None,
    ) -> None:
        """Add one resolved claim and its surrounding nodes/edges.

        The per-add (incremental) path. ``edges`` are pre-built bitemporal
        edges; the store upserts nodes idempotently by id.
        """
        ...

    def bulk_load(
        self,
        *,
        speakers: list[SpeakerNode],
        entities: list[EntityNode],
        claims: list[ClaimNode],
        edges: list[GraphEdge],
        bypass_llm_dedup: bool = True,
    ) -> int:
        """Load PRE-RESOLVED nodes/edges in bulk; return claims loaded.

        ``bypass_llm_dedup=True`` is the explicit fast path (spec §7.5/§7.6):
        resolution already happened in our batch module, so the backend must
        NOT run a per-add LLM node-dedup call. Returns the number of claim
        nodes loaded.
        """
        ...

    def query(
        self,
        *,
        subject_canonical_id: str | None = None,
        speaker_id: str | None = None,
        predicate: Predicate | None = None,
        as_of: datetime | None = None,
        include_invalidated: bool = False,
    ) -> list[QueryResult]:
        """Return claim rows matching the filters.

        ``as_of`` applies a bitemporal point-in-time filter on edge validity;
        ``include_invalidated=False`` (default) returns only live edges
        (current-state). Filters combine with AND.
        """
        ...

    def invalidate(self, edge_id: str, *, at: datetime) -> bool:
        """Invalidate an edge (set ``valid_to`` + flag), never delete it.

        Returns ``True`` if an edge was invalidated, ``False`` if not found or
        already invalidated. History is preserved (spec §6).
        """
        ...


# --------------------------------------------------------------------------- #
# Real backend — Graphiti / Neo4j (lazy heavy imports)
# --------------------------------------------------------------------------- #
class GraphitiStore:
    """Graphiti-on-Neo4j implementation of :class:`GraphStore`.

    Every ``graphiti-core`` / ``neo4j`` symbol is imported *inside* a method,
    never at module top level, so importing this module requires only the core
    dependency group. Construct via :meth:`connect`.

    This is a thin adapter: the bulk path is wired to bypass Graphiti's per-add
    LLM node-dedup (spec §7.5/§7.6) because resolution already happened in our
    batch module. At PoC scale the real backend is exercised by the spike, not
    by unit tests — tests use :class:`~dlogos.graph.fake_store.FakeGraphStore`.
    """

    def __init__(self, client: Any, driver: Any) -> None:
        self._client = client
        self._driver = driver

    # -- construction ------------------------------------------------------- #
    @classmethod
    def connect(cls, settings: "Settings | None" = None) -> "GraphitiStore":
        """Open a Graphiti client + Neo4j driver from settings.

        Heavy imports happen here, lazily. Raises a clear error if the optional
        ``graph`` extra is not installed.
        """
        if settings is None:
            from dlogos.config import settings as default_settings

            settings = default_settings
        try:
            from graphiti_core import Graphiti  # noqa: F401
            from neo4j import GraphDatabase
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "GraphitiStore requires the optional 'graph' extra "
                "(graphiti-core, neo4j). Install with: uv sync --extra graph"
            ) from exc

        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        client = Graphiti(
            settings.neo4j_uri,
            settings.neo4j_user,
            settings.neo4j_password,
        )
        return cls(client=client, driver=driver)

    # -- contract methods --------------------------------------------------- #
    def add_claim_triplet(
        self,
        claim: ClaimNode,
        *,
        speaker: SpeakerNode,
        subject: EntityNode,
        edges: list[GraphEdge],
        mentions: list[EntityNode] | None = None,
    ) -> None:  # pragma: no cover - requires a live Neo4j; exercised in the spike
        nodes_cypher = _UPSERT_NODES_CYPHER
        with self._driver.session() as session:
            session.run(
                nodes_cypher,
                speakers=[s.model_dump(mode="json") for s in [speaker]],
                entities=[
                    e.model_dump(mode="json") for e in [subject, *(mentions or [])]
                ],
                claims=[claim.model_dump(mode="json")],
            )
            session.run(
                _UPSERT_EDGES_CYPHER,
                edges=[_edge_payload(e) for e in edges],
            )

    def bulk_load(
        self,
        *,
        speakers: list[SpeakerNode],
        entities: list[EntityNode],
        claims: list[ClaimNode],
        edges: list[GraphEdge],
        bypass_llm_dedup: bool = True,
    ) -> int:  # pragma: no cover - requires a live Neo4j; exercised in the spike
        if not bypass_llm_dedup:
            raise NotImplementedError(
                "GraphitiStore.bulk_load only supports the dedup-bypass fast "
                "path; per-add LLM dedup is intentionally disabled on bulk "
                "(spec §7.5/§7.6). Use add_claim_triplet for the per-add path."
            )
        with self._driver.session() as session:
            session.run(
                _UPSERT_NODES_CYPHER,
                speakers=[s.model_dump(mode="json") for s in speakers],
                entities=[e.model_dump(mode="json") for e in entities],
                claims=[c.model_dump(mode="json") for c in claims],
            )
            session.run(
                _UPSERT_EDGES_CYPHER,
                edges=[_edge_payload(e) for e in edges],
            )
        return len(claims)

    def query(
        self,
        *,
        subject_canonical_id: str | None = None,
        speaker_id: str | None = None,
        predicate: Predicate | None = None,
        as_of: datetime | None = None,
        include_invalidated: bool = False,
    ) -> list[QueryResult]:  # pragma: no cover - requires a live Neo4j
        raise NotImplementedError(
            "GraphitiStore.query is wired during the retrieval spike; unit "
            "tests use FakeGraphStore."
        )

    def invalidate(self, edge_id: str, *, at: datetime) -> bool:  # pragma: no cover
        with self._driver.session() as session:
            result = session.run(
                _INVALIDATE_EDGE_CYPHER, edge_id=edge_id, at=at.isoformat()
            )
            return result.single() is not None

    def close(self) -> None:  # pragma: no cover - requires a live Neo4j
        self._driver.close()


def _edge_payload(edge: GraphEdge) -> dict[str, Any]:
    """Flatten a :class:`GraphEdge` to a Cypher-friendly param map."""
    data = edge.model_dump(mode="json")
    data["type"] = edge.type.value
    return data


# Parameterized, dedup-free Cypher (Approach-B bulk path). Kept as module
# constants for readability; only used by the real backend.
_UPSERT_NODES_CYPHER = """
UNWIND $speakers AS s
  MERGE (n:Speaker {speaker_id: s.speaker_id})
  SET n += s
WITH 1 AS _
UNWIND $entities AS e
  MERGE (n:Entity {canonical_id: e.canonical_id})
  SET n += e
WITH 1 AS _
UNWIND $claims AS c
  MERGE (n:Claim {claim_id: c.claim_id})
  SET n += c
"""

_UPSERT_EDGES_CYPHER = """
UNWIND $edges AS e
  MATCH (src {`__id__`: e.src_id})
  MATCH (dst {`__id__`: e.dst_id})
  MERGE (src)-[r:REL {edge_id: e.edge_id}]->(dst)
  SET r += e
"""

_INVALIDATE_EDGE_CYPHER = """
MATCH ()-[r:REL {edge_id: $edge_id}]->()
WHERE r.invalidated = false
SET r.invalidated = true, r.valid_to = $at
RETURN r.edge_id AS edge_id
"""
