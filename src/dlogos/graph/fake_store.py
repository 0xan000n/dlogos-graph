"""In-memory :class:`GraphStore` for tests — no Graphiti, no Neo4j, no network.

A faithful, dependency-light implementation of the store contract so the
loader, temporal helpers, retrieval, and eval arms can be unit-tested
deterministically with the core dependency group alone. It mirrors the real
backend's *semantics* (idempotent upsert by id, bitemporal invalidate-not-
delete, current-state filtering) without any of its infrastructure.

History is preserved: invalidation flips ``invalidated`` + sets ``valid_to`` on
the stored edge but never removes it, so a point-in-time query that predates the
invalidation still sees the fact (spec §6).
"""

from __future__ import annotations

from datetime import datetime

from dlogos.graph.store import (
    ClaimNode,
    EntityNode,
    GraphEdge,
    QueryResult,
    SpeakerNode,
)
from dlogos.graph.temporal import is_live_at
from dlogos.schema import Predicate


class FakeGraphStore:
    """Deterministic in-memory graph store implementing :class:`GraphStore`.

    Storage is plain dicts keyed by id. The class is intentionally a structural
    (duck-typed) match for the ``GraphStore`` Protocol — ``isinstance(store,
    GraphStore)`` is ``True`` via ``@runtime_checkable`` — so tests can assert
    the contract is satisfied.
    """

    def __init__(self) -> None:
        self.speakers: dict[str, SpeakerNode] = {}
        self.entities: dict[str, EntityNode] = {}
        self.claims: dict[str, ClaimNode] = {}
        self.edges: dict[str, GraphEdge] = {}
        # Diagnostics so tests/spike can assert the dedup-bypass fast path ran
        # without any LLM call.
        self.bulk_load_calls: int = 0
        self.llm_dedup_invocations: int = 0

    # -- internal helpers --------------------------------------------------- #
    def _upsert_speaker(self, speaker: SpeakerNode) -> None:
        self.speakers[speaker.speaker_id] = speaker

    def _upsert_entity(self, entity: EntityNode) -> None:
        existing = self.entities.get(entity.canonical_id)
        if existing is None:
            self.entities[entity.canonical_id] = entity
            return
        merged = list(dict.fromkeys([*existing.aliases, *entity.aliases]))
        self.entities[entity.canonical_id] = existing.model_copy(
            update={"aliases": merged}
        )

    def _upsert_claim(self, claim: ClaimNode) -> None:
        self.claims[claim.claim_id] = claim

    def _upsert_edge(self, edge: GraphEdge) -> None:
        # Idempotent by id; do NOT clobber an already-invalidated edge's
        # invalidation when the same logical edge is re-loaded.
        existing = self.edges.get(edge.edge_id)
        if existing is not None and existing.invalidated and not edge.invalidated:
            return
        self.edges[edge.edge_id] = edge

    # -- contract ----------------------------------------------------------- #
    def add_claim_triplet(
        self,
        claim: ClaimNode,
        *,
        speaker: SpeakerNode,
        subject: EntityNode,
        edges: list[GraphEdge],
        mentions: list[EntityNode] | None = None,
    ) -> None:
        """Per-add path: upsert the claim and its surrounding nodes/edges."""
        self._upsert_speaker(speaker)
        self._upsert_entity(subject)
        for ent in mentions or []:
            self._upsert_entity(ent)
        self._upsert_claim(claim)
        for edge in edges:
            self._upsert_edge(edge)

    def bulk_load(
        self,
        *,
        speakers: list[SpeakerNode],
        entities: list[EntityNode],
        claims: list[ClaimNode],
        edges: list[GraphEdge],
        bypass_llm_dedup: bool = True,
    ) -> int:
        """Bulk fast path: upsert a pre-resolved batch; return claims loaded.

        When ``bypass_llm_dedup`` is ``True`` (the backfill default) NO
        per-add LLM dedup is simulated — ``llm_dedup_invocations`` stays at
        zero, which tests assert. When ``False`` we record that the (slow,
        costly) per-add resolution path *would* have run, so the spike can
        compare. Either way the data load is identical: resolution already
        happened upstream.
        """
        self.bulk_load_calls += 1
        if not bypass_llm_dedup:
            # Simulate the redundant per-add LLM resolution call the spec wants
            # bypassed (spec §7.5/§7.6) — one notional call per claim.
            self.llm_dedup_invocations += len(claims)
        for speaker in speakers:
            self._upsert_speaker(speaker)
        for entity in entities:
            self._upsert_entity(entity)
        for claim in claims:
            self._upsert_claim(claim)
        for edge in edges:
            self._upsert_edge(edge)
        return len(claims)

    def query(
        self,
        *,
        subject_canonical_id: str | None = None,
        speaker_id: str | None = None,
        predicate: Predicate | None = None,
        as_of: datetime | None = None,
        include_invalidated: bool = False,
    ) -> list[QueryResult]:
        """Return claim rows matching the filters, current-state by default.

        A claim is included only if it has at least one live ``asserts`` edge
        (Speaker -> Claim) under the requested temporal view, unless
        ``include_invalidated`` is set. This is what makes "current state"
        return only live edges while a point-in-time ``as_of`` still surfaces
        history.
        """
        # Index live asserts-edges by the claim they point at.
        live_claim_ids: set[str] = set()
        for edge in self.edges.values():
            if edge.type.value != "asserts":
                continue
            if include_invalidated or is_live_at(edge, as_of):
                live_claim_ids.add(edge.dst_id)

        results: list[QueryResult] = []
        for claim in self.claims.values():
            if claim.claim_id not in live_claim_ids:
                continue
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
                    speaker=self.speakers.get(claim.speaker_id),
                    subject=self.entities.get(claim.subject_canonical_id),
                    event_time=self._claim_event_time(claim.claim_id),
                )
            )
        # Deterministic ordering for stable tests.
        results.sort(key=lambda r: r.claim.claim_id)
        return results

    def invalidate(self, edge_id: str, *, at: datetime) -> bool:
        """Invalidate an edge in place (preserve history); return success."""
        edge = self.edges.get(edge_id)
        if edge is None or edge.invalidated:
            return False
        self.edges[edge_id] = edge.model_copy(
            update={"valid_to": at, "invalidated": True}
        )
        return True

    # -- read helpers (test-friendly; not part of the Protocol) ------------- #
    def _claim_event_time(self, claim_id: str) -> datetime | None:
        """Event-time of a claim via its ``asserts`` edge, if present."""
        for edge in self.edges.values():
            if edge.type.value == "asserts" and edge.dst_id == claim_id:
                return edge.event_time
        return None

    def live_edges(self, as_of: datetime | None = None) -> list[GraphEdge]:
        """All live edges under the requested temporal view (current-state)."""
        return [e for e in self.edges.values() if is_live_at(e, as_of)]

    def claim_count(self) -> int:
        """Total reified claim nodes stored."""
        return len(self.claims)
