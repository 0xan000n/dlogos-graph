"""Approach-B loader: resolved :class:`ExtractedClaim` -> reified graph records.

This is the spec's Approach-B shape (§7.6(b)): *our* extraction + resolution
produces pre-formed, resolved claims, and Graphiti is used as store + temporal
manager + retrieval — NOT as the extractor. The loader maps each resolved
:class:`~dlogos.schema.ExtractedClaim` to:

- a reified ``Claim`` node (carries stance/sentiment/confidence/source span),
- a ``Speaker`` node (resolved id required),
- canonical ``Entity`` node(s) (``canonical_id`` required — resolution ran),
- bitemporal edges: ``Speaker -asserts-> Claim``, ``Claim -about-> Entity``,
  ``Speaker -appears_in-> Episode``-style references via denormalized ids.

Two entry points (spec §7.5):

- :meth:`ClaimLoader.load_claim` — the per-add path (one resolved claim).
- :meth:`ClaimLoader.bulk_load` — the explicit fast path that loads a whole
  PRE-RESOLVED batch *without* per-add LLM node-dedup, because batch resolution
  already ran in our resolution module before the graph load.

Ids are deterministic (content-hashed) so re-loading the same claim is
idempotent — re-runs of the backfill don't double-count.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Iterable

from pydantic import BaseModel, ConfigDict

from dlogos.graph.store import (
    ClaimNode,
    EdgeType,
    EntityNode,
    GraphEdge,
    SpeakerNode,
)
from dlogos.schema import ExtractedClaim

if TYPE_CHECKING:  # pragma: no cover - typing only
    from dlogos.graph.store import GraphStore


class GraphTriplet(BaseModel):
    """The fully-expanded graph records for a single resolved claim.

    A small value object so the loader's mapping is testable in isolation
    (without a store) and so the bulk path can collect, de-duplicate, and load
    these in one shot.
    """

    model_config = ConfigDict(extra="forbid")

    claim: ClaimNode
    speaker: SpeakerNode
    subject: EntityNode
    edges: list[GraphEdge]


def _stable_id(prefix: str, *parts: str) -> str:
    """Deterministic short id from ``parts`` (content hash).

    Used so the same logical claim/edge always gets the same id, making loads
    idempotent across backfill re-runs.
    """
    digest = hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


class ClaimLoader:
    """Maps resolved claims to graph records and loads them into a store.

    The loader is backend-agnostic: it depends only on the
    :class:`~dlogos.graph.store.GraphStore` Protocol. It assumes resolution has
    already run — every claim must carry a resolved speaker id and a canonical
    entity id; :meth:`to_triplet` raises otherwise, surfacing a resolution gap
    early rather than silently fragmenting the graph.

    ``event_times`` maps ``episode_id -> event-time`` (publish/recording date),
    kept separate because event-time is an episode property (see the test
    fixtures). ``ingestion_time`` defaults to "now" but is injectable for
    deterministic tests.
    """

    def __init__(
        self,
        *,
        event_times: dict[str, datetime] | None = None,
        ingestion_time: datetime | None = None,
    ) -> None:
        self._event_times = dict(event_times or {})
        self._ingestion_time = ingestion_time

    # -- mapping ------------------------------------------------------------ #
    def _resolve_event_time(self, episode_id: str) -> datetime:
        if episode_id in self._event_times:
            return self._event_times[episode_id]
        # Fall back to ingestion time if event-time is unknown; never crash a
        # load on a missing publish date (the loader records what it has).
        return self._now()

    def _now(self) -> datetime:
        return self._ingestion_time or datetime.now(timezone.utc)

    def to_triplet(self, claim: ExtractedClaim) -> GraphTriplet:
        """Expand one *resolved* claim into its graph records.

        Raises ``ValueError`` if the claim is not resolved (missing speaker
        ``resolved_id`` or entity ``canonical_id``) — the bulk path bypasses
        LLM dedup precisely because resolution is guaranteed to have run, so an
        unresolved claim here is a programming error, not a graph to dedup.
        """
        speaker_id = claim.speaker.resolved_id
        if not speaker_id:
            raise ValueError(
                "ClaimLoader requires a resolved speaker id; got an unresolved "
                f"SpeakerRef(label={claim.speaker.label!r}). Run cross-episode "
                "speaker identity before loading."
            )
        canonical_id = claim.subject_entity.canonical_id
        if not canonical_id:
            raise ValueError(
                "ClaimLoader requires a canonical entity id; got an unresolved "
                f"Entity(name={claim.subject_entity.name!r}). Run subject-entity "
                "resolution before loading."
            )

        episode_id = claim.source_span.episode_id
        event_time = self._resolve_event_time(episode_id)
        ingestion_time = self._now()

        claim_id = _stable_id(
            "claim",
            speaker_id,
            claim.predicate.value,
            canonical_id,
            claim.object,
            episode_id,
            f"{claim.source_span.t_start:.3f}",
        )

        speaker = SpeakerNode(
            speaker_id=speaker_id,
            name=claim.speaker.name,
            is_host=False,
        )
        subject = EntityNode(
            canonical_id=canonical_id,
            name=claim.subject_entity.name,
            type=claim.subject_entity.type,
            aliases=[claim.subject_entity.name],
        )
        claim_node = ClaimNode(
            claim_id=claim_id,
            predicate=claim.predicate,
            stance=claim.stance,
            object=claim.object,
            sentiment=claim.sentiment,
            confidence=claim.confidence,
            source_span=claim.source_span,
            speaker_id=speaker_id,
            subject_canonical_id=canonical_id,
        )

        def _edge(edge_type: EdgeType, src: str, dst: str) -> GraphEdge:
            return GraphEdge(
                edge_id=_stable_id("edge", edge_type.value, src, dst),
                type=edge_type,
                src_id=src,
                dst_id=dst,
                event_time=event_time,
                ingestion_time=ingestion_time,
                valid_from=event_time,
            )

        edges = [
            _edge(EdgeType.asserts, speaker_id, claim_id),
            _edge(EdgeType.about, claim_id, canonical_id),
            _edge(EdgeType.appears_in, speaker_id, episode_id),
        ]
        return GraphTriplet(
            claim=claim_node, speaker=speaker, subject=subject, edges=edges
        )

    # -- per-add path ------------------------------------------------------- #
    def load_claim(self, store: "GraphStore", claim: ExtractedClaim) -> str:
        """Map + load a single resolved claim via the per-add store path.

        Returns the deterministic ``claim_id``. Use this for the incremental,
        low-volume going-forward path (spec §7.5); the bulk path is for backfill.
        """
        triplet = self.to_triplet(claim)
        store.add_claim_triplet(
            triplet.claim,
            speaker=triplet.speaker,
            subject=triplet.subject,
            edges=triplet.edges,
        )
        return triplet.claim.claim_id

    # -- bulk fast path ----------------------------------------------------- #
    def bulk_load(
        self,
        store: "GraphStore",
        claims: Iterable[ExtractedClaim],
        *,
        bypass_llm_dedup: bool = True,
    ) -> int:
        """Load a PRE-RESOLVED batch in one shot, bypassing per-add LLM dedup.

        This is the explicit backfill fast path (spec §7.5/§7.6). Because batch
        resolution already ran in our resolution module, the store must NOT run
        Graphiti's per-add LLM node-dedup — ``bypass_llm_dedup`` is threaded
        straight through to :meth:`GraphStore.bulk_load`.

        Nodes are de-duplicated *by id* here (speakers by ``speaker_id``,
        entities by ``canonical_id``, claims/edges by their deterministic ids)
        so the load is a simple idempotent upsert without any LLM call. Returns
        the number of distinct claims loaded.
        """
        speakers: dict[str, SpeakerNode] = {}
        entities: dict[str, EntityNode] = {}
        claim_nodes: dict[str, ClaimNode] = {}
        edges: dict[str, GraphEdge] = {}

        for claim in claims:
            triplet = self.to_triplet(claim)
            speakers.setdefault(triplet.speaker.speaker_id, triplet.speaker)
            # Merge aliases when the same canonical entity recurs with a new
            # surface form so the node keeps the full alias set.
            existing = entities.get(triplet.subject.canonical_id)
            if existing is None:
                entities[triplet.subject.canonical_id] = triplet.subject
            else:
                merged = list(
                    dict.fromkeys([*existing.aliases, *triplet.subject.aliases])
                )
                entities[triplet.subject.canonical_id] = existing.model_copy(
                    update={"aliases": merged}
                )
            claim_nodes.setdefault(triplet.claim.claim_id, triplet.claim)
            for edge in triplet.edges:
                edges.setdefault(edge.edge_id, edge)

        return store.bulk_load(
            speakers=list(speakers.values()),
            entities=list(entities.values()),
            claims=list(claim_nodes.values()),
            edges=list(edges.values()),
            bypass_llm_dedup=bypass_llm_dedup,
        )
