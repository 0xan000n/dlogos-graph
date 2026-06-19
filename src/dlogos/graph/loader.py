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
from typing import TYPE_CHECKING, Callable, Iterable

from pydantic import BaseModel, ConfigDict

from dlogos.graph.relations import derive_relation_edges, mention_edges
from dlogos.graph.store import (
    ClaimNode,
    EdgeType,
    EntityNode,
    GraphEdge,
    SpeakerNode,
)
from dlogos.schema import Entity, ExtractedClaim

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
    # Resolved non-subject entities this claim mentions, plus the structural
    # ``mentions`` edges to them. Empty unless secondary entities were supplied.
    mentions: list[EntityNode] = []


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

    def to_triplet(
        self,
        claim: ExtractedClaim,
        *,
        secondary_entities: list[Entity] | None = None,
    ) -> GraphTriplet:
        """Expand one *resolved* claim into its graph records.

        ``secondary_entities`` are the resolved non-subject entities the claim
        references (the ``Claim -mentions-> Entity`` ontology edge, spec §6);
        each becomes an :class:`EntityNode` plus a ``mentions`` edge. They are a
        side input because :class:`~dlogos.schema.ExtractedClaim` carries only
        the subject entity — secondary entities come from the extraction /
        resolution stage and are threaded through here, never invented.

        Raises ``ValueError`` if the claim is not resolved (missing speaker
        ``resolved_id`` or entity ``canonical_id``) — the bulk path bypasses
        LLM dedup precisely because resolution is guaranteed to have run, so an
        unresolved claim here is a programming error, not a graph to dedup. A
        secondary entity missing a ``canonical_id`` is likewise rejected.
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

        # Secondary (non-subject) entities -> mention nodes + mentions edges.
        mention_nodes: list[EntityNode] = []
        if secondary_entities:
            seen: set[str] = set()
            for ent in secondary_entities:
                ent_cid = ent.canonical_id
                if not ent_cid:
                    raise ValueError(
                        "ClaimLoader requires a canonical id on every secondary "
                        f"entity; got an unresolved Entity(name={ent.name!r}). "
                        "Run subject-entity resolution before loading."
                    )
                # The subject is already linked via `about`; never re-link it as
                # a mention, and collapse duplicates by canonical id.
                if ent_cid == canonical_id or ent_cid in seen:
                    continue
                seen.add(ent_cid)
                mention_nodes.append(
                    EntityNode(
                        canonical_id=ent_cid,
                        name=ent.name,
                        type=ent.type,
                        aliases=[ent.name],
                    )
                )
            edges.extend(
                mention_edges(
                    claim_node,
                    mention_nodes,
                    event_time=event_time,
                    ingestion_time=ingestion_time,
                )
            )

        return GraphTriplet(
            claim=claim_node,
            speaker=speaker,
            subject=subject,
            edges=edges,
            mentions=mention_nodes,
        )

    # -- per-add path ------------------------------------------------------- #
    def load_claim(
        self,
        store: "GraphStore",
        claim: ExtractedClaim,
        *,
        secondary_entities: list[Entity] | None = None,
    ) -> str:
        """Map + load a single resolved claim via the per-add store path.

        ``secondary_entities`` (optional) are the claim's resolved non-subject
        entities; they are upserted and linked with ``mentions`` edges. Returns
        the deterministic ``claim_id``. Use this for the incremental, low-volume
        going-forward path (spec §7.5); the bulk path is for backfill.
        """
        triplet = self.to_triplet(claim, secondary_entities=secondary_entities)
        store.add_claim_triplet(
            triplet.claim,
            speaker=triplet.speaker,
            subject=triplet.subject,
            edges=triplet.edges,
            mentions=triplet.mentions,
        )
        return triplet.claim.claim_id

    # -- bulk fast path ----------------------------------------------------- #
    def bulk_load(
        self,
        store: "GraphStore",
        claims: Iterable[ExtractedClaim],
        *,
        bypass_llm_dedup: bool = True,
        secondary_entities: "Callable[[ExtractedClaim], list[Entity]] | None" = None,
    ) -> int:
        """Load a PRE-RESOLVED batch in one shot, bypassing per-add LLM dedup.

        This is the explicit backfill fast path (spec §7.5/§7.6). Because batch
        resolution already ran in our resolution module, the store must NOT run
        Graphiti's per-add LLM node-dedup — ``bypass_llm_dedup`` is threaded
        straight through to :meth:`GraphStore.bulk_load`.

        Nodes are de-duplicated *by id* here (speakers by ``speaker_id``,
        entities by ``canonical_id``, claims/edges by their deterministic ids)
        so the load is a simple idempotent upsert without any LLM call.

        Beyond the per-claim structural edges, this batch path also derives the
        *cross-claim* dialogue-ontology edges that only exist between claims
        (spec §6): ``agrees_with`` / ``disputes`` between claims on the same
        canonical subject by stance polarity, and ``supersedes`` when one
        speaker reverses position on a subject over time (newer event-time
        supersedes older). ``secondary_entities``, if given, maps each claim to
        its resolved non-subject entities, yielding ``mentions`` edges + nodes.

        Returns the number of distinct claims loaded.
        """
        speakers: dict[str, SpeakerNode] = {}
        entities: dict[str, EntityNode] = {}
        claim_nodes: dict[str, ClaimNode] = {}
        edges: dict[str, GraphEdge] = {}
        # claim_id -> resolved secondary EntityNodes (for cross-claim derivation).
        mentions_by_claim: dict[str, list[EntityNode]] = {}
        ingestion_time = self._now()

        def _merge_entity(node: EntityNode) -> None:
            existing = entities.get(node.canonical_id)
            if existing is None:
                entities[node.canonical_id] = node
                return
            # Merge aliases when the same canonical entity recurs with a new
            # surface form so the node keeps the full alias set.
            merged = list(dict.fromkeys([*existing.aliases, *node.aliases]))
            entities[node.canonical_id] = existing.model_copy(
                update={"aliases": merged}
            )

        for claim in claims:
            secondary = secondary_entities(claim) if secondary_entities else None
            triplet = self.to_triplet(claim, secondary_entities=secondary)
            speakers.setdefault(triplet.speaker.speaker_id, triplet.speaker)
            _merge_entity(triplet.subject)
            for mention in triplet.mentions:
                _merge_entity(mention)
            claim_nodes.setdefault(triplet.claim.claim_id, triplet.claim)
            if triplet.mentions:
                mentions_by_claim[triplet.claim.claim_id] = triplet.mentions
            for edge in triplet.edges:
                edges.setdefault(edge.edge_id, edge)

        # Cross-claim relations (agrees_with / disputes / supersedes) plus the
        # mention edges are derived over the WHOLE batch — they cannot be built
        # one claim at a time. Mention edges are already in `edges` from the
        # triplets; derive_relation_edges re-adds them idempotently (by id) and
        # contributes the cross-claim edges.
        for edge in derive_relation_edges(
            list(claim_nodes.values()),
            self._event_times,
            mentions=mentions_by_claim,
            ingestion_time=ingestion_time,
        ):
            edges.setdefault(edge.edge_id, edge)

        return store.bulk_load(
            speakers=list(speakers.values()),
            entities=list(entities.values()),
            claims=list(claim_nodes.values()),
            edges=list(edges.values()),
            bypass_llm_dedup=bypass_llm_dedup,
        )
