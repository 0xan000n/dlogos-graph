"""Cross-claim and mention edge derivation (spec §6 dialogue ontology).

The loader (:mod:`dlogos.graph.loader`) emits the *structural* edges for a
single resolved claim — ``Speaker -asserts-> Claim``, ``Claim -about-> Entity``,
``Speaker -appears_in-> Episode``. The *relational* edges of the dialogue
ontology, however, only exist **between** claims (or between a claim and its
secondary entities) and so cannot be derived one claim at a time:

- ``Claim -mentions-> Entity`` — a claim's *non-subject* entities (everything it
  references that is not the thing it is *about*).
- ``Claim -agrees_with / disputes-> Claim`` — two claims about the **same
  canonical subject** whose stance polarity aligns / opposes.
- ``Claim -supersedes-> Claim`` — the **same speaker** changing position on the
  same canonical subject over time; the newer (later event-time) claim
  supersedes the older. This is the contradiction archetype's backbone.

These are pure functions over the already-built graph records (so they need no
store and no backend) plus an injectable clock for the ingestion-time stamp.
They never *delete* anything: superseding adds a ``supersedes`` edge and lets
the caller invalidate the prior ``asserts`` edge (invalidate-not-delete, §6).
The bitemporal stamps on the derived edges are chosen so the relation's
event-time is the **newer** of the two claims (when the relationship became
true): a dispute/agreement/supersession exists from the moment the second claim
was said, not before.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from dlogos.graph.store import (
    ClaimNode,
    EdgeType,
    EntityNode,
    GraphEdge,
)
from dlogos.schema import Stance

# --------------------------------------------------------------------------- #
# Stance polarity.
#
# Reuses the spec's stance-direction convention (mirrors retrieval.consensus):
# an assertion/prediction pushes a claim *toward* its object, a dispute/retract
# pushes *against* it, a hedge is neutral. Combined with the signed sentiment we
# get a single polarity sign per claim, which is what makes two claims about the
# same subject classifiable as agreeing vs. opposing.
# --------------------------------------------------------------------------- #
_STANCE_SIGN: dict[Stance, float] = {
    Stance.asserts: 1.0,
    Stance.predicts: 0.5,
    Stance.hedges: 0.0,
    Stance.disputes: -1.0,
    Stance.retracts: -1.0,
}


def _stable_id(prefix: str, *parts: str) -> str:
    """Deterministic short id from ``parts`` (content hash).

    Matches :func:`dlogos.graph.loader._stable_id` so a re-derivation of the
    same logical relation always yields the same edge id — derived edges are
    idempotent across backfill re-runs, exactly like the structural edges.
    """
    digest = hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def claim_polarity(claim: ClaimNode) -> float:
    """Signed polarity of a claim about its subject in ``[-1, 1]``-ish range.

    The product of the stance sign and the sentiment magnitude. A positive
    value means "holds the position favourably", negative means "against",
    ``0.0`` means neutral/hedged (and so cannot agree or dispute with anything).

    Sentiment of exactly ``0`` is treated as mildly affirming when the stance is
    affirming (and vice-versa) so a confident bare assertion with neutral
    sentiment still has a non-zero polarity and can participate in
    agreement/dispute; only a genuinely neutral *stance* (hedge) yields ``0``.
    """
    stance_sign = _STANCE_SIGN.get(claim.stance, 0.0)
    if stance_sign == 0.0:
        return 0.0
    # Sentiment magnitude scales the stance sign; a neutral (0.0) sentiment
    # still leaves the stance sign intact so an assertion is not silenced.
    magnitude = abs(claim.sentiment) if claim.sentiment != 0.0 else 1.0
    return stance_sign * magnitude


def _polarity_sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_edge(
    edge_type: EdgeType,
    src_id: str,
    dst_id: str,
    *,
    event_time: datetime,
    ingestion_time: datetime,
) -> GraphEdge:
    """Build one bitemporal derived edge with a deterministic id."""
    return GraphEdge(
        edge_id=_stable_id("edge", edge_type.value, src_id, dst_id),
        type=edge_type,
        src_id=src_id,
        dst_id=dst_id,
        event_time=event_time,
        ingestion_time=ingestion_time,
        valid_from=event_time,
    )


# --------------------------------------------------------------------------- #
# mentions
# --------------------------------------------------------------------------- #
def mention_edges(
    claim: ClaimNode,
    secondary_entities: list[EntityNode],
    *,
    event_time: datetime,
    ingestion_time: datetime | None = None,
) -> list[GraphEdge]:
    """``Claim -mentions-> Entity`` for each *non-subject* secondary entity.

    ``secondary_entities`` are the resolved entities a claim references beyond
    its subject (the loader passes these through from the extraction/resolution
    stage). Any entity whose ``canonical_id`` equals the claim's
    ``subject_canonical_id`` is dropped — that relationship is already an
    ``about`` edge, not a mention. Duplicates collapse by canonical id so a
    subject mentioned twice yields a single edge.
    """
    ingestion_time = ingestion_time or _now()
    edges: list[GraphEdge] = []
    seen: set[str] = set()
    for ent in secondary_entities:
        cid = ent.canonical_id
        if cid == claim.subject_canonical_id:
            continue
        if cid in seen:
            continue
        seen.add(cid)
        edges.append(
            _make_edge(
                EdgeType.mentions,
                claim.claim_id,
                cid,
                event_time=event_time,
                ingestion_time=ingestion_time,
            )
        )
    return edges


# --------------------------------------------------------------------------- #
# agrees_with / disputes
# --------------------------------------------------------------------------- #
def stance_relation_edges(
    claims: list[ClaimNode],
    event_times: dict[str, datetime],
    *,
    ingestion_time: datetime | None = None,
) -> list[GraphEdge]:
    """Derive ``agrees_with`` / ``disputes`` edges between claims on one subject.

    For every unordered pair of claims about the **same canonical subject** with
    a *different* speaker, compare the signed :func:`claim_polarity`:

    - same sign  -> ``agrees_with``
    - opposite   -> ``disputes``
    - either neutral (hedge / zero polarity) -> no edge

    Same-speaker pairs are skipped here: a single speaker holding two positions
    on a subject is a *supersession* over time (handled by
    :func:`supersession_edges`), not a self-dispute. The edge is directed from
    the **newer** claim to the older (by event-time, ties broken by claim id)
    and carries the newer claim's event-time, since that is when the relation
    became true.
    """
    ingestion_time = ingestion_time or _now()
    by_subject: dict[str, list[ClaimNode]] = {}
    for claim in claims:
        by_subject.setdefault(claim.subject_canonical_id, []).append(claim)

    edges: list[GraphEdge] = []
    for subject_claims in by_subject.values():
        # Deterministic order so pairing/ids are stable across runs.
        ordered = sorted(subject_claims, key=lambda c: c.claim_id)
        # Collapse the claim-pair explosion: emit ONE relation per distinct
        # (speaker-pair, polarity-relation) on a subject — "person A agrees /
        # disputes person B about subject S", once — not one edge per claim pair.
        # Without this a subject discussed by many speakers yields O(claims^2)
        # near-duplicate agrees_with edges (56k on the 20-episode slice).
        seen: set[tuple[str, str, bool]] = set()
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                a, b = ordered[i], ordered[j]
                if a.speaker_id == b.speaker_id:
                    continue
                sign_a = _polarity_sign(claim_polarity(a))
                sign_b = _polarity_sign(claim_polarity(b))
                if sign_a == 0 or sign_b == 0:
                    continue
                same = sign_a == sign_b
                lo, hi = sorted((a.speaker_id, b.speaker_id))
                if (lo, hi, same) in seen:
                    continue
                seen.add((lo, hi, same))
                edge_type = EdgeType.agrees_with if same else EdgeType.disputes
                newer, older = _order_by_time(a, b, event_times)
                edges.append(
                    _make_edge(
                        edge_type,
                        newer.claim_id,
                        older.claim_id,
                        event_time=_event_time_of(newer, event_times),
                        ingestion_time=ingestion_time,
                    )
                )
    return edges


# --------------------------------------------------------------------------- #
# supersedes
# --------------------------------------------------------------------------- #
def supersession_edges(
    claims: list[ClaimNode],
    event_times: dict[str, datetime],
    *,
    ingestion_time: datetime | None = None,
) -> list[GraphEdge]:
    """Derive ``supersedes`` edges where one speaker changes stance over time.

    Group by ``(speaker_id, subject_canonical_id)``. Within a group, order the
    claims by event-time; whenever the polarity **sign flips** from one claim to
    the next-in-time, the later claim ``supersedes`` the earlier one (newer
    event-time supersedes older — the bitemporal rule). Same-polarity
    restatements do not supersede (the position did not change).

    This NEVER deletes the superseded claim: the prior claim and its
    ``asserts`` edge remain in the graph (history is preserved). The caller is
    responsible for invalidating the prior ``asserts`` edge if it wants the
    current-state view to reflect only the latest position — that is the
    invalidate-not-delete contract (§6), kept distinct from edge derivation.
    """
    ingestion_time = ingestion_time or _now()
    by_speaker_subject: dict[tuple[str, str], list[ClaimNode]] = {}
    for claim in claims:
        key = (claim.speaker_id, claim.subject_canonical_id)
        by_speaker_subject.setdefault(key, []).append(claim)

    edges: list[GraphEdge] = []
    for group in by_speaker_subject.values():
        if len(group) < 2:
            continue
        # Order in time; ties broken by claim id for determinism.
        ordered = sorted(
            group, key=lambda c: (_event_time_of(c, event_times), c.claim_id)
        )
        for prev, cur in zip(ordered, ordered[1:]):
            prev_sign = _polarity_sign(claim_polarity(prev))
            cur_sign = _polarity_sign(claim_polarity(cur))
            # Only a genuine reversal supersedes; a neutral->something or a
            # restatement of the same position does not.
            if prev_sign == 0 or cur_sign == 0:
                continue
            if prev_sign == cur_sign:
                continue
            edges.append(
                _make_edge(
                    EdgeType.supersedes,
                    cur.claim_id,
                    prev.claim_id,
                    event_time=_event_time_of(cur, event_times),
                    ingestion_time=ingestion_time,
                )
            )
    return edges


# --------------------------------------------------------------------------- #
# shared time helpers
# --------------------------------------------------------------------------- #
def _event_time_of(claim: ClaimNode, event_times: dict[str, datetime]) -> datetime:
    """Event-time for a claim via its episode; falls back to "now" if unknown.

    Keeps the relations module self-contained: it reads the episode id off the
    claim's source span and looks it up in the same ``episode_id -> event_time``
    mapping the loader uses, so derived edges share the structural edges' clock.
    """
    episode_id = claim.source_span.episode_id
    return event_times.get(episode_id) or _now()


def _order_by_time(
    a: ClaimNode, b: ClaimNode, event_times: dict[str, datetime]
) -> tuple[ClaimNode, ClaimNode]:
    """Return ``(newer, older)`` by event-time; ties broken by claim id."""
    ta, tb = _event_time_of(a, event_times), _event_time_of(b, event_times)
    if ta > tb:
        return a, b
    if tb > ta:
        return b, a
    # Same event-time: deterministic tiebreak so direction is stable.
    return (a, b) if a.claim_id >= b.claim_id else (b, a)


def derive_relation_edges(
    claims: list[ClaimNode],
    event_times: dict[str, datetime],
    *,
    mentions: dict[str, list[EntityNode]] | None = None,
    ingestion_time: datetime | None = None,
) -> list[GraphEdge]:
    """All derived edges (mentions + agrees/disputes + supersedes) for a batch.

    The single entry point the loader's bulk path calls. ``mentions`` maps a
    ``claim_id`` to the claim's resolved secondary entities; absent entries
    simply yield no mention edges. Edges are de-duplicated by ``edge_id`` so a
    relation derived from two symmetric vantage points collapses to one record.
    """
    ingestion_time = ingestion_time or _now()
    mentions = mentions or {}

    out: dict[str, GraphEdge] = {}

    def _add(edge_list: list[GraphEdge]) -> None:
        for edge in edge_list:
            out.setdefault(edge.edge_id, edge)

    for claim in claims:
        secondary = mentions.get(claim.claim_id, [])
        if secondary:
            _add(
                mention_edges(
                    claim,
                    secondary,
                    event_time=_event_time_of(claim, event_times),
                    ingestion_time=ingestion_time,
                )
            )

    _add(stance_relation_edges(claims, event_times, ingestion_time=ingestion_time))
    _add(supersession_edges(claims, event_times, ingestion_time=ingestion_time))
    return list(out.values())
