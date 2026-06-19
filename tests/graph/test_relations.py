"""Tests for cross-claim & mention edge derivation (spec §6 dialogue ontology).

These close GAP 3: ``EdgeType`` defines ``mentions`` / ``agrees_with`` /
``disputes`` / ``supersedes`` but the loader previously only emitted the
structural ``asserts`` / ``about`` / ``appears_in`` edges, so the contradiction
archetype had no graph. The tests drive the pure derivation functions in
``dlogos.graph.relations`` directly *and* through the loader's bulk path against
the in-memory ``FakeGraphStore`` (no Graphiti, no Neo4j, no network).
"""

from __future__ import annotations

from datetime import datetime, timezone

from dlogos.graph.fake_store import FakeGraphStore
from dlogos.graph.loader import ClaimLoader
from dlogos.graph.relations import (
    claim_polarity,
    derive_relation_edges,
    mention_edges,
    stance_relation_edges,
    supersession_edges,
)
from dlogos.graph.store import ClaimNode, EdgeType, EntityNode
from dlogos.schema import (
    Entity,
    EntityType,
    ExtractedClaim,
    Predicate,
    SourceSpan,
    SpeakerRef,
    Stance,
)

INGEST = datetime(2026, 6, 18, tzinfo=timezone.utc)


def _dt(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Builders — self-contained so the test owns its inputs.
# --------------------------------------------------------------------------- #
def _claim_node(
    *,
    claim_id: str,
    speaker_id: str,
    subject_canonical_id: str = "ent-apple",
    stance: Stance = Stance.asserts,
    sentiment: float = 0.6,
    predicate: Predicate = Predicate.rates_positive,
    episode_id: str = "ep-0001",
    t_start: float = 0.0,
) -> ClaimNode:
    return ClaimNode(
        claim_id=claim_id,
        predicate=predicate,
        stance=stance,
        object="apple is great",
        sentiment=sentiment,
        confidence=0.9,
        source_span=SourceSpan(
            episode_id=episode_id, t_start=t_start, t_end=t_start + 5
        ),
        speaker_id=speaker_id,
        subject_canonical_id=subject_canonical_id,
    )


def _extracted(
    *,
    speaker_id: str,
    name: str,
    canonical_id: str = "ent-apple",
    subject: str = "Apple",
    stance: Stance = Stance.asserts,
    sentiment: float = 0.6,
    predicate: Predicate = Predicate.rates_positive,
    episode_id: str = "ep-0001",
    t_start: float = 0.0,
    obj: str = "apple is great",
) -> ExtractedClaim:
    return ExtractedClaim(
        speaker=SpeakerRef(label="SPEAKER_X", resolved_id=speaker_id, name=name),
        predicate=predicate,
        subject_entity=Entity(
            name=subject, type=EntityType.organization, canonical_id=canonical_id
        ),
        object=obj,
        stance=stance,
        sentiment=sentiment,
        confidence=0.9,
        source_span=SourceSpan(
            episode_id=episode_id, t_start=t_start, t_end=t_start + 5
        ),
    )


def _entity_node(cid: str, name: str, type_: EntityType) -> EntityNode:
    return EntityNode(canonical_id=cid, name=name, type=type_)


# --------------------------------------------------------------------------- #
# polarity
# --------------------------------------------------------------------------- #
def test_polarity_sign_reflects_stance_and_sentiment() -> None:
    pos = _claim_node(
        claim_id="c1", speaker_id="s1", stance=Stance.asserts, sentiment=0.7
    )
    neg = _claim_node(
        claim_id="c2", speaker_id="s2", stance=Stance.disputes, sentiment=0.7
    )
    assert claim_polarity(pos) > 0
    assert claim_polarity(neg) < 0


def test_hedge_has_zero_polarity() -> None:
    hedge = _claim_node(
        claim_id="c1", speaker_id="s1", stance=Stance.hedges, sentiment=0.9
    )
    assert claim_polarity(hedge) == 0.0


def test_bare_assertion_with_neutral_sentiment_is_not_silenced() -> None:
    # A confident assertion with 0 sentiment still has a sign so it can agree.
    c = _claim_node(
        claim_id="c1", speaker_id="s1", stance=Stance.asserts, sentiment=0.0
    )
    assert claim_polarity(c) > 0


# --------------------------------------------------------------------------- #
# disputes / agrees_with
# --------------------------------------------------------------------------- #
def test_opposing_stance_on_same_subject_yields_disputes() -> None:
    a = _claim_node(
        claim_id="c-pro", speaker_id="s-a", stance=Stance.asserts, sentiment=0.8
    )
    b = _claim_node(
        claim_id="c-con", speaker_id="s-b", stance=Stance.disputes, sentiment=0.8
    )
    edges = stance_relation_edges([a, b], {"ep-0001": _dt(2026, 1, 1)})
    types = {e.type for e in edges}
    assert EdgeType.disputes in types
    assert EdgeType.agrees_with not in types


def test_same_stance_on_same_subject_yields_agrees_with() -> None:
    a = _claim_node(
        claim_id="c1", speaker_id="s-a", stance=Stance.asserts, sentiment=0.6
    )
    b = _claim_node(
        claim_id="c2", speaker_id="s-b", stance=Stance.asserts, sentiment=0.7
    )
    edges = stance_relation_edges([a, b], {"ep-0001": _dt(2026, 1, 1)})
    types = {e.type for e in edges}
    assert EdgeType.agrees_with in types
    assert EdgeType.disputes not in types


def test_claims_on_different_subjects_have_no_stance_edge() -> None:
    a = _claim_node(
        claim_id="c1", speaker_id="s-a", subject_canonical_id="ent-apple"
    )
    b = _claim_node(
        claim_id="c2",
        speaker_id="s-b",
        subject_canonical_id="ent-openai",
        stance=Stance.disputes,
    )
    edges = stance_relation_edges([a, b], {"ep-0001": _dt(2026, 1, 1)})
    assert edges == []


def test_same_speaker_pair_is_not_a_stance_relation() -> None:
    # Same speaker, opposing stance -> supersession territory, not self-dispute.
    a = _claim_node(
        claim_id="c1", speaker_id="s-a", stance=Stance.asserts, sentiment=0.8
    )
    b = _claim_node(
        claim_id="c2", speaker_id="s-a", stance=Stance.disputes, sentiment=0.8
    )
    edges = stance_relation_edges([a, b], {"ep-0001": _dt(2026, 1, 1)})
    assert edges == []


def test_hedge_does_not_dispute_or_agree() -> None:
    a = _claim_node(
        claim_id="c1", speaker_id="s-a", stance=Stance.hedges, sentiment=0.0
    )
    b = _claim_node(
        claim_id="c2", speaker_id="s-b", stance=Stance.asserts, sentiment=0.8
    )
    edges = stance_relation_edges([a, b], {"ep-0001": _dt(2026, 1, 1)})
    assert edges == []


def test_dispute_edge_points_from_newer_to_older_claim() -> None:
    older = _claim_node(
        claim_id="c-old",
        speaker_id="s-a",
        stance=Stance.asserts,
        sentiment=0.8,
        episode_id="ep-old",
    )
    newer = _claim_node(
        claim_id="c-new",
        speaker_id="s-b",
        stance=Stance.disputes,
        sentiment=0.8,
        episode_id="ep-new",
    )
    event_times = {"ep-old": _dt(2024, 1, 1), "ep-new": _dt(2026, 1, 1)}
    edges = stance_relation_edges([older, newer], event_times)
    (edge,) = [e for e in edges if e.type is EdgeType.disputes]
    assert edge.src_id == "c-new"
    assert edge.dst_id == "c-old"
    # The relation became true when the second (newer) claim was said.
    assert edge.event_time == _dt(2026, 1, 1)
    assert edge.valid_from == _dt(2026, 1, 1)


# --------------------------------------------------------------------------- #
# supersedes
# --------------------------------------------------------------------------- #
def test_same_speaker_stance_change_over_time_yields_supersedes() -> None:
    early = _claim_node(
        claim_id="c-early",
        speaker_id="s-a",
        stance=Stance.asserts,
        sentiment=0.8,
        episode_id="ep-2024",
    )
    late = _claim_node(
        claim_id="c-late",
        speaker_id="s-a",
        stance=Stance.disputes,
        sentiment=0.8,
        episode_id="ep-2026",
    )
    event_times = {"ep-2024": _dt(2024, 1, 1), "ep-2026": _dt(2026, 1, 1)}
    edges = supersession_edges([early, late], event_times)
    (edge,) = edges
    assert edge.type is EdgeType.supersedes
    # Newer event-time supersedes older (bitemporal rule).
    assert edge.src_id == "c-late"
    assert edge.dst_id == "c-early"
    assert edge.event_time == _dt(2026, 1, 1)


def test_same_speaker_consistent_position_does_not_supersede() -> None:
    a = _claim_node(
        claim_id="c1",
        speaker_id="s-a",
        stance=Stance.asserts,
        sentiment=0.6,
        episode_id="ep-a",
    )
    b = _claim_node(
        claim_id="c2",
        speaker_id="s-a",
        stance=Stance.asserts,
        sentiment=0.7,
        episode_id="ep-b",
    )
    event_times = {"ep-a": _dt(2024, 1, 1), "ep-b": _dt(2026, 1, 1)}
    assert supersession_edges([a, b], event_times) == []


def test_different_speakers_do_not_supersede() -> None:
    a = _claim_node(
        claim_id="c1",
        speaker_id="s-a",
        stance=Stance.asserts,
        sentiment=0.8,
        episode_id="ep-a",
    )
    b = _claim_node(
        claim_id="c2",
        speaker_id="s-b",
        stance=Stance.disputes,
        sentiment=0.8,
        episode_id="ep-b",
    )
    event_times = {"ep-a": _dt(2024, 1, 1), "ep-b": _dt(2026, 1, 1)}
    # Cross-speaker disagreement is a dispute, never a supersession.
    assert supersession_edges([a, b], event_times) == []


# --------------------------------------------------------------------------- #
# mentions
# --------------------------------------------------------------------------- #
def test_mention_edges_link_non_subject_entities() -> None:
    claim = _claim_node(
        claim_id="c1", speaker_id="s-a", subject_canonical_id="ent-apple"
    )
    secondary = [
        _entity_node("ent-openai", "OpenAI", EntityType.organization),
        _entity_node("ent-iphone", "iPhone", EntityType.work),
    ]
    edges = mention_edges(
        claim, secondary, event_time=_dt(2026, 1, 1), ingestion_time=INGEST
    )
    assert {e.type for e in edges} == {EdgeType.mentions}
    assert all(e.src_id == "c1" for e in edges)
    assert {e.dst_id for e in edges} == {"ent-openai", "ent-iphone"}


def test_mention_edges_skip_the_subject_entity() -> None:
    claim = _claim_node(
        claim_id="c1", speaker_id="s-a", subject_canonical_id="ent-apple"
    )
    secondary = [
        _entity_node("ent-apple", "Apple", EntityType.organization),
        _entity_node("ent-openai", "OpenAI", EntityType.organization),
    ]
    edges = mention_edges(
        claim, secondary, event_time=_dt(2026, 1, 1), ingestion_time=INGEST
    )
    # The subject is linked via `about`, never re-linked as a mention.
    assert {e.dst_id for e in edges} == {"ent-openai"}


# --------------------------------------------------------------------------- #
# combined derivation + idempotency
# --------------------------------------------------------------------------- #
def test_derive_relation_edges_is_deterministic_and_idempotent() -> None:
    a = _claim_node(
        claim_id="c1", speaker_id="s-a", stance=Stance.asserts, sentiment=0.8
    )
    b = _claim_node(
        claim_id="c2", speaker_id="s-b", stance=Stance.disputes, sentiment=0.8
    )
    mentions = {
        "c1": [_entity_node("ent-openai", "OpenAI", EntityType.organization)]
    }
    et = {"ep-0001": _dt(2026, 1, 1)}
    first = derive_relation_edges(
        [a, b], et, mentions=mentions, ingestion_time=INGEST
    )
    second = derive_relation_edges(
        [a, b], et, mentions=mentions, ingestion_time=INGEST
    )
    assert {e.edge_id for e in first} == {e.edge_id for e in second}
    types = {e.type for e in first}
    assert EdgeType.disputes in types
    assert EdgeType.mentions in types


# --------------------------------------------------------------------------- #
# loader.bulk_load wiring -> FakeGraphStore
# --------------------------------------------------------------------------- #
def _loader() -> ClaimLoader:
    return ClaimLoader(
        event_times={
            "ep-2024": _dt(2024, 1, 1),
            "ep-2026": _dt(2026, 1, 1),
            "ep-0001": _dt(2026, 1, 1),
        },
        ingestion_time=INGEST,
    )


def test_bulk_load_emits_dispute_edge_for_opposing_claims() -> None:
    store = FakeGraphStore()
    loader = _loader()
    claims = [
        _extracted(
            speaker_id="s-a",
            name="Ann",
            stance=Stance.asserts,
            sentiment=0.8,
            obj="apple wins",
            t_start=1.0,
        ),
        _extracted(
            speaker_id="s-b",
            name="Bob",
            stance=Stance.disputes,
            sentiment=0.8,
            predicate=Predicate.rates_negative,
            obj="apple loses",
            t_start=2.0,
        ),
    ]
    loader.bulk_load(store, claims)
    dispute_edges = [
        e for e in store.edges.values() if e.type is EdgeType.disputes
    ]
    assert len(dispute_edges) == 1


def test_bulk_load_emits_supersedes_for_same_speaker_over_time() -> None:
    store = FakeGraphStore()
    loader = _loader()
    claims = [
        _extracted(
            speaker_id="s-a",
            name="Ann",
            stance=Stance.asserts,
            sentiment=0.8,
            episode_id="ep-2024",
            obj="was bullish",
        ),
        _extracted(
            speaker_id="s-a",
            name="Ann",
            stance=Stance.disputes,
            sentiment=0.8,
            predicate=Predicate.rates_negative,
            episode_id="ep-2026",
            obj="now bearish",
        ),
    ]
    loader.bulk_load(store, claims)
    sup = [e for e in store.edges.values() if e.type is EdgeType.supersedes]
    assert len(sup) == 1
    edge = sup[0]
    # Newer event-time claim supersedes the older; nothing is deleted.
    newer_claim_id = next(
        cid
        for cid, c in store.claims.items()
        if c.source_span.episode_id == "ep-2026"
    )
    older_claim_id = next(
        cid
        for cid, c in store.claims.items()
        if c.source_span.episode_id == "ep-2024"
    )
    assert edge.src_id == newer_claim_id
    assert edge.dst_id == older_claim_id
    # invalidate-not-delete: both claims remain in the store.
    assert len(store.claims) == 2


def test_bulk_load_emits_mentions_for_secondary_entities() -> None:
    store = FakeGraphStore()
    loader = _loader()
    claim = _extracted(
        speaker_id="s-a", name="Ann", subject="Apple", canonical_id="ent-apple"
    )

    def secondary(_c: ExtractedClaim) -> list[Entity]:
        return [
            Entity(
                name="OpenAI",
                type=EntityType.organization,
                canonical_id="ent-openai",
            ),
            # The subject itself must NOT become a mention.
            Entity(
                name="Apple",
                type=EntityType.organization,
                canonical_id="ent-apple",
            ),
        ]

    loader.bulk_load(store, [claim], secondary_entities=secondary)
    mentions = [e for e in store.edges.values() if e.type is EdgeType.mentions]
    assert {e.dst_id for e in mentions} == {"ent-openai"}
    # The mentioned secondary entity node is upserted.
    assert "ent-openai" in store.entities


def test_supersede_then_invalidate_preserves_history() -> None:
    # Wiring the contradiction archetype end to end: supersession adds an edge,
    # and invalidating the older asserts edge respects invalidate-not-delete.
    store = FakeGraphStore()
    loader = _loader()
    early = _extracted(
        speaker_id="s-a",
        name="Ann",
        stance=Stance.asserts,
        sentiment=0.8,
        episode_id="ep-2024",
        obj="bullish",
    )
    late = _extracted(
        speaker_id="s-a",
        name="Ann",
        stance=Stance.disputes,
        sentiment=0.8,
        predicate=Predicate.rates_negative,
        episode_id="ep-2026",
        obj="bearish",
    )
    loader.bulk_load(store, [early, late])

    older_claim_id = next(
        cid
        for cid, c in store.claims.items()
        if c.source_span.episode_id == "ep-2024"
    )
    older_asserts = next(
        e
        for e in store.edges.values()
        if e.type is EdgeType.asserts and e.dst_id == older_claim_id
    )
    assert store.invalidate(older_asserts.edge_id, at=_dt(2026, 1, 1)) is True
    # History preserved: the edge is still present, just flagged invalidated.
    assert store.edges[older_asserts.edge_id].invalidated is True
    assert store.edges[older_asserts.edge_id].valid_to == _dt(2026, 1, 1)
    # A point-in-time query before invalidation still sees the older claim.
    before = store.query(subject_canonical_id="ent-apple", as_of=_dt(2025, 1, 1))
    assert any(r.claim.claim_id == older_claim_id for r in before)
