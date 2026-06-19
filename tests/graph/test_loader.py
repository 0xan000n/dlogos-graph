"""Tests for the Approach-B loader (resolved claim -> reified graph records)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from dlogos.graph.fake_store import FakeGraphStore
from dlogos.graph.loader import ClaimLoader
from dlogos.graph.store import EdgeType
from dlogos.schema import (
    Entity,
    EntityType,
    ExtractedClaim,
    Predicate,
    SourceSpan,
    SpeakerRef,
    Stance,
)


def _resolved_claim(
    *,
    speaker_id: str = "spk-analyst",
    canonical_id: str = "ent-apple",
    subject: str = "Apple",
    obj: str = "hardware innovation has plateaued",
    episode_id: str = "ep-0001",
    t_start: float = 4.5,
    predicate: Predicate = Predicate.rates_negative,
) -> ExtractedClaim:
    return ExtractedClaim(
        speaker=SpeakerRef(label="SPEAKER_01", resolved_id=speaker_id, name="Jane"),
        predicate=predicate,
        subject_entity=Entity(
            name=subject, type=EntityType.organization, canonical_id=canonical_id
        ),
        object=obj,
        stance=Stance.asserts,
        sentiment=-0.6,
        confidence=0.82,
        source_span=SourceSpan(episode_id=episode_id, t_start=t_start, t_end=10.0),
    )


def _loader() -> ClaimLoader:
    return ClaimLoader(
        event_times={"ep-0001": datetime(2026, 1, 10, tzinfo=timezone.utc)},
        ingestion_time=datetime(2026, 6, 18, tzinfo=timezone.utc),
    )


def test_to_triplet_builds_reified_claim_node() -> None:
    triplet = _loader().to_triplet(_resolved_claim())

    # Reified claim carries stance/sentiment/confidence + source span (spec §6).
    assert triplet.claim.stance is Stance.asserts
    assert triplet.claim.predicate is Predicate.rates_negative
    assert triplet.claim.sentiment == -0.6
    assert triplet.claim.confidence == 0.82
    assert triplet.claim.source_span.episode_id == "ep-0001"
    assert triplet.claim.speaker_id == "spk-analyst"
    assert triplet.claim.subject_canonical_id == "ent-apple"

    # Canonical entity node carries the resolution id + the surface form alias.
    assert triplet.subject.canonical_id == "ent-apple"
    assert "Apple" in triplet.subject.aliases


def test_to_triplet_builds_bitemporal_edges() -> None:
    triplet = _loader().to_triplet(_resolved_claim())
    edge_types = {e.type for e in triplet.edges}
    assert EdgeType.asserts in edge_types  # Speaker -> Claim
    assert EdgeType.about in edge_types  # Claim -> Entity (subject)
    assert EdgeType.appears_in in edge_types  # Speaker -> Episode

    for edge in triplet.edges:
        # event-time is the episode publish date; ingestion-time is injected.
        assert edge.event_time == datetime(2026, 1, 10, tzinfo=timezone.utc)
        assert edge.ingestion_time == datetime(2026, 6, 18, tzinfo=timezone.utc)
        assert edge.valid_from == edge.event_time
        assert edge.valid_to is None
        assert edge.invalidated is False


def test_claim_id_is_deterministic_and_idempotent() -> None:
    loader = _loader()
    a = loader.to_triplet(_resolved_claim())
    b = loader.to_triplet(_resolved_claim())
    assert a.claim.claim_id == b.claim.claim_id
    # Same logical edges get the same ids too.
    assert {e.edge_id for e in a.edges} == {e.edge_id for e in b.edges}


def test_unresolved_speaker_rejected() -> None:
    claim = _resolved_claim()
    claim = claim.model_copy(
        update={"speaker": SpeakerRef(label="SPEAKER_01", resolved_id=None)}
    )
    with pytest.raises(ValueError, match="resolved speaker id"):
        _loader().to_triplet(claim)


def test_unresolved_entity_rejected() -> None:
    claim = _resolved_claim()
    claim = claim.model_copy(
        update={
            "subject_entity": Entity(name="Apple", type=EntityType.organization)
        }
    )
    with pytest.raises(ValueError, match="canonical entity id"):
        _loader().to_triplet(claim)


def test_load_claim_per_add_path() -> None:
    store = FakeGraphStore()
    loader = _loader()
    claim_id = loader.load_claim(store, _resolved_claim())
    assert store.claim_count() == 1
    assert claim_id in store.claims
    assert "spk-analyst" in store.speakers
    assert "ent-apple" in store.entities
    # Per-add path is not the bulk path: no bulk_load call recorded.
    assert store.bulk_load_calls == 0


def test_bulk_load_count_and_dedup_bypass() -> None:
    store = FakeGraphStore()
    loader = _loader()
    claims = [
        _resolved_claim(obj="hardware innovation has plateaued", t_start=4.5),
        _resolved_claim(
            obj="services growth offsets hardware",
            t_start=120.0,
            predicate=Predicate.rates_positive,
        ),
        _resolved_claim(
            speaker_id="spk-guest-b",
            obj="strongest product cycle in years",
            t_start=45.0,
        ),
    ]
    loaded = loader.bulk_load(store, claims)

    assert loaded == 3
    assert store.claim_count() == 3
    assert store.bulk_load_calls == 1
    # The whole point of the fast path: NO per-add LLM dedup ran (spec §7.5/§7.6).
    assert store.llm_dedup_invocations == 0
    # Distinct speakers and one canonical entity (clustering already collapsed).
    assert set(store.speakers) == {"spk-analyst", "spk-guest-b"}
    assert set(store.entities) == {"ent-apple"}


def test_bulk_load_is_idempotent_on_reload() -> None:
    store = FakeGraphStore()
    loader = _loader()
    claims = [_resolved_claim(), _resolved_claim(obj="b", t_start=120.0)]
    first = loader.bulk_load(store, claims)
    second = loader.bulk_load(store, claims)
    assert first == 2
    assert second == 2
    # Re-loading the identical batch does not double-count (deterministic ids).
    assert store.claim_count() == 2


def test_bulk_load_without_bypass_records_llm_cost() -> None:
    """The non-bypass path simulates the redundant per-add LLM dedup cost."""
    store = FakeGraphStore()
    loader = _loader()
    claims = [_resolved_claim(), _resolved_claim(obj="b", t_start=120.0)]
    loaded = loader.bulk_load(store, claims, bypass_llm_dedup=False)
    assert loaded == 2
    # Data still loads identically, but the costly path is recorded for the spike.
    assert store.llm_dedup_invocations == 2


def test_bulk_load_merges_aliases_for_same_canonical_entity() -> None:
    store = FakeGraphStore()
    loader = _loader()
    claims = [
        _resolved_claim(subject="Apple", obj="a", t_start=4.5),
        _resolved_claim(subject="iPhone", obj="b", t_start=120.0),
        _resolved_claim(subject="Apple hardware", obj="c", t_start=600.0),
    ]
    loader.bulk_load(store, claims)
    # All three surface forms clustered to one canonical id with merged aliases.
    assert set(store.entities) == {"ent-apple"}
    aliases = set(store.entities["ent-apple"].aliases)
    assert {"Apple", "iPhone", "Apple hardware"} <= aliases
