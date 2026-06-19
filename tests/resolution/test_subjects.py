"""Tests for subject-entity embedding clustering (resolution §7.4a, lever i).

All deterministic: the injected ``fake_embedder`` (conftest) maps known surface
forms to fixed vectors, so the Apple-family collapses and Apple/Microsoft don't,
with no model or network.
"""

from __future__ import annotations

import pytest

from dlogos.resolution.subjects import (
    DEFAULT_SIMILARITY_THRESHOLD,
    Embedder,
    EntityCluster,
    cluster_entities,
    resolve_subjects,
)
from dlogos.schema import Entity, EntityType


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _org(name: str) -> Entity:
    return Entity(name=name, type=EntityType.organization)


def _cluster_for(clusters: list[EntityCluster], member: str) -> EntityCluster:
    for c in clusters:
        if any(m.casefold() == member.casefold() for m in c.members):
            return c
    raise AssertionError(f"no cluster contains {member!r}: {clusters}")


# --------------------------------------------------------------------------- #
# Core behaviour: Apple family collapses, distinct orgs don't
# --------------------------------------------------------------------------- #
def test_apple_family_collapses_to_one_cluster(fake_embedder):
    entities = [_org("Apple"), _org("the iPhone"), _org("Apple hardware")]
    clusters = cluster_entities(entities, fake_embedder)

    assert len(clusters) == 1, clusters
    only = clusters[0]
    assert set(m.casefold() for m in only.members) == {
        "apple",
        "the iphone",
        "apple hardware",
    }
    # All three surface forms share exactly one canonical id.
    assert only.canonical_id.startswith("ent-")


def test_apple_and_openai_do_not_merge(fake_embedder):
    # conftest fake-embedder places OpenAI orthogonal to Apple (cosine 0).
    entities = [_org("Apple"), _org("OpenAI")]
    clusters = cluster_entities(entities, fake_embedder)

    assert len(clusters) == 2, clusters
    apple = _cluster_for(clusters, "Apple")
    openai = _cluster_for(clusters, "OpenAI")
    assert apple.canonical_id != openai.canonical_id


def test_microsoft_unknown_form_stays_separate_from_apple(fake_embedder):
    # "Microsoft" is not in the fake table -> a stable pseudo-random vector,
    # which must not collapse into the Apple cluster.
    entities = [_org("Apple"), _org("the iPhone"), _org("Microsoft")]
    clusters = cluster_entities(entities, fake_embedder)

    apple = _cluster_for(clusters, "Apple")
    microsoft = _cluster_for(clusters, "Microsoft")
    assert apple.canonical_id != microsoft.canonical_id
    assert "apple" in {m.casefold() for m in apple.members}
    assert "the iphone" in {m.casefold() for m in apple.members}


# --------------------------------------------------------------------------- #
# canonical_id stability
# --------------------------------------------------------------------------- #
def test_canonical_id_is_order_independent(fake_embedder):
    a = cluster_entities(
        [_org("Apple"), _org("the iPhone"), _org("Apple hardware")], fake_embedder
    )
    b = cluster_entities(
        [_org("Apple hardware"), _org("Apple"), _org("the iPhone")], fake_embedder
    )
    assert len(a) == len(b) == 1
    # Same canonical id regardless of input ordering.
    assert a[0].canonical_id == b[0].canonical_id


def test_canonical_id_stable_across_runs(fake_embedder):
    first = cluster_entities([_org("Apple"), _org("OpenAI")], fake_embedder)
    second = cluster_entities([_org("Apple"), _org("OpenAI")], fake_embedder)
    ids_first = {c.canonical_id for c in first}
    ids_second = {c.canonical_id for c in second}
    assert ids_first == ids_second


def test_casing_and_whitespace_variants_collapse(fake_embedder):
    # Exact-but-for-casing/whitespace forms must merge even before embedding.
    entities = [_org("Apple"), _org("  apple "), _org("APPLE")]
    clusters = cluster_entities(entities, fake_embedder)
    assert len(clusters) == 1
    assert len(clusters[0].members) == 1  # all collapse to one display form


# --------------------------------------------------------------------------- #
# Type-awareness: different EntityType never merges
# --------------------------------------------------------------------------- #
def test_different_entity_types_never_merge(fake_embedder):
    # Same surface form, different types -> two clusters, never one.
    person_apple = Entity(name="Apple", type=EntityType.person)
    org_apple = Entity(name="Apple", type=EntityType.organization)
    clusters = cluster_entities([person_apple, org_apple], fake_embedder)
    assert len(clusters) == 2
    types = {c.entity_type for c in clusters}
    assert types == {EntityType.person, EntityType.organization}


# --------------------------------------------------------------------------- #
# Threshold behaviour
# --------------------------------------------------------------------------- #
def test_high_threshold_keeps_family_separate(fake_embedder):
    # Apple-vs-iPhone cosine ~0.994; a threshold above that refuses the merge.
    clusters = cluster_entities(
        [_org("Apple"), _org("the iPhone")], fake_embedder, threshold=0.999
    )
    assert len(clusters) == 2


def test_default_threshold_value():
    # The tuned default keeps the Apple family together with margin.
    assert 0.8 <= DEFAULT_SIMILARITY_THRESHOLD < 0.97


# --------------------------------------------------------------------------- #
# resolve_subjects: stamps canonical_id onto claim subjects without mutating
# --------------------------------------------------------------------------- #
def test_resolve_subjects_stamps_canonical_id(synthetic_claims, fake_embedder):
    # All synthetic claims are about "Apple" -> one canonical id across them.
    result = resolve_subjects(synthetic_claims, fake_embedder)

    canonical_ids = {c.subject_entity.canonical_id for c in result.claims}
    assert len(canonical_ids) == 1
    assert next(iter(canonical_ids)) is not None
    assert next(iter(canonical_ids)).startswith("ent-")

    # Exactly one Apple cluster.
    assert len(result.clusters) == 1
    assert result.clusters[0].entity_type == EntityType.organization


def test_resolve_subjects_does_not_mutate_inputs(synthetic_claims, fake_embedder):
    before = [c.subject_entity.canonical_id for c in synthetic_claims]
    resolve_subjects(synthetic_claims, fake_embedder)
    after = [c.subject_entity.canonical_id for c in synthetic_claims]
    # Inputs untouched (all still None).
    assert before == after == [None] * len(synthetic_claims)


def test_resolve_subjects_mixed_subjects(fake_embedder):
    from dlogos.schema import (
        ExtractedClaim,
        Predicate,
        SourceSpan,
        SpeakerRef,
        Stance,
    )

    def claim(subject: str) -> ExtractedClaim:
        return ExtractedClaim(
            speaker=SpeakerRef(label="SPEAKER_00"),
            predicate=Predicate.rates_positive,
            subject_entity=_org(subject),
            object="x",
            stance=Stance.asserts,
            sentiment=0.1,
            confidence=0.5,
            source_span=SourceSpan(episode_id="ep", t_start=0.0, t_end=1.0),
        )

    claims = [claim("Apple"), claim("the iPhone"), claim("OpenAI")]
    result = resolve_subjects(claims, fake_embedder)

    by_name = {c.subject_entity.name: c.subject_entity.canonical_id for c in result.claims}
    # Apple and the iPhone share an id; OpenAI is distinct.
    assert by_name["Apple"] == by_name["the iPhone"]
    assert by_name["OpenAI"] != by_name["Apple"]
    # surface_to_id covers every surface form, keyed by type+surface.
    assert len(result.surface_to_id) == 3


def test_surface_to_id_round_trips(synthetic_claims, fake_embedder):
    result = resolve_subjects(synthetic_claims, fake_embedder)
    # Every resolved claim's id appears as a value in surface_to_id.
    ids_on_claims = {c.subject_entity.canonical_id for c in result.claims}
    assert ids_on_claims <= set(result.surface_to_id.values())


# --------------------------------------------------------------------------- #
# Injection contract
# --------------------------------------------------------------------------- #
def test_fake_embedder_satisfies_protocol(fake_embedder):
    assert isinstance(fake_embedder, Embedder)


def test_embedder_without_batch_method_still_works():
    """Embedders exposing only ``embed`` (no ``embed_batch``) must work."""

    class OnlyEmbed:
        def embed(self, text: str) -> list[float]:
            table = {
                "Apple": [1.0, 0.0, 0.0],
                "the iPhone": [0.95, 0.05, 0.0],
                "OpenAI": [0.0, 1.0, 0.0],
            }
            return table.get(text, [0.0, 0.0, 1.0])

    clusters = cluster_entities(
        [_org("Apple"), _org("the iPhone"), _org("OpenAI")], OnlyEmbed()
    )
    # Apple + iPhone merge; OpenAI separate.
    assert len(clusters) == 2
