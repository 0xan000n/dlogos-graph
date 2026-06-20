"""Tests for the shared domain model in ``dlogos.schema``."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dlogos.schema import (
    Entity,
    EntityType,
    ExtractedClaim,
    Predicate,
    SourceSpan,
    SpeakerRef,
    Stance,
)


def _good_claim(**overrides) -> ExtractedClaim:
    base = dict(
        speaker=SpeakerRef(label="SPEAKER_01", resolved_id="spk-1", name="Jane"),
        predicate=Predicate.rates_negative,
        subject_entity=Entity(name="Apple", type=EntityType.organization),
        object="hardware innovation has plateaued",
        stance=Stance.asserts,
        sentiment=-0.5,
        confidence=0.8,
        source_span=SourceSpan(episode_id="ep-1", t_start=4.5, t_end=10.0),
    )
    base.update(overrides)
    return ExtractedClaim(**base)


def test_valid_extracted_claim() -> None:
    claim = _good_claim()
    assert claim.predicate is Predicate.rates_negative
    assert claim.stance is Stance.asserts
    assert claim.subject_entity.type is EntityType.organization
    assert claim.subject_entity.canonical_id is None  # unresolved off the extractor
    assert claim.speaker.resolved_id == "spk-1"
    assert claim.source_span.transcript_offset is None


def test_entity_carries_optional_wikidata_qid() -> None:
    # Wikidata-anchored resolution stamps a QID onto the entity; it defaults to
    # None straight off the extractor.
    anchored = Entity(name="Apple", type=EntityType.organization, qid="Q312")
    assert anchored.qid == "Q312"
    assert Entity(name="Apple", type=EntityType.organization).qid is None


@pytest.mark.parametrize("bad_sentiment", [-1.5, 1.5, 2.0, -42.0])
def test_sentiment_out_of_range_rejected(bad_sentiment: float) -> None:
    with pytest.raises(ValidationError):
        _good_claim(sentiment=bad_sentiment)


@pytest.mark.parametrize("bad_confidence", [-0.1, 1.1, 5.0])
def test_confidence_out_of_range_rejected(bad_confidence: float) -> None:
    with pytest.raises(ValidationError):
        _good_claim(confidence=bad_confidence)


def test_sentiment_boundaries_accepted() -> None:
    assert _good_claim(sentiment=-1.0).sentiment == -1.0
    assert _good_claim(sentiment=1.0).sentiment == 1.0


def test_confidence_boundaries_accepted() -> None:
    assert _good_claim(confidence=0.0).confidence == 0.0
    assert _good_claim(confidence=1.0).confidence == 1.0


def test_predicate_is_closed_enum() -> None:
    # A value outside the controlled vocabulary must be rejected.
    with pytest.raises(ValidationError):
        _good_claim(predicate="invents")

    # The controlled vocabulary has the expected ~15 normalized predicates.
    expected = {
        "expects",
        "rates_positive",
        "rates_negative",
        "predicts",
        "recommends",
        "criticizes",
        "compares",
        "explains",
        "attributes",
        "forecasts",
        "endorses",
        "rejects",
        "questions",
        "agrees",
        "disagrees",
    }
    assert {p.value for p in Predicate} == expected


def test_stance_is_closed_enum() -> None:
    with pytest.raises(ValidationError):
        _good_claim(stance="speculates")
    assert {s.value for s in Stance} == {
        "asserts",
        "disputes",
        "hedges",
        "predicts",
        "retracts",
    }


def test_negative_timestamp_rejected() -> None:
    with pytest.raises(ValidationError):
        SourceSpan(episode_id="ep-1", t_start=-1.0, t_end=10.0)


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        _good_claim(unexpected_field="boom")


def test_synthetic_fixtures_load(synthetic_transcript, synthetic_claims) -> None:
    assert synthetic_transcript.episode_id == "ep-0001"
    assert len(synthetic_transcript.segments) == 6
    assert len(synthetic_claims) == 4
    assert all(isinstance(c, ExtractedClaim) for c in synthetic_claims)


def test_fake_embedder_deterministic(fake_embedder) -> None:
    a1 = fake_embedder.embed("Apple")
    a2 = fake_embedder.embed("Apple")
    assert a1 == a2
    assert len(a1) == fake_embedder.DIM
    # Apple and iPhone are near; Apple and OpenAI are far.
    import numpy as np

    apple = np.asarray(fake_embedder.embed("Apple"))
    iphone = np.asarray(fake_embedder.embed("the iPhone"))
    openai = np.asarray(fake_embedder.embed("OpenAI"))
    assert float(apple @ iphone) > float(apple @ openai)
