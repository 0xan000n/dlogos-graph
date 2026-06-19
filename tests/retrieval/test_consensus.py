"""Tests for the consensus-over-time primitive (spec §8).

All deterministic: claims and their event-times are the hand-authored synthetic
fixtures from ``tests/conftest.py``, plus a couple of locally-built claims for
the cases the shared fixtures don't cover (a clean monotonic shift, a reversal,
and canonical-id-based merging of surface forms).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from dlogos.retrieval.consensus import (
    ConsensusTrend,
    TrendDirection,
    consensus_over_time,
    speaker_key,
    subject_key,
)
from dlogos.schema import (
    Entity,
    EntityType,
    ExtractedClaim,
    Predicate,
    SourceSpan,
    SpeakerRef,
    Stance,
)


def _dt(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)


def _claim(
    *,
    speaker_id: str,
    subject: str,
    stance: Stance,
    sentiment: float,
    episode_id: str,
    canonical_id: str | None = None,
    confidence: float = 0.8,
    predicate: Predicate = Predicate.rates_positive,
) -> ExtractedClaim:
    return ExtractedClaim(
        speaker=SpeakerRef(label="SPEAKER_X", resolved_id=speaker_id),
        predicate=predicate,
        subject_entity=Entity(
            name=subject, type=EntityType.organization, canonical_id=canonical_id
        ),
        object="…",
        stance=stance,
        sentiment=sentiment,
        confidence=confidence,
        source_span=SourceSpan(episode_id=episode_id, t_start=0.0, t_end=1.0),
    )


# --------------------------------------------------------------------------- #
# Keys
# --------------------------------------------------------------------------- #
def test_subject_key_prefers_canonical_id() -> None:
    c = _claim(
        speaker_id="s1",
        subject="the iPhone",
        canonical_id="ent-apple",
        stance=Stance.asserts,
        sentiment=0.1,
        episode_id="e1",
    )
    assert subject_key(c) == "ent-apple"


def test_subject_key_falls_back_to_surface_form() -> None:
    c = _claim(
        speaker_id="s1",
        subject="Nvidia",
        stance=Stance.asserts,
        sentiment=0.1,
        episode_id="e1",
    )
    assert subject_key(c) == "Nvidia"


def test_speaker_key_prefers_resolved_then_name_then_label() -> None:
    resolved = ExtractedClaim(
        speaker=SpeakerRef(label="SPEAKER_00", resolved_id="spk-1", name="Jane"),
        predicate=Predicate.rates_positive,
        subject_entity=Entity(name="Apple", type=EntityType.organization),
        object="x",
        stance=Stance.asserts,
        sentiment=0.1,
        confidence=0.5,
        source_span=SourceSpan(episode_id="e", t_start=0.0, t_end=1.0),
    )
    named = resolved.model_copy(
        update={"speaker": SpeakerRef(label="SPEAKER_00", name="Jane")}
    )
    bare = resolved.model_copy(update={"speaker": SpeakerRef(label="SPEAKER_00")})
    assert speaker_key(resolved) == "spk-1"
    assert speaker_key(named) == "Jane"
    assert speaker_key(bare) == "SPEAKER_00"


# --------------------------------------------------------------------------- #
# The headline: a consensus shift over time, with per-bucket attribution.
# --------------------------------------------------------------------------- #
def test_rising_consensus_trend_with_attributed_speakers() -> None:
    """Negative → positive over three monthly buckets = a rising trend.

    Each bucket must carry exactly the speakers who spoke in it.
    """

    claims = [
        # Bucket 0 (Jan): two skeptics, strongly negative.
        _claim(
            speaker_id="alice",
            subject="Apple",
            stance=Stance.asserts,
            sentiment=-0.7,
            episode_id="jan-a",
        ),
        _claim(
            speaker_id="bob",
            subject="Apple",
            stance=Stance.asserts,
            sentiment=-0.5,
            episode_id="jan-b",
        ),
        # Bucket 1 (Feb): warming up, mildly positive.
        _claim(
            speaker_id="carol",
            subject="Apple",
            stance=Stance.asserts,
            sentiment=0.2,
            episode_id="feb-a",
        ),
        # Bucket 2 (Mar): clearly positive.
        _claim(
            speaker_id="alice",
            subject="Apple",
            stance=Stance.asserts,
            sentiment=0.8,
            episode_id="mar-a",
        ),
    ]
    event_times = {
        "jan-a": _dt(2026, 1, 5),
        "jan-b": _dt(2026, 1, 20),
        "feb-a": _dt(2026, 2, 10),
        "mar-a": _dt(2026, 3, 15),
    }

    trend = consensus_over_time(
        claims, event_times, subject="Apple", bucket=timedelta(days=30)
    )

    assert isinstance(trend, ConsensusTrend)
    assert trend.subject == "Apple"
    assert trend.direction is TrendDirection.rising
    assert trend.sentiment_delta > 0

    populated = [b for b in trend.buckets if b.claim_count > 0]
    assert len(populated) == 3

    # Net sentiment increases bucket over bucket.
    sentiments = [b.net_sentiment for b in populated]
    assert sentiments[0] < sentiments[1] < sentiments[2]
    assert sentiments[0] < 0 < sentiments[2]

    # Per-bucket attributed speakers are exactly who spoke in that window.
    assert set(populated[0].speakers) == {"alice", "bob"}
    assert set(populated[1].speakers) == {"carol"}
    assert set(populated[2].speakers) == {"alice"}

    # Bucket 0's confidence-weighted mean of -0.7 and -0.5 (equal confidence).
    assert populated[0].net_sentiment == -0.6

    # The full cast, ordered by first appearance.
    assert trend.all_speakers == ("alice", "bob", "carol")


def test_falling_consensus_trend() -> None:
    claims = [
        _claim(
            speaker_id="a",
            subject="Hype",
            stance=Stance.asserts,
            sentiment=0.9,
            episode_id="e1",
        ),
        _claim(
            speaker_id="b",
            subject="Hype",
            stance=Stance.disputes,
            sentiment=-0.8,
            episode_id="e2",
        ),
    ]
    event_times = {"e1": _dt(2026, 1, 1), "e2": _dt(2026, 3, 1)}
    trend = consensus_over_time(claims, event_times, subject="Hype")
    assert trend.direction is TrendDirection.falling
    assert trend.sentiment_delta < 0


def test_reversal_is_mixed() -> None:
    """Up then down (non-monotonic) is reported as mixed, not rising/falling."""

    claims = [
        _claim(
            speaker_id="a",
            subject="X",
            stance=Stance.asserts,
            sentiment=-0.5,
            episode_id="m0",
        ),
        _claim(
            speaker_id="b",
            subject="X",
            stance=Stance.asserts,
            sentiment=0.7,
            episode_id="m1",
        ),
        _claim(
            speaker_id="c",
            subject="X",
            stance=Stance.asserts,
            sentiment=-0.4,
            episode_id="m2",
        ),
    ]
    # Spaced so each lands in a distinct 30-day bucket aligned to Jan 1.
    event_times = {
        "m0": _dt(2026, 1, 1),
        "m1": _dt(2026, 2, 5),
        "m2": _dt(2026, 3, 15),
    }
    trend = consensus_over_time(claims, event_times, subject="X")
    populated = [b for b in trend.buckets if b.claim_count > 0]
    assert len(populated) == 3
    assert trend.direction is TrendDirection.mixed


def test_flat_trend_when_change_below_threshold() -> None:
    claims = [
        _claim(
            speaker_id="a",
            subject="Steady",
            stance=Stance.asserts,
            sentiment=0.30,
            episode_id="e1",
        ),
        _claim(
            speaker_id="b",
            subject="Steady",
            stance=Stance.asserts,
            sentiment=0.33,
            episode_id="e2",
        ),
    ]
    event_times = {"e1": _dt(2026, 1, 1), "e2": _dt(2026, 3, 1)}
    trend = consensus_over_time(claims, event_times, subject="Steady")
    assert trend.direction is TrendDirection.flat


def test_single_bucket_is_insufficient() -> None:
    claims = [
        _claim(
            speaker_id="a",
            subject="Solo",
            stance=Stance.asserts,
            sentiment=0.4,
            episode_id="e1",
        )
    ]
    event_times = {"e1": _dt(2026, 1, 1)}
    trend = consensus_over_time(claims, event_times, subject="Solo")
    assert trend.direction is TrendDirection.insufficient
    assert len(trend.buckets) == 1


# --------------------------------------------------------------------------- #
# Canonical-id merging: surface forms must aggregate, not fragment (§7.4a/§8).
# --------------------------------------------------------------------------- #
def test_canonical_id_merges_surface_forms() -> None:
    """Apple / iPhone / Apple hardware under one canonical_id form one subject."""

    claims = [
        _claim(
            speaker_id="a",
            subject="Apple",
            canonical_id="ent-apple",
            stance=Stance.asserts,
            sentiment=-0.5,
            episode_id="e1",
        ),
        _claim(
            speaker_id="b",
            subject="the iPhone",
            canonical_id="ent-apple",
            stance=Stance.asserts,
            sentiment=0.6,
            episode_id="e2",
        ),
    ]
    event_times = {"e1": _dt(2026, 1, 1), "e2": _dt(2026, 3, 1)}

    # Selecting by canonical id catches both surface forms in one trend.
    trend = consensus_over_time(claims, event_times, subject="ent-apple")
    populated = [b for b in trend.buckets if b.claim_count > 0]
    assert len(populated) == 2
    assert set(trend.all_speakers) == {"a", "b"}

    # Without a canonical id they would fragment into two single-claim subjects.
    fragmented = [
        c.model_copy(
            update={
                "subject_entity": Entity(
                    name=c.subject_entity.name, type=EntityType.organization
                )
            }
        )
        for c in claims
    ]
    apple_only = consensus_over_time(fragmented, event_times, subject="Apple")
    assert sum(b.claim_count for b in apple_only.buckets) == 1


# --------------------------------------------------------------------------- #
# Subject auto-selection + the shared synthetic fixtures.
# --------------------------------------------------------------------------- #
def test_subject_auto_selected_to_most_claimed() -> None:
    claims = [
        _claim(
            speaker_id="a",
            subject="Apple",
            stance=Stance.asserts,
            sentiment=0.1,
            episode_id="e1",
        ),
        _claim(
            speaker_id="b",
            subject="Apple",
            stance=Stance.asserts,
            sentiment=0.2,
            episode_id="e2",
        ),
        _claim(
            speaker_id="c",
            subject="Tesla",
            stance=Stance.asserts,
            sentiment=0.9,
            episode_id="e3",
        ),
    ]
    event_times = {
        "e1": _dt(2026, 1, 1),
        "e2": _dt(2026, 2, 1),
        "e3": _dt(2026, 1, 1),
    }
    trend = consensus_over_time(claims, event_times)  # subject=None
    assert trend.subject == "Apple"


def test_consensus_over_shared_synthetic_fixtures(
    synthetic_claims, claim_event_times
) -> None:
    """The conftest synthetic claims about Apple span Jan→May 2026.

    They start negative (Jan, the analyst rating hardware down) and end positive
    (May, guest-b's strongest-cycle assertion), so the helper reports a rise and
    surfaces every attributed speaker across the timeline.
    """

    trend = consensus_over_time(
        synthetic_claims,
        claim_event_times,
        subject="Apple",
        bucket=timedelta(days=30),
    )
    populated = [b for b in trend.buckets if b.claim_count > 0]
    assert len(populated) == 4  # one claim per month, distinct buckets

    # First bucket is the analyst's negative rating; last is guest-b's positive.
    assert populated[0].net_sentiment < 0
    assert populated[-1].net_sentiment > 0
    assert trend.direction is TrendDirection.rising

    # Every resolved speaker shows up in the cast.
    assert set(trend.all_speakers) == {
        "spk-analyst",
        "spk-host",
        "spk-guest-b",
    }
    # The analyst appears in two different buckets (Jan and April).
    analyst_buckets = [b for b in populated if "spk-analyst" in b.speakers]
    assert len(analyst_buckets) == 2


def test_claim_with_missing_event_time_is_skipped() -> None:
    claims = [
        _claim(
            speaker_id="a",
            subject="Z",
            stance=Stance.asserts,
            sentiment=0.5,
            episode_id="known",
        ),
        _claim(
            speaker_id="b",
            subject="Z",
            stance=Stance.asserts,
            sentiment=-0.5,
            episode_id="unknown",
        ),
    ]
    event_times = {"known": _dt(2026, 1, 1)}  # 'unknown' absent on purpose
    trend = consensus_over_time(claims, event_times, subject="Z")
    assert sum(b.claim_count for b in trend.buckets) == 1
    assert trend.all_speakers == ("a",)


def test_confidence_weighting_pulls_net_toward_confident_claim() -> None:
    """Within a bucket, a higher-confidence claim dominates the net sentiment."""

    claims = [
        _claim(
            speaker_id="confident",
            subject="W",
            stance=Stance.asserts,
            sentiment=1.0,
            confidence=0.95,
            episode_id="e1",
        ),
        _claim(
            speaker_id="unsure",
            subject="W",
            stance=Stance.asserts,
            sentiment=-1.0,
            confidence=0.05,
            episode_id="e1b",
        ),
    ]
    event_times = {"e1": _dt(2026, 1, 1), "e1b": _dt(2026, 1, 2)}
    trend = consensus_over_time(claims, event_times, subject="W")
    # Both land in one bucket; net should lean positive toward the confident one.
    bucket = next(b for b in trend.buckets if b.claim_count > 0)
    assert bucket.claim_count == 2
    assert bucket.net_sentiment > 0
