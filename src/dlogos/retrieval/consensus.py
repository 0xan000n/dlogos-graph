"""Consensus-over-time — the headline retrieval primitive (spec §8).

The thing dLogos is supposed to do that a frozen model and a dumb vector index
cannot: take the resolved, stance-tagged claims about a *subject* and show **how
the position on it moved over time across multiple attributed speakers**.

This module is deliberately a set of **pure functions** over the shared schema
types — no graph, no embedder, no I/O. That keeps it trivially testable and
makes it reusable both by the live retrieval path (over claims pulled from the
graph) and by the eval harness (over a fixed synthetic set).

Design intent from the spec:

- Bucketing is by the resolved subject ``canonical_id`` so claims about
  *Apple* / *iPhone* / *Apple hardware* aggregate into one subject instead of
  fragmenting and undercounting the consensus (§7.4a, §8). Callers pass the
  claims that the resolution stage already grouped; here we simply *use* the
  ``canonical_id`` when present and fall back to the surface ``name`` otherwise.
- Each bucket carries the **attributed speakers** — the per-bucket "who said
  what" — because attribution is the dimension the eval elevates (§9). A
  speaker is identified by their resolved id where available, else by name,
  else by the raw diarization label.
- A signed **net sentiment**, weighted by extraction confidence, plus a
  signed **net stance** (assertions push positive, disputes/retracts push
  negative, hedges neutral), give a per-bucket scalar; the trend across buckets
  is the headline "the consensus moved from X to Y".

Event-time (when the claim was *said*) is an episode property and is kept out
of the claim record in this codebase (see ``tests/conftest.py``), so the
caller supplies an ``episode_id -> event_time`` mapping. This mirrors how the
graph load joins a Claim's source span to its Episode's publish date.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum

from dlogos.schema import ExtractedClaim, Stance

# --------------------------------------------------------------------------- #
# How a stance maps onto a signed direction.
#
# The net-stance scalar smooths extraction noise (spec §11): a single mislabeled
# claim cannot flip a bucket that has several agreeing claims. Magnitudes are
# intentionally simple and symmetric.
# --------------------------------------------------------------------------- #
_STANCE_SIGN: dict[Stance, float] = {
    Stance.asserts: 1.0,
    Stance.predicts: 0.5,
    Stance.hedges: 0.0,
    Stance.disputes: -1.0,
    Stance.retracts: -1.0,
}


class TrendDirection(str, Enum):
    """Coarse direction of the consensus shift across the time buckets."""

    rising = "rising"  # net sentiment moved up (more positive) over time
    falling = "falling"  # net sentiment moved down (more negative) over time
    flat = "flat"  # no meaningful change
    mixed = "mixed"  # non-monotonic — moved both ways across buckets
    insufficient = "insufficient"  # fewer than two populated buckets


# --------------------------------------------------------------------------- #
# Result objects (plain dataclasses — no pydantic needed, these are outputs).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ConsensusBucket:
    """One time window's worth of consensus about the subject.

    ``net_sentiment`` is the confidence-weighted mean sentiment in the window;
    ``net_stance`` is the confidence-weighted mean stance sign. ``speakers`` is
    the ordered, de-duplicated set of attributed speakers contributing to the
    bucket — the per-bucket "who said what" the eval cares about.
    """

    start: datetime
    end: datetime
    claim_count: int
    net_sentiment: float
    net_stance: float
    speakers: tuple[str, ...]
    # Per-speaker contribution within the bucket: speaker -> (count, mean_sentiment).
    speaker_breakdown: dict[str, tuple[int, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class ConsensusTrend:
    """The full consensus-over-time answer for a subject."""

    subject: str
    buckets: tuple[ConsensusBucket, ...]
    direction: TrendDirection
    # Net-sentiment delta from the first to the last *populated* bucket.
    sentiment_delta: float
    # Every distinct attributed speaker across all buckets, ordered by first
    # appearance — the cast of the story.
    all_speakers: tuple[str, ...]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def subject_key(claim: ExtractedClaim) -> str:
    """The key a claim aggregates under.

    Prefer the resolved ``canonical_id`` (so surface variants merge); fall back
    to the raw surface form when resolution has not run.
    """

    ent = claim.subject_entity
    return ent.canonical_id or ent.name


def speaker_key(claim: ExtractedClaim) -> str:
    """Stable, human-meaningful identifier for the attributed speaker.

    Resolved id first (merges a speaker across episodes), then name, then the
    raw per-episode diarization label as a last resort.
    """

    spk = claim.speaker
    return spk.resolved_id or spk.name or spk.label


def _as_aware(dt: datetime) -> datetime:
    """Treat naive datetimes as UTC so comparisons never raise."""

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


# --------------------------------------------------------------------------- #
# The primitive
# --------------------------------------------------------------------------- #
def consensus_over_time(
    claims: list[ExtractedClaim],
    event_times: dict[str, datetime],
    *,
    subject: str | None = None,
    bucket: timedelta = timedelta(days=30),
    drop_empty_buckets: bool = True,
    flat_threshold: float = 0.1,
) -> ConsensusTrend:
    """Bucket claims about a subject over time and characterize the shift.

    Pure: no I/O, no global state, deterministic for fixed inputs.

    Parameters
    ----------
    claims:
        Resolved, stance-tagged claims. Claims whose subject does not match
        ``subject`` are ignored. Event-time comes from ``event_times`` keyed by
        the claim's ``source_span.episode_id``; a claim whose episode is absent
        from ``event_times`` is skipped (we cannot place it on the timeline).
    event_times:
        ``episode_id -> event_time`` (the episode publish/recording date). This
        is the validity anchor; see the module docstring for why it is passed
        in rather than read off the claim.
    subject:
        Which subject to summarize, matched against
        :func:`subject_key`. If ``None``, the most-claimed subject is chosen so
        a caller can pass a pre-filtered list without naming it.
    bucket:
        Width of each time window. Buckets are aligned to the earliest event
        time so the boundaries are deterministic and independent of "now".
    drop_empty_buckets:
        When ``True`` (default), windows with no claims are omitted from the
        result; the trend is computed over populated buckets only.
    flat_threshold:
        Absolute net-sentiment delta below which the trend is reported as
        ``flat`` rather than rising/falling.

    Returns
    -------
    ConsensusTrend
        Ordered buckets (oldest first), the coarse direction, the first→last
        sentiment delta, and the full ordered speaker cast.
    """

    # 1) Filter to the subject of interest and attach event-time.
    placed: list[tuple[datetime, ExtractedClaim]] = []
    counts: dict[str, int] = {}
    for c in claims:
        key = subject_key(c)
        counts[key] = counts.get(key, 0) + 1

    if subject is None:
        if not counts:
            return ConsensusTrend(
                subject="",
                buckets=(),
                direction=TrendDirection.insufficient,
                sentiment_delta=0.0,
                all_speakers=(),
            )
        # Most-claimed subject; ties broken by name for determinism.
        subject = max(sorted(counts), key=lambda k: counts[k])

    for c in claims:
        if subject_key(c) != subject:
            continue
        et = event_times.get(c.source_span.episode_id)
        if et is None:
            continue  # cannot place on the timeline
        placed.append((_as_aware(et), c))

    if not placed:
        return ConsensusTrend(
            subject=subject,
            buckets=(),
            direction=TrendDirection.insufficient,
            sentiment_delta=0.0,
            all_speakers=(),
        )

    placed.sort(key=lambda pair: pair[0])
    origin = placed[0][0]

    # 2) Assign each claim to a deterministic, origin-aligned bucket index.
    bucket_seconds = bucket.total_seconds()
    if bucket_seconds <= 0:
        raise ValueError("bucket width must be positive")

    grouped: dict[int, list[ExtractedClaim]] = {}
    for et, c in placed:
        idx = int((et - origin).total_seconds() // bucket_seconds)
        grouped.setdefault(idx, []).append(c)

    # 3) Build a bucket for each index. When not dropping empties, fill the gaps
    #    between the first and last populated index with zero-claim buckets so
    #    the timeline is contiguous.
    min_idx = min(grouped)
    max_idx = max(grouped)
    indices = (
        sorted(grouped)
        if drop_empty_buckets
        else list(range(min_idx, max_idx + 1))
    )

    buckets: list[ConsensusBucket] = []
    all_speakers: list[str] = []
    seen_speakers: set[str] = set()

    for idx in indices:
        start = origin + timedelta(seconds=idx * bucket_seconds)
        end = start + bucket
        bucket_claims = grouped.get(idx, [])

        if not bucket_claims:
            buckets.append(
                ConsensusBucket(
                    start=start,
                    end=end,
                    claim_count=0,
                    net_sentiment=0.0,
                    net_stance=0.0,
                    speakers=(),
                    speaker_breakdown={},
                )
            )
            continue

        weight_sum = 0.0
        sent_acc = 0.0
        stance_acc = 0.0
        # Per-speaker accumulation (insertion-ordered).
        spk_count: dict[str, int] = {}
        spk_sent_sum: dict[str, float] = {}
        spk_order: list[str] = []

        for c in bucket_claims:
            w = max(c.confidence, 0.0)
            # Even a zero-confidence claim should count as present; nudge weight
            # so it contributes to the (unweighted) mean rather than vanishing.
            eff_w = w if w > 0 else 1e-9
            weight_sum += eff_w
            sent_acc += eff_w * c.sentiment
            stance_acc += eff_w * _STANCE_SIGN[c.stance]

            sk = speaker_key(c)
            if sk not in spk_count:
                spk_count[sk] = 0
                spk_sent_sum[sk] = 0.0
                spk_order.append(sk)
            spk_count[sk] += 1
            spk_sent_sum[sk] += c.sentiment

            if sk not in seen_speakers:
                seen_speakers.add(sk)
                all_speakers.append(sk)

        net_sentiment = sent_acc / weight_sum if weight_sum else 0.0
        net_stance = stance_acc / weight_sum if weight_sum else 0.0
        breakdown = {
            sk: (spk_count[sk], spk_sent_sum[sk] / spk_count[sk])
            for sk in spk_order
        }

        buckets.append(
            ConsensusBucket(
                start=start,
                end=end,
                claim_count=len(bucket_claims),
                net_sentiment=net_sentiment,
                net_stance=net_stance,
                speakers=tuple(spk_order),
                speaker_breakdown=breakdown,
            )
        )

    # 4) Characterize the trend over *populated* buckets.
    populated = [b for b in buckets if b.claim_count > 0]
    direction, delta = _classify_trend(populated, flat_threshold)

    return ConsensusTrend(
        subject=subject,
        buckets=tuple(buckets),
        direction=direction,
        sentiment_delta=delta,
        all_speakers=tuple(all_speakers),
    )


def _classify_trend(
    populated: list[ConsensusBucket], flat_threshold: float
) -> tuple[TrendDirection, float]:
    """Direction + first→last delta over the populated buckets.

    ``mixed`` is reported when the net-sentiment series moves both up and down
    by more than ``flat_threshold`` between consecutive buckets (non-monotonic),
    so a genuine reversal is not flattened into a single rising/falling label.
    """

    if len(populated) < 2:
        return TrendDirection.insufficient, 0.0

    series = [b.net_sentiment for b in populated]
    delta = series[-1] - series[0]

    rose = falls = False
    for prev, cur in zip(series, series[1:]):
        step = cur - prev
        if step > flat_threshold:
            rose = True
        elif step < -flat_threshold:
            falls = True

    if rose and falls:
        return TrendDirection.mixed, delta
    if abs(delta) < flat_threshold:
        return TrendDirection.flat, delta
    if delta > 0:
        return TrendDirection.rising, delta
    return TrendDirection.falling, delta
