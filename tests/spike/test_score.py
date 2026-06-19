"""Deterministic tests for the spike scorers (spec §7.6).

Build :class:`ApproachArtifacts` directly from synthetic inputs so each metric
is exercised in isolation — no orchestration, no network, no clock. The span /
attribution checks are validated against hand-authored diarization ground-truth.
"""

from __future__ import annotations

import json

from dlogos.schema import (
    Entity,
    EntityType,
    ExtractedClaim,
    Predicate,
    SourceSpan,
    SpeakerRef,
    Stance,
    TranscriptSegment,
)
from dlogos.spike.run_comparison import (
    ApproachArtifacts,
    ComparisonResult,
    Pricing,
)
from dlogos.spike.score import (
    SpikeReport,
    emit_report,
    render_report_md,
    score_approach,
    score_comparison,
)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _seg(speaker: str, t0: float, t1: float) -> TranscriptSegment:
    return TranscriptSegment(speaker=speaker, text="...", t_start=t0, t_end=t1)


def _claim(
    *,
    speaker_label: str = "SPEAKER_00",
    episode_id: str = "ep-1",
    t_start: float = 0.0,
    t_end: float = 4.0,
    subject: str = "Apple",
) -> ExtractedClaim:
    return ExtractedClaim(
        speaker=SpeakerRef(label=speaker_label),
        predicate=Predicate.rates_negative,
        subject_entity=Entity(name=subject, type=EntityType.organization),
        object="hardware plateaued",
        stance=Stance.asserts,
        sentiment=-0.5,
        confidence=0.8,
        source_span=SourceSpan(
            episode_id=episode_id, t_start=t_start, t_end=t_end
        ),
    )


def _artifacts(**overrides: object) -> ApproachArtifacts:
    defaults: dict[str, object] = dict(
        approach="B",
        label="test",
        episodes=2,
        claims=[],
        wall_clock_seconds=30.0,
        transcript_segments={
            "ep-1": [_seg("SPEAKER_00", 0.0, 5.0), _seg("SPEAKER_01", 5.0, 10.0)],
            "ep-2": [_seg("SPEAKER_00", 0.0, 5.0)],
        },
    )
    defaults.update(overrides)
    return ApproachArtifacts(**defaults)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Throughput
# --------------------------------------------------------------------------- #
def test_throughput_episodes_per_min() -> None:
    art = _artifacts(episodes=4, wall_clock_seconds=120.0)
    score = score_approach(art)
    # 4 episodes in 120s == 2 episodes/min.
    assert score.throughput_episodes_per_min == 2.0


def test_throughput_zero_wall_clock_is_zero_not_error() -> None:
    art = _artifacts(episodes=3, wall_clock_seconds=0.0)
    score = score_approach(art)
    assert score.throughput_episodes_per_min == 0.0


# --------------------------------------------------------------------------- #
# Cost per episode
# --------------------------------------------------------------------------- #
def test_cost_per_episode_from_tokens_and_dedup() -> None:
    pricing = Pricing(
        extraction_prompt_usd_per_token=1e-6,
        extraction_completion_usd_per_token=2e-6,
        dedup_call_usd=0.01,
    )
    art = _artifacts(
        episodes=2,
        prompt_tokens=1_000_000,  # $1.00
        completion_tokens=500_000,  # $1.00
        llm_dedup_calls=10,  # $0.10
        pricing=pricing,
    )
    score = score_approach(art)
    # ($1.00 + $1.00 + $0.10) / 2 episodes = $1.05/episode.
    assert abs(score.cost_per_episode_usd - 1.05) < 1e-9


def test_bypassed_dedup_costs_nothing() -> None:
    pricing = Pricing(dedup_call_usd=0.05)
    art = _artifacts(episodes=2, llm_dedup_calls=0, pricing=pricing)
    score = score_approach(art)
    assert score.cost_per_episode_usd == 0.0
    assert score.bypassed_llm_dedup is True


def test_non_bypassed_dedup_flags_not_bypassed() -> None:
    art = _artifacts(llm_dedup_calls=5)
    score = score_approach(art)
    assert score.bypassed_llm_dedup is False
    assert score.llm_dedup_calls == 5


# --------------------------------------------------------------------------- #
# Claims/episode
# --------------------------------------------------------------------------- #
def test_claims_per_episode() -> None:
    claims = [_claim() for _ in range(6)]
    art = _artifacts(episodes=2, claims=claims)
    score = score_approach(art)
    assert score.total_claims == 6
    assert score.claims_per_episode == 3.0


# --------------------------------------------------------------------------- #
# Valid source-span
# --------------------------------------------------------------------------- #
def test_valid_source_span_rate_counts_in_window_spans() -> None:
    claims = [
        # Inside SPEAKER_00's segment [0,5].
        _claim(t_start=0.5, t_end=4.5, speaker_label="SPEAKER_00"),
        # Fabricated: span [40,50] is outside any segment in ep-1.
        _claim(t_start=40.0, t_end=50.0, speaker_label="SPEAKER_00"),
    ]
    art = _artifacts(episodes=1, claims=claims)
    score = score_approach(art)
    assert score.valid_source_span_rate == 0.5


def test_span_eps_tolerance_accepts_boundary() -> None:
    # Span exactly on the segment boundary (0..5) should validate.
    claims = [_claim(t_start=0.0, t_end=5.0, speaker_label="SPEAKER_00")]
    art = _artifacts(episodes=1, claims=claims)
    score = score_approach(art)
    assert score.valid_source_span_rate == 1.0


# --------------------------------------------------------------------------- #
# Valid speaker-attribution (the misattribution check, spec §11)
# --------------------------------------------------------------------------- #
def test_speaker_attribution_rejects_misattribution() -> None:
    claims = [
        # Correct: SPEAKER_00 speaks at [0,5].
        _claim(t_start=1.0, t_end=2.0, speaker_label="SPEAKER_00"),
        # Wrong speaker at a real timestamp: SPEAKER_01 owns [5,10], not [1,2].
        _claim(t_start=1.0, t_end=2.0, speaker_label="SPEAKER_01"),
    ]
    art = _artifacts(episodes=1, claims=claims)
    score = score_approach(art)
    # Both spans are valid (in-window), but only one attribution is correct.
    assert score.valid_source_span_rate == 1.0
    assert score.valid_speaker_attribution_rate == 0.5


def test_attribution_ignores_claims_with_no_matching_segment() -> None:
    # An out-of-window span has no ground-truth segment, so it cannot count as
    # a correct attribution.
    claims = [_claim(t_start=100.0, t_end=110.0, speaker_label="SPEAKER_00")]
    art = _artifacts(episodes=1, claims=claims)
    score = score_approach(art)
    assert score.valid_speaker_attribution_rate == 0.0


# --------------------------------------------------------------------------- #
# JSON-parse success rate (Approach B only)
# --------------------------------------------------------------------------- #
def test_json_parse_rate_for_approach_b() -> None:
    art = _artifacts(approach="B", parse_attempts=10, parse_successes=9)
    score = score_approach(art)
    assert score.json_parse_success_rate == 0.9


def test_json_parse_rate_none_when_no_attempts() -> None:
    art = _artifacts(approach="A", parse_attempts=0, parse_successes=0)
    score = score_approach(art)
    assert score.json_parse_success_rate is None


# --------------------------------------------------------------------------- #
# Empty inputs never divide by zero
# --------------------------------------------------------------------------- #
def test_zero_claims_rates_are_zero() -> None:
    art = _artifacts(episodes=0, claims=[], wall_clock_seconds=0.0)
    score = score_approach(art)
    assert score.claims_per_episode == 0.0
    assert score.valid_source_span_rate == 0.0
    assert score.valid_speaker_attribution_rate == 0.0
    assert score.cost_per_episode_usd == 0.0


# --------------------------------------------------------------------------- #
# Recommendation (the throughput/$ gate, spec §7.6)
# --------------------------------------------------------------------------- #
def _comparison(*, a: ApproachArtifacts, b: ApproachArtifacts) -> ComparisonResult:
    return ComparisonResult(
        episodes=max(a.episodes, b.episodes), approach_a=a, approach_b=b
    )


def test_recommend_b_when_b_cheaper_at_equal_quality() -> None:
    pricing = Pricing(dedup_call_usd=0.10)
    good = [_claim(t_start=1.0, t_end=2.0, speaker_label="SPEAKER_00")]
    # A pays per-add dedup; B bypasses it. Equal claim quality.
    a = _artifacts(
        approach="A",
        episodes=2,
        claims=list(good),
        wall_clock_seconds=60.0,
        llm_dedup_calls=20,
        pricing=pricing,
    )
    b = _artifacts(
        approach="B",
        episodes=2,
        claims=list(good),
        wall_clock_seconds=60.0,
        llm_dedup_calls=0,
        pricing=pricing,
    )
    report = score_comparison(_comparison(a=a, b=b))
    assert report.recommended == "B"
    assert "§7.6" in report.rationale


def test_recommend_ties_break_toward_b() -> None:
    # Identical artifacts: the spec's working assumption is B.
    a = _artifacts(approach="A", claims=[_claim()], wall_clock_seconds=10.0)
    b = _artifacts(approach="B", claims=[_claim()], wall_clock_seconds=10.0)
    report = score_comparison(_comparison(a=a, b=b))
    assert report.recommended == "B"


def test_recommend_a_when_a_strictly_better() -> None:
    # B is both far more expensive AND far slower at equal quality -> A wins.
    pricing = Pricing(dedup_call_usd=1.0)
    good = [_claim(t_start=1.0, t_end=2.0, speaker_label="SPEAKER_00")]
    a = _artifacts(
        approach="A",
        episodes=4,
        claims=list(good),
        wall_clock_seconds=10.0,  # fast
        llm_dedup_calls=0,  # cheap
        pricing=pricing,
    )
    b = _artifacts(
        approach="B",
        episodes=4,
        claims=list(good),
        wall_clock_seconds=400.0,  # slow
        llm_dedup_calls=100,  # expensive
        pricing=pricing,
    )
    report = score_comparison(_comparison(a=a, b=b))
    assert report.recommended == "A"


# --------------------------------------------------------------------------- #
# Report rendering + emission
# --------------------------------------------------------------------------- #
def _sample_report() -> SpikeReport:
    a = _artifacts(approach="A", episodes=2, claims=[_claim()], wall_clock_seconds=20.0)
    b = _artifacts(approach="B", episodes=2, claims=[_claim()], wall_clock_seconds=10.0)
    return score_comparison(_comparison(a=a, b=b))


def test_render_report_md_contains_axes() -> None:
    md = render_report_md(_sample_report())
    assert "Throughput (episodes/min)" in md
    assert "Cost ($/episode)" in md
    assert "Valid speaker-attribution" in md
    assert "Recommendation: Approach" in md


def test_emit_report_writes_both_files(tmp_path) -> None:
    report = _sample_report()
    paths = emit_report(report, tmp_path)
    assert paths["md"].exists()
    assert paths["json"].exists()

    # JSON round-trips to the same model.
    loaded = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert loaded["recommended"] == report.recommended
    assert loaded["approach_a"]["approach"] == "A"
    assert loaded["approach_b"]["approach"] == "B"

    md = paths["md"].read_text(encoding="utf-8")
    assert md.startswith("# Graphiti")


def test_report_json_is_deterministic() -> None:
    report = _sample_report()
    assert report.to_json() == report.to_json()
