"""Spike scoring (spec §7.6): the throughput / $ gate, not only quality.

Turns the per-approach :class:`~dlogos.spike.run_comparison.ApproachArtifacts`
captured by the runner into the metrics the spec's decision rule needs, then
emits ``report.md`` + ``report.json``. Every metric here is a **pure function
of the artifacts** — no clock, no network, no randomness — so the scorers are
trivially deterministic to unit-test on synthetic inputs.

The metrics (spec §7.6, axes 2-3 plus the supporting quality counts):

- **throughput** — episodes/min through the approach's load path (the
  ``wall_clock_seconds`` is captured by the runner and injected, never read
  from a real clock here).
- **$/episode** — all-in (extraction + load) estimated cost per episode, from
  token counts × injected unit prices plus any per-episode Graphiti per-add
  LLM-dedup calls the approach did *not* bypass.
- **claims/episode** — extraction yield.
- **%valid-source-span** — fraction of claims whose ``source_span`` falls
  inside its episode's true time window AND inside one real diarized segment
  (a fabricated citation fails this; the eval's speaker-verified check depends
  on spans being real — spec §6/§9).
- **%valid-speaker-attribution** — fraction of claims whose attributed speaker
  matches the speaker actually talking at the claim's ``source_span`` in the
  diarized transcript. Diarization → confident MISATTRIBUTION is the top risk
  (spec §11), so this is scored against ground-truth, not topic presence.
- **json_parse_success_rate** — Approach B only: parses ok / parse attempts.
  Open-weight structured output wobbles; this is the reliability number that
  justifies (or not) owning extraction (Approach A has no separate parse stage
  we observe, so it is ``None`` there).

A small tolerance absorbs floating-point timestamp rounding when checking a
span against a segment window.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from dlogos.spike.run_comparison import ApproachArtifacts, ComparisonResult

# Tolerance (seconds) when testing whether a claim span sits inside a segment.
_SPAN_EPS = 0.05


# --------------------------------------------------------------------------- #
# Result models
# --------------------------------------------------------------------------- #
class ApproachScore(BaseModel):
    """The scored metrics for a single approach (A or B)."""

    model_config = ConfigDict(extra="forbid")

    approach: str = Field(description="'A' or 'B'.")
    label: str = Field(description="Human-readable approach description.")
    episodes: int = Field(ge=0)
    total_claims: int = Field(ge=0)
    wall_clock_seconds: float = Field(ge=0.0)

    throughput_episodes_per_min: float = Field(ge=0.0)
    cost_per_episode_usd: float = Field(ge=0.0)
    claims_per_episode: float = Field(ge=0.0)
    valid_source_span_rate: float = Field(ge=0.0, le=1.0)
    valid_speaker_attribution_rate: float = Field(ge=0.0, le=1.0)
    # None for Approach A (no separate JSON parse stage we observe).
    json_parse_success_rate: float | None = Field(default=None)

    # Diagnostics the spike's dedup-bypass gate reads.
    llm_dedup_calls: int = Field(
        default=0,
        ge=0,
        description="Per-add LLM node-dedup calls that were NOT bypassed.",
    )
    bypassed_llm_dedup: bool = Field(
        default=True,
        description="Did the bulk load bypass Graphiti per-add LLM dedup?",
    )


class SpikeReport(BaseModel):
    """The full spike report: both approaches scored + the recommendation."""

    model_config = ConfigDict(extra="forbid")

    episodes: int = Field(ge=0)
    approach_a: ApproachScore
    approach_b: ApproachScore
    recommended: str = Field(description="'A' or 'B' — the spike's pick.")
    rationale: str

    def to_json(self) -> str:
        """Stable, pretty JSON (sorted keys) for the report.json artifact."""

        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True)


# --------------------------------------------------------------------------- #
# Per-claim validity checks (pure)
# --------------------------------------------------------------------------- #
def _span_in_window(t_start: float, t_end: float, lo: float, hi: float) -> bool:
    return (t_end >= t_start) and (t_start >= lo - _SPAN_EPS) and (
        t_end <= hi + _SPAN_EPS
    )


def _segment_for_span(
    artifacts: ApproachArtifacts, episode_id: str, t_start: float, t_end: float
) -> tuple[str, float, float] | None:
    """Return the ``(speaker, t_start, t_end)`` of the diarized segment whose
    window contains ``[t_start, t_end]``, or ``None`` if no segment does.

    The ground-truth diarization lives on the runner's captured transcripts so
    scoring can verify both span-realness and speaker-attribution against the
    actual audio timeline — not against mere topic presence.
    """

    segments = artifacts.transcript_segments.get(episode_id, [])
    for seg in segments:
        if _span_in_window(t_start, t_end, seg.t_start, seg.t_end):
            return (seg.speaker, seg.t_start, seg.t_end)
    return None


def _valid_span_count(artifacts: ApproachArtifacts) -> int:
    n = 0
    for claim in artifacts.claims:
        span = claim.source_span
        seg = _segment_for_span(
            artifacts, span.episode_id, span.t_start, span.t_end
        )
        if seg is not None:
            n += 1
    return n


def _valid_attribution_count(artifacts: ApproachArtifacts) -> int:
    """Count claims whose attributed speaker matches the speaker truly talking.

    The attributed *label* (``source_span``-anchored diarization label) is
    compared to the diarization ground-truth segment's speaker. This is the
    misattribution check (spec §11): a claim sourced to the wrong speaker at the
    right timestamp fails here even though its topic is present.
    """

    n = 0
    for claim in artifacts.claims:
        span = claim.source_span
        seg = _segment_for_span(
            artifacts, span.episode_id, span.t_start, span.t_end
        )
        if seg is None:
            continue
        true_speaker = seg[0]
        if claim.speaker.label == true_speaker:
            n += 1
    return n


# --------------------------------------------------------------------------- #
# Cost model (pure)
# --------------------------------------------------------------------------- #
def _approach_cost_usd(artifacts: ApproachArtifacts) -> float:
    """All-in extraction + load cost from token counts × injected unit prices.

    - extraction: ``(prompt_tokens, completion_tokens)`` × per-token prices.
    - per-add LLM node-dedup: a flat per-call price × the number of dedup calls
      the approach did NOT bypass (the hidden $/episode line item the spec
      flags — §7.5/§7.6). A bulk load that bypasses dedup contributes zero here.
    """

    p = artifacts.pricing
    extraction = (
        artifacts.prompt_tokens * p.extraction_prompt_usd_per_token
        + artifacts.completion_tokens * p.extraction_completion_usd_per_token
    )
    dedup = artifacts.llm_dedup_calls * p.dedup_call_usd
    return extraction + dedup


# --------------------------------------------------------------------------- #
# Scorers
# --------------------------------------------------------------------------- #
def _safe_div(num: float, den: float) -> float:
    return (num / den) if den else 0.0


def score_approach(artifacts: ApproachArtifacts) -> ApproachScore:
    """Score one approach's captured artifacts into :class:`ApproachScore`.

    Pure: every output is a function of ``artifacts``. The runner is the only
    place a wall clock is read; here it is just an input number.
    """

    episodes = artifacts.episodes
    total_claims = len(artifacts.claims)
    valid_spans = _valid_span_count(artifacts)
    valid_attr = _valid_attribution_count(artifacts)

    throughput = (
        _safe_div(episodes, artifacts.wall_clock_seconds) * 60.0
        if artifacts.wall_clock_seconds > 0
        else 0.0
    )
    cost = _approach_cost_usd(artifacts)

    json_rate: float | None
    if artifacts.parse_attempts > 0:
        json_rate = _safe_div(artifacts.parse_successes, artifacts.parse_attempts)
    else:
        json_rate = None

    return ApproachScore(
        approach=artifacts.approach,
        label=artifacts.label,
        episodes=episodes,
        total_claims=total_claims,
        wall_clock_seconds=artifacts.wall_clock_seconds,
        throughput_episodes_per_min=throughput,
        cost_per_episode_usd=_safe_div(cost, episodes),
        claims_per_episode=_safe_div(total_claims, episodes),
        valid_source_span_rate=_safe_div(valid_spans, total_claims),
        valid_speaker_attribution_rate=_safe_div(valid_attr, total_claims),
        json_parse_success_rate=json_rate,
        llm_dedup_calls=artifacts.llm_dedup_calls,
        bypassed_llm_dedup=artifacts.llm_dedup_calls == 0,
    )


def _recommend(a: ApproachScore, b: ApproachScore) -> tuple[str, str]:
    """Pick the winner per the spec's decision rule (§7.6).

    The gate is throughput / $ AND quality — a shape that produces good claims
    but is too slow or too expensive does not pass. We score each approach on
    three normalized axes (cost: lower better; throughput: higher better;
    quality: attribution × span-validity, higher better) and pick the higher
    aggregate, breaking ties toward B (the spec's working assumption — keeps
    open-weight extraction first-class and swappable, trivially bypasses dedup).
    """

    def quality(s: ApproachScore) -> float:
        return s.valid_speaker_attribution_rate * s.valid_source_span_rate

    # Normalize cost to a [0,1] "cheapness" score (lower cost -> higher score).
    max_cost = max(a.cost_per_episode_usd, b.cost_per_episode_usd, 1e-9)
    a_cheap = 1.0 - _safe_div(a.cost_per_episode_usd, max_cost)
    b_cheap = 1.0 - _safe_div(b.cost_per_episode_usd, max_cost)

    max_thru = max(
        a.throughput_episodes_per_min, b.throughput_episodes_per_min, 1e-9
    )
    a_thru = _safe_div(a.throughput_episodes_per_min, max_thru)
    b_thru = _safe_div(b.throughput_episodes_per_min, max_thru)

    a_score = quality(a) + a_cheap + a_thru
    b_score = quality(b) + b_cheap + b_thru

    if a_score > b_score:
        winner = "A"
    else:
        winner = "B"  # ties -> B per the spec's working assumption

    rationale = (
        f"Approach {winner} wins on the combined throughput/$/quality gate "
        f"(§7.6): "
        f"A[quality={quality(a):.2f}, ${a.cost_per_episode_usd:.4f}/ep, "
        f"{a.throughput_episodes_per_min:.1f} ep/min], "
        f"B[quality={quality(b):.2f}, ${b.cost_per_episode_usd:.4f}/ep, "
        f"{b.throughput_episodes_per_min:.1f} ep/min]. "
    )
    if winner == "B":
        rationale += (
            "B keeps open-weight extraction first-class/swappable and "
            "trivially bypasses Graphiti's per-add LLM node-dedup."
        )
    else:
        rationale += (
            "A clears the gate despite owning less of the pipeline; recheck "
            "the dedup-bypass config before scaling."
        )
    return winner, rationale


def score_comparison(comparison: ComparisonResult) -> SpikeReport:
    """Score both approaches and produce the full :class:`SpikeReport`."""

    a = score_approach(comparison.approach_a)
    b = score_approach(comparison.approach_b)
    winner, rationale = _recommend(a, b)
    return SpikeReport(
        episodes=comparison.episodes,
        approach_a=a,
        approach_b=b,
        recommended=winner,
        rationale=rationale,
    )


# --------------------------------------------------------------------------- #
# Report rendering + emission
# --------------------------------------------------------------------------- #
def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _fmt_json_rate(x: float | None) -> str:
    return "n/a" if x is None else _fmt_pct(x)


def render_report_md(report: SpikeReport) -> str:
    """Render the human-readable ``report.md`` for the spike artifact.

    A compact side-by-side table over the spec's three axes plus the supporting
    quality counts, then the recommendation. Deterministic string output so the
    emitter is testable without touching disk.
    """

    a = report.approach_a
    b = report.approach_b
    lines = [
        "# Graphiti × open-weight extraction spike — report",
        "",
        f"Episodes compared: **{report.episodes}**",
        "",
        "Decision rule (spec §7.6): pick a shape that clears the "
        "**throughput / $ gate**, not only a quality bar.",
        "",
        "| Metric | A: " + a.label + " | B: " + b.label + " |",
        "| --- | --- | --- |",
        f"| Throughput (episodes/min) | {a.throughput_episodes_per_min:.2f} "
        f"| {b.throughput_episodes_per_min:.2f} |",
        f"| Cost ($/episode) | ${a.cost_per_episode_usd:.4f} "
        f"| ${b.cost_per_episode_usd:.4f} |",
        f"| Claims/episode | {a.claims_per_episode:.2f} "
        f"| {b.claims_per_episode:.2f} |",
        f"| Valid source-span | {_fmt_pct(a.valid_source_span_rate)} "
        f"| {_fmt_pct(b.valid_source_span_rate)} |",
        f"| Valid speaker-attribution | "
        f"{_fmt_pct(a.valid_speaker_attribution_rate)} "
        f"| {_fmt_pct(b.valid_speaker_attribution_rate)} |",
        f"| JSON-parse success (B) | {_fmt_json_rate(a.json_parse_success_rate)} "
        f"| {_fmt_json_rate(b.json_parse_success_rate)} |",
        f"| Per-add LLM dedup calls | {a.llm_dedup_calls} | {b.llm_dedup_calls} |",
        f"| Bypassed per-add LLM dedup | {a.bypassed_llm_dedup} "
        f"| {b.bypassed_llm_dedup} |",
        "",
        f"## Recommendation: Approach {report.recommended}",
        "",
        report.rationale,
        "",
    ]
    return "\n".join(lines)


def emit_report(report: SpikeReport, out_dir: str | Path) -> dict[str, Path]:
    """Write ``report.md`` + ``report.json`` into ``out_dir``; return the paths.

    Creates ``out_dir`` if needed. The only side-effecting function in this
    module; scoring stays pure so tests can assert on the models directly and
    only the emission test touches a tmp path.
    """

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md_path = out / "report.md"
    json_path = out / "report.json"
    md_path.write_text(render_report_md(report), encoding="utf-8")
    json_path.write_text(report.to_json(), encoding="utf-8")
    return {"md": md_path, "json": json_path}
