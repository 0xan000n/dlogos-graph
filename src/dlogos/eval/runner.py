"""The four-arm head-to-head eval runner (spec §9) — the greenlight artifact.

Ties the eval subpackage together into one orchestration:

    golden queries
        -> run all FOUR arms per query (model-alone / +web-search /
           +naive-vector-RAG / +dLogos)            [dlogos.eval.arms]
        -> blind the four answers per query          [dlogos.eval.blind]
        -> a blinded rater scores each on the rubric  [dlogos.eval.rubric]
           (with the speaker-verified citation check capping attribution)
        -> unblind the per-label scores back to arms  [dlogos.eval.blind]
        -> optional second rater on a subset -> agreement [dlogos.eval.agreement]
        -> a side-by-side artifact (JSON + Markdown)

Everything that could be non-deterministic is **injected**: the four arms (each
an async callable, fakes in tests), and the *rater* — a callable that, given a
blinded answer + its query, returns the rater's raw [0, 1] judgment per rubric
:class:`~dlogos.eval.rubric.Dimension`. The rater sees only the *blinded* answer
(arm identity stripped), which is the blind-scoring credibility control (§9).

Nothing heavy is imported at module top level; the runner uses ``asyncio`` from
the stdlib and the eval/retrieval modules only.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from dlogos.eval.agreement import agreement_report, bin_scores, cohen_kappa
from dlogos.eval.arms import ALL_ARMS, Answer
from dlogos.eval.blind import BlindedAnswer, blind_answers, unblind_scores
from dlogos.eval.golden import GoldenQuery
from dlogos.eval.rubric import (
    DEFAULT_WEIGHTS,
    Dimension,
    RubricResult,
    score_answer,
)
from dlogos.schema import Transcript


# --------------------------------------------------------------------------- #
# Injected collaborators
# --------------------------------------------------------------------------- #
@runtime_checkable
class Arm(Protocol):
    """An eval arm: an async callable mapping a query to an :class:`Answer`.

    Matches the arms in :mod:`dlogos.eval.arms` (each exposes ``name`` and is
    awaitable). Kept structural so fakes satisfy it in tests.
    """

    name: str

    async def __call__(self, query: GoldenQuery) -> Answer: ...


@runtime_checkable
class Rater(Protocol):
    """A blinded rater: scores one blinded answer on the rubric dimensions.

    Given the :class:`~dlogos.eval.golden.GoldenQuery` (so the rater can compare
    against the pre-registered answer shape) and the *blinded* answer (arm
    identity stripped), return the rater's raw judgment in [0, 1] per
    :class:`~dlogos.eval.rubric.Dimension`. A human rater or an LLM-judge would
    implement this; tests inject a deterministic fake. The rater never sees the
    arm name, preserving blind scoring (§9).
    """

    def __call__(
        self, query: GoldenQuery, blinded: BlindedAnswer
    ) -> dict[Dimension, float]: ...


# --------------------------------------------------------------------------- #
# Per-query / aggregate result records (serializable)
# --------------------------------------------------------------------------- #
class ArmScore(BaseModel):
    """One arm's scored outcome for one query (after unblinding)."""

    model_config = ConfigDict(extra="forbid")

    arm: str
    label: str  # the blinded label this arm was scored under
    total: float
    raw: dict[str, float]
    verified_citations: int
    rejected_citations: int


class QueryResult(BaseModel):
    """The full scored outcome for one golden query across the four arms."""

    model_config = ConfigDict(extra="forbid")

    query_id: str
    archetype: str
    domain: str
    query_text: str
    seed: int
    answers: dict[str, str] = Field(
        description="arm -> answer text (verbatim, for the artifact)."
    )
    citation_counts: dict[str, int] = Field(
        description="arm -> number of citations the arm returned."
    )
    scores: list[ArmScore]
    label_to_arm: dict[str, str]

    def total_for(self, arm: str) -> float:
        for s in self.scores:
            if s.arm == arm:
                return s.total
        raise KeyError(arm)


class AgreementResult(BaseModel):
    """Inter-rater agreement over the second-rater subset (spec §9 control)."""

    model_config = ConfigDict(extra="forbid")

    n: int
    percent_agreement: float
    cohen_kappa: float
    bands: int = Field(description="Number of ordinal bands totals were binned into.")


class EvalReport(BaseModel):
    """The committed four-arm side-by-side artifact (spec §9 output)."""

    model_config = ConfigDict(extra="forbid")

    per_query: list[QueryResult]
    mean_total_by_arm: dict[str, float]
    win_count_by_arm: dict[str, int] = Field(
        description="How many queries each arm scored strictly highest on."
    )
    coverage: dict[str, list[str]] = Field(
        description="Domains and archetypes the query set touched (the spread)."
    )
    agreement: AgreementResult | None = None


# --------------------------------------------------------------------------- #
# Speaker-verified citation context (for the rubric's attribution cap)
# --------------------------------------------------------------------------- #
@dataclass
class CitationContext:
    """The diarized ground-truth the speaker-verified check reads (spec §9).

    ``transcripts`` and ``segment_speaker_ids`` are keyed by ``episode_id``; the
    latter maps each transcript segment index to its resolved speaker id. When
    empty, the citation check is skipped and attribution is taken at the rater's
    word (still useful for the model-alone / web-search arms, which carry no
    podcast-span citations).
    """

    transcripts: dict[str, Transcript] = field(default_factory=dict)
    segment_speaker_ids: dict[str, dict[int, str]] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #
class EvalRunner:
    """Orchestrates the four-arm head-to-head over the golden set (spec §9).

    Parameters
    ----------
    arms:
        The four arms to run per query (each an async :class:`Arm`). Order does
        not matter — answers are keyed/blinded per query.
    rater:
        The primary blinded :class:`Rater`.
    second_rater:
        Optional second independent rater for the agreement control. When given,
        a subset of queries is scored by both and agreement is reported.
    citation_context:
        Diarized ground-truth for the speaker-verified citation check; when
        present the attribution-precision raw score is capped by the verified
        fraction (spec §9 / §11 top risk).
    weights:
        Rubric weights (defaults to :data:`~dlogos.eval.rubric.DEFAULT_WEIGHTS`).
    seed:
        Base seed for the deterministic per-query blinding.
    """

    def __init__(
        self,
        arms: list[Arm],
        rater: Rater,
        *,
        second_rater: Rater | None = None,
        citation_context: CitationContext | None = None,
        weights: dict[Dimension, float] | None = None,
        seed: int = 1234,
    ) -> None:
        self._arms = list(arms)
        self._rater = rater
        self._second_rater = second_rater
        self._cite = citation_context or CitationContext()
        self._weights = weights or DEFAULT_WEIGHTS
        self._seed = seed

    # -- per-query ---------------------------------------------------------- #
    async def _run_arms(self, query: GoldenQuery) -> list[Answer]:
        """Run every arm on one query concurrently, preserving arm order."""

        results = await asyncio.gather(*(arm(query) for arm in self._arms))
        return list(results)

    def _score_blinded(
        self, query: GoldenQuery, blinded: BlindedAnswer, rater: Rater
    ) -> RubricResult:
        """Score one blinded answer with a rater + the rubric (cite-check on)."""

        dim_scores = rater(query, blinded)
        return score_answer(
            blinded.answer,
            dim_scores,
            transcripts=self._cite.transcripts or None,
            segment_speaker_ids=self._cite.segment_speaker_ids or None,
            weights=self._weights,
        )

    async def run_query(self, query: GoldenQuery) -> QueryResult:
        """Run + blind + score one golden query across all four arms."""

        answers = await self._run_arms(query)
        blinded_query = blind_answers(query.id, answers, seed=self._seed)

        # Score each blinded answer; collect per-label rubric totals.
        per_label_total: dict[str, float] = {}
        per_label_result: dict[str, RubricResult] = {}
        for ba in blinded_query.blinded:
            result = self._score_blinded(query, ba, self._rater)
            per_label_total[ba.label] = result.total
            per_label_result[ba.label] = result

        # Unblind: map blinded labels back to true arms.
        per_arm_total = unblind_scores(blinded_query.unblind, per_label_total)

        # Build the per-arm scored rows.
        arm_scores: list[ArmScore] = []
        answers_by_arm = {a.arm: a for a in answers}
        for label, arm in blinded_query.unblind.label_to_arm.items():
            result = per_label_result[label]
            arm_scores.append(
                ArmScore(
                    arm=arm,
                    label=label,
                    total=result.total,
                    raw={d.value: result.raw[d] for d in Dimension},
                    verified_citations=result.verified_citations,
                    rejected_citations=result.rejected_citations,
                )
            )
        arm_scores.sort(key=lambda s: s.arm)

        return QueryResult(
            query_id=query.id,
            archetype=query.archetype.value,
            domain=query.domain.value,
            query_text=query.query_text,
            seed=blinded_query.seed,
            answers={a.arm: a.text for a in answers},
            citation_counts={a.arm: len(a.citations) for a in answers},
            scores=arm_scores,
            label_to_arm=dict(blinded_query.unblind.label_to_arm),
        )

    # -- whole run ---------------------------------------------------------- #
    async def run(
        self,
        queries: list[GoldenQuery],
        *,
        agreement_subset: int = 0,
    ) -> EvalReport:
        """Run the whole golden set and produce the side-by-side report.

        ``agreement_subset`` (when > 0 and a ``second_rater`` is injected) is the
        number of leading queries also scored by the second rater; inter-rater
        agreement (percent + Cohen's kappa over ordinal-binned totals) is
        reported over those shared items.
        """

        per_query = [await self.run_query(q) for q in queries]

        mean_total_by_arm = self._mean_totals(per_query)
        win_count_by_arm = self._win_counts(per_query)
        agreement = await self._compute_agreement(queries, agreement_subset)

        domains = sorted({q.domain.value for q in queries})
        archetypes = sorted({q.archetype.value for q in queries})

        return EvalReport(
            per_query=per_query,
            mean_total_by_arm=mean_total_by_arm,
            win_count_by_arm=win_count_by_arm,
            coverage={"domains": domains, "archetypes": archetypes},
            agreement=agreement,
        )

    # -- aggregation helpers ----------------------------------------------- #
    @staticmethod
    def _mean_totals(per_query: list[QueryResult]) -> dict[str, float]:
        sums: dict[str, float] = {arm: 0.0 for arm in ALL_ARMS}
        counts: dict[str, int] = {arm: 0 for arm in ALL_ARMS}
        for qr in per_query:
            for s in qr.scores:
                sums.setdefault(s.arm, 0.0)
                counts.setdefault(s.arm, 0)
                sums[s.arm] += s.total
                counts[s.arm] += 1
        return {
            arm: (sums[arm] / counts[arm] if counts[arm] else 0.0)
            for arm in sums
        }

    @staticmethod
    def _win_counts(per_query: list[QueryResult]) -> dict[str, int]:
        wins: dict[str, int] = {arm: 0 for arm in ALL_ARMS}
        for qr in per_query:
            if not qr.scores:
                continue
            best = max(qr.scores, key=lambda s: s.total)
            # Strict win only — a tie credits no one (conservative).
            top = [s for s in qr.scores if s.total == best.total]
            if len(top) == 1:
                wins.setdefault(top[0].arm, 0)
                wins[top[0].arm] += 1
        return wins

    async def _compute_agreement(
        self, queries: list[GoldenQuery], subset: int
    ) -> AgreementResult | None:
        """Score the leading ``subset`` queries with both raters; report kappa."""

        if subset <= 0 or self._second_rater is None:
            return None

        shared = queries[: min(subset, len(queries))]
        totals_a: list[float] = []
        totals_b: list[float] = []
        for query in shared:
            answers = await self._run_arms(query)
            blinded_query = blind_answers(query.id, answers, seed=self._seed)
            for ba in blinded_query.blinded:
                totals_a.append(self._score_blinded(query, ba, self._rater).total)
                totals_b.append(
                    self._score_blinded(query, ba, self._second_rater).total
                )

        if not totals_a:
            return None

        # Bin continuous totals into ordinal bands for kappa (spec §9 / agreement).
        edges = [0.33, 0.66]
        bands_a = bin_scores(totals_a, edges)
        bands_b = bin_scores(totals_b, edges)
        report = agreement_report(bands_a, bands_b)
        return AgreementResult(
            n=int(report["n"]),
            percent_agreement=report["percent_agreement"],
            cohen_kappa=report["cohen_kappa"],
            bands=len(edges) + 1,
        )


# --------------------------------------------------------------------------- #
# Artifact rendering (JSON + Markdown)
# --------------------------------------------------------------------------- #
def report_to_json(report: EvalReport, *, indent: int = 2) -> str:
    """Serialize the report to stable JSON (the committed machine artifact)."""

    return json.dumps(report.model_dump(mode="json"), indent=indent, sort_keys=False)


_ARM_TITLES = {
    "model_alone": "Model alone",
    "model_web_search": "Model + web search",
    "model_naive_rag": "Model + naive vector-RAG",
    "model_dlogos": "Model + dLogos",
}


def report_to_markdown(report: EvalReport) -> str:
    """Render the human-readable side-by-side scorecard (spec §9 artifact).

    A summary table (mean total + wins per arm), then one section per query with
    the four arms' verbatim answers and their unblinded rubric totals.
    """

    lines: list[str] = ["# dLogos four-arm head-to-head\n"]

    # Spread / framing line (existence proof, not generalization proof — §9).
    domains = ", ".join(report.coverage.get("domains", []))
    archetypes = ", ".join(report.coverage.get("archetypes", []))
    lines.append(
        f"_Existence proof across a spread — domains: {domains}; "
        f"archetypes: {archetypes}._\n"
    )

    # Summary table.
    lines.append("## Summary (blinded, then unblinded)\n")
    lines.append("| Arm | Mean total | Wins |")
    lines.append("|---|---|---|")
    for arm in ALL_ARMS:
        title = _ARM_TITLES.get(arm, arm)
        mean = report.mean_total_by_arm.get(arm, 0.0)
        wins = report.win_count_by_arm.get(arm, 0)
        lines.append(f"| {title} | {mean:.3f} | {wins} |")
    lines.append("")

    if report.agreement is not None:
        a = report.agreement
        lines.append(
            f"**Inter-rater agreement** (n={a.n}, {a.bands} bands): "
            f"percent={a.percent_agreement:.3f}, Cohen's κ={a.cohen_kappa:.3f}\n"
        )

    # Per-query side-by-side.
    for qr in report.per_query:
        lines.append(f"## {qr.query_id} · {qr.domain} · {qr.archetype}\n")
        lines.append(f"> {qr.query_text}\n")
        score_by_arm = {s.arm: s for s in qr.scores}
        lines.append("| Arm | Total | Verified cites | Rejected cites |")
        lines.append("|---|---|---|---|")
        for arm in ALL_ARMS:
            title = _ARM_TITLES.get(arm, arm)
            s = score_by_arm.get(arm)
            if s is None:
                lines.append(f"| {title} | — | — | — |")
            else:
                lines.append(
                    f"| {title} | {s.total:.3f} | {s.verified_citations} | "
                    f"{s.rejected_citations} |"
                )
        lines.append("")
        for arm in ALL_ARMS:
            title = _ARM_TITLES.get(arm, arm)
            text = qr.answers.get(arm, "_(no answer)_").strip() or "_(empty)_"
            lines.append(f"### {title}\n")
            lines.append(text + "\n")

    return "\n".join(lines)
