"""Tests for the four-arm eval runner (spec §9).

Drives :class:`~dlogos.eval.runner.EvalRunner` with fake arms + a fake blinded
rater (no network, deterministic). Asserts: all four arms scored, blinding +
unblinding round-trips, the speaker-verified citation cap bites on a
misattributed citation, aggregate wins/means, agreement, and the JSON/Markdown
artifact render.
"""

from __future__ import annotations

import json

from dlogos.eval.arms import (
    ALL_ARMS,
    ARM_DLOGOS,
    ARM_MODEL_ALONE,
    Answer,
    Citation,
)
from dlogos.eval.golden import (
    AnswerShape,
    Archetype,
    Domain,
    GoldenQuery,
    starter_golden_queries,
)
from dlogos.eval.rubric import Dimension
from dlogos.eval.runner import (
    CitationContext,
    EvalRunner,
    report_to_json,
    report_to_markdown,
)
from dlogos.schema import Transcript, TranscriptSegment


def _make_arm(name: str, *, citations=None):
    class _Arm:
        def __init__(self) -> None:
            self.name = name

        async def __call__(self, query: GoldenQuery) -> Answer:
            return Answer(
                arm=name,
                text=f"{name} answers {query.id}",
                citations=list(citations or []),
            )

    return _Arm()


def _arms(*, dlogos_citations=None):
    return [
        _make_arm(
            a,
            citations=dlogos_citations if a == ARM_DLOGOS else None,
        )
        for a in ALL_ARMS
    ]


def _rater_prefers_cited(query: GoldenQuery, blinded) -> dict[Dimension, float]:
    """A rater that scores answers carrying citations higher (the dLogos arm).

    It sees only the *blinded* answer (arm identity stripped); it uses the
    presence of citations as its proxy, which is exactly the structural signal
    the dLogos arm carries.
    """

    base = 0.9 if blinded.answer.citations else 0.3
    return {d: base for d in Dimension}


def _query() -> GoldenQuery:
    return GoldenQuery(
        id="gq-x",
        archetype=Archetype.temporal_consensus,
        domain=Domain.technology,
        query_text="How has consensus on X moved?",
        pre_registered_answer_shape=AnswerShape(min_attributed_sources=1),
    )


async def test_run_query_scores_all_four_arms_and_unblinds() -> None:
    cit = Citation(
        episode_id="ep-0001", t_start=4.5, t_end=10.0, speaker_id="spk-analyst"
    )
    runner = EvalRunner(_arms(dlogos_citations=[cit]), _rater_prefers_cited, seed=7)
    qr = await runner.run_query(_query())

    assert {s.arm for s in qr.scores} == set(ALL_ARMS)
    # Blinding round-trips: each label maps to exactly one arm.
    assert set(qr.label_to_arm.values()) == set(ALL_ARMS)
    # The dLogos arm (the only one with a citation) scores highest.
    assert qr.total_for(ARM_DLOGOS) > qr.total_for(ARM_MODEL_ALONE)


async def test_citation_cap_bites_on_misattribution() -> None:
    # dLogos cites spk-analyst at [4.5,10] but the transcript says spk-host.
    cit = Citation(
        episode_id="ep-0001", t_start=4.5, t_end=10.0, speaker_id="spk-analyst"
    )
    transcript = Transcript(
        episode_id="ep-0001",
        language="en",
        duration_s=10.0,
        segments=[
            TranscriptSegment(speaker="SPEAKER_01", text="x", t_start=4.5, t_end=10.0)
        ],
    )
    ctx = CitationContext(
        transcripts={"ep-0001": transcript},
        segment_speaker_ids={"ep-0001": {0: "spk-host"}},  # not the analyst
    )

    def full_marks(query, blinded):
        return {d: 1.0 for d in Dimension}

    runner = EvalRunner(
        _arms(dlogos_citations=[cit]), full_marks, citation_context=ctx, seed=7
    )
    qr = await runner.run_query(_query())
    dlogos = next(s for s in qr.scores if s.arm == ARM_DLOGOS)
    # Even with a rater giving full marks, attribution precision is capped to 0
    # because 0/1 citations passed the speaker-verified check.
    assert dlogos.raw["attribution_precision"] == 0.0
    assert dlogos.verified_citations == 0
    assert dlogos.rejected_citations == 1


async def test_full_run_aggregates_and_renders_artifact() -> None:
    cit = Citation(
        episode_id="ep-0001", t_start=4.5, t_end=10.0, speaker_id="spk-analyst"
    )
    runner = EvalRunner(
        _arms(dlogos_citations=[cit]),
        _rater_prefers_cited,
        second_rater=_rater_prefers_cited,
        seed=99,
    )
    queries = starter_golden_queries()[:3]
    report = await runner.run(queries, agreement_subset=2)

    # dLogos wins every query and has the highest mean.
    assert report.win_count_by_arm[ARM_DLOGOS] == 3
    assert report.mean_total_by_arm[ARM_DLOGOS] == max(
        report.mean_total_by_arm.values()
    )
    # Coverage spread is reported.
    assert report.coverage["domains"]
    assert report.coverage["archetypes"]
    # Agreement over the subset (identical raters -> perfect).
    assert report.agreement is not None
    assert report.agreement.cohen_kappa == 1.0

    # Artifacts render and the JSON parses.
    md = report_to_markdown(report)
    assert "four-arm head-to-head" in md
    assert "Model + dLogos" in md
    parsed = json.loads(report_to_json(report))
    assert parsed["win_count_by_arm"][ARM_DLOGOS] == 3
