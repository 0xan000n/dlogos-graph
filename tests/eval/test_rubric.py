"""Tests for the reweighted rubric + speaker-verified citation check (spec §9)."""

from __future__ import annotations

import pytest

from dlogos.eval.arms import Answer, Citation
from dlogos.eval.rubric import (
    DEFAULT_WEIGHTS,
    DEMOTED_DIMENSIONS,
    ELEVATED_DIMENSION,
    Dimension,
    count_verified_citations,
    score_answer,
    validate_weights,
    verify_citation,
)
from dlogos.schema import Transcript, TranscriptSegment


# --------------------------------------------------------------------------- #
# Weighting math
# --------------------------------------------------------------------------- #
def test_default_weights_honor_reweighting_intent() -> None:
    # Does not raise: elevated dim is largest, demoted dims strictly smallest.
    validate_weights(DEFAULT_WEIGHTS)
    elevated = DEFAULT_WEIGHTS[ELEVATED_DIMENSION]
    assert all(
        DEFAULT_WEIGHTS[d] < elevated
        for d in Dimension
        if d is not ELEVATED_DIMENSION
    )
    demoted = [DEFAULT_WEIGHTS[d] for d in DEMOTED_DIMENSIONS]
    non_demoted = [
        DEFAULT_WEIGHTS[d] for d in Dimension if d not in DEMOTED_DIMENSIONS
    ]
    assert max(demoted) < min(non_demoted)


def test_validate_weights_rejects_non_demoted_recency() -> None:
    bad = dict(DEFAULT_WEIGHTS)
    # Pump recency above a high dimension -> violates the demotion contract.
    bad[Dimension.recency] = 0.5
    with pytest.raises(ValueError):
        validate_weights(bad)


def test_validate_weights_rejects_missing_dimension() -> None:
    bad = {Dimension.temporal_consensus_synthesis: 1.0}
    with pytest.raises(ValueError):
        validate_weights(bad)


def test_score_is_weighted_normalized_sum() -> None:
    answer = Answer(arm="model_dlogos", text="...", citations=[])
    raws = {
        Dimension.temporal_consensus_synthesis: 1.0,
        Dimension.attribution_precision: 0.5,
        Dimension.provenance_integrity: 0.0,
        Dimension.recency: 1.0,
        Dimension.couldnt_have_known: 1.0,
    }
    result = score_answer(answer, raws)
    # Hand-computed: sum(w*raw)/sum(w) with DEFAULT_WEIGHTS.
    w = DEFAULT_WEIGHTS
    expected_num = (
        w[Dimension.temporal_consensus_synthesis] * 1.0
        + w[Dimension.attribution_precision] * 0.5
        + w[Dimension.provenance_integrity] * 0.0
        + w[Dimension.recency] * 1.0
        + w[Dimension.couldnt_have_known] * 1.0
    )
    expected = expected_num / sum(w.values())
    assert result.total == pytest.approx(expected)


def test_temporal_synthesis_outweighs_recency_for_equal_raw() -> None:
    # Two answers: one strong only on the elevated dim, one strong only on the
    # demoted recency dim. The elevated-dim answer must score higher.
    answer = Answer(arm="x", text="...", citations=[])
    elevated_only = score_answer(
        answer,
        {Dimension.temporal_consensus_synthesis: 1.0},
    )
    recency_only = score_answer(
        answer,
        {Dimension.recency: 1.0},
    )
    assert elevated_only.total > recency_only.total


def test_score_rejects_out_of_range_raw() -> None:
    answer = Answer(arm="x", text="...", citations=[])
    with pytest.raises(ValueError):
        score_answer(answer, {Dimension.recency: 1.5})


# --------------------------------------------------------------------------- #
# Speaker-verified citation check
# --------------------------------------------------------------------------- #
def _transcript() -> Transcript:
    return Transcript(
        episode_id="ep-0001",
        language="en",
        duration_s=28.0,
        segments=[
            TranscriptSegment(speaker="SPEAKER_00", text="host intro", t_start=0.0, t_end=4.5),
            TranscriptSegment(
                speaker="SPEAKER_01", text="iPhone plateaued", t_start=4.5, t_end=10.0
            ),
            TranscriptSegment(speaker="SPEAKER_00", text="and OpenAI?", t_start=10.0, t_end=14.0),
        ],
    )


# segment index -> resolved canonical speaker id (supplied by speaker-identity)
_IDS = {0: "spk-host", 1: "spk-analyst", 2: "spk-host"}


def test_citation_passes_when_attributed_speaker_is_actually_speaking() -> None:
    cit = Citation(
        episode_id="ep-0001",
        t_start=5.0,
        t_end=9.0,
        speaker_id="spk-analyst",  # correct: segment 1 is the analyst
    )
    verdict = verify_citation(cit, _transcript(), _IDS)
    assert verdict.passed
    assert verdict.actual_speaker_id == "spk-analyst"


def test_topic_present_but_wrong_speaker_is_rejected() -> None:
    # The topic ("iPhone plateaued") IS at 4.5-10.0s, but it was the ANALYST who
    # said it. An answer that attributes that span to the HOST must be rejected,
    # even though the topic is present at the timestamp (spec §9 / §11 risk).
    cit = Citation(
        episode_id="ep-0001",
        t_start=5.0,
        t_end=9.0,
        speaker_id="spk-host",  # wrong: misattribution
    )
    verdict = verify_citation(cit, _transcript(), _IDS)
    assert not verdict.passed
    assert verdict.actual_speaker_id == "spk-analyst"
    assert "misattribution" in verdict.reason


def test_citation_to_wrong_episode_is_rejected() -> None:
    cit = Citation(
        episode_id="ep-9999", t_start=5.0, t_end=9.0, speaker_id="spk-analyst"
    )
    verdict = verify_citation(cit, _transcript(), _IDS)
    assert not verdict.passed


def test_hallucinated_timestamp_outside_all_segments_is_rejected() -> None:
    cit = Citation(
        episode_id="ep-0001",
        t_start=500.0,
        t_end=510.0,
        speaker_id="spk-analyst",
    )
    verdict = verify_citation(cit, _transcript(), _IDS)
    assert not verdict.passed
    assert "no diarized segment" in verdict.reason


def test_count_verified_splits_pass_and_reject() -> None:
    good = Citation(episode_id="ep-0001", t_start=5.0, t_end=9.0, speaker_id="spk-analyst")
    bad = Citation(episode_id="ep-0001", t_start=5.0, t_end=9.0, speaker_id="spk-host")
    answer = Answer(arm="x", text="...", citations=[good, bad])
    verified, rejected = count_verified_citations(
        answer, {"ep-0001": _transcript()}, {"ep-0001": _IDS}
    )
    assert (verified, rejected) == (1, 1)


def test_attribution_capped_by_verified_fraction() -> None:
    # Rater generously gave full attribution credit, but half the citations are
    # misattributions -> the cap pulls attribution_precision down to 0.5.
    good = Citation(episode_id="ep-0001", t_start=5.0, t_end=9.0, speaker_id="spk-analyst")
    bad = Citation(episode_id="ep-0001", t_start=5.0, t_end=9.0, speaker_id="spk-host")
    answer = Answer(arm="x", text="...", citations=[good, bad])
    result = score_answer(
        answer,
        {Dimension.attribution_precision: 1.0},
        transcripts={"ep-0001": _transcript()},
        segment_speaker_ids={"ep-0001": _IDS},
    )
    assert result.verified_citations == 1
    assert result.rejected_citations == 1
    assert result.raw[Dimension.attribution_precision] == pytest.approx(0.5)
