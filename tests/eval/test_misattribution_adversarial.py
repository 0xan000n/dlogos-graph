"""The teeth: the speaker-verified citation check catches a REAL diarization swap.

``tests/eval/test_rubric.py`` proves ``verify_citation`` rejects a *synthetic*
misattribution (a hand-set wrong ``speaker_id`` against a clean transcript).
That is necessary but circular-looking — the wrongness is asserted into
existence. These tests close the loop: they take the misattribution produced by
the adversarial diarization slice (panel / remote / ad-saturated) and feed the
*resulting* citation through ``verify_citation``, verified against an
independent GROUND-TRUTH transcript, proving the §9 check rejects a swap the
diarization mapper genuinely committed — not one we typed in.

The pipeline modeled here:

  1. Adversarial diarization maps the probe words to the WRONG label
     (``tests/fixtures/adversarial`` / proven in
     ``tests/asr/test_adversarial_diarization.py``).
  2. Extraction/retrieval builds a Citation from the *diarized* transcript: it
     credits the span to the human that the wrong label resolves to. This is
     the confident misattribution shipping downstream.
  3. ``verify_citation`` checks that citation against the GROUND-TRUTH speaker
     ids (an oracle NOT derived from the diarization) and must REJECT it,
     naming the human who really spoke there.

Because the ground truth is independent of the diarizer, a rejection is real
evidence the check works — and the contrasting "correct attribution passes"
test rules out a check that simply rejects everything.
"""

from __future__ import annotations

import pytest

from dlogos.asr.diarization import map_words_to_speakers
from dlogos.eval.arms import Answer, Citation
from dlogos.eval.rubric import (
    Dimension,
    count_verified_citations,
    score_answer,
    verify_citation,
)
from tests.fixtures.adversarial import (
    ALL_SCENARIOS,
    ad_saturated_scenario,
    panel_scenario,
    remote_scenario,
)


def _misattributed_citation(scenario) -> Citation:
    """Build the citation that the diarization swap actually produces.

    Runs the probe words through ``map_words_to_speakers`` (the real swap),
    resolves the winning label to its human via the pipeline's (flawed)
    ``label_to_human``, and emits a Citation crediting that WRONG human for the
    probe span — exactly what a retriever reading the diarized transcript carries
    into an answer.
    """

    mapped = map_words_to_speakers(scenario.probe.words, scenario.diarization)
    labels = {w["speaker"] for w in mapped}
    assert len(labels) == 1, f"{scenario.name}: probe split across labels {labels}"
    wrong_label = next(iter(labels))
    wrong_human = scenario.label_to_human[wrong_label]
    # Sanity: this really is the wrong human (the swap happened).
    assert wrong_human != scenario.probe.expected_human_id
    return Citation(
        episode_id=scenario.episode_id,
        t_start=scenario.probe.t_start,
        t_end=scenario.probe.t_end,
        speaker_id=wrong_human,
    )


def _correct_citation(scenario) -> Citation:
    """A citation crediting the TRUE speaker of the probe span (the oracle)."""

    return Citation(
        episode_id=scenario.episode_id,
        t_start=scenario.probe.t_start,
        t_end=scenario.probe.t_end,
        speaker_id=scenario.probe.expected_human_id,
    )


# --------------------------------------------------------------------------- #
# The core teeth test, over all three adversarial scenarios
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("make_scenario", ALL_SCENARIOS, ids=lambda f: f().name)
def test_verify_citation_rejects_real_diarization_swap(make_scenario) -> None:
    scenario = make_scenario()
    citation = _misattributed_citation(scenario)

    # Verify against the GROUND TRUTH (independent oracle), not the diarization.
    transcript = scenario.ground_truth_transcript()
    seg_ids = scenario.ground_truth_segment_ids()

    verdict = verify_citation(citation, transcript, seg_ids)

    # The check rejects the misattribution the diarizer produced.
    assert not verdict.passed, (
        f"{scenario.name}: speaker-verified check FAILED to catch the swap "
        f"(citation credited {citation.speaker_id!r})"
    )
    assert "misattribution" in verdict.reason
    # It names who REALLY spoke at the cited span (the ground-truth speaker).
    assert verdict.actual_speaker_id == scenario.probe.expected_human_id


@pytest.mark.parametrize("make_scenario", ALL_SCENARIOS, ids=lambda f: f().name)
def test_correct_attribution_passes_same_check(make_scenario) -> None:
    # The check is not a reject-everything stub: a citation crediting the TRUE
    # speaker of the same span passes against the same ground-truth transcript.
    scenario = make_scenario()
    verdict = verify_citation(
        _correct_citation(scenario),
        scenario.ground_truth_transcript(),
        scenario.ground_truth_segment_ids(),
    )
    assert verdict.passed
    assert verdict.actual_speaker_id == scenario.probe.expected_human_id


# --------------------------------------------------------------------------- #
# The swap propagates into rubric scoring: attribution credit is capped
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("make_scenario", ALL_SCENARIOS, ids=lambda f: f().name)
def test_swap_caps_attribution_precision_in_scorer(make_scenario) -> None:
    # An answer whose ONLY citation is the diarization-swap misattribution
    # cannot earn attribution-precision credit, even if a blinded rater was
    # fooled into giving it full marks (topic IS present at the timestamp).
    scenario = make_scenario()
    answer = Answer(
        arm="model_dlogos",
        text="...attributes the claim to the wrong speaker...",
        citations=[_misattributed_citation(scenario)],
    )
    result = score_answer(
        answer,
        {Dimension.attribution_precision: 1.0},
        transcripts={scenario.episode_id: scenario.ground_truth_transcript()},
        segment_speaker_ids={scenario.episode_id: scenario.ground_truth_segment_ids()},
    )
    assert result.verified_citations == 0
    assert result.rejected_citations == 1
    # Capped to the verified fraction (0/1 == 0.0), overriding the rater's 1.0.
    assert result.raw[Dimension.attribution_precision] == pytest.approx(0.0)


def test_mixed_answer_partial_attribution_credit() -> None:
    # A realistic answer mixing one correct citation and one swapped citation
    # earns exactly half attribution credit (1 of 2 verified). Uses the remote
    # scenario; both citations live in its single episode.
    scenario = remote_scenario()
    answer = Answer(
        arm="model_dlogos",
        text="one right, one wrong attribution",
        citations=[_correct_citation(scenario), _misattributed_citation(scenario)],
    )
    transcripts = {scenario.episode_id: scenario.ground_truth_transcript()}
    seg_ids = {scenario.episode_id: scenario.ground_truth_segment_ids()}

    verified, rejected = count_verified_citations(answer, transcripts, seg_ids)
    assert (verified, rejected) == (1, 1)

    result = score_answer(
        answer,
        {Dimension.attribution_precision: 1.0},
        transcripts=transcripts,
        segment_speaker_ids=seg_ids,
    )
    assert result.raw[Dimension.attribution_precision] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Per-scenario specifics — name the wrong human the swap credits
# --------------------------------------------------------------------------- #
def test_panel_swap_credits_neighbor_panelist() -> None:
    scenario = panel_scenario()
    citation = _misattributed_citation(scenario)
    # B's interjection is credited to panelist A.
    assert citation.speaker_id == "spk-panel-a"
    verdict = verify_citation(
        citation, scenario.ground_truth_transcript(), scenario.ground_truth_segment_ids()
    )
    assert not verdict.passed
    assert verdict.actual_speaker_id == "spk-panel-b"


def test_remote_swap_credits_guest_for_host_words() -> None:
    scenario = remote_scenario()
    citation = _misattributed_citation(scenario)
    assert citation.speaker_id == "spk-guest"
    verdict = verify_citation(
        citation, scenario.ground_truth_transcript(), scenario.ground_truth_segment_ids()
    )
    assert not verdict.passed
    assert verdict.actual_speaker_id == "spk-host"


def test_ad_swap_credits_ad_persona_for_real_claim() -> None:
    scenario = ad_saturated_scenario()
    citation = _misattributed_citation(scenario)
    assert citation.speaker_id == "spk-ad-read"
    verdict = verify_citation(
        citation, scenario.ground_truth_transcript(), scenario.ground_truth_segment_ids()
    )
    assert not verdict.passed
    # The real claim belongs to the host, not the ad-read persona.
    assert verdict.actual_speaker_id == "spk-host"
