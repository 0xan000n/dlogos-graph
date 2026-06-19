"""Adversarial diarization-mapping tests (spec §9 / §11, the top risk).

The clean fixtures in ``tests/asr/test_diarization.py`` prove the happy path:
two well-separated speakers map correctly. These tests prove the *failure* path
is real — that under the three adversarial cases the spec names (panel show,
remote-heavy interview, ad-saturated show) ``map_words_to_speakers`` actually
attributes a word window to the WRONG speaker. That swap is the input to the
companion eval test (``tests/eval/test_misattribution_adversarial.py``), which
proves the speaker-verified citation check catches it.

The fixtures are honest: each carries a ground-truth oracle of who really spoke
(NOT derived from the diarization), so comparing the mapping against it is a
real test rather than a tautology.
"""

from __future__ import annotations

import pytest

from dlogos.asr.base import drop_low_talk_time_speakers, talk_time_by_speaker
from dlogos.asr.diarization import (
    crosstalk_regions,
    map_words_to_speakers,
)
from tests.fixtures.adversarial import (
    ALL_SCENARIOS,
    ad_saturated_scenario,
    panel_scenario,
    remote_scenario,
)


def _resolved_speakers_for_probe(scenario) -> set[str]:
    """Map the scenario's probe words and resolve labels → canonical humans."""

    mapped = map_words_to_speakers(scenario.probe.words, scenario.diarization)
    labels = {w["speaker"] for w in mapped}
    return {scenario.label_to_human.get(lbl, lbl) for lbl in labels}


# --------------------------------------------------------------------------- #
# The core adversarial assertion, run over all three scenarios
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("make_scenario", ALL_SCENARIOS, ids=lambda f: f().name)
def test_probe_window_is_misattributed_under_failure(make_scenario) -> None:
    """Each adversarial case attributes the probe to the WRONG human.

    This is the documented failure actually occurring: the diarization-driven
    mapping lands the probe's words on a label that resolves to someone other
    than the ground-truth speaker.
    """

    scenario = make_scenario()

    # Oracle sanity: the probe's declared expected speaker matches the ground
    # truth derived independently of the diarization.
    assert scenario.true_human_at(scenario.probe.t_start, scenario.probe.t_end) == (
        scenario.probe.expected_human_id
    ), f"{scenario.name}: probe oracle inconsistent with ground truth"

    resolved = _resolved_speakers_for_probe(scenario)

    # The mapping does NOT recover the true speaker — it is a misattribution.
    assert scenario.probe.expected_human_id not in resolved, (
        f"{scenario.name}: expected a misattribution but the mapping recovered "
        f"the true speaker {scenario.probe.expected_human_id!r}"
    )
    # And it lands on exactly the wrong resolved human.
    assert len(resolved) == 1, f"{scenario.name}: probe spans multiple labels: {resolved}"


# --------------------------------------------------------------------------- #
# (a) PANEL — absorbed interjection + crosstalk detection
# --------------------------------------------------------------------------- #
def test_panel_interjection_absorbed_into_neighbor() -> None:
    scenario = panel_scenario()

    mapped = map_words_to_speakers(scenario.probe.words, scenario.diarization)
    # B's interjection words all land on SPEAKER_01 (panelist A's collapsed turn).
    assert {w["speaker"] for w in mapped} == {"SPEAKER_01"}
    assert scenario.label_to_human["SPEAKER_01"] == "spk-panel-a"  # not B


def test_panel_talk_time_does_not_save_the_interjection() -> None:
    # Both panelists are substantial speakers, so talk-time pruning cannot flag
    # the absorbed interjection — only the speaker-verified check can. We assert
    # the collapsed SPEAKER_01 is well above the 5% drop threshold.
    scenario = panel_scenario()
    transcript = scenario.transcript()
    stats = talk_time_by_speaker(transcript)
    assert stats.fractions["SPEAKER_01"] > 0.05
    pruned = drop_low_talk_time_speakers(transcript, min_fraction=0.05)
    kept = {seg.speaker for seg in pruned.segments}
    assert "SPEAKER_01" in kept  # the offending merged label survives pruning


def test_panel_raw_crosstalk_region_is_detected() -> None:
    # When the diarizer emits the overlap (rather than collapsing it), the
    # crosstalk_regions helper surfaces the contested window so the attribution
    # there can be flagged low-confidence — the failure the talk-time helper
    # cannot see (both speakers are genuinely present).
    scenario = panel_scenario()
    raw = scenario.extra["raw_overlapping_diarization"]
    regions = crosstalk_regions(raw)
    assert (8.0, 9.2) in regions

    # A word in that window is flagged, while a word outside it is not.
    flagged = map_words_to_speakers(
        [{"start": 8.4, "end": 8.8, "word": "no"}], raw, flag_crosstalk=True
    )
    assert flagged[0]["crosstalk"] is True
    clean = map_words_to_speakers(
        [{"start": 1.0, "end": 2.0, "word": "welcome"}], raw, flag_crosstalk=True
    )
    assert clean[0]["crosstalk"] is False


def test_crosstalk_regions_ignores_same_label_splits() -> None:
    # Two turns with the SAME label (a diarizer splitting one speaker) is NOT
    # crosstalk — only distinct labels overlapping count.
    from dlogos.asr.diarization import DiarizationTurn

    same = [
        DiarizationTurn("SPEAKER_01", 0.0, 5.0),
        DiarizationTurn("SPEAKER_01", 4.0, 8.0),  # same speaker, overlapping
    ]
    assert crosstalk_regions(same) == []


# --------------------------------------------------------------------------- #
# (b) REMOTE — fragmentation + boundary overrun
# --------------------------------------------------------------------------- #
def test_remote_guest_fragmented_across_multiple_labels() -> None:
    scenario = remote_scenario()
    frags = scenario.extra["guest_fragment_labels"]
    # All fragment labels resolve to the SAME guest human (one human, many labels).
    assert {scenario.label_to_human[f] for f in frags} == {"spk-guest"}
    assert len(frags) >= 3


def test_remote_overrun_steals_host_first_reply_words() -> None:
    scenario = remote_scenario()
    mapped = map_words_to_speakers(scenario.probe.words, scenario.diarization)
    # The host's first reply words land on the overrunning guest fragment.
    assert {w["speaker"] for w in mapped} == {"SPEAKER_03"}
    assert scenario.label_to_human["SPEAKER_03"] == "spk-guest"  # not the host


# --------------------------------------------------------------------------- #
# (c) AD-SATURATED — spurious ad label + content contamination
# --------------------------------------------------------------------------- #
def test_ad_label_overrun_contaminates_real_claim() -> None:
    scenario = ad_saturated_scenario()
    mapped = map_words_to_speakers(scenario.probe.words, scenario.diarization)
    # The host's first post-ad CLAIM words land on the spurious ad label.
    assert {w["speaker"] for w in mapped} == {"SPEAKER_09"}
    assert scenario.label_to_human["SPEAKER_09"] == "spk-ad-read"  # not the host


def test_ad_label_is_not_pruned_by_talk_time() -> None:
    # The ad read is long enough that drop_low_talk_time_speakers keeps the
    # spurious SPEAKER_09 — so talk-time pruning is NOT the safety net here; the
    # speaker-verified citation check must be.
    scenario = ad_saturated_scenario()
    transcript = scenario.transcript()
    stats = talk_time_by_speaker(transcript)
    assert stats.fractions["SPEAKER_09"] > 0.05
    pruned = drop_low_talk_time_speakers(transcript, min_fraction=0.05)
    assert "SPEAKER_09" in {seg.speaker for seg in pruned.segments}
