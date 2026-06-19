"""The adversarial diarization validation slice (spec §9 / §11, top risk).

The diarization → speaker-mapping path is the PoC's #1 correctness risk: a
swapped label produces a *confident, sourced* misattribution — the worst
failure of all. The clean 2-turn fixtures in ``tests/conftest.py`` exercise the
happy path; this module builds the three deliberately *adversarial* cases the
spec names (§9 "Adversarial validation slice", §11 risk table) so the
attribution machinery is validated under stress before it is trusted at corpus
scale:

  (a) **PANEL show** — 3+ speakers with overlapping / absorbed interjections.
  (b) **REMOTE-heavy interview** — one guest fragmented across several
      diarization labels by unstable (phone-quality) voiceprints.
  (c) **AD-SATURATED show** — a spurious ad-read speaker label spawned mid
      episode, plus a real claim near the ad boundary that inherits the wrong
      label.

Construction is *honest*, not circular. Each scenario carries two independent
things:

- ``ground_truth``: the intervals of who ACTUALLY spoke, tagged with the real
  canonical human id. This is the oracle — it is NOT derived from the
  diarization, so feeding the diarization back through the mapper and comparing
  to it is a real test, not a tautology.
- ``diarization``: the (flawed) :class:`DiarizationTurn` list a real diarizer
  would emit under the failure condition — collapsed crosstalk, fragmented
  remote leg, spurious ad label. This is what the mapper actually sees.

A scenario also exposes a ``probe`` word window whose ground-truth speaker is
known, so a test can assert ``map_words_to_speakers`` attributes it to the
WRONG label (the documented failure actually occurs) and then that the
resulting misattributed citation is REJECTED by ``eval.rubric.verify_citation``.

Heavy-dep-free: only stdlib + ``dlogos.schema`` + the pure
``dlogos.asr.diarization`` types. No torch / pyannote / network / randomness.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dlogos.asr.diarization import DiarizationTurn
from dlogos.schema import Transcript, TranscriptSegment


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GroundTruthInterval:
    """Who *actually* spoke over ``[start, end]`` — the oracle, not the diarizer.

    ``human_id`` is the real canonical speaker identity (e.g. ``"spk-host"``),
    independent of any per-episode diarization label. The adversarial tests
    treat this as truth and check the diarization-driven mapping against it.
    """

    human_id: str
    start: float
    end: float
    text: str = ""


@dataclass(frozen=True)
class ProbeWindow:
    """A word window with a known ground-truth speaker, used to trigger a swap.

    ``expected_human_id`` is who truly speaks here; ``words`` are word dicts (the
    shape ``map_words_to_speakers`` consumes) covering ``[t_start, t_end]``.
    """

    t_start: float
    t_end: float
    expected_human_id: str
    words: list[dict]


@dataclass(frozen=True)
class AdversarialScenario:
    """One adversarial diarization case end to end.

    Fields
    ------
    name:
        Short scenario id (``"panel"`` / ``"remote"`` / ``"ad_saturated"``).
    episode_id:
        Episode the spans belong to.
    ground_truth:
        Who actually spoke, when (the oracle).
    diarization:
        The flawed turns a real diarizer would emit under this failure.
    label_to_human:
        How a (correct) speaker-identity stage *would* resolve each diarization
        label to a canonical human — i.e. the **majority** human behind that
        label. Under the failure, a single label can cover words from more than
        one human; this map captures the dominant resolution, which is exactly
        what makes the minority words a misattribution.
    probe:
        The word window whose attribution flips to the wrong human.
    failure_mode:
        Human-readable description of the documented failure being reproduced.
    """

    name: str
    episode_id: str
    ground_truth: list[GroundTruthInterval]
    diarization: list[DiarizationTurn]
    label_to_human: dict[str, str]
    probe: ProbeWindow
    failure_mode: str
    extra: dict = field(default_factory=dict)

    # -- derived oracles --------------------------------------------------- #
    def true_human_at(self, t_start: float, t_end: float) -> str | None:
        """Ground-truth human whose interval overlaps ``[t_start, t_end]`` most."""

        best: str | None = None
        best_ov = 0.0
        for iv in self.ground_truth:
            ov = max(0.0, min(t_end, iv.end) - max(t_start, iv.start))
            if ov > best_ov:
                best_ov = ov
                best = iv.human_id
        return best

    def transcript(self) -> Transcript:
        """A diarized :class:`Transcript`, one segment per diarization turn.

        Segment order follows the diarization turns sorted by time, which is how
        the real pipeline emits segments. ``verify_citation`` indexes segments
        positionally, so :meth:`segment_speaker_ids` is keyed to this order.
        """

        ordered = sorted(self.diarization, key=lambda t: (t.start, t.end, t.speaker))
        segments = [
            TranscriptSegment(
                speaker=t.speaker,
                text=self._text_for_turn(t),
                t_start=t.start,
                t_end=t.end,
            )
            for t in ordered
        ]
        duration = max((t.end for t in ordered), default=0.0)
        return Transcript(
            episode_id=self.episode_id,
            language="en",
            segments=segments,
            duration_s=duration,
        )

    def segment_speaker_ids(self) -> dict[int, str]:
        """Map each *diarized* segment index → human the pipeline RESOLVED it to.

        Resolution is by ``label_to_human`` (the dominant human behind a label),
        mirroring what the speaker-identity stage produces from the (flawed)
        diarization. This is what the pipeline *believes*. Under the failure it
        disagrees with the ground truth for the absorbed / fragmented /
        ad-contaminated words.
        """

        ordered = sorted(self.diarization, key=lambda t: (t.start, t.end, t.speaker))
        return {
            idx: self.label_to_human[t.speaker]
            for idx, t in enumerate(ordered)
            if t.speaker in self.label_to_human
        }

    def ground_truth_transcript(self) -> Transcript:
        """A transcript whose segments are the GROUND-TRUTH intervals.

        This is the oracle the §9 check should be verified against: it encodes
        who *really* spoke when, independent of the diarizer. Segment order is by
        time, matching :meth:`ground_truth_segment_ids`.
        """

        ordered = sorted(self.ground_truth, key=lambda iv: (iv.start, iv.end))
        segments = [
            TranscriptSegment(
                speaker=iv.human_id,
                text=iv.text,
                t_start=iv.start,
                t_end=iv.end,
            )
            for iv in ordered
        ]
        duration = max((iv.end for iv in ordered), default=0.0)
        return Transcript(
            episode_id=self.episode_id,
            language="en",
            segments=segments,
            duration_s=duration,
        )

    def ground_truth_segment_ids(self) -> dict[int, str]:
        """Map each ground-truth segment index → the real human id (the oracle).

        Keyed to :meth:`ground_truth_transcript`'s segment order. This is the
        truth the speaker-verified citation check is verified against, so a
        rejection is real evidence the swap was caught — not a tautology.
        """

        ordered = sorted(self.ground_truth, key=lambda iv: (iv.start, iv.end))
        return {idx: iv.human_id for idx, iv in enumerate(ordered)}

    def _text_for_turn(self, turn: DiarizationTurn) -> str:
        """Best-effort text for a turn (from overlapping ground truth)."""

        parts = [
            iv.text
            for iv in self.ground_truth
            if iv.text and max(0.0, min(turn.end, iv.end) - max(turn.start, iv.start)) > 0.0
        ]
        return " ".join(parts)


def _w(start: float, end: float, word: str = "x") -> dict:
    return {"start": start, "end": end, "word": word}


# --------------------------------------------------------------------------- #
# (a) PANEL show — 3+ speakers, overlapping / absorbed interjection
# --------------------------------------------------------------------------- #
def panel_scenario() -> AdversarialScenario:
    """Panel show: a short interjection is *absorbed* into a crosstalk turn.

    Cast: a moderator (``spk-mod``) and two panelists (``spk-panel-a``,
    ``spk-panel-b``). At ~8.0-9.2s panelist B cuts in with a one-line
    interjection WHILE panelist A is still finishing — genuine crosstalk. The
    diarizer, faced with overlapping voices, collapses the whole 6.0-12.0s
    stretch into a single ``SPEAKER_01`` turn dominated by panelist A. B's
    interjection is thereby absorbed into A's label.

    Documented failure (§11): "short interjections get absorbed into a
    neighbour's turn." A word window over B's interjection maps to
    ``SPEAKER_01`` → resolves to panelist A, not B. The talk-time helper cannot
    catch this (both A and B are real, substantial speakers); only the
    speaker-verified citation check can.
    """

    episode_id = "ep-panel-001"

    ground_truth = [
        GroundTruthInterval("spk-mod", 0.0, 6.0, "Welcome to the panel. Let's open it up."),
        # Panelist A holds the floor 6.0-12.0, but B cuts in at 8.0-9.2.
        GroundTruthInterval("spk-panel-a", 6.0, 8.0, "The regulation will slow everyone down,"),
        GroundTruthInterval("spk-panel-b", 8.0, 9.2, "No, that's exactly backwards."),
        GroundTruthInterval("spk-panel-a", 9.2, 12.0, "as I was saying, it slows the incumbents."),
        GroundTruthInterval("spk-mod", 12.0, 15.0, "Let's let each of you finish."),
    ]

    # Flawed diarization: crosstalk collapses 6.0-12.0 into ONE turn (A-dominant);
    # B's interjection never gets its own label. Two true speakers, one label.
    diarization = [
        DiarizationTurn("SPEAKER_00", 0.0, 6.0),       # moderator
        DiarizationTurn("SPEAKER_01", 6.0, 12.0),      # "panelist A" — actually A+B
        DiarizationTurn("SPEAKER_00", 12.0, 15.0),     # moderator
    ]

    # SPEAKER_01 is dominated by panelist A (4.8s of 6.0s) → resolves to A.
    label_to_human = {
        "SPEAKER_00": "spk-mod",
        "SPEAKER_01": "spk-panel-a",
    }

    # The *raw* diarizer output before crosstalk collapse: B's interjection
    # briefly gets its own label (SPEAKER_02) overlapping A's SPEAKER_01 turn.
    # This is a genuine crosstalk region that crosstalk_regions() must surface
    # so the attribution there can be flagged low-confidence. The mitigation
    # collapses it (above) — but a diarizer that emits the overlap instead hands
    # the mapper two distinct labels fighting over the same instant.
    raw_overlapping_diarization = [
        DiarizationTurn("SPEAKER_00", 0.0, 6.0),
        DiarizationTurn("SPEAKER_01", 6.0, 12.0),      # panelist A
        DiarizationTurn("SPEAKER_02", 8.0, 9.2),       # panelist B, OVERLAPS A
        DiarizationTurn("SPEAKER_00", 12.0, 15.0),
    ]

    # Probe: panelist B's interjection words. Ground truth = spk-panel-b.
    probe = ProbeWindow(
        t_start=8.1,
        t_end=9.1,
        expected_human_id="spk-panel-b",
        words=[_w(8.1, 8.5, "No"), _w(8.6, 9.1, "backwards")],
    )

    return AdversarialScenario(
        name="panel",
        episode_id=episode_id,
        ground_truth=ground_truth,
        diarization=diarization,
        label_to_human=label_to_human,
        probe=probe,
        failure_mode=(
            "crosstalk: panelist B's interjection is absorbed into panelist A's "
            "collapsed turn (SPEAKER_01), so B's words are attributed to A"
        ),
        extra={
            "raw_overlapping_diarization": raw_overlapping_diarization,
            "crosstalk_window": (8.0, 9.2),
        },
    )


# --------------------------------------------------------------------------- #
# (b) REMOTE-heavy interview — one guest split across several labels
# --------------------------------------------------------------------------- #
def remote_scenario() -> AdversarialScenario:
    """Remote interview: one guest fragmented across several diarization labels.

    A single remote guest (``spk-guest``) on a phone-quality / echo-cancelled
    leg has an unstable voiceprint, so the diarizer splits the *same* human
    across three labels (``SPEAKER_01``, ``SPEAKER_02``, ``SPEAKER_03``) over the
    interview. The host (``spk-host``) is local and stable.

    The danger is the *adjacent* failure the fragmentation enables: at a
    host↔guest handoff the diarizer mis-segments — it extends the guest's third
    fragment (``SPEAKER_03``) slightly past the true boundary, swallowing the
    first words of the host's reply. Those host words land on ``SPEAKER_03``,
    which resolves to the guest.

    Documented failure (§11): "the diarizer splits one remote guest into several
    labels"; the fragmentation at a boundary then steals a neighbour's words.
    Probe = the host's first reply words, which map to a guest fragment.
    """

    episode_id = "ep-remote-001"

    ground_truth = [
        GroundTruthInterval("spk-host", 0.0, 5.0, "Thanks for calling in. How's the connection?"),
        GroundTruthInterval("spk-guest", 5.0, 12.0, "A bit echoey, but I can hear you fine."),
        GroundTruthInterval("spk-host", 12.0, 16.0, "Great. So tell me about the new model."),
        GroundTruthInterval("spk-guest", 16.0, 24.0, "It's faster, and cheaper to run at scale."),
        # Host replies; the true boundary is 24.0, but diarization runs late.
        GroundTruthInterval("spk-host", 24.0, 30.0, "Cheaper how — is that the inference cost?"),
    ]

    # Flawed diarization: the guest is fragmented into SPEAKER_01/02/03, AND the
    # last guest fragment (SPEAKER_03) runs 1.5s past the true 24.0 boundary,
    # swallowing the host's first reply words (24.0-25.5).
    diarization = [
        DiarizationTurn("SPEAKER_00", 0.0, 5.0),       # host
        DiarizationTurn("SPEAKER_01", 5.0, 12.0),      # guest fragment 1
        DiarizationTurn("SPEAKER_00", 12.0, 16.0),     # host
        DiarizationTurn("SPEAKER_02", 16.0, 20.0),     # guest fragment 2
        DiarizationTurn("SPEAKER_03", 20.0, 25.5),     # guest fragment 3, OVERRUNS
        DiarizationTurn("SPEAKER_00", 25.5, 30.0),     # host (truncated start)
    ]

    # A correct identity stage resolves all three guest fragments to one human.
    # (This is the *mitigation* working — yet the overrun still misattributes.)
    label_to_human = {
        "SPEAKER_00": "spk-host",
        "SPEAKER_01": "spk-guest",
        "SPEAKER_02": "spk-guest",
        "SPEAKER_03": "spk-guest",
    }

    # Probe: the host's first reply words at 24.2-25.0, inside the overrun.
    probe = ProbeWindow(
        t_start=24.2,
        t_end=25.0,
        expected_human_id="spk-host",
        words=[_w(24.2, 24.6, "Cheaper"), _w(24.7, 25.0, "how")],
    )

    return AdversarialScenario(
        name="remote",
        episode_id=episode_id,
        ground_truth=ground_truth,
        diarization=diarization,
        label_to_human=label_to_human,
        probe=probe,
        failure_mode=(
            "remote fragmentation: a guest fragment (SPEAKER_03) overruns the "
            "host's reply boundary, so the host's first words are attributed to "
            "the guest"
        ),
        extra={"guest_fragment_labels": ["SPEAKER_01", "SPEAKER_02", "SPEAKER_03"]},
    )


# --------------------------------------------------------------------------- #
# (c) AD-SATURATED show — spurious ad-read speaker label mid-episode
# --------------------------------------------------------------------------- #
def ad_saturated_scenario() -> AdversarialScenario:
    """Ad-saturated show: a spurious ad-read label, and a real claim near it.

    The host (``spk-host``) reads a dynamically-stitched sponsorship in a
    different register / from a different voice actor. The diarizer spawns a
    spurious label (``SPEAKER_09``) for the ad voice. The real risk is at the
    ad↔content boundary: when the host resumes with a genuine claim right after
    the ad, the diarizer's ad turn runs slightly long and swallows the first
    words of that claim — so real content inherits the ad label.

    Documented failure (§11): "a stitched-in dynamic ad with a third voice
    spawns a spurious label ... and a real claim near the ad boundary may
    inherit the ad's label." The probe = the host's first post-ad content words,
    which map to ``SPEAKER_09`` (the ad voice), not ``spk-host``.

    Note: ``SPEAKER_09`` carries enough talk time (the full ad read) that it is
    NOT pruned by ``drop_low_talk_time_speakers`` — so talk-time pruning does not
    save us here; the speaker-verified check must.
    """

    episode_id = "ep-ad-001"

    ground_truth = [
        GroundTruthInterval("spk-host", 0.0, 20.0, "Before the break, markets looked shaky."),
        # Host reads the ad (different register) 20.0-40.0 — still the host, but
        # the diarizer hears a different voiceprint.
        GroundTruthInterval(
            "spk-host", 20.0, 40.0,
            "This episode is brought to you by Acme. Use code POD for ten percent off.",
        ),
        # Host resumes with a REAL claim immediately after the ad.
        GroundTruthInterval(
            "spk-host", 40.0, 50.0,
            "Anyway — I think the Fed will cut rates by September.",
        ),
    ]

    # Flawed diarization: a spurious SPEAKER_09 for the ad voice that OVERRUNS
    # the ad boundary (40.0 → 41.8), swallowing the host's first post-ad words.
    diarization = [
        DiarizationTurn("SPEAKER_00", 0.0, 20.0),      # host (pre-ad)
        DiarizationTurn("SPEAKER_09", 20.0, 41.8),     # "ad voice" — OVERRUNS into content
        DiarizationTurn("SPEAKER_00", 41.8, 50.0),     # host (post-ad, truncated start)
    ]

    # The identity stage resolves the spurious ad label to its own pseudo-id
    # (an "ad-read" persona / unknown), distinct from the host. A citation that
    # credits the host for words inside SPEAKER_09 is therefore a misattribution.
    label_to_human = {
        "SPEAKER_00": "spk-host",
        "SPEAKER_09": "spk-ad-read",
    }

    # Probe: the host's first real-claim words after the ad, at 40.2-41.2,
    # caught inside the ad turn's overrun.
    probe = ProbeWindow(
        t_start=40.2,
        t_end=41.2,
        expected_human_id="spk-host",
        words=[_w(40.2, 40.6, "the"), _w(40.7, 41.2, "Fed")],
    )

    return AdversarialScenario(
        name="ad_saturated",
        episode_id=episode_id,
        ground_truth=ground_truth,
        diarization=diarization,
        label_to_human=label_to_human,
        probe=probe,
        failure_mode=(
            "ad-read contamination: the spurious ad label (SPEAKER_09) overruns "
            "into the host's first post-ad claim words, attributing real content "
            "to the ad-read persona"
        ),
    )


ALL_SCENARIOS = (
    panel_scenario,
    remote_scenario,
    ad_saturated_scenario,
)
