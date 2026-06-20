"""Tests for post-hoc claim grounding (src/dlogos/extraction/grounding.py).

The extractor sometimes emits a ``source_span`` on a coarse estimated grid and
a speaker label that disagrees with the diarization at the cited time. The
grounding pass snaps each claim back onto the transcript segment its evidence
actually came from, fixing BOTH the span and the speaker. These tests pin that
behavior down deterministically on a small synthetic transcript with three
distinctly-worded segments by different speakers — no model, no network.
"""

from __future__ import annotations

from dlogos.extraction.grounding import (
    DEFAULT_THRESHOLD,
    ground_claim,
    ground_claims,
    match_score,
)
from dlogos.schema import (
    Entity,
    EntityType,
    ExtractedClaim,
    Predicate,
    SourceSpan,
    SpeakerRef,
    Stance,
    Transcript,
    TranscriptSegment,
)

EPISODE_ID = "ep-ground"


def _transcript() -> Transcript:
    """Three distinctly-worded segments, each by a different speaker.

    The wordings share no salient content words across segments, so a claim's
    evidence can only match the one segment it was drawn from — that is what
    makes the regrounding target unambiguous.
    """

    segments = [
        TranscriptSegment(
            speaker="SPEAKER_00",
            text="The new electric pickup truck has incredible towing capacity.",
            t_start=0.0,
            t_end=6.0,
        ),
        TranscriptSegment(
            speaker="SPEAKER_01",
            text="Honestly, sourdough bread needs a long cold fermentation to "
            "develop flavor.",
            t_start=6.0,
            t_end=12.5,
        ),
        TranscriptSegment(
            speaker="SPEAKER_02",
            text="Quantum computers will eventually break most public-key "
            "cryptography schemes.",
            t_start=12.5,
            t_end=19.0,
        ),
    ]
    return Transcript(
        episode_id=EPISODE_ID,
        language="en",
        segments=segments,
        duration_s=19.0,
    )


def _claim(
    *,
    obj: str,
    label: str = "SPEAKER_99",
    t_start: float = 190.0,
    t_end: float = 200.0,
) -> ExtractedClaim:
    """A claim whose ``object`` is the evidence text and whose span/label are
    the defective (estimated / mis-attributed) values to be corrected."""

    return ExtractedClaim(
        speaker=SpeakerRef(label=label),
        predicate=Predicate.rates_positive,
        subject_entity=Entity(name="thing", type=EntityType.concept),
        object=obj,
        stance=Stance.asserts,
        sentiment=0.0,
        confidence=0.8,
        source_span=SourceSpan(
            episode_id=EPISODE_ID, t_start=t_start, t_end=t_end
        ),
    )


# --------------------------------------------------------------------------- #
# Core regrounding behavior
# --------------------------------------------------------------------------- #
def test_claim_matching_segment_two_gets_its_span_and_speaker() -> None:
    """A claim whose evidence is segment-2's text snaps to segment-2 exactly."""

    transcript = _transcript()
    claim = _claim(
        obj="sourdough bread needs a long cold fermentation to develop flavor",
        label="SPEAKER_00",  # wrong: the audio there is SPEAKER_01
    )

    [grounded] = ground_claims([claim], transcript)

    seg = transcript.segments[1]
    assert grounded.source_span.t_start == seg.t_start == 6.0
    assert grounded.source_span.t_end == seg.t_end == 12.5
    # Speaker label corrected to the diarized label at that time.
    assert grounded.speaker.label == "SPEAKER_01"


def test_claim_matching_a_different_segment_is_regrounded_there() -> None:
    """A claim drawn from segment-3 reground onto segment-3, not segment-2."""

    transcript = _transcript()
    claim = _claim(
        obj="quantum computers will eventually break public-key cryptography",
        label="SPEAKER_00",
    )

    [grounded] = ground_claims([claim], transcript)

    seg = transcript.segments[2]
    assert (grounded.source_span.t_start, grounded.source_span.t_end) == (
        seg.t_start,
        seg.t_end,
    )
    assert grounded.speaker.label == "SPEAKER_02"


def test_first_segment_match_is_regrounded() -> None:
    transcript = _transcript()
    claim = _claim(
        obj="the new electric pickup truck has incredible towing capacity",
        label="SPEAKER_02",
    )

    [grounded] = ground_claims([claim], transcript)

    assert grounded.speaker.label == "SPEAKER_00"
    assert grounded.source_span.t_start == 0.0
    assert grounded.source_span.t_end == 6.0


def test_unmatchable_claim_is_left_unchanged() -> None:
    """Evidence that matches no segment leaves span AND speaker untouched."""

    transcript = _transcript()
    claim = _claim(
        obj="the migratory patterns of arctic terns span both hemispheres",
        label="SPEAKER_99",
        t_start=190.0,
        t_end=200.0,
    )

    [grounded] = ground_claims([claim], transcript)

    # Unchanged: original estimated span and original (foreign) label survive.
    assert grounded.source_span.t_start == 190.0
    assert grounded.source_span.t_end == 200.0
    assert grounded.speaker.label == "SPEAKER_99"
    # And it is the very same object (no needless copy on a non-match).
    assert grounded is claim


# --------------------------------------------------------------------------- #
# Purity / non-mutation
# --------------------------------------------------------------------------- #
def test_inputs_are_not_mutated() -> None:
    transcript = _transcript()
    claim = _claim(
        obj="sourdough bread needs a long cold fermentation to develop flavor",
        label="SPEAKER_00",
        t_start=190.0,
        t_end=200.0,
    )

    ground_claims([claim], transcript)

    # The original claim still carries its pre-grounding span and label.
    assert claim.source_span.t_start == 190.0
    assert claim.source_span.t_end == 200.0
    assert claim.speaker.label == "SPEAKER_00"


def test_other_claim_fields_are_preserved_on_regrounding() -> None:
    transcript = _transcript()
    claim = _claim(
        obj="quantum computers will eventually break public-key cryptography",
        label="SPEAKER_00",
    )
    claim = claim.model_copy(
        update={
            "predicate": Predicate.predicts,
            "stance": Stance.predicts,
            "sentiment": -0.4,
            "confidence": 0.66,
        }
    )

    [grounded] = ground_claims([claim], transcript)

    assert grounded.predicate == Predicate.predicts
    assert grounded.stance == Stance.predicts
    assert grounded.sentiment == -0.4
    assert grounded.confidence == 0.66
    assert grounded.object == claim.object
    assert grounded.subject_entity == claim.subject_entity
    # Episode id on the span is preserved; only the timings move.
    assert grounded.source_span.episode_id == EPISODE_ID


def test_resolved_speaker_id_and_name_are_carried_through() -> None:
    """Regrounding corrects only the diarization label, not a resolved id/name.

    Grounding runs before cross-episode identity stamps a resolved id, but if a
    claim already carries resolved fields the copy must keep them.
    """

    transcript = _transcript()
    claim = _claim(
        obj="the new electric pickup truck has incredible towing capacity",
        label="SPEAKER_02",
    )
    claim = claim.model_copy(
        update={
            "speaker": SpeakerRef(
                label="SPEAKER_02", resolved_id="spk-7", name="Dana"
            )
        }
    )

    [grounded] = ground_claims([claim], transcript)

    assert grounded.speaker.label == "SPEAKER_00"  # corrected
    assert grounded.speaker.resolved_id == "spk-7"  # preserved
    assert grounded.speaker.name == "Dana"  # preserved


# --------------------------------------------------------------------------- #
# Determinism / ties / edge cases
# --------------------------------------------------------------------------- #
def test_ties_break_toward_the_earlier_segment() -> None:
    """When two segments score identically, the earlier one wins, repeatably."""

    segments = [
        TranscriptSegment(
            speaker="SPEAKER_A",
            text="alpha beta gamma delta",
            t_start=0.0,
            t_end=2.0,
        ),
        TranscriptSegment(
            speaker="SPEAKER_B",
            text="alpha beta gamma delta",  # identical text, later in time
            t_start=2.0,
            t_end=4.0,
        ),
    ]
    transcript = Transcript(
        episode_id=EPISODE_ID,
        language="en",
        segments=segments,
        duration_s=4.0,
    )
    claim = _claim(obj="alpha beta gamma delta", label="SPEAKER_Z")

    grounded_a = ground_claim(claim, transcript)
    grounded_b = ground_claim(claim, transcript)

    # Both runs pick the same (earliest) segment.
    assert grounded_a.speaker.label == grounded_b.speaker.label == "SPEAKER_A"
    assert grounded_a.source_span.t_start == 0.0
    assert grounded_a.source_span.t_end == 2.0


def test_empty_transcript_leaves_claims_unchanged() -> None:
    transcript = Transcript(
        episode_id=EPISODE_ID, language="en", segments=[], duration_s=0.0
    )
    claim = _claim(obj="anything at all", label="SPEAKER_5")

    [grounded] = ground_claims([claim], transcript)

    assert grounded is claim


def test_empty_claim_list_returns_empty_list() -> None:
    transcript = _transcript()
    assert ground_claims([], transcript) == []


def test_threshold_gates_a_weak_match() -> None:
    """A borderline match below an explicit high threshold is not regrounded."""

    transcript = _transcript()
    # Shares only the single weak token "the" with segment-1; well below bar.
    claim = _claim(obj="the", label="SPEAKER_X")

    [grounded] = ground_claims([claim], transcript, threshold=0.9)

    assert grounded is claim
    assert grounded.speaker.label == "SPEAKER_X"


def test_score_is_symmetric_and_bounded() -> None:
    """match_score is in [0, 1] and identical on identical inputs (no drift)."""

    a = "quantum computers will break cryptography"
    b = "quantum computers will eventually break public-key cryptography"
    score1 = match_score(a, b)
    score2 = match_score(a, b)
    assert score1 == score2
    assert 0.0 <= score1 <= 1.0
    # Identical strings score a perfect 1.0.
    assert match_score(a, a) == 1.0
    # Disjoint strings score below the default acceptance bar.
    assert match_score("xylophone zeppelin", "submarine artichoke") < DEFAULT_THRESHOLD
