"""Ground each extracted claim to the transcript segment it came from (§7.4).

Two attribution defects surface when an open-weight extractor emits a claim's
``source_span`` and speaker label by *guessing* rather than by pointing at a
real segment:

1. **Estimated spans.** The model is asked for ``[t_start, t_end]`` inside the
   chunk window, but it often snaps to a coarse mental grid (e.g. a round
   ``[190, 200]``) that does not line up with any diarized segment. The eval's
   speaker-verified citation check then reads a span that covers minutes of
   unrelated talk instead of the one utterance the claim came from.
2. **Wrong speaker.** The label the model attaches can disagree with the
   diarization at the cited time — the claim says ``SPEAKER_00`` but the audio
   in that window is ``SPEAKER_01``.

This module fixes both *post hoc*, deterministically, with no model call and no
dependency beyond the standard library. For each claim we take its evidence
text (the verbatim-ish :attr:`~dlogos.schema.ExtractedClaim.object` the claim
asserts) and find the transcript :class:`~dlogos.schema.TranscriptSegment`
whose text best matches it, scoring with :class:`difflib.SequenceMatcher` ratio
blended with token-overlap (Jaccard) so neither a character-level near-miss nor
a bag-of-words coincidence alone decides the match. If the best segment clears
a threshold we rewrite the claim's ``source_span`` to that segment's *real*
``[t_start, t_end]`` and correct the claim's diarization label to that
segment's speaker. If nothing clears the threshold the claim is returned
unchanged — we never invent a span.

The function is **pure**: it does not mutate its inputs (claims are rebuilt via
:meth:`pydantic.BaseModel.model_copy`) and depends only on ``(claims,
transcript)``.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from dlogos.schema import ExtractedClaim, Transcript, TranscriptSegment

# Default acceptance threshold for the blended match score. Below this we treat
# the best segment as "not a real match" and leave the claim untouched rather
# than regrounding it onto a coincidental near-match.
DEFAULT_THRESHOLD = 0.45

# Weight on the sequence-ratio component of the blended score; the remaining
# weight goes to the token-overlap (Jaccard) component. Both are in [0, 1], so
# the blend is too. Sequence ratio rewards a near-verbatim quote; token overlap
# rewards a paraphrase that reuses the same content words — blending them means
# a claim's evidence matches its segment whether the extractor quoted or
# paraphrased, without either signal alone carrying a coincidence over the bar.
_RATIO_WEIGHT = 0.5

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    """Lowercased alphanumeric token set used for the Jaccard component."""

    return set(_TOKEN_RE.findall(text.lower()))


def _token_overlap(a: str, b: str) -> float:
    """Jaccard overlap of the two token sets, in ``[0, 1]``.

    ``0.0`` when either side has no tokens (an empty evidence string can never
    overlap), so an empty ``object`` cannot accidentally score against a
    whitespace segment.
    """

    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def _sequence_ratio(a: str, b: str) -> float:
    """:class:`difflib.SequenceMatcher` ratio over lowercased text, ``[0, 1]``."""

    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def match_score(evidence: str, segment_text: str) -> float:
    """Blended match score between a claim's evidence and a segment's text.

    A weighted mean of the sequence-ratio and token-overlap components, each in
    ``[0, 1]``; the result is in ``[0, 1]``. Pure and symmetric in the
    individual components (the same inputs always yield the same score), so
    ties are reproducible.
    """

    ratio = _sequence_ratio(evidence, segment_text)
    overlap = _token_overlap(evidence, segment_text)
    return _RATIO_WEIGHT * ratio + (1.0 - _RATIO_WEIGHT) * overlap


def _best_segment(
    evidence: str, segments: list[TranscriptSegment]
) -> tuple[int, float]:
    """Index + score of the best-matching segment for ``evidence``.

    Ties are broken deterministically toward the **earlier** segment (lower
    index), so ordering is stable regardless of how Python iterates. Returns
    ``(-1, 0.0)`` when there are no segments to match against.
    """

    best_index = -1
    best_score = 0.0
    for index, segment in enumerate(segments):
        score = match_score(evidence, segment.text)
        # Strict ``>`` keeps the first (earliest) segment on a tie.
        if score > best_score:
            best_score = score
            best_index = index
    return best_index, best_score


def ground_claim(
    claim: ExtractedClaim,
    transcript: Transcript,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> ExtractedClaim:
    """Ground a single claim to its best-matching transcript segment.

    Returns a regrounded **copy** when the best segment's blended score clears
    ``threshold`` — the copy's ``source_span`` carries that segment's real
    ``[t_start, t_end]`` (episode id and any character offset preserved) and its
    speaker ``label`` is corrected to the segment's diarization label. When
    nothing clears the bar (or the transcript has no segments) the original
    claim object is returned unchanged. Never mutates its input.
    """

    index, score = _best_segment(claim.object, transcript.segments)
    if index < 0 or score < threshold:
        return claim

    segment = transcript.segments[index]
    grounded_span = claim.source_span.model_copy(
        update={"t_start": segment.t_start, "t_end": segment.t_end}
    )
    grounded_speaker = claim.speaker.model_copy(
        update={"label": segment.speaker}
    )
    return claim.model_copy(
        update={"source_span": grounded_span, "speaker": grounded_speaker}
    )


def ground_claims(
    claims: list[ExtractedClaim],
    transcript: Transcript,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[ExtractedClaim]:
    """Reground a batch of claims onto the transcript they were extracted from.

    Pure over ``(claims, transcript)``: produces a new list of claims, each
    either a regrounded copy (its ``source_span`` snapped to the real segment
    timings and its speaker label corrected to that segment's diarization label)
    or the original claim untouched when no segment matched its evidence above
    ``threshold``. Input order is preserved one-to-one. Inputs are never
    mutated.

    Parameters
    ----------
    claims:
        The extractor's claims, whose ``object`` text is the evidence matched
        against segment text and whose ``source_span`` / speaker ``label`` are
        the fields corrected on a match.
    transcript:
        The diarized transcript the claims were extracted from; its segments
        supply the authoritative ``[t_start, t_end]`` and speaker labels.
    threshold:
        Minimum blended match score (sequence-ratio blended with token-overlap,
        both in ``[0, 1]``) for a segment to be accepted as a claim's true
        source. Below it the claim is left unchanged.
    """

    return [
        ground_claim(claim, transcript, threshold=threshold) for claim in claims
    ]
