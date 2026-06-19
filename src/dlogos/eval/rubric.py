"""The reweighted rubric scorer + speaker-verified citation check (spec §9).

The rubric deliberately **elevates** temporal-consensus synthesis across
attributed sources and **demotes** recency / couldn't-have-known, because the
web-search arm can also be recent -- a recency-only win is not a *structural*
win (spec §9, scoring table). Weights live in :data:`DEFAULT_WEIGHTS`; the
scorer is a weighted sum of per-dimension [0, 1] scores, returning a normalized
[0, 1] total so arms are comparable.

The speaker-verified citation check (:func:`verify_citation`) is the teeth on
the §11 top risk -- diarization error -> confident *misattribution*. A citation
passes ONLY if the speaker the answer attributes the span to is the one actually
speaking at that timestamp in the diarized transcript -- not merely that the
topic appears nearby.

Import-light: pydantic v2 + stdlib only. The diarized transcript is the shared
:class:`~dlogos.schema.Transcript`; ``resolved_id`` on its segments is supplied
by the speaker-identity stage. For the check we accept a per-episode map of
``segment -> resolved speaker id`` so this module stays decoupled from how
resolution attaches ids.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from dlogos.eval.arms import Answer, Citation
from dlogos.schema import Transcript


# --------------------------------------------------------------------------- #
# Rubric dimensions + weights
# --------------------------------------------------------------------------- #
class Dimension(str, Enum):
    """Scored dimensions (spec §9 scoring table)."""

    temporal_consensus_synthesis = "temporal_consensus_synthesis"
    attribution_precision = "attribution_precision"
    provenance_integrity = "provenance_integrity"
    recency = "recency"
    couldnt_have_known = "couldnt_have_known"


# Weights encode the reweighting intent: temporal-consensus synthesis is the
# heaviest (the dimension that survives web-search and vector-RAG competitors);
# attribution + provenance are high; recency and couldn't-have-known are demoted.
DEFAULT_WEIGHTS: dict[Dimension, float] = {
    Dimension.temporal_consensus_synthesis: 0.40,
    Dimension.attribution_precision: 0.25,
    Dimension.provenance_integrity: 0.20,
    Dimension.recency: 0.075,
    Dimension.couldnt_have_known: 0.075,
}

# Dimensions whose weight the reweighting deliberately demotes (spec §9). Used
# by a sanity assertion: every demoted dim must weigh strictly less than every
# elevated/high dim.
DEMOTED_DIMENSIONS = frozenset(
    {Dimension.recency, Dimension.couldnt_have_known}
)
ELEVATED_DIMENSION = Dimension.temporal_consensus_synthesis


class DimensionScore(BaseModel):
    """A raw [0, 1] judgment on one dimension, with an optional note."""

    model_config = ConfigDict(extra="forbid")

    dimension: Dimension
    raw: float = Field(ge=0.0, le=1.0)
    note: str = Field(default="")


class RubricResult(BaseModel):
    """The scored outcome for one answer."""

    model_config = ConfigDict(extra="forbid")

    total: float = Field(
        ge=0.0, le=1.0, description="Weighted, weight-normalized score in [0, 1]."
    )
    weighted: dict[Dimension, float] = Field(
        description="Per-dimension weight*raw contribution (pre-normalization)."
    )
    raw: dict[Dimension, float] = Field(description="Per-dimension raw [0,1] scores.")
    verified_citations: int = Field(
        ge=0, description="Citations that passed the speaker-verified check."
    )
    rejected_citations: int = Field(
        ge=0, description="Citations rejected (wrong speaker / no span)."
    )


def validate_weights(weights: dict[Dimension, float]) -> None:
    """Assert the weights actually implement the reweighting intent (spec §9).

    - All five dimensions present and non-negative.
    - The elevated dimension is the single largest weight.
    - Every demoted dimension weighs strictly less than every non-demoted one.

    Raises ``ValueError`` if the weights do not honor the rubric's intent. This
    is what the weighting-math test asserts against.
    """

    missing = set(Dimension) - set(weights)
    if missing:
        raise ValueError(f"weights missing dimensions: {sorted(d.value for d in missing)}")
    if any(w < 0 for w in weights.values()):
        raise ValueError("weights must be non-negative")
    if sum(weights.values()) <= 0:
        raise ValueError("weights must sum to a positive number")

    elevated_w = weights[ELEVATED_DIMENSION]
    if any(
        w >= elevated_w for d, w in weights.items() if d is not ELEVATED_DIMENSION
    ):
        raise ValueError("elevated dimension must carry the single largest weight")

    non_demoted = [w for d, w in weights.items() if d not in DEMOTED_DIMENSIONS]
    demoted = [w for d, w in weights.items() if d in DEMOTED_DIMENSIONS]
    if max(demoted) >= min(non_demoted):
        raise ValueError(
            "every demoted dimension must weigh strictly less than every "
            "non-demoted dimension"
        )


# --------------------------------------------------------------------------- #
# Speaker-verified citation check (spec §9 / §11 top risk)
# --------------------------------------------------------------------------- #
class CitationVerdict(BaseModel):
    """Outcome of verifying one citation against the diarized transcript."""

    model_config = ConfigDict(extra="forbid")

    passed: bool
    reason: str
    actual_speaker_id: str | None = Field(
        default=None, description="Who is actually speaking at the cited span."
    )


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Length of the temporal overlap between [a_start,a_end] and [b_start,b_end]."""

    lo = max(a_start, b_start)
    hi = min(a_end, b_end)
    return max(0.0, hi - lo)


def verify_citation(
    citation: Citation,
    transcript: Transcript,
    segment_speaker_ids: dict[int, str],
    *,
    min_overlap_s: float = 0.0,
) -> CitationVerdict:
    """Confirm the cited span is actually spoken by the attributed speaker.

    A citation passes ONLY if the diarized segment(s) covering the cited
    ``[t_start, t_end]`` are attributed (via ``segment_speaker_ids``, keyed by
    segment index) to the SAME ``speaker_id`` the answer claims. Topic presence
    is irrelevant -- this is a *who-is-speaking* check (spec §9).

    The "actual" speaker is the one whose segment overlaps the cited span the
    most. ``min_overlap_s`` rejects citations whose span barely grazes any
    segment (a hallucinated timestamp).

    Returns a :class:`CitationVerdict`; never raises on a mismatch.
    """

    if citation.episode_id != transcript.episode_id:
        return CitationVerdict(
            passed=False,
            reason=(
                f"citation episode {citation.episode_id!r} != transcript "
                f"{transcript.episode_id!r}"
            ),
        )

    best_idx: int | None = None
    best_overlap = 0.0
    for idx, seg in enumerate(transcript.segments):
        ov = _overlaps(citation.t_start, citation.t_end, seg.t_start, seg.t_end)
        if ov > best_overlap:
            best_overlap = ov
            best_idx = idx

    if best_idx is None or best_overlap <= min_overlap_s:
        return CitationVerdict(
            passed=False,
            reason="no diarized segment overlaps the cited timestamp span",
        )

    actual = segment_speaker_ids.get(best_idx)
    if actual is None:
        return CitationVerdict(
            passed=False,
            reason=f"no resolved speaker for segment {best_idx}",
        )

    if actual != citation.speaker_id:
        return CitationVerdict(
            passed=False,
            reason=(
                f"misattribution: answer credits {citation.speaker_id!r} but "
                f"{actual!r} is speaking at the cited span"
            ),
            actual_speaker_id=actual,
        )

    return CitationVerdict(
        passed=True,
        reason="speaker at cited timestamp matches the attribution",
        actual_speaker_id=actual,
    )


def count_verified_citations(
    answer: Answer,
    transcripts: dict[str, Transcript],
    segment_speaker_ids: dict[str, dict[int, str]],
    *,
    min_overlap_s: float = 0.0,
) -> tuple[int, int]:
    """Verify every citation in an answer; return (verified, rejected) counts.

    ``transcripts`` / ``segment_speaker_ids`` are keyed by ``episode_id``. A
    citation whose episode is unknown is rejected (cannot be verified -> does not
    count toward attribution precision).
    """

    verified = 0
    rejected = 0
    for cit in answer.citations:
        transcript = transcripts.get(cit.episode_id)
        ids = segment_speaker_ids.get(cit.episode_id, {})
        if transcript is None:
            rejected += 1
            continue
        verdict = verify_citation(
            cit, transcript, ids, min_overlap_s=min_overlap_s
        )
        if verdict.passed:
            verified += 1
        else:
            rejected += 1
    return verified, rejected


# --------------------------------------------------------------------------- #
# Scorer
# --------------------------------------------------------------------------- #
def score_answer(
    answer: Answer,
    dimension_scores: dict[Dimension, float],
    *,
    transcripts: dict[str, Transcript] | None = None,
    segment_speaker_ids: dict[str, dict[int, str]] | None = None,
    weights: dict[Dimension, float] | None = None,
    min_overlap_s: float = 0.0,
    cap_attribution_to_verified: bool = True,
) -> RubricResult:
    """Score one answer with the reweighted rubric.

    ``dimension_scores`` are the rater's raw [0, 1] judgments per dimension
    (typically produced by a blinded human/LLM rater). When transcripts are
    supplied, the speaker-verified citation check runs and, if
    ``cap_attribution_to_verified`` is set, the attribution-precision raw score
    is **capped** by the fraction of citations that pass the check -- so an
    answer with a confidently misattributed citation cannot score full
    attribution credit even if a rater was fooled by topic presence (spec §9).

    Returns a :class:`RubricResult` with the normalized total in [0, 1].
    """

    weights = weights or DEFAULT_WEIGHTS
    validate_weights(weights)

    raw = {dim: float(dimension_scores.get(dim, 0.0)) for dim in Dimension}
    for dim, val in raw.items():
        if not 0.0 <= val <= 1.0:
            raise ValueError(f"raw score for {dim.value} out of [0,1]: {val}")

    verified = rejected = 0
    if transcripts is not None and segment_speaker_ids is not None:
        verified, rejected = count_verified_citations(
            answer,
            transcripts,
            segment_speaker_ids,
            min_overlap_s=min_overlap_s,
        )
        total_cited = verified + rejected
        if cap_attribution_to_verified and total_cited > 0:
            verified_fraction = verified / total_cited
            raw[Dimension.attribution_precision] = min(
                raw[Dimension.attribution_precision], verified_fraction
            )

    weighted = {dim: weights[dim] * raw[dim] for dim in Dimension}
    weight_sum = sum(weights.values())
    total = sum(weighted.values()) / weight_sum

    return RubricResult(
        total=total,
        weighted=weighted,
        raw=raw,
        verified_citations=verified,
        rejected_citations=rejected,
    )
