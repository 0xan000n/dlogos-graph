"""Golden query set for the four-arm head-to-head eval (spec §9).

A :class:`GoldenQuery` carries the query text plus a *pre-registered*
``answer_shape`` — the expected speakers / stance / timeframe / citations,
frozen **before** any arm output is seen so scoring is against a fixed target
(spec §9, "Pre-registered good-answer shapes"). The starter set spreads across
the eight knowledge-economy domains and the five query archetypes.

Framing (spec §9): 12-15 queries cannot cover 8 domains x 5 archetypes (40
cells). This set is an **existence proof across a spread**, not a generalization
proof. :func:`coverage` reports the (domain, archetype) cells touched so the
spread is auditable rather than asserted.

Import-light: only pydantic v2 + stdlib. No heavy/optional deps.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Archetype(str, Enum):
    """The five query archetypes the eval samples (spec §9)."""

    temporal_consensus = "temporal_consensus"
    belief_tracking = "belief_tracking"
    contradiction = "contradiction"
    consensus_outlier = "consensus_outlier"
    provenance = "provenance"


class Domain(str, Enum):
    """The eight knowledge-economy domains the corpus spans (spec §3/§4)."""

    science = "science"
    technology = "technology"
    philosophy = "philosophy"
    engineering = "engineering"
    product = "product"
    business = "business"
    finance = "finance"
    politics = "politics"


class AnswerShape(BaseModel):
    """The pre-registered expected shape of a good answer (spec §9).

    Written down and frozen before any arm output is generated. Every field is
    optional because not all archetypes constrain all dimensions (a provenance
    query cares about citations; a belief-tracking query cares about a position
    *changing*). The rubric scorer reads these as the fixed target.
    """

    model_config = ConfigDict(extra="forbid")

    expected_speakers: list[str] = Field(
        default_factory=list,
        description="Canonical speaker ids/names expected to appear, attributed.",
    )
    expected_subjects: list[str] = Field(
        default_factory=list,
        description="Canonical subject ids/names the answer should be about.",
    )
    expected_stance_shift: bool = Field(
        default=False,
        description="True if a good answer must describe a position MOVING over time.",
    )
    timeframe_start: str | None = Field(
        default=None, description="ISO date; earliest event-time a good answer covers."
    )
    timeframe_end: str | None = Field(
        default=None, description="ISO date; latest event-time a good answer covers."
    )
    min_attributed_sources: int = Field(
        default=1,
        ge=1,
        description="How many distinct attributed speakers a good answer must cite.",
    )
    requires_citations: bool = Field(
        default=True,
        description="Whether a good answer must carry episode+timestamp citations.",
    )
    notes: str = Field(
        default="", description="Free-text rationale, frozen at pre-registration."
    )


class GoldenQuery(BaseModel):
    """One hero query in the golden set, with its frozen good-answer shape."""

    model_config = ConfigDict(extra="forbid")

    id: str
    archetype: Archetype
    domain: Domain
    query_text: str
    pre_registered_answer_shape: AnswerShape
    deep_tier: bool = Field(
        default=False,
        description="From the ~18-24mo deep subset (§4b) that carries temporal shifts.",
    )


def _q(
    qid: str,
    archetype: Archetype,
    domain: Domain,
    text: str,
    shape: AnswerShape,
    *,
    deep_tier: bool = False,
) -> GoldenQuery:
    return GoldenQuery(
        id=qid,
        archetype=archetype,
        domain=domain,
        query_text=text,
        pre_registered_answer_shape=shape,
        deep_tier=deep_tier,
    )


# --------------------------------------------------------------------------- #
# Starter golden set — 15 hero queries spanning 8 domains x 5 archetypes.
# Temporal/belief queries are flagged deep_tier (need the 18-24mo window, §9).
# --------------------------------------------------------------------------- #
def starter_golden_queries() -> list[GoldenQuery]:
    """The pre-registered 15-query starter set (spec §9).

    Constructed fresh each call (no shared mutable module state). The set
    deliberately hits every archetype at least twice and every domain at least
    once, so :func:`coverage` shows a real spread.
    """

    return [
        _q(
            "gq-01",
            Archetype.temporal_consensus,
            Domain.technology,
            "Who's been discussing the limits of LLM scaling in the last "
            "18 months, and how has the framing shifted?",
            AnswerShape(
                expected_subjects=["scaling laws", "large language models"],
                expected_stance_shift=True,
                timeframe_start="2024-12-01",
                timeframe_end="2026-06-01",
                min_attributed_sources=3,
                notes="Should show a move from pure-scale optimism toward "
                "data/efficiency framing across several attributed speakers.",
            ),
            deep_tier=True,
        ),
        _q(
            "gq-02",
            Archetype.temporal_consensus,
            Domain.finance,
            "How has the consensus on a 2025-2026 soft landing for the US "
            "economy moved over the past 18 months?",
            AnswerShape(
                expected_subjects=["soft landing", "US economy"],
                expected_stance_shift=True,
                timeframe_start="2024-12-01",
                timeframe_end="2026-06-01",
                min_attributed_sources=3,
                notes="A good answer tracks the shift, not just the latest take.",
            ),
            deep_tier=True,
        ),
        _q(
            "gq-03",
            Archetype.belief_tracking,
            Domain.business,
            "What does this recurring guest believe about remote vs. in-office "
            "work, and has their position changed?",
            AnswerShape(
                expected_speakers=["spk-recurring-guest"],
                expected_subjects=["remote work"],
                expected_stance_shift=True,
                timeframe_start="2024-12-01",
                timeframe_end="2026-06-01",
                min_attributed_sources=1,
                notes="Per-speaker belief over time; guest, not only host.",
            ),
            deep_tier=True,
        ),
        _q(
            "gq-04",
            Archetype.belief_tracking,
            Domain.science,
            "What is this host's stated view on the replication crisis, and "
            "how has it evolved across episodes?",
            AnswerShape(
                expected_speakers=["spk-host"],
                expected_subjects=["replication crisis"],
                expected_stance_shift=True,
                timeframe_start="2024-12-01",
                timeframe_end="2026-06-01",
                min_attributed_sources=1,
            ),
            deep_tier=True,
        ),
        _q(
            "gq-05",
            Archetype.contradiction,
            Domain.philosophy,
            "Who disagrees with the claim that AI systems can be conscious, "
            "and what do they argue instead?",
            AnswerShape(
                expected_subjects=["machine consciousness"],
                min_attributed_sources=2,
                notes="Must surface at least one disputes/disagrees stance with "
                "the counter-argument, attributed.",
            ),
        ),
        _q(
            "gq-06",
            Archetype.contradiction,
            Domain.engineering,
            "Who disputes that microservices are the right default "
            "architecture, and on what grounds?",
            AnswerShape(
                expected_subjects=["microservices"],
                min_attributed_sources=2,
            ),
        ),
        _q(
            "gq-07",
            Archetype.consensus_outlier,
            Domain.technology,
            "What's the emerging consensus on open-weight vs. closed frontier "
            "models, and who's the contrarian?",
            AnswerShape(
                expected_subjects=["open-weight models", "frontier models"],
                min_attributed_sources=3,
                notes="Identify both the majority position and the named outlier.",
            ),
            deep_tier=True,
        ),
        _q(
            "gq-08",
            Archetype.consensus_outlier,
            Domain.finance,
            "What's the prevailing view on Bitcoin as a treasury reserve "
            "asset, and who dissents?",
            AnswerShape(
                expected_subjects=["Bitcoin", "treasury reserve"],
                min_attributed_sources=3,
            ),
        ),
        _q(
            "gq-09",
            Archetype.provenance,
            Domain.product,
            "Where was the failure of feature-factory roadmaps discussed, with "
            "episode and timestamp?",
            AnswerShape(
                expected_subjects=["feature factory", "product roadmap"],
                min_attributed_sources=1,
                requires_citations=True,
                notes="Provenance: a real episode + timestamp span is the answer.",
            ),
        ),
        _q(
            "gq-10",
            Archetype.provenance,
            Domain.politics,
            "Where was the durability of the 2024-2026 industrial-policy "
            "consensus discussed, with episode and timestamp?",
            AnswerShape(
                expected_subjects=["industrial policy"],
                min_attributed_sources=1,
                requires_citations=True,
            ),
        ),
        _q(
            "gq-11",
            Archetype.temporal_consensus,
            Domain.science,
            "How has expert framing of GLP-1 drugs shifted over the past "
            "18 months across the shows?",
            AnswerShape(
                expected_subjects=["GLP-1", "Ozempic"],
                expected_stance_shift=True,
                timeframe_start="2024-12-01",
                timeframe_end="2026-06-01",
                min_attributed_sources=3,
            ),
            deep_tier=True,
        ),
        _q(
            "gq-12",
            Archetype.belief_tracking,
            Domain.finance,
            "What does this recurring guest believe about the Fed's rate path, "
            "and has the view changed?",
            AnswerShape(
                expected_speakers=["spk-recurring-guest"],
                expected_subjects=["federal funds rate"],
                expected_stance_shift=True,
                timeframe_start="2024-12-01",
                timeframe_end="2026-06-01",
            ),
            deep_tier=True,
        ),
        _q(
            "gq-13",
            Archetype.contradiction,
            Domain.business,
            "Who pushes back on the 'AI will replace most knowledge work soon' "
            "claim, and what's their counter-case?",
            AnswerShape(
                expected_subjects=["AI automation", "knowledge work"],
                min_attributed_sources=2,
            ),
        ),
        _q(
            "gq-14",
            Archetype.consensus_outlier,
            Domain.philosophy,
            "What's the consensus on longtermism as an ethical framework, and "
            "who's the prominent skeptic?",
            AnswerShape(
                expected_subjects=["longtermism"],
                min_attributed_sources=3,
            ),
        ),
        _q(
            "gq-15",
            Archetype.provenance,
            Domain.engineering,
            "Where was the claim that formal verification is finally practical "
            "made, with episode and timestamp?",
            AnswerShape(
                expected_subjects=["formal verification"],
                min_attributed_sources=1,
                requires_citations=True,
            ),
        ),
    ]


def coverage(queries: list[GoldenQuery]) -> dict[str, set[str]]:
    """Report the spread: which domains and archetypes the set touches.

    Returns a dict with ``"domains"`` and ``"archetypes"`` sets of the values
    present. Used to make the "existence proof across a spread" claim (spec §9)
    auditable rather than asserted.
    """

    return {
        "domains": {q.domain.value for q in queries},
        "archetypes": {q.archetype.value for q in queries},
    }
