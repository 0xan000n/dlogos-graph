"""dLogos four-arm evaluation harness (spec §9).

This package builds the head-to-head greenlight artifact: a golden query set
with pre-registered answer shapes (:mod:`.golden`), four answer-producing arms
behind one interface (:mod:`.arms`), a reweighted rubric scorer plus the
speaker-verified citation check (:mod:`.rubric`), deterministic blinding of arm
identity (:mod:`.blind`), and inter-rater agreement (:mod:`.agreement`).

``runner.py`` (wiring the arms + scoring into the committed artifact) is owned
by the integration layer and is intentionally NOT part of this subpackage.

Everything here is import-light: pydantic v2 / numpy / stdlib only, so importing
the harness never pulls a heavy/optional dep.
"""

from __future__ import annotations

from dlogos.eval.agreement import (
    agreement_report,
    bin_scores,
    cohen_kappa,
    percent_agreement,
)
from dlogos.eval.arms import (
    ALL_ARMS,
    ARM_DLOGOS,
    ARM_MODEL_ALONE,
    ARM_NAIVE_RAG,
    ARM_WEB_SEARCH,
    Answer,
    Citation,
    ModelAloneArm,
    ModelDLogosArm,
    ModelNaiveRagArm,
    ModelWebSearchArm,
)
from dlogos.eval.blind import (
    BlindedAnswer,
    BlindedQuery,
    UnblindMap,
    blind_answers,
    unblind_scores,
)
from dlogos.eval.golden import (
    AnswerShape,
    Archetype,
    Domain,
    GoldenQuery,
    coverage,
    starter_golden_queries,
)
from dlogos.eval.rubric import (
    DEFAULT_WEIGHTS,
    CitationVerdict,
    Dimension,
    RubricResult,
    count_verified_citations,
    score_answer,
    validate_weights,
    verify_citation,
)

__all__ = [
    # golden
    "AnswerShape",
    "Archetype",
    "Domain",
    "GoldenQuery",
    "coverage",
    "starter_golden_queries",
    # arms
    "ALL_ARMS",
    "ARM_DLOGOS",
    "ARM_MODEL_ALONE",
    "ARM_NAIVE_RAG",
    "ARM_WEB_SEARCH",
    "Answer",
    "Citation",
    "ModelAloneArm",
    "ModelDLogosArm",
    "ModelNaiveRagArm",
    "ModelWebSearchArm",
    # rubric
    "DEFAULT_WEIGHTS",
    "CitationVerdict",
    "Dimension",
    "RubricResult",
    "count_verified_citations",
    "score_answer",
    "validate_weights",
    "verify_citation",
    # blind
    "BlindedAnswer",
    "BlindedQuery",
    "UnblindMap",
    "blind_answers",
    "unblind_scores",
    # agreement
    "agreement_report",
    "bin_scores",
    "cohen_kappa",
    "percent_agreement",
]
