"""Controlled predicate vocabulary helpers (§7.4, decision (iii)).

The :class:`~dlogos.schema.Predicate` enum is a *closed* vocabulary enforced at
extraction time — there is no separate post-hoc normalization pass. In
practice an open-weight model still occasionally emits a near-synonym or a
surface variant (``"is positive about"`` instead of ``rates_positive``,
``"forecast"`` instead of ``forecasts``). These helpers map such free output
onto the enum, and — crucially — **reject anything unmappable** rather than
silently coercing it to a wrong relation. A rejected predicate drops the claim;
a wrongly coerced one corrupts consensus aggregation, which is worse.

Mapping order (deterministic, no model call):

1. **Exact enum value** — already canonical.
2. **Synonym table** — a curated alias → canonical map (normalized).
3. **Nearest-by-rule** — light morphological / substring rules over the
   normalized form (drop a trailing ``s``, collapse separators, look for a
   canonical token as a word).

If none match, raise :class:`PredicateMappingError`.
"""

from __future__ import annotations

import re

from dlogos.schema import Predicate


class PredicateMappingError(ValueError):
    """Raised when a free predicate string cannot be mapped to the vocabulary."""

    def __init__(self, raw: str) -> None:
        self.raw = raw
        super().__init__(
            f"predicate {raw!r} is not in the controlled vocabulary and "
            f"has no mappable synonym"
        )


# Curated synonym table: free surface form -> canonical Predicate.
# Keys are matched after normalization (lowercased, separators -> single space,
# stripped). Keep this tight and auditable — broad fuzzy matching is exactly
# what corrupts the relation vocabulary.
PREDICATE_SYNONYMS: dict[str, Predicate] = {
    # expects
    "expect": Predicate.expects,
    "anticipates": Predicate.expects,
    "anticipate": Predicate.expects,
    "looks forward to": Predicate.expects,
    # rates_positive
    "rates positive": Predicate.rates_positive,
    "rates positively": Predicate.rates_positive,
    "is positive about": Predicate.rates_positive,
    "positive on": Predicate.rates_positive,
    "praises": Predicate.rates_positive,
    "likes": Predicate.rates_positive,
    "favorable": Predicate.rates_positive,
    "bullish": Predicate.rates_positive,
    "bullish on": Predicate.rates_positive,
    # rates_negative
    "rates negative": Predicate.rates_negative,
    "rates negatively": Predicate.rates_negative,
    "is negative about": Predicate.rates_negative,
    "negative on": Predicate.rates_negative,
    "dislikes": Predicate.rates_negative,
    "unfavorable": Predicate.rates_negative,
    "bearish": Predicate.rates_negative,
    "bearish on": Predicate.rates_negative,
    # predicts
    "predict": Predicate.predicts,
    "prediction": Predicate.predicts,
    # recommends
    "recommend": Predicate.recommends,
    "recommendation": Predicate.recommends,
    "suggests": Predicate.recommends,
    "advises": Predicate.recommends,
    # criticizes
    "criticize": Predicate.criticizes,
    "criticises": Predicate.criticizes,
    "critiques": Predicate.criticizes,
    "knocks": Predicate.criticizes,
    # compares
    "compare": Predicate.compares,
    "comparison": Predicate.compares,
    "contrasts": Predicate.compares,
    # explains
    "explain": Predicate.explains,
    "explanation": Predicate.explains,
    "clarifies": Predicate.explains,
    "describes": Predicate.explains,
    # attributes
    "attribute": Predicate.attributes,
    "attribution": Predicate.attributes,
    "credits": Predicate.attributes,
    "ascribes": Predicate.attributes,
    # forecasts
    "forecast": Predicate.forecasts,
    "projects": Predicate.forecasts,
    "projection": Predicate.forecasts,
    # endorses
    "endorse": Predicate.endorses,
    "endorsement": Predicate.endorses,
    "backs": Predicate.endorses,
    "supports": Predicate.endorses,
    # rejects
    "reject": Predicate.rejects,
    "rejection": Predicate.rejects,
    "denies": Predicate.rejects,
    "dismisses": Predicate.rejects,
    # questions
    "question": Predicate.questions,
    "questioning": Predicate.questions,
    "doubts": Predicate.questions,
    "casts doubt on": Predicate.questions,
    # agrees
    "agree": Predicate.agrees,
    "agreement": Predicate.agrees,
    "concurs": Predicate.agrees,
    "agrees with": Predicate.agrees,
    # disagrees
    "disagree": Predicate.disagrees,
    "disagreement": Predicate.disagrees,
    "disputes": Predicate.disagrees,
    "disagrees with": Predicate.disagrees,
}

# Canonical enum values, kept for the nearest-by-rule pass.
_CANONICAL_VALUES: set[str] = {p.value for p in Predicate}
# Tokens that appear in canonical values, for word-level matching.
_CANONICAL_BY_TOKEN: dict[str, Predicate] = {}
for _p in Predicate:
    for _tok in _p.value.split("_"):
        # Only register unambiguous single-token handles ("criticizes",
        # "forecasts", etc.). "rates"/"positive"/"negative" are ambiguous on
        # their own and are intentionally left out so a bare "rates" does not
        # silently win.
        if _tok in {"rates", "positive", "negative"}:
            continue
        _CANONICAL_BY_TOKEN.setdefault(_tok, _p)


def _normalize(raw: str) -> str:
    """Lowercase, collapse separators (``_``/``-``/whitespace) to one space."""

    return re.sub(r"[\s_\-]+", " ", raw.strip().lower()).strip()


def try_map_predicate(raw: str) -> Predicate | None:
    """Map a free predicate string onto the vocabulary, or ``None`` if unmappable.

    Pure, deterministic, no model call. See module docstring for the ordering.
    """

    if raw is None:
        return None
    norm = _normalize(raw)
    if not norm:
        return None

    # 1. Exact canonical value (after normalization, "_" became spaces).
    compact = norm.replace(" ", "_")
    if compact in _CANONICAL_VALUES:
        return Predicate(compact)

    # 2. Curated synonym table.
    if norm in PREDICATE_SYNONYMS:
        return PREDICATE_SYNONYMS[norm]

    # 3. Nearest-by-rule.
    #    3a. Singularize a trailing "s" then retry exact + synonyms.
    if norm.endswith("s"):
        singular = norm[:-1]
        s_compact = singular.replace(" ", "_")
        if s_compact in _CANONICAL_VALUES:
            return Predicate(s_compact)
        if singular in PREDICATE_SYNONYMS:
            return PREDICATE_SYNONYMS[singular]
    #    3b. Pluralize and retry (handles "forecast" -> "forecasts").
    plural = norm + "s"
    p_compact = plural.replace(" ", "_")
    if p_compact in _CANONICAL_VALUES:
        return Predicate(p_compact)

    #    3c. Word-level: a canonical single-token handle appears as a word.
    words = norm.split(" ")
    for w in words:
        if w in _CANONICAL_BY_TOKEN:
            return _CANONICAL_BY_TOKEN[w]
        if w.endswith("s") and w[:-1] in _CANONICAL_BY_TOKEN:
            return _CANONICAL_BY_TOKEN[w[:-1]]

    return None


def map_predicate(raw: str) -> Predicate:
    """Map a free predicate string onto the vocabulary or raise.

    Raises
    ------
    PredicateMappingError
        If ``raw`` is not in the controlled vocabulary and no synonym or rule
        maps it. Callers drop the claim rather than coerce it.
    """

    mapped = try_map_predicate(raw)
    if mapped is None:
        raise PredicateMappingError(raw)
    return mapped
