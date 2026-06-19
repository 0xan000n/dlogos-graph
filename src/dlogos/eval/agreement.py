"""Inter-rater agreement for the second-rater credibility control (spec §9).

A second independent rater scores an overlapping subset; we report agreement so
the artifact is not self-graded. The headline statistic is **Cohen's kappa**
for categorical/ordinal labels, which corrects observed agreement for the
agreement expected by chance.

For continuous rubric totals (a [0, 1] score), kappa needs categories: use
:func:`bin_scores` to bucket scores into ordinal bands first, then
:func:`cohen_kappa`. :func:`percent_agreement` is reported alongside as the
uncorrected baseline.

Import-light: numpy + stdlib only.
"""

from __future__ import annotations

import numpy as np


def percent_agreement(rater_a: list, rater_b: list) -> float:
    """Raw fraction of items on which the two raters assigned the same label.

    Both lists must be the same non-zero length. Returns a value in [0, 1].
    """

    _check_pair(rater_a, rater_b)
    matches = sum(1 for x, y in zip(rater_a, rater_b) if x == y)
    return matches / len(rater_a)


def cohen_kappa(rater_a: list, rater_b: list) -> float:
    """Cohen's kappa for two raters over a shared set of items.

    ``rater_a`` / ``rater_b`` are aligned label sequences (same length, one
    entry per shared item). Labels may be any hashable category.

    Returns kappa in [-1, 1]: 1.0 = perfect agreement, 0.0 = agreement no
    better than chance, negative = worse than chance. When both raters assign a
    single constant label to every item (no observable variation) and they
    agree, kappa is defined here as 1.0 (perfect agreement); if that lone label
    differs, kappa is 0.0.
    """

    _check_pair(rater_a, rater_b)
    n = len(rater_a)

    categories = sorted(set(rater_a) | set(rater_b), key=repr)
    index = {c: i for i, c in enumerate(categories)}
    k = len(categories)

    conf = np.zeros((k, k), dtype=float)
    for x, y in zip(rater_a, rater_b):
        conf[index[x], index[y]] += 1.0

    p_observed = np.trace(conf) / n

    row_marg = conf.sum(axis=1) / n
    col_marg = conf.sum(axis=0) / n
    p_expected = float(np.dot(row_marg, col_marg))

    if np.isclose(p_expected, 1.0):
        # Degenerate: both raters used a single category. Perfect observed
        # agreement is perfect agreement; otherwise no agreement.
        return 1.0 if np.isclose(p_observed, 1.0) else 0.0

    return float((p_observed - p_expected) / (1.0 - p_expected))


def bin_scores(scores: list[float], edges: list[float]) -> list[int]:
    """Bucket continuous [0, 1] scores into ordinal bands for kappa.

    ``edges`` are the interior cut points (e.g. ``[0.33, 0.66]`` -> bands
    0/1/2). A score equal to an edge falls into the upper band. Returns the band
    index per score.
    """

    sorted_edges = sorted(edges)
    return [int(np.searchsorted(sorted_edges, s, side="right")) for s in scores]


def agreement_report(rater_a: list, rater_b: list) -> dict[str, float]:
    """Convenience bundle: percent agreement + Cohen's kappa + n."""

    return {
        "n": float(len(rater_a)),
        "percent_agreement": percent_agreement(rater_a, rater_b),
        "cohen_kappa": cohen_kappa(rater_a, rater_b),
    }


def _check_pair(rater_a: list, rater_b: list) -> None:
    if len(rater_a) != len(rater_b):
        raise ValueError(
            f"rater label lists must be equal length: "
            f"{len(rater_a)} != {len(rater_b)}"
        )
    if not rater_a:
        raise ValueError("cannot compute agreement on zero items")
