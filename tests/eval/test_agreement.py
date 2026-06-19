"""Tests for inter-rater agreement (Cohen's kappa) on known inputs (spec §9)."""

from __future__ import annotations

import pytest

from dlogos.eval.agreement import (
    agreement_report,
    bin_scores,
    cohen_kappa,
    percent_agreement,
)


def test_perfect_agreement_is_kappa_one() -> None:
    a = ["hi", "lo", "hi", "mid", "lo"]
    b = list(a)
    assert cohen_kappa(a, b) == pytest.approx(1.0)
    assert percent_agreement(a, b) == pytest.approx(1.0)


def test_known_kappa_value() -> None:
    # Classic worked example. Two raters, binary labels, 50 items:
    #   both yes = 20, both no = 15, a=yes/b=no = 5, a=no/b=yes = 10.
    # p_o = (20+15)/50 = 0.70
    # p_yes_a = 25/50=0.5, p_yes_b = 30/50=0.6 -> 0.30
    # p_no_a  = 25/50=0.5, p_no_b  = 20/50=0.4 -> 0.20 ; p_e = 0.50
    # kappa = (0.70-0.50)/(1-0.50) = 0.40
    a = ["yes"] * 20 + ["yes"] * 5 + ["no"] * 15 + ["no"] * 10
    b = ["yes"] * 20 + ["no"] * 5 + ["no"] * 15 + ["yes"] * 10
    assert percent_agreement(a, b) == pytest.approx(0.70)
    assert cohen_kappa(a, b) == pytest.approx(0.40)


def test_chance_level_agreement_is_near_zero() -> None:
    # Independent-looking labels arranged so observed ~= expected agreement.
    a = ["x", "y", "x", "y"]
    b = ["x", "x", "y", "y"]
    # p_o = 2/4 = 0.5 ; marginals all 0.5 -> p_e = 0.5 ; kappa = 0.
    assert cohen_kappa(a, b) == pytest.approx(0.0)


def test_worse_than_chance_is_negative() -> None:
    a = ["x", "x", "y", "y"]
    b = ["y", "y", "x", "x"]
    assert cohen_kappa(a, b) < 0.0


def test_degenerate_single_category_agreeing_is_one() -> None:
    a = ["same", "same", "same"]
    b = ["same", "same", "same"]
    assert cohen_kappa(a, b) == pytest.approx(1.0)


def test_bin_scores_buckets_continuous_totals() -> None:
    scores = [0.1, 0.33, 0.5, 0.66, 0.9]
    bands = bin_scores(scores, edges=[0.33, 0.66])
    # Edge falls into the upper band (side="right").
    assert bands == [0, 1, 1, 2, 2]


def test_kappa_on_binned_rubric_totals() -> None:
    # Two raters score the same answers; bin into ordinal bands, then kappa.
    rater_a_totals = [0.2, 0.8, 0.5, 0.9, 0.1]
    rater_b_totals = [0.25, 0.75, 0.4, 0.85, 0.15]
    edges = [0.33, 0.66]
    ba = bin_scores(rater_a_totals, edges)
    bb = bin_scores(rater_b_totals, edges)
    # They bucket identically here -> perfect agreement on bands.
    assert ba == bb
    assert cohen_kappa(ba, bb) == pytest.approx(1.0)


def test_report_bundles_n_percent_and_kappa() -> None:
    a = ["hi", "lo", "hi"]
    b = ["hi", "lo", "lo"]
    rep = agreement_report(a, b)
    assert rep["n"] == 3.0
    assert rep["percent_agreement"] == pytest.approx(2 / 3)
    assert "cohen_kappa" in rep


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError):
        cohen_kappa(["a"], ["a", "b"])


def test_empty_raises() -> None:
    with pytest.raises(ValueError):
        percent_agreement([], [])
