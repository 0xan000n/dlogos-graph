"""Tests for the controlled predicate vocabulary mapping."""

from __future__ import annotations

import pytest

from dlogos.extraction.predicates import (
    PredicateMappingError,
    map_predicate,
    try_map_predicate,
)
from dlogos.schema import Predicate


@pytest.mark.parametrize("p", list(Predicate))
def test_canonical_values_map_to_themselves(p: Predicate) -> None:
    assert map_predicate(p.value) == p


def test_case_and_separator_insensitive() -> None:
    assert map_predicate("Rates_Positive") == Predicate.rates_positive
    assert map_predicate("rates positive") == Predicate.rates_positive
    assert map_predicate("RATES-POSITIVE") == Predicate.rates_positive
    assert map_predicate("  forecasts  ") == Predicate.forecasts


def test_synonym_table_mappings() -> None:
    assert map_predicate("anticipates") == Predicate.expects
    assert map_predicate("bullish on") == Predicate.rates_positive
    assert map_predicate("bearish") == Predicate.rates_negative
    assert map_predicate("praises") == Predicate.rates_positive
    assert map_predicate("dismisses") == Predicate.rejects
    assert map_predicate("concurs") == Predicate.agrees
    assert map_predicate("disputes") == Predicate.disagrees
    assert map_predicate("casts doubt on") == Predicate.questions


def test_singular_plural_rules() -> None:
    # Model emits singular verb -> mapped to the plural canonical form.
    assert map_predicate("forecast") == Predicate.forecasts
    assert map_predicate("predict") == Predicate.predicts
    assert map_predicate("recommend") == Predicate.recommends
    assert map_predicate("endorse") == Predicate.endorses
    assert map_predicate("criticize") == Predicate.criticizes


def test_word_level_handle() -> None:
    # A canonical single-token handle embedded in a phrase still resolves.
    assert map_predicate("strongly criticizes") == Predicate.criticizes
    assert map_predicate("clearly forecasts") == Predicate.forecasts


def test_ambiguous_bare_tokens_do_not_silently_win() -> None:
    # "rates"/"positive"/"negative" are intentionally not single-token handles.
    assert try_map_predicate("rates") is None
    assert try_map_predicate("positive") is None
    assert try_map_predicate("negative") is None


def test_unmappable_raises() -> None:
    with pytest.raises(PredicateMappingError):
        map_predicate("flibbertigibbets")
    with pytest.raises(PredicateMappingError):
        map_predicate("")
    with pytest.raises(PredicateMappingError):
        map_predicate("   ")


def test_try_map_returns_none_for_unmappable() -> None:
    assert try_map_predicate("flibbertigibbets") is None
    assert try_map_predicate("") is None
    assert try_map_predicate(None) is None  # type: ignore[arg-type]


def test_mapping_error_carries_raw() -> None:
    try:
        map_predicate("nonsense_predicate")
    except PredicateMappingError as exc:
        assert exc.raw == "nonsense_predicate"
    else:  # pragma: no cover
        pytest.fail("expected PredicateMappingError")
