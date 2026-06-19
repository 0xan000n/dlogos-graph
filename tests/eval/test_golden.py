"""Tests for the golden query set (spec §9)."""

from __future__ import annotations

import pytest

from dlogos.eval.golden import (
    AnswerShape,
    Archetype,
    Domain,
    GoldenQuery,
    coverage,
    starter_golden_queries,
)


def test_starter_set_spans_all_domains_and_archetypes() -> None:
    queries = starter_golden_queries()
    cov = coverage(queries)
    # Existence proof across a spread: every archetype and every domain present.
    assert cov["archetypes"] == {a.value for a in Archetype}
    assert cov["domains"] == {d.value for d in Domain}


def test_starter_set_is_12_to_15_hero_queries() -> None:
    queries = starter_golden_queries()
    assert 12 <= len(queries) <= 15


def test_query_ids_are_unique() -> None:
    queries = starter_golden_queries()
    ids = [q.id for q in queries]
    assert len(ids) == len(set(ids))


def test_set_is_constructed_fresh_each_call() -> None:
    # No shared mutable module state between calls.
    a = starter_golden_queries()
    b = starter_golden_queries()
    assert a is not b
    assert a[0] is not b[0]
    assert a == b


def test_temporal_and_belief_queries_are_deep_tier() -> None:
    queries = starter_golden_queries()
    for q in queries:
        if q.archetype in (Archetype.temporal_consensus, Archetype.belief_tracking):
            assert q.deep_tier, f"{q.id} should be deep_tier (carries the shift)"


def test_temporal_shapes_require_stance_shift_and_multiple_sources() -> None:
    queries = starter_golden_queries()
    temporal = [q for q in queries if q.archetype is Archetype.temporal_consensus]
    assert temporal
    for q in temporal:
        shape = q.pre_registered_answer_shape
        assert shape.expected_stance_shift is True
        assert shape.min_attributed_sources >= 3


def test_provenance_shapes_require_citations() -> None:
    queries = starter_golden_queries()
    prov = [q for q in queries if q.archetype is Archetype.provenance]
    assert prov
    for q in prov:
        assert q.pre_registered_answer_shape.requires_citations is True


def test_answer_shape_rejects_unknown_fields() -> None:
    with pytest.raises(Exception):
        AnswerShape(bogus_field=1)  # type: ignore[call-arg]


def test_golden_query_is_immutable_value_object_round_trips() -> None:
    q = GoldenQuery(
        id="gq-x",
        archetype=Archetype.provenance,
        domain=Domain.product,
        query_text="where?",
        pre_registered_answer_shape=AnswerShape(min_attributed_sources=1),
    )
    again = GoldenQuery.model_validate(q.model_dump())
    assert again == q
