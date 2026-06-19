"""Tests for deterministic, reversible blinding (spec §9)."""

from __future__ import annotations

from dlogos.eval.arms import (
    ARM_DLOGOS,
    ARM_MODEL_ALONE,
    ARM_NAIVE_RAG,
    ARM_WEB_SEARCH,
    Answer,
)
from dlogos.eval.blind import blind_answers, unblind_scores


def _answers() -> list[Answer]:
    return [
        Answer(arm=ARM_MODEL_ALONE, text="floor answer"),
        Answer(arm=ARM_WEB_SEARCH, text="web answer"),
        Answer(arm=ARM_NAIVE_RAG, text="rag answer"),
        Answer(arm=ARM_DLOGOS, text="dlogos answer"),
    ]


def test_blinding_strips_arm_identity_from_grader_view() -> None:
    blinded = blind_answers("gq-01", _answers(), seed=7)
    for b in blinded.blinded:
        # The grader sees text + citations but NOT which arm produced it.
        assert b.answer.arm == ""
        assert b.label.startswith("answer_")
    # Texts survive (the grader needs them).
    seen_text = {b.answer.text for b in blinded.blinded}
    assert seen_text == {"floor answer", "web answer", "rag answer", "dlogos answer"}


def test_blinding_is_seed_deterministic() -> None:
    a = blind_answers("gq-01", _answers(), seed=42)
    b = blind_answers("gq-01", _answers(), seed=42)
    assert [x.label for x in a.blinded] == [x.label for x in b.blinded]
    assert [x.answer.text for x in a.blinded] == [x.answer.text for x in b.blinded]
    assert a.unblind.label_to_arm == b.unblind.label_to_arm


def test_different_seed_changes_layout() -> None:
    a = blind_answers("gq-01", _answers(), seed=1)
    b = blind_answers("gq-01", _answers(), seed=999)
    # Extremely unlikely to coincide for 4 items across two distinct seeds.
    assert a.unblind.label_to_arm != b.unblind.label_to_arm


def test_different_query_id_changes_layout() -> None:
    a = blind_answers("gq-01", _answers(), seed=42)
    b = blind_answers("gq-02", _answers(), seed=42)
    # Per-query seed derivation -> queries don't share a fixed ordering.
    assert a.unblind.label_to_arm != b.unblind.label_to_arm


def test_blinding_is_reversible() -> None:
    blinded = blind_answers("gq-01", _answers(), seed=7)
    # Every original arm is recoverable, exactly once.
    recovered = set(blinded.unblind.label_to_arm.values())
    assert recovered == {ARM_MODEL_ALONE, ARM_WEB_SEARCH, ARM_NAIVE_RAG, ARM_DLOGOS}
    # And the label->arm mapping aligns with the shuffled order.
    for b in blinded.blinded:
        true_arm = blinded.unblind.arm_for(b.label)
        # Find the original answer with that arm; its text must match.
        original = next(a for a in _answers() if a.arm == true_arm)
        assert b.answer.text == original.text


def test_unblind_scores_maps_labels_back_to_arms() -> None:
    blinded = blind_answers("gq-01", _answers(), seed=7)
    # Grader records a score per opaque label.
    scores_by_label = {b.label: float(i) for i, b in enumerate(blinded.blinded)}
    by_arm = unblind_scores(blinded.unblind, scores_by_label)
    assert set(by_arm) == {ARM_MODEL_ALONE, ARM_WEB_SEARCH, ARM_NAIVE_RAG, ARM_DLOGOS}
    # Round-trip: the score the grader gave a label lands on the right arm.
    for label, score in scores_by_label.items():
        assert by_arm[blinded.unblind.arm_for(label)] == score
