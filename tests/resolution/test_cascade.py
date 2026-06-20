"""Tests for the rules -> embedding -> LLM-adjudication cascade matcher.

The store is INJECTED via a tiny in-test fake that implements the
``CanonicalEntityStore`` Protocol surface the cascade depends on
(``by_qid`` / ``by_exact_name`` / ``candidates``). No real store, no
embedding model, no network: every tier is exercised over canned data so
the decision logic is verified in isolation.

The LLM adjudicator is likewise injected. ``llm_adjudicator_from_client``
is tested against a fake synchronous OpenAI-compatible client (same shape
as ``extractor._call`` uses, but sync) so no network is touched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dlogos.resolution.cascade import (
    DEFAULT_THRESHOLDS,
    MatchDecision,
    TypeThresholds,
    llm_adjudicator_from_client,
    match_entity,
)
from dlogos.resolution.subjects import _normalize_surface
from dlogos.schema import Entity, EntityType


# --------------------------------------------------------------------------- #
# Tiny fake canonical-entity record + store (codes to the plan's Protocol)
# --------------------------------------------------------------------------- #
@dataclass
class _Cand:
    """Stand-in for ``CanonicalEntity`` — only the fields the cascade reads."""

    canonical_id: str
    canonical_name: str
    entity_type: EntityType
    qid: str | None = None
    embedding: list[float] | None = None
    aliases: list[str] = field(default_factory=list)


class FakeStore:
    """Minimal ``CanonicalEntityStore`` for the cascade.

    ``candidates`` returns a pre-canned, already-scored list (the cascade does
    not re-score; it consumes ``(entity, score)`` pairs), so each test can pin
    the exact similarity tier it wants to exercise.
    """

    def __init__(
        self,
        *,
        by_qid: dict[tuple[str, EntityType], _Cand] | None = None,
        by_name: dict[tuple[str, EntityType], _Cand] | None = None,
        candidates: list[tuple[_Cand, float]] | None = None,
    ) -> None:
        self._by_qid = by_qid or {}
        self._by_name = by_name or {}
        self._candidates = candidates or []
        self.candidates_called = False

    def by_qid(self, qid: str, entity_type: EntityType) -> _Cand | None:
        return self._by_qid.get((qid, entity_type))

    def by_exact_name(self, norm_name: str, entity_type: EntityType) -> _Cand | None:
        return self._by_name.get((norm_name, entity_type))

    def candidates(
        self, embedding: list[float], entity_type: EntityType, k: int = 5
    ) -> list[tuple[_Cand, float]]:
        self.candidates_called = True
        return [
            (c, s) for c, s in self._candidates if c.entity_type == entity_type
        ][:k]


def _org(name: str, cid: str, *, qid: str | None = None) -> _Cand:
    return _Cand(
        canonical_id=cid,
        canonical_name=name,
        entity_type=EntityType.organization,
        qid=qid,
        aliases=[name],
    )


# --------------------------------------------------------------------------- #
# Tier 1 — rules (exact name + qid)
# --------------------------------------------------------------------------- #
def test_rules_exact_name_match_skips_embedding() -> None:
    apple = _org("Apple", "ent-apple")
    store = FakeStore(by_name={(_normalize_surface("Apple"), EntityType.organization): apple})

    ent = Entity(name="apple", type=EntityType.organization)
    decision = match_entity(ent, [0.0, 0.0], store)

    assert isinstance(decision, MatchDecision)
    assert decision.canonical_id == "ent-apple"
    assert decision.reason == "exact-name"
    # No embedding compare happened on an exact hit.
    assert store.candidates_called is False


def test_rules_qid_match_wins_over_distant_surface() -> None:
    apple = _org("Apple", "ent-apple", qid="Q312")
    store = FakeStore(by_qid={("Q312", EntityType.organization): apple})

    # Surface form embeds far away, but the QID anchor pins it.
    ent = Entity(name="the iPhone maker", type=EntityType.organization, qid="Q312")
    decision = match_entity(ent, [0.0, 1.0], store, qid="Q312")

    assert decision.canonical_id == "ent-apple"
    assert decision.reason == "qid"
    assert store.candidates_called is False


# --------------------------------------------------------------------------- #
# Tier 2 — embedding (per-type thresholds)
# --------------------------------------------------------------------------- #
def test_embedding_high_score_matches() -> None:
    apple = _org("Apple", "ent-apple")
    # Score above the org HIGH threshold (0.86).
    store = FakeStore(candidates=[(apple, 0.95)])

    ent = Entity(name="Apple Inc.", type=EntityType.organization)
    decision = match_entity(ent, [1.0, 0.0], store)

    assert decision.canonical_id == "ent-apple"
    assert decision.reason.startswith("embed-")


def test_embedding_low_score_is_new() -> None:
    other = _Cand(
        canonical_id="ent-other",
        canonical_name="Inflation",
        entity_type=EntityType.concept,
    )
    # Score below the concept LOW threshold (0.70).
    store = FakeStore(candidates=[(other, 0.40)])

    ent = Entity(name="Deflation", type=EntityType.concept)
    decision = match_entity(ent, [1.0, 0.0], store)

    assert decision.canonical_id is None
    assert decision.reason.startswith("new-")


def test_empty_candidates_is_new() -> None:
    store = FakeStore(candidates=[])
    ent = Entity(name="Brand New Co", type=EntityType.organization)

    decision = match_entity(ent, [1.0, 0.0], store)

    assert decision.canonical_id is None
    assert decision.reason == "new-empty"


# --------------------------------------------------------------------------- #
# Tier 3 — ambiguous middle (LLM adjudication / conservative fallback)
# --------------------------------------------------------------------------- #
def test_ambiguous_with_adjudicator_yes_matches() -> None:
    apple = _org("Apple", "ent-apple")
    # Between org LOW (0.62) and HIGH (0.86).
    store = FakeStore(candidates=[(apple, 0.75)])
    ent = Entity(name="Apple Computer", type=EntityType.organization)

    decision = match_entity(ent, [1.0, 0.0], store, llm_adjudicator=lambda a, b: True)

    assert decision.canonical_id == "ent-apple"
    assert decision.reason == "llm-yes"


def test_ambiguous_with_adjudicator_no_is_new() -> None:
    apple = _org("Apple", "ent-apple")
    store = FakeStore(candidates=[(apple, 0.75)])
    ent = Entity(name="Apricot Inc.", type=EntityType.organization)

    decision = match_entity(ent, [1.0, 0.0], store, llm_adjudicator=lambda a, b: False)

    assert decision.canonical_id is None
    assert decision.reason == "ambiguous-conservative"


def test_ambiguous_without_adjudicator_is_conservative_new() -> None:
    apple = _org("Apple", "ent-apple")
    store = FakeStore(candidates=[(apple, 0.75)])
    ent = Entity(name="Apple Computer", type=EntityType.organization)

    # No adjudicator injected: conservative -> NEW.
    decision = match_entity(ent, [1.0, 0.0], store)

    assert decision.canonical_id is None
    assert decision.reason == "ambiguous-conservative"


# --------------------------------------------------------------------------- #
# Per-type thresholds are wired
# --------------------------------------------------------------------------- #
def test_default_thresholds_present_for_all_types() -> None:
    for etype in EntityType:
        assert etype in DEFAULT_THRESHOLDS
        t = DEFAULT_THRESHOLDS[etype]
        assert isinstance(t, TypeThresholds)
        assert 0.0 < t.low < t.high <= 1.0


def test_custom_thresholds_override() -> None:
    apple = _org("Apple", "ent-apple")
    store = FakeStore(candidates=[(apple, 0.80)])
    ent = Entity(name="Apple Inc.", type=EntityType.organization)

    # Lower the org HIGH bar so 0.80 now counts as a confident match.
    thresholds = dict(DEFAULT_THRESHOLDS)
    thresholds[EntityType.organization] = TypeThresholds(high=0.75, low=0.50)

    decision = match_entity(ent, [1.0, 0.0], store, thresholds=thresholds)

    assert decision.canonical_id == "ent-apple"
    assert decision.reason.startswith("embed-")


# --------------------------------------------------------------------------- #
# Task 1.6 — the real LLM adjudicator (injected sync OpenAI-compatible client)
# --------------------------------------------------------------------------- #
class _FakeCompletions:
    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"choices": [{"message": {"content": self._content}}]}


class _FakeChat:
    def __init__(self, content: str) -> None:
        self.completions = _FakeCompletions(content)


class FakeSyncClient:
    """Synchronous OpenAI-compatible client returning one canned completion."""

    def __init__(self, content: str) -> None:
        self.chat = _FakeChat(content)


class _BoomCompletions:
    def create(self, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("network down")


class _BoomChat:
    def __init__(self) -> None:
        self.completions = _BoomCompletions()


class BoomClient:
    def __init__(self) -> None:
        self.chat = _BoomChat()


def test_adjudicator_returns_true_on_same_true() -> None:
    client = FakeSyncClient('{"same": true}')
    adj = llm_adjudicator_from_client(client, model="meta/test")

    a = _org("Apple", "ent-apple")
    b = _org("Apple Inc.", "ent-apple-inc")
    assert adj(a, b) is True
    # The model id was forwarded.
    assert client.chat.completions.calls[0]["model"] == "meta/test"


def test_adjudicator_returns_false_on_same_false() -> None:
    client = FakeSyncClient('{"same": false}')
    adj = llm_adjudicator_from_client(client, model="meta/test")

    a = _org("Apple", "ent-apple")
    b = _org("Apricot", "ent-apricot")
    assert adj(a, b) is False


def test_adjudicator_malformed_completion_is_false() -> None:
    client = FakeSyncClient("not json at all {{{")
    adj = llm_adjudicator_from_client(client, model="meta/test")

    a = _org("Apple", "ent-apple")
    b = _org("Apple Inc.", "ent-apple-inc")
    assert adj(a, b) is False


def test_adjudicator_missing_key_is_false() -> None:
    client = FakeSyncClient('{"verdict": "yes"}')
    adj = llm_adjudicator_from_client(client, model="meta/test")
    assert adj(_org("Apple", "a"), _org("Apple Inc.", "b")) is False


def test_adjudicator_http_error_is_false() -> None:
    adj = llm_adjudicator_from_client(BoomClient(), model="meta/test")
    assert adj(_org("Apple", "a"), _org("Apple Inc.", "b")) is False


def test_adjudicator_works_with_entity_and_candidate_aliases() -> None:
    # The candidate carries aliases; the prompt should not crash on them and
    # a positive verdict still resolves to True.
    client = FakeSyncClient('{"same": true}')
    adj = llm_adjudicator_from_client(client, model="meta/test")

    a = Entity(name="the iPhone maker", type=EntityType.organization)
    b = _Cand(
        canonical_id="ent-apple",
        canonical_name="Apple",
        entity_type=EntityType.organization,
        aliases=["Apple", "Apple Inc.", "AAPL"],
    )
    assert adj(a, b) is True
    # Both surface forms made it into the user prompt.
    user_msg = client.chat.completions.calls[0]["messages"][-1]["content"]
    assert "iPhone maker" in user_msg
    assert "Apple" in user_msg
