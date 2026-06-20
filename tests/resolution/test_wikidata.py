"""Tests for lightweight Wikidata linking (resolution §7.4a, lever ii / §7.3).

The HTTP client is INJECTED — a tiny fake returning canned candidate lists —
so no network is touched and behaviour is deterministic.
"""

from __future__ import annotations

from typing import Any

from dlogos.resolution.wikidata import (
    WikidataClient,
    WikidataLinker,
    WikidataMatch,
    anchor_entity,
    link_entities,
)
from dlogos.schema import Entity, EntityType


# --------------------------------------------------------------------------- #
# Fake client
# --------------------------------------------------------------------------- #
class FakeWikidataClient:
    """Deterministic, offline Wikidata client returning canned candidates."""

    _DB: dict[str, list[dict[str, Any]]] = {
        # Realistic relevance ordering: wbsearchentities ranks the intended
        # sense first (Apple Inc. before the fruit), even though the fruit's
        # label is the exact lowercase query.
        "apple": [
            {"id": "Q312", "label": "Apple Inc.", "description": "technology company"},
            {"id": "Q89", "label": "apple", "description": "fruit"},
        ],
        "openai": [
            {"id": "Q21708200", "label": "OpenAI", "description": "AI research lab"},
        ],
        "tyler cowen": [
            {"id": "Q7860590", "label": "Tyler Cowen", "description": "economist"},
        ],
    }

    def __init__(self) -> None:
        self.calls: list[tuple[str, EntityType | None]] = []

    def search(
        self, name: str, *, entity_type: EntityType | None = None, limit: int = 5
    ) -> list[dict[str, Any]]:
        self.calls.append((name, entity_type))
        return list(self._DB.get(name.strip().casefold(), []))


# --------------------------------------------------------------------------- #
# Basic linking
# --------------------------------------------------------------------------- #
def test_links_known_org_to_qid():
    linker = WikidataLinker(FakeWikidataClient())
    match = linker.link("Apple", EntityType.organization)
    assert match.qid == "Q312"
    assert match.label == "Apple Inc."


def test_links_known_person_to_qid():
    linker = WikidataLinker(FakeWikidataClient())
    match = linker.link("Tyler Cowen", EntityType.person)
    assert match.qid == "Q7860590"
    assert match.description == "economist"


def test_unknown_name_returns_none_qid():
    linker = WikidataLinker(FakeWikidataClient())
    match = linker.link("Nonexistent Person", EntityType.person)
    assert isinstance(match, WikidataMatch)
    assert match.qid is None


def test_relevance_first_wins_over_homonym_label():
    # "Apple" returns Apple Inc. (Q312) first and the fruit (Q89, whose label
    # exactly equals the lowercase query) second. The conservative matcher
    # trusts relevance ordering and must NOT let the exact-label homonym hijack
    # the link -- that would be a wrong-canonical-id misattribution.
    linker = WikidataLinker(FakeWikidataClient())
    match = linker.link("Apple", EntityType.organization)
    assert match.qid == "Q312"


# --------------------------------------------------------------------------- #
# Conservatism: concepts/works are not linked; empty names skip
# --------------------------------------------------------------------------- #
def test_concept_type_not_linked():
    fake = FakeWikidataClient()
    linker = WikidataLinker(fake)
    match = linker.link("Apple", EntityType.concept)
    assert match.qid is None
    # Conservative: we never even call the client for non-linkable types.
    assert fake.calls == []


def test_work_type_not_linked():
    fake = FakeWikidataClient()
    linker = WikidataLinker(fake)
    match = linker.link("Apple", EntityType.work)
    assert match.qid is None
    assert fake.calls == []


def test_empty_name_skips_lookup():
    fake = FakeWikidataClient()
    linker = WikidataLinker(fake)
    match = linker.link("   ", EntityType.person)
    assert match.qid is None
    assert fake.calls == []


# --------------------------------------------------------------------------- #
# Caching: repeated guests cost one lookup
# --------------------------------------------------------------------------- #
def test_repeated_lookup_is_cached():
    fake = FakeWikidataClient()
    linker = WikidataLinker(fake)
    linker.link("Tyler Cowen", EntityType.person)
    linker.link("tyler  cowen", EntityType.person)  # casing/spacing variant
    assert len(fake.calls) == 1  # second hit served from cache


# --------------------------------------------------------------------------- #
# link_entity: fills canonical_id only when empty
# --------------------------------------------------------------------------- #
def test_link_entity_fills_canonical_id():
    linker = WikidataLinker(FakeWikidataClient())
    ent = Entity(name="OpenAI", type=EntityType.organization)
    linked = linker.link_entity(ent)
    assert linked.canonical_id == "Q21708200"
    # Original untouched.
    assert ent.canonical_id is None


def test_link_entity_does_not_clobber_existing_canonical_id():
    linker = WikidataLinker(FakeWikidataClient())
    ent = Entity(
        name="Apple", type=EntityType.organization, canonical_id="ent-fromclustering"
    )
    linked = linker.link_entity(ent)
    # Pre-existing clustering id is preserved, QID does NOT overwrite it.
    assert linked.canonical_id == "ent-fromclustering"


def test_link_entity_unmatched_leaves_canonical_id_none():
    linker = WikidataLinker(FakeWikidataClient())
    ent = Entity(name="Unknown Org", type=EntityType.organization)
    linked = linker.link_entity(ent)
    assert linked.canonical_id is None


# --------------------------------------------------------------------------- #
# Batch helper
# --------------------------------------------------------------------------- #
def test_link_entities_batch_is_deterministic_and_ordered():
    entities = [
        Entity(name="Apple", type=EntityType.organization),
        Entity(name="Tyler Cowen", type=EntityType.person),
        Entity(name="Some Concept", type=EntityType.concept),
    ]
    matches = link_entities(entities, FakeWikidataClient())
    assert [m.qid for m in matches] == ["Q312", "Q7860590", None]
    # Names preserved in order.
    assert [m.name for m in matches] == ["Apple", "Tyler Cowen", "Some Concept"]


# --------------------------------------------------------------------------- #
# Domain-context disambiguation (recurring-guest lever; spec §7.3)
# --------------------------------------------------------------------------- #
class _MultiSenseClient:
    """Returns two same-name senses so context can pick the intended one."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, EntityType | None]] = []

    def search(
        self, name: str, *, entity_type: EntityType | None = None, limit: int = 5
    ) -> list[dict[str, Any]]:
        self.calls.append((name, entity_type))
        if name.strip().casefold() == "jane doe":
            return [
                {"id": "Q999", "label": "Jane Doe", "description": "actor"},
                {"id": "Q111", "label": "Jane Doe", "description": "finance economist"},
            ]
        return []


def test_context_disambiguates_to_matching_sense():
    linker = WikidataLinker(_MultiSenseClient())
    match = linker.link("Jane Doe", EntityType.person, context=["finance"])
    # The "finance" hint overrides relevance ordering (which would pick Q999).
    assert match.qid == "Q111"
    assert match.description == "finance economist"


def test_no_context_trusts_relevance_ordering():
    linker = WikidataLinker(_MultiSenseClient())
    match = linker.link("Jane Doe", EntityType.person)
    # Without context the top relevance hit wins.
    assert match.qid == "Q999"


def test_context_without_overlap_falls_back_to_relevance():
    linker = WikidataLinker(_MultiSenseClient())
    match = linker.link("Jane Doe", EntityType.person, context=["sports"])
    # No description overlaps "sports" → keep the relevance-first hit.
    assert match.qid == "Q999"


# --------------------------------------------------------------------------- #
# anchor_entity: subject-entity QID anchoring for person/org only
# --------------------------------------------------------------------------- #
def test_anchor_entity_returns_qid_for_org():
    linker = WikidataLinker(FakeWikidataClient())
    qid = anchor_entity(Entity(name="Apple", type=EntityType.organization), linker)
    assert qid == "Q312"


def test_anchor_entity_returns_qid_for_person():
    linker = WikidataLinker(FakeWikidataClient())
    qid = anchor_entity(Entity(name="Tyler Cowen", type=EntityType.person), linker)
    assert qid == "Q7860590"


def test_anchor_entity_concept_returns_none_without_client_call():
    fake = FakeWikidataClient()
    linker = WikidataLinker(fake)
    qid = anchor_entity(Entity(name="Apple", type=EntityType.concept), linker)
    assert qid is None
    # Concepts are not QID-anchored: never touch the client.
    assert fake.calls == []


def test_anchor_entity_work_returns_none_without_client_call():
    fake = FakeWikidataClient()
    linker = WikidataLinker(fake)
    qid = anchor_entity(Entity(name="Apple", type=EntityType.work), linker)
    assert qid is None
    assert fake.calls == []


def test_anchor_entity_unknown_name_returns_none():
    linker = WikidataLinker(FakeWikidataClient())
    qid = anchor_entity(
        Entity(name="Nonexistent Person", type=EntityType.person), linker
    )
    assert qid is None


# --------------------------------------------------------------------------- #
# Protocol + no-network guarantees
# --------------------------------------------------------------------------- #
def test_fake_client_satisfies_protocol():
    assert isinstance(FakeWikidataClient(), WikidataClient)


def test_importing_module_opens_no_network():
    # Constructing a linker without a client must NOT build a real httpx client
    # or hit the network until a linkable lookup is actually attempted.
    linker = WikidataLinker()  # no client injected
    # A non-linkable lookup short-circuits before any client is built.
    match = linker.link("anything", EntityType.concept)
    assert match.qid is None
    assert linker._client is None  # real client never constructed
