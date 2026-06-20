"""Tests for the persistent canonical entity store (plan Tasks 1.1 + 1.2).

The store is the ANN-ready entity index that the incremental resolver upserts
into and queries via ``candidates()`` (the sole similarity seam). Two backends
must satisfy the same Protocol behaviour:

- :class:`InMemoryCanonicalStore` — process-local dict (Task 1.1).
- :class:`SqliteCanonicalStore` — stdlib sqlite3, persistent across instances
  (Task 1.2): re-opening the same DB path sees prior upserts, and re-upserting
  the same ``canonical_id`` merges aliases rather than duplicating.

All offline: no network, no heavy deps.
"""

from __future__ import annotations

import pytest

from dlogos.resolution.canonical_store import (
    CanonicalEntity,
    CanonicalEntityStore,
    InMemoryCanonicalStore,
    SqliteCanonicalStore,
)
from dlogos.schema import EntityType


def _apple() -> CanonicalEntity:
    return CanonicalEntity(
        canonical_id="ent-apple",
        canonical_name="Apple",
        entity_type=EntityType.organization,
        embedding=[1.0, 0.0],
        aliases=["Apple"],
    )


def _openai() -> CanonicalEntity:
    return CanonicalEntity(
        canonical_id="ent-openai",
        canonical_name="OpenAI",
        entity_type=EntityType.organization,
        embedding=[0.0, 1.0],
        aliases=["OpenAI"],
    )


# --------------------------------------------------------------------------- #
# Task 1.1 — record + in-memory impl
# --------------------------------------------------------------------------- #
def test_inmemory_is_a_canonical_entity_store() -> None:
    assert isinstance(InMemoryCanonicalStore(), CanonicalEntityStore)


def test_candidates_returns_nearest_org_with_high_score() -> None:
    store = InMemoryCanonicalStore()
    store.upsert(_apple())
    store.upsert(_openai())

    cands = store.candidates([0.99, 0.01], EntityType.organization, k=1)
    assert len(cands) == 1
    top, score = cands[0]
    assert top.canonical_id == "ent-apple"
    assert score == pytest.approx(1.0, abs=1e-2)


def test_by_qid_finds_entity_after_qid_upsert() -> None:
    store = InMemoryCanonicalStore()
    store.upsert(_apple())  # no qid yet
    assert store.by_qid("Q312", EntityType.organization) is None

    # Upsert the same canonical id, now carrying a qid.
    store.upsert(
        CanonicalEntity(
            canonical_id="ent-apple",
            canonical_name="Apple",
            entity_type=EntityType.organization,
            qid="Q312",
            embedding=[1.0, 0.0],
            aliases=["Apple"],
        )
    )
    hit = store.by_qid("Q312", EntityType.organization)
    assert hit is not None
    assert hit.canonical_id == "ent-apple"


def test_type_partition_excludes_other_types_from_candidates_and_qid() -> None:
    store = InMemoryCanonicalStore()
    # A *person* who happens to be named "Apple", anchored to a qid, with an
    # embedding identical to the org probe — must never surface for an org query.
    store.upsert(
        CanonicalEntity(
            canonical_id="per-apple",
            canonical_name="Apple",
            entity_type=EntityType.person,
            qid="Q999",
            embedding=[1.0, 0.0],
            aliases=["Apple"],
        )
    )
    assert store.candidates([1.0, 0.0], EntityType.organization, k=5) == []
    assert store.by_qid("Q999", EntityType.organization) is None
    # But it is findable as a person.
    assert store.by_qid("Q999", EntityType.person) is not None


def test_by_exact_name_normalizes_and_partitions_by_type() -> None:
    store = InMemoryCanonicalStore()
    store.upsert(_apple())
    # Casefold + whitespace collapse via _normalize_surface (DRY w/ subjects).
    assert store.by_exact_name("apple", EntityType.organization) is not None
    assert store.by_exact_name("  APPLE ", EntityType.organization) is not None
    assert store.by_exact_name("apple", EntityType.person) is None


def test_get_and_all() -> None:
    store = InMemoryCanonicalStore()
    store.upsert(_apple())
    store.upsert(_openai())
    assert store.get("ent-apple").canonical_id == "ent-apple"
    assert store.get("missing") is None
    assert {e.canonical_id for e in store.all()} == {"ent-apple", "ent-openai"}


# --------------------------------------------------------------------------- #
# Task 1.2 — sqlite-backed persistence + alias merge on re-upsert
# --------------------------------------------------------------------------- #
def test_sqlite_is_a_canonical_entity_store(tmp_path) -> None:
    assert isinstance(
        SqliteCanonicalStore(tmp_path / "ent.db"), CanonicalEntityStore
    )


def test_sqlite_persists_across_two_store_instances(tmp_path) -> None:
    path = tmp_path / "ent.db"
    first = SqliteCanonicalStore(path)
    first.upsert(
        CanonicalEntity(
            canonical_id="ent-apple",
            canonical_name="Apple",
            entity_type=EntityType.organization,
            qid="Q312",
            embedding=[1.0, 0.0],
            aliases=["Apple"],
        )
    )

    # A *separate* store instance on the same path must see the persisted row.
    second = SqliteCanonicalStore(path)
    by_qid = second.by_qid("Q312", EntityType.organization)
    assert by_qid is not None
    assert by_qid.canonical_id == "ent-apple"

    cands = second.candidates([0.99, 0.01], EntityType.organization, k=1)
    assert cands and cands[0][0].canonical_id == "ent-apple"
    assert second.by_exact_name("apple", EntityType.organization) is not None


def test_sqlite_reupsert_merges_aliases_not_duplicates(tmp_path) -> None:
    path = tmp_path / "ent.db"
    store = SqliteCanonicalStore(path)
    store.upsert(
        CanonicalEntity(
            canonical_id="ent-apple",
            canonical_name="Apple",
            entity_type=EntityType.organization,
            embedding=[1.0, 0.0],
            aliases=["Apple"],
        )
    )
    # Re-upsert the same canonical id with a new alias + a qid.
    store.upsert(
        CanonicalEntity(
            canonical_id="ent-apple",
            canonical_name="Apple",
            entity_type=EntityType.organization,
            qid="Q312",
            embedding=[1.0, 0.0],
            aliases=["Apple Inc."],
        )
    )

    # Re-open to prove persistence + the merge.
    reopened = SqliteCanonicalStore(path)
    assert len(reopened.all()) == 1  # not duplicated
    merged = reopened.get("ent-apple")
    assert merged is not None
    assert set(merged.aliases) == {"Apple", "Apple Inc."}
    assert merged.qid == "Q312"


def test_sqlite_type_partition(tmp_path) -> None:
    store = SqliteCanonicalStore(tmp_path / "ent.db")
    store.upsert(
        CanonicalEntity(
            canonical_id="per-apple",
            canonical_name="Apple",
            entity_type=EntityType.person,
            embedding=[1.0, 0.0],
            aliases=["Apple"],
        )
    )
    store.upsert(_apple())  # org "Apple"
    cands = store.candidates([1.0, 0.0], EntityType.organization, k=5)
    assert [ent.canonical_id for ent, _score in cands] == ["ent-apple"]
