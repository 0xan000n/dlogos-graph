"""Tests for the incremental, Wikidata-anchored, cascade resolver (Task 1.5).

This is the *cross-episode* property that makes a multi-episode graph correct:
the SAME real-world entity must get the SAME ``canonical_id`` even when it is
resolved by **separate** ``resolve`` calls against a persistent store — that is
episode N resolving against episodes 1..N-1. The whole point of the
:class:`IncrementalResolver` is to fix the fragmentation the batch
``resolve_subjects`` cannot (it only sees one episode's claims at a time).

All deterministic and offline: the injected ``fake_embedder`` (conftest) maps
known surface forms to fixed vectors; the Wikidata linker is a tiny in-test
fake returning canned QIDs. No model, no network.
"""

from __future__ import annotations

import numpy as np

from dlogos.resolution.canonical_store import InMemoryCanonicalStore
from dlogos.resolution.incremental import IncrementalResolver
from dlogos.resolution.subjects import (
    SubjectResolution,
    _normalize_surface,
    _surface_key,
)
from dlogos.schema import (
    Entity,
    EntityType,
    ExtractedClaim,
    Predicate,
    SourceSpan,
    SpeakerRef,
    Stance,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _claim(
    subject: str,
    *,
    subject_type: EntityType = EntityType.organization,
    qid: str | None = None,
    episode_id: str = "ep-0001",
) -> ExtractedClaim:
    return ExtractedClaim(
        speaker=SpeakerRef(label="SPEAKER_00"),
        predicate=Predicate.rates_positive,
        subject_entity=Entity(name=subject, type=subject_type, qid=qid),
        object="something",
        stance=Stance.asserts,
        sentiment=0.1,
        confidence=0.8,
        source_span=SourceSpan(episode_id=episode_id, t_start=0.0, t_end=1.0),
    )


class _FamilyEmbedder:
    """Embedder where the Apple family ("Apple"/"Apple Inc.") embeds ~identical.

    The conftest ``fake_embedder`` only knows "Apple"/"the iPhone" and hashes
    everything else to a far-away random vector — so "Apple Inc." would not
    converge by embedding alone there. The plan's convergence test specifies a
    fake embedder where Apple / Apple Inc. embed ≈identical; this is it. Known
    surface forms get hand-placed unit vectors; unknowns hash deterministically
    (so OpenAI stays orthogonal and never falsely merges).
    """

    DIM = 4
    _TABLE = {
        "apple": [1.0, 0.0, 0.0, 0.0],
        "apple inc.": [0.99, 0.01, 0.0, 0.0],
        "apple inc": [0.99, 0.01, 0.0, 0.0],
        "the iphone maker": [0.98, 0.02, 0.0, 0.0],
        "openai": [0.0, 1.0, 0.0, 0.0],
    }

    def _norm(self, text: str) -> str:
        return _normalize_surface(text)

    def embed(self, text: str) -> list[float]:
        key = self._norm(text)
        if key in self._TABLE:
            v = np.asarray(self._TABLE[key], dtype=float)
            return (v / (np.linalg.norm(v) or 1.0)).tolist()
        rng = np.random.default_rng(abs(hash(key)) % (2**32))
        v = rng.standard_normal(self.DIM)
        return (v / (np.linalg.norm(v) or 1.0)).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class _FakeLinker:
    """Minimal ``WikidataLinker`` stand-in: name(+type) -> canned QID.

    ``anchor_entity`` calls ``linker.link(name, type).qid``; we mirror that
    surface so the resolver can drive a real ``anchor_entity`` over a fake.
    Concepts/works are never anchored (``_LINKABLE_TYPES``), so the table only
    needs person/org entries.
    """

    def __init__(self, table: dict[str, str]) -> None:
        # key on normalized name -> qid
        self._table = {_normalize_surface(k): v for k, v in table.items()}

    def link(self, name: str, entity_type: EntityType, **_: object):
        from dlogos.resolution.wikidata import WikidataMatch

        qid = self._table.get(_normalize_surface(name))
        return WikidataMatch(name=name, qid=qid)


def _id_for(res: SubjectResolution, surface: str, etype: EntityType) -> str:
    key = _surface_key(etype, _normalize_surface(surface))
    cid = res.surface_to_id.get(key)
    assert cid is not None, f"no canonical id for {surface!r}: {res.surface_to_id}"
    return cid


# --------------------------------------------------------------------------- #
# Drop-in shape
# --------------------------------------------------------------------------- #
def test_returns_subject_resolution_shape(fake_embedder) -> None:
    store = InMemoryCanonicalStore()
    resolver = IncrementalResolver(store, fake_embedder)

    res = resolver.resolve([_claim("Apple")])

    assert isinstance(res, SubjectResolution)
    # claims carry a stamped canonical id on their subject entity.
    assert len(res.claims) == 1
    stamped = res.claims[0].subject_entity
    assert stamped.canonical_id is not None
    assert stamped.canonical_id.startswith("ent-")
    # surface_to_id + clusters are populated like resolve_subjects.
    assert res.surface_to_id
    assert res.clusters
    # input not mutated.
    assert res.claims[0].subject_entity is not None


# --------------------------------------------------------------------------- #
# THE cross-episode property: convergence across SEPARATE resolve calls
# --------------------------------------------------------------------------- #
def test_cross_episode_convergence_separate_calls() -> None:
    store = InMemoryCanonicalStore()
    resolver = IncrementalResolver(store, _FamilyEmbedder())

    # "Episode 1": resolve a batch mentioning "Apple".
    res1 = resolver.resolve([_claim("Apple", episode_id="ep-1")])
    id1 = res1.claims[0].subject_entity.canonical_id

    # "Episode 2": a SEPARATE resolve call mentioning "Apple Inc." — which the
    # embedder places ~identical to "Apple", so the embedding tier converges
    # them across calls even though the surface form differs.
    res2 = resolver.resolve([_claim("Apple Inc.", episode_id="ep-2")])
    id2 = res2.claims[0].subject_entity.canonical_id

    assert id1 == id2, "Apple / Apple Inc. must share one canonical id across calls"

    # The store holds exactly ONE Apple-org canonical with both aliases.
    orgs = [e for e in store.all() if e.entity_type == EntityType.organization]
    assert len(orgs) == 1, orgs
    norm_aliases = {_normalize_surface(a) for a in orgs[0].aliases}
    assert "apple" in norm_aliases
    assert "apple inc." in norm_aliases


def test_cross_episode_convergence_via_embedding(fake_embedder) -> None:
    # "Apple" and "the iPhone" embed at cosine ~0.99 in the fake embedder, well
    # above the org HIGH threshold (0.86) — so even without an alias hit, the
    # embedding tier converges them across separate calls.
    store = InMemoryCanonicalStore()
    resolver = IncrementalResolver(store, fake_embedder)

    res1 = resolver.resolve([_claim("Apple", episode_id="ep-1")])
    res2 = resolver.resolve([_claim("the iPhone", episode_id="ep-2")])

    assert (
        res1.claims[0].subject_entity.canonical_id
        == res2.claims[0].subject_entity.canonical_id
    )
    assert len([e for e in store.all()]) == 1


# --------------------------------------------------------------------------- #
# QID convergence: same QID, far-apart embeddings -> one canonical id
# --------------------------------------------------------------------------- #
def test_qid_convergence_overrides_distant_embedding(fake_embedder) -> None:
    store = InMemoryCanonicalStore()
    linker = _FakeLinker({"Apple": "Q312", "the iPhone maker": "Q312"})
    resolver = IncrementalResolver(store, fake_embedder, wikidata_linker=linker)

    # Episode 1: "Apple" anchors to Q312, minted as wd-Q312.
    res1 = resolver.resolve([_claim("Apple", episode_id="ep-1")])
    id1 = res1.claims[0].subject_entity.canonical_id
    assert id1 == "wd-Q312"
    assert res1.claims[0].subject_entity.qid == "Q312"

    # Episode 2: "the iPhone maker" embeds far from "Apple" but shares Q312, so
    # the qid anchor wins -> same canonical id.
    res2 = resolver.resolve([_claim("the iPhone maker", episode_id="ep-2")])
    id2 = res2.claims[0].subject_entity.canonical_id
    assert id2 == "wd-Q312"
    assert res2.claims[0].subject_entity.qid == "Q312"

    assert id1 == id2
    orgs = [e for e in store.all() if e.entity_type == EntityType.organization]
    assert len(orgs) == 1
    assert orgs[0].qid == "Q312"


# --------------------------------------------------------------------------- #
# No false merge
# --------------------------------------------------------------------------- #
def test_no_false_merge_openai_vs_apple(fake_embedder) -> None:
    store = InMemoryCanonicalStore()
    resolver = IncrementalResolver(store, fake_embedder)

    res1 = resolver.resolve([_claim("Apple", episode_id="ep-1")])
    res2 = resolver.resolve([_claim("OpenAI", episode_id="ep-2")])

    assert (
        res1.claims[0].subject_entity.canonical_id
        != res2.claims[0].subject_entity.canonical_id
    )
    assert len(store.all()) == 2


def test_distinct_subjects_within_one_batch(fake_embedder) -> None:
    store = InMemoryCanonicalStore()
    resolver = IncrementalResolver(store, fake_embedder)

    res = resolver.resolve(
        [_claim("Apple", episode_id="ep-1"), _claim("OpenAI", episode_id="ep-1")]
    )

    apple_id = _id_for(res, "Apple", EntityType.organization)
    openai_id = _id_for(res, "OpenAI", EntityType.organization)
    assert apple_id != openai_id
    assert len(store.all()) == 2


# --------------------------------------------------------------------------- #
# Persistence interaction: a SqliteCanonicalStore survives the resolver too
# --------------------------------------------------------------------------- #
def test_convergence_against_sqlite_store(tmp_path) -> None:
    from dlogos.resolution.canonical_store import SqliteCanonicalStore

    db = tmp_path / "ent.db"
    # First resolver instance / "process".
    store1 = SqliteCanonicalStore(db)
    res1 = IncrementalResolver(store1, _FamilyEmbedder()).resolve(
        [_claim("Apple", episode_id="ep-1")]
    )
    id1 = res1.claims[0].subject_entity.canonical_id
    store1.close()

    # Second resolver instance on the SAME db path resolves against episode 1.
    store2 = SqliteCanonicalStore(db)
    res2 = IncrementalResolver(store2, _FamilyEmbedder()).resolve(
        [_claim("Apple Inc.", episode_id="ep-2")]
    )
    id2 = res2.claims[0].subject_entity.canonical_id
    assert id1 == id2
    assert len([e for e in store2.all() if e.entity_type == EntityType.organization]) == 1
    store2.close()


# --------------------------------------------------------------------------- #
# Concept subjects (not QID-anchored) still resolve and converge
# --------------------------------------------------------------------------- #
def test_concept_subject_mints_deterministic_id(fake_embedder) -> None:
    store = InMemoryCanonicalStore()
    linker = _FakeLinker({})  # nothing anchors
    resolver = IncrementalResolver(store, fake_embedder, wikidata_linker=linker)

    res = resolver.resolve(
        [_claim("inflation", subject_type=EntityType.concept, episode_id="ep-1")]
    )
    cid = res.claims[0].subject_entity.canonical_id
    assert cid is not None
    assert cid.startswith("ent-")  # deterministic content-addressed, not wd-
    assert res.claims[0].subject_entity.qid is None
