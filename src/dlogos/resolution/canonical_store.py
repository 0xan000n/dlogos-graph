"""Persistent canonical entity store — the ANN-ready entity index (plan Phase 1).

The incremental resolver (``resolution/incremental.py``) anchors each episode's
subject entities against the *accumulated* canonical set: episode N resolves
against episodes 1..N-1, stably and re-runnably. That accumulation lives here.

Two backends satisfy one :class:`CanonicalEntityStore` ``Protocol``:

- :class:`InMemoryCanonicalStore` — a process-local dict; the default for tests
  and single-shot runs.
- :class:`SqliteCanonicalStore` — stdlib ``sqlite3`` (NO new dependency),
  persistent across process boundaries: re-opening the same DB path sees every
  prior upsert, so cross-episode resolution survives separate runs.

**``candidates()`` is the only similarity entry point.** At 20-episode scale it
is a type-partitioned brute-force cosine over a few hundred rows (reusing
:func:`dlogos.resolution.subjects._cosine` — DRY). For >100k entities, swap an
ANN index (FAISS/HNSW) *behind this method* — nothing else in the resolver
changes. Likewise the SQLite table is the columnar/vector-extension seam where a
larger external store later lives. Both are deliberate YAGNI boundaries: simple
now, one local swap later.

Heavy/optional deps stay out: ``numpy`` is pulled lazily inside ``_cosine``;
``sqlite3`` is stdlib. Importing this module costs nothing extra.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from dlogos.resolution.subjects import _cosine, _normalize_surface
from dlogos.schema import EntityType


# --------------------------------------------------------------------------- #
# Record
# --------------------------------------------------------------------------- #
class CanonicalEntity(BaseModel):
    """One canonical real-world entity, accumulated across episodes.

    ``canonical_id`` is the stable cluster id every merged surface form shares
    (minted ``wd-<QID>`` when Wikidata-anchored, else the content-addressed
    ``ent-...`` from :func:`dlogos.resolution.subjects._canonical_id`).
    ``embedding`` is the representative vector ``candidates()`` cosines against;
    ``aliases`` accumulates every surface form merged into this entity.
    """

    canonical_id: str
    canonical_name: str
    entity_type: EntityType
    qid: str | None = None
    embedding: list[float] | None = None
    aliases: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #
@runtime_checkable
class CanonicalEntityStore(Protocol):
    """The persistent canonical-entity index contract.

    The cascade matcher and incremental resolver depend only on this Protocol,
    never on a concrete backend. Every lookup is partitioned by
    :class:`~dlogos.schema.EntityType` so a *person* named "Apple" can never be
    returned for an *organization* query.
    """

    def get(self, canonical_id: str) -> CanonicalEntity | None: ...

    def by_qid(self, qid: str, entity_type: EntityType) -> CanonicalEntity | None: ...

    def by_exact_name(
        self, norm_name: str, entity_type: EntityType
    ) -> CanonicalEntity | None: ...

    def candidates(
        self, embedding: list[float], entity_type: EntityType, k: int = 5
    ) -> list[tuple[CanonicalEntity, float]]:  # the ANN seam (brute-force now)
        ...

    def upsert(self, entity: CanonicalEntity) -> None: ...

    def all(self) -> list[CanonicalEntity]: ...


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _merge_aliases(existing: Sequence[str], incoming: Sequence[str]) -> list[str]:
    """Union two alias lists, de-duped on normalized form, order-stable.

    Keeps the first-seen display form for each normalized key so casing/spacing
    variants ("Apple" / "apple ") collapse to one entry without losing the
    original display surface.
    """

    out: list[str] = []
    seen: set[str] = set()
    for alias in [*existing, *incoming]:
        norm = _normalize_surface(alias)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(alias)
    return out


def _top_k_candidates(
    rows: Sequence[CanonicalEntity], embedding: list[float], k: int
) -> list[tuple[CanonicalEntity, float]]:
    """Brute-force cosine over already type-filtered rows; top-k, desc.

    Rows without an embedding are skipped (nothing to compare). This is the
    shared body behind both backends' ``candidates()`` so the similarity
    semantics are identical regardless of where rows are stored.
    """

    scored: list[tuple[CanonicalEntity, float]] = [
        (row, _cosine(embedding, row.embedding))
        for row in rows
        if row.embedding
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:k]


# --------------------------------------------------------------------------- #
# In-memory backend (Task 1.1)
# --------------------------------------------------------------------------- #
class InMemoryCanonicalStore:
    """Process-local :class:`CanonicalEntityStore` backed by a dict.

    Lookups are linear scans (``by_qid`` / ``by_exact_name``) and brute-force
    cosine (``candidates``) — both fine at this scale and identical in behaviour
    to the SQLite backend. ``upsert`` merges aliases for an existing id.
    """

    def __init__(self) -> None:
        self._entities: dict[str, CanonicalEntity] = {}

    def get(self, canonical_id: str) -> CanonicalEntity | None:
        return self._entities.get(canonical_id)

    def by_qid(self, qid: str, entity_type: EntityType) -> CanonicalEntity | None:
        for ent in self._entities.values():
            if ent.entity_type == entity_type and ent.qid == qid:
                return ent
        return None

    def by_exact_name(
        self, norm_name: str, entity_type: EntityType
    ) -> CanonicalEntity | None:
        target = _normalize_surface(norm_name)
        for ent in self._entities.values():
            if ent.entity_type != entity_type:
                continue
            if _normalize_surface(ent.canonical_name) == target:
                return ent
            if any(_normalize_surface(a) == target for a in ent.aliases):
                return ent
        return None

    def candidates(
        self, embedding: list[float], entity_type: EntityType, k: int = 5
    ) -> list[tuple[CanonicalEntity, float]]:
        rows = [
            e for e in self._entities.values() if e.entity_type == entity_type
        ]
        return _top_k_candidates(rows, embedding, k)

    def upsert(self, entity: CanonicalEntity) -> None:
        existing = self._entities.get(entity.canonical_id)
        if existing is None:
            self._entities[entity.canonical_id] = entity.model_copy(deep=True)
            return
        merged = existing.model_copy(
            update={
                "canonical_name": entity.canonical_name or existing.canonical_name,
                "entity_type": entity.entity_type,
                "qid": entity.qid if entity.qid is not None else existing.qid,
                "embedding": entity.embedding
                if entity.embedding is not None
                else existing.embedding,
                "aliases": _merge_aliases(existing.aliases, entity.aliases),
            }
        )
        self._entities[entity.canonical_id] = merged

    def all(self) -> list[CanonicalEntity]:
        return list(self._entities.values())


# --------------------------------------------------------------------------- #
# SQLite backend (Task 1.2)
# --------------------------------------------------------------------------- #
_SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    canonical_id   TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    entity_type    TEXT NOT NULL,
    qid            TEXT,
    embedding_json TEXT,
    aliases_json   TEXT NOT NULL
)
"""


class SqliteCanonicalStore:
    """Persistent :class:`CanonicalEntityStore` over stdlib ``sqlite3``.

    One table ``entities(canonical_id PK, canonical_name, entity_type, qid,
    embedding_json, aliases_json)``. ``candidates`` loads the type's rows and
    brute-force cosines them in Python — acceptable at 20-episode scale; the
    table is the seam where a vector extension / external store later lives.
    ``upsert`` is ``INSERT ... ON CONFLICT(canonical_id) DO UPDATE`` merging the
    alias sets, so re-upserting the same id never duplicates and accumulates
    surface forms.

    Re-opening the same path is the cross-run persistence the incremental
    resolver relies on: episode N+1 (a separate process) resolves against
    episodes 1..N. Same Protocol/behaviour as the in-memory store.
    """

    def __init__(self, path: str | Path) -> None:
        import sqlite3

        self._path = str(path)
        self._conn = sqlite3.connect(self._path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    # -- row <-> record ----------------------------------------------------- #
    @staticmethod
    def _to_record(row) -> CanonicalEntity:
        import json

        return CanonicalEntity(
            canonical_id=row["canonical_id"],
            canonical_name=row["canonical_name"],
            entity_type=EntityType(row["entity_type"]),
            qid=row["qid"],
            embedding=(
                json.loads(row["embedding_json"])
                if row["embedding_json"] is not None
                else None
            ),
            aliases=json.loads(row["aliases_json"]),
        )

    # -- reads -------------------------------------------------------------- #
    def get(self, canonical_id: str) -> CanonicalEntity | None:
        row = self._conn.execute(
            "SELECT * FROM entities WHERE canonical_id = ?", (canonical_id,)
        ).fetchone()
        return self._to_record(row) if row is not None else None

    def by_qid(self, qid: str, entity_type: EntityType) -> CanonicalEntity | None:
        row = self._conn.execute(
            "SELECT * FROM entities WHERE qid = ? AND entity_type = ?",
            (qid, entity_type.value),
        ).fetchone()
        return self._to_record(row) if row is not None else None

    def by_exact_name(
        self, norm_name: str, entity_type: EntityType
    ) -> CanonicalEntity | None:
        target = _normalize_surface(norm_name)
        for ent in self._rows_of_type(entity_type):
            if _normalize_surface(ent.canonical_name) == target:
                return ent
            if any(_normalize_surface(a) == target for a in ent.aliases):
                return ent
        return None

    def candidates(
        self, embedding: list[float], entity_type: EntityType, k: int = 5
    ) -> list[tuple[CanonicalEntity, float]]:
        return _top_k_candidates(self._rows_of_type(entity_type), embedding, k)

    def all(self) -> list[CanonicalEntity]:
        rows = self._conn.execute("SELECT * FROM entities").fetchall()
        return [self._to_record(r) for r in rows]

    def _rows_of_type(self, entity_type: EntityType) -> list[CanonicalEntity]:
        rows = self._conn.execute(
            "SELECT * FROM entities WHERE entity_type = ?", (entity_type.value,)
        ).fetchall()
        return [self._to_record(r) for r in rows]

    # -- write -------------------------------------------------------------- #
    def upsert(self, entity: CanonicalEntity) -> None:
        import json

        existing = self.get(entity.canonical_id)
        aliases = (
            _merge_aliases(existing.aliases, entity.aliases)
            if existing is not None
            else _merge_aliases([], entity.aliases)
        )
        # Preserve a prior qid/embedding when the incoming record omits them.
        qid = entity.qid
        if qid is None and existing is not None:
            qid = existing.qid
        embedding = entity.embedding
        if embedding is None and existing is not None:
            embedding = existing.embedding

        self._conn.execute(
            """
            INSERT INTO entities
                (canonical_id, canonical_name, entity_type, qid,
                 embedding_json, aliases_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(canonical_id) DO UPDATE SET
                canonical_name = excluded.canonical_name,
                entity_type    = excluded.entity_type,
                qid            = excluded.qid,
                embedding_json = excluded.embedding_json,
                aliases_json   = excluded.aliases_json
            """,
            (
                entity.canonical_id,
                entity.canonical_name,
                entity.entity_type.value,
                qid,
                json.dumps(embedding) if embedding is not None else None,
                json.dumps(aliases),
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
