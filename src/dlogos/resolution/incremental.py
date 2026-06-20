"""Incremental, Wikidata-anchored, cascade-matched subject resolution (Task 1.5).

The batch :func:`dlogos.resolution.subjects.resolve_subjects` clusters one
episode's subject entities against *each other* — it has no memory across
episodes, so the same real-world thing ("Apple" in episode 1, "Apple Inc." in
episode 7) shatters into separate canonical nodes and consensus fragments.

:class:`IncrementalResolver` is the drop-in fix. It is a **DROP-IN for
``resolve_subjects``** (same call shape, same :class:`SubjectResolution`
return), but each distinct subject is resolved against the *accumulated*
canonical set held in a persistent :class:`CanonicalEntityStore` — so episode N
resolves against episodes 1..N-1, stably and re-runnably. The per-subject path:

    embed -> anchor_entity (Wikidata QID, optional)
          -> match_entity (rules -> embedding -> optional LLM adjudication)
          -> reuse the matched canonical_id, else MINT
                 ("wd-<qid>" when anchored, else the deterministic
                  content-addressed ``ent-...`` from subjects._canonical_id)
          -> upsert (merging aliases / qid / embedding)

The resolved ``canonical_id`` **and** ``qid`` are stamped onto copies of every
claim's ``subject_entity`` (inputs are never mutated), exactly like
``resolve_subjects`` plus the QID.

Everything reusable comes from siblings — DRY:
``EntityCluster`` / ``SubjectResolution`` / ``_normalize_surface`` /
``_canonical_id`` / ``_embed_all`` from :mod:`dlogos.resolution.subjects`,
``match_entity`` / ``DEFAULT_THRESHOLDS`` from :mod:`dlogos.resolution.cascade`,
``anchor_entity`` from :mod:`dlogos.resolution.wikidata`, and the store
record/Protocol from :mod:`dlogos.resolution.canonical_store`. Nothing heavy is
imported at module top; the store handles its own lazy deps.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from dlogos.resolution.canonical_store import CanonicalEntity, CanonicalEntityStore
from dlogos.resolution.cascade import DEFAULT_THRESHOLDS, TypeThresholds, match_entity
from dlogos.resolution.subjects import (
    Embedder,
    EntityCluster,
    SubjectResolution,
    _canonical_id,
    _embed_all,
    _normalize_surface,
    _surface_key,
)
from dlogos.resolution.wikidata import WikidataLinker, anchor_entity
from dlogos.schema import Entity, EntityType, ExtractedClaim


class IncrementalResolver:
    """Resolve a claim batch's subject entities against a persistent store.

    Constructed with a :class:`CanonicalEntityStore` (the accumulator) and an
    :class:`~dlogos.resolution.subjects.Embedder` (injected; a fake in tests).
    Optional collaborators are defaulted off so the resolver is usable with the
    bare in-memory store and the conftest fake embedder:

    - ``wikidata_linker`` — when present, each person/org subject is anchored to
      a QID via :func:`~dlogos.resolution.wikidata.anchor_entity`; the QID is the
      strongest cross-episode signal (two surface forms sharing a QID are one
      node regardless of embedding distance). When ``None``, no anchoring.
    - ``llm_adjudicator`` — the injected yes/no callable the cascade escalates to
      only for the ambiguous similarity band; ``None`` resolves that band
      conservatively to NEW.
    - ``thresholds`` — per-:class:`EntityType` cosine bars (defaulted to
      :data:`~dlogos.resolution.cascade.DEFAULT_THRESHOLDS`).

    ``resolve(claims)`` returns a :class:`SubjectResolution` so the pipeline can
    call it exactly where it calls ``resolve_subjects``.
    """

    def __init__(
        self,
        store: CanonicalEntityStore,
        embedder: Embedder,
        *,
        wikidata_linker: WikidataLinker | None = None,
        llm_adjudicator: Callable[[Any, Any], bool] | None = None,
        thresholds: dict[EntityType, TypeThresholds] = DEFAULT_THRESHOLDS,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._linker = wikidata_linker
        self._adjudicator = llm_adjudicator
        self._thresholds = thresholds

    # ----------------------------------------------------------------------- #
    # Drop-in for resolve_subjects
    # ----------------------------------------------------------------------- #
    def resolve(self, claims: Sequence[ExtractedClaim]) -> SubjectResolution:
        """Resolve each distinct subject against the store; stamp claims.

        Distinct ``(type, normalized-name)`` forms are collapsed first (duplicate
        mentions don't re-embed), each distinct display form is embedded in one
        batch, then each is anchored + cascade-matched against the accumulated
        store. A match reuses the existing ``canonical_id``; a miss mints a new
        one (``wd-<qid>`` when anchored, else the deterministic ``ent-...``) and
        upserts it so the *next* distinct form — and the next episode — resolves
        against it. ``surface_to_id``/``clusters`` are assembled from the store
        entries touched; ``canonical_id`` and ``qid`` are stamped onto copies of
        every claim's ``subject_entity``.
        """

        # Step 1: collapse to distinct (type, normalized) -> (entity, display).
        # Keep the first-seen Entity for its type; the display form drives embed.
        distinct: dict[tuple[EntityType, str], Entity] = {}
        order: list[tuple[EntityType, str]] = []
        for claim in claims:
            ent = claim.subject_entity
            norm = _normalize_surface(ent.name)
            if not norm:
                continue
            key = (ent.type, norm)
            if key not in distinct:
                distinct[key] = ent
                order.append(key)

        displays = [distinct[key].name.strip() for key in order]
        vectors = _embed_all(self._embedder, displays) if displays else []

        # Step 2: resolve each distinct form against the accumulated store.
        # surface_to_id keys on (type, normalized surface) -> canonical_id.
        # Track which canonical entities were touched (for the clusters table).
        surface_to_id: dict[str, str] = {}
        qid_by_key: dict[tuple[EntityType, str], str | None] = {}
        touched_ids: list[str] = []

        for idx, key in enumerate(order):
            entity = distinct[key]
            emb = vectors[idx]

            qid = (
                anchor_entity(entity, self._linker)
                if self._linker is not None
                else entity.qid
            )
            qid_by_key[key] = qid

            decision = match_entity(
                entity,
                emb,
                self._store,
                qid=qid,
                thresholds=self._thresholds,
                llm_adjudicator=self._adjudicator,
            )

            if decision.canonical_id is not None:
                canonical_id = decision.canonical_id
            elif qid:
                canonical_id = f"wd-{qid}"
            else:
                canonical_id = _canonical_id(entity.name, entity.type)

            # Upsert so later distinct forms (and later episodes) see this entity.
            # The store merges aliases / preserves prior qid+embedding on conflict.
            self._store.upsert(
                CanonicalEntity(
                    canonical_id=canonical_id,
                    canonical_name=entity.name.strip(),
                    entity_type=entity.type,
                    qid=qid,
                    embedding=emb,
                    aliases=[entity.name.strip()],
                )
            )

            surface_to_id[_surface_key(entity.type, key[1])] = canonical_id
            if canonical_id not in touched_ids:
                touched_ids.append(canonical_id)

        # Step 3: stamp canonical_id + qid onto copies of each claim's subject.
        resolved: list[ExtractedClaim] = []
        for claim in claims:
            ent = claim.subject_entity
            norm = _normalize_surface(ent.name)
            key = (ent.type, norm)
            canonical_id = surface_to_id.get(_surface_key(ent.type, norm))
            qid = qid_by_key.get(key, ent.qid)
            new_entity = ent.model_copy(
                update={
                    "canonical_id": canonical_id,
                    # Prefer the freshly anchored qid; fall back to any the
                    # extractor already carried so we never drop a known QID.
                    "qid": qid if qid is not None else ent.qid,
                }
            )
            resolved.append(claim.model_copy(update={"subject_entity": new_entity}))

        # Step 4: assemble the clusters table from the store entries touched.
        clusters: list[EntityCluster] = []
        for canonical_id in touched_ids:
            rec = self._store.get(canonical_id)
            if rec is None:
                continue
            clusters.append(
                EntityCluster(
                    canonical_id=rec.canonical_id,
                    canonical_name=rec.canonical_name,
                    entity_type=rec.entity_type,
                    members=sorted(set(rec.aliases)) or [rec.canonical_name],
                )
            )
        clusters.sort(key=lambda c: c.canonical_id)

        return SubjectResolution(
            claims=resolved,
            clusters=clusters,
            surface_to_id=surface_to_id,
        )
