"""Subject-entity embedding clustering (spec §7.4a, lever i).

Many extracted claims name the same real-world thing with different surface
forms — *Apple* / *iPhone* / *Apple hardware*. If those stay distinct nodes,
"consensus about Apple" fragments across them and the consensus helper (§8)
undercounts. This module embeds each distinct surface form, clusters
near-duplicates by cosine similarity, and assigns every member of a cluster a
shared, deterministic ``canonical_id``.

Design choices, straight from the spec:

- **Batch, before graph load.** This runs over the *whole* extracted-claim set
  in one pass so the Graphiti bulk loader can bypass per-add LLM node-dedup
  (§7.5/§7.6). It is not an online/per-add resolver.
- **Conservative.** "Ambiguous merges are conservative (prefer leaving separate
  over wrongly merging two distinct entities)." A single similarity threshold
  governs the merge; the greedy single-link pass below only merges a candidate
  into a cluster when it is similar to that cluster's *seed* (representative),
  which is stricter than transitive single-linkage and avoids drift-chaining
  two distinct entities together through a bridge.
- **Type-aware.** Entities of different :class:`~dlogos.schema.EntityType` are
  never merged (a *person* named "Apple" must not collapse into the *org*).
- **Embedder injected.** No model or network in unit tests — any object with
  ``embed(text) -> list[float]`` (optionally ``embed_batch``) works; the heavy
  real embedder (BGE-M3) is built lazily via :func:`default_embedder`.
- **Deterministic.** Surface forms are processed in a stable sorted order and
  ``canonical_id`` is derived from the cluster's chosen canonical name, so the
  same input always yields the same ids regardless of input ordering.

The output ``canonical_id`` is written back onto every claim's
``subject_entity`` (a copy — inputs are not mutated) by :func:`resolve_subjects`.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from dlogos.schema import Entity, EntityType, ExtractedClaim

# Tuned on the validation slice (conftest fake-embedder geometry): the
# Apple-family surface forms sit at cosine ~0.97-0.99 of each other, while
# Apple vs OpenAI is ~0.0. A 0.85 floor cleanly merges the family and refuses
# cross-entity merges, with comfortable margin on both sides.
DEFAULT_SIMILARITY_THRESHOLD: float = 0.85

# Stable id prefix so canonical ids are self-describing in the graph.
_CANONICAL_PREFIX = "ent"


# --------------------------------------------------------------------------- #
# Injected-embedder contract
# --------------------------------------------------------------------------- #
@runtime_checkable
class Embedder(Protocol):
    """Minimal embedder interface the clustering depends on.

    Any object exposing ``embed(text) -> list[float]`` satisfies it. An
    optional ``embed_batch`` is used when present for efficiency; otherwise we
    fall back to per-string ``embed``. The test ``FakeEmbedder`` (conftest)
    implements both — inject it to keep tests network-free and deterministic.
    """

    def embed(self, text: str) -> list[float]:  # pragma: no cover - protocol
        ...


def _embed_all(embedder: Embedder, texts: Sequence[str]) -> list[list[float]]:
    """Embed many strings, preferring a batch method when the embedder has one."""

    batch = getattr(embedder, "embed_batch", None)
    if callable(batch):
        return [list(map(float, v)) for v in batch(list(texts))]
    return [list(map(float, embedder.embed(t))) for t in texts]


def default_embedder() -> Embedder:
    """Build the real open embedding model (BGE-M3) — imported LAZILY.

    Never called by unit tests (which inject a fake). Importing this module
    therefore never pulls FlagEmbedding / sentence-transformers / torch.
    """

    try:  # pragma: no cover - exercised only with the optional ``embed`` extra
        from FlagEmbedding import BGEM3FlagModel  # type: ignore import-not-found
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Subject-entity clustering needs an embedding model. Install the "
            "'embed' extra (FlagEmbedding) or inject an Embedder. For tests, "
            "inject the conftest FakeEmbedder."
        ) from exc

    class _BGEEmbedder:  # pragma: no cover - requires the heavy extra
        def __init__(self) -> None:
            self._model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

        def embed(self, text: str) -> list[float]:
            return self.embed_batch([text])[0]

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            out = self._model.encode(list(texts))["dense_vecs"]
            return [list(map(float, v)) for v in out]

    return _BGEEmbedder()


# --------------------------------------------------------------------------- #
# Result models
# --------------------------------------------------------------------------- #
class EntityCluster(BaseModel):
    """One resolved canonical entity: a group of merged surface forms."""

    model_config = ConfigDict(extra="forbid")

    canonical_id: str = Field(description="Stable id shared by all members.")
    canonical_name: str = Field(
        description="Chosen representative surface form for the cluster."
    )
    entity_type: EntityType
    members: list[str] = Field(
        description="Distinct surface forms merged into this canonical entity."
    )


class SubjectResolution(BaseModel):
    """Output of :func:`resolve_subjects`.

    ``claims`` are copies of the inputs with ``subject_entity.canonical_id``
    filled in. ``clusters`` is the canonical-entity table. ``surface_to_id``
    maps each ``(type, surface form)`` -> ``canonical_id`` for reuse by the
    loader / retrieval consensus bucketing.
    """

    model_config = ConfigDict(extra="forbid")

    claims: list[ExtractedClaim]
    clusters: list[EntityCluster]
    surface_to_id: dict[str, str] = Field(
        description="'<type>\\x1f<surface>' -> canonical_id."
    )


# --------------------------------------------------------------------------- #
# Math + helpers (numpy is a CORE dep, fine to import at module top)
# --------------------------------------------------------------------------- #
def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    import numpy as np

    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def _normalize_surface(name: str) -> str:
    """Casefold + collapse whitespace so exact-but-for-spacing forms collapse."""

    text = unicodedata.normalize("NFKC", name).strip()
    text = re.sub(r"\s+", " ", text)
    return text.casefold()


def _surface_key(entity_type: EntityType, surface: str) -> str:
    # \x1f (unit separator) can't appear in a normal name, so this is collision-safe.
    return f"{entity_type.value}\x1f{surface}"


def _canonical_id(canonical_name: str, entity_type: EntityType) -> str:
    """Deterministic id from the chosen canonical name + type.

    Content-addressed (not positional) so ids are stable across runs and
    independent of input ordering: the same canonical name always maps to the
    same id. Short hex digest keeps it readable in the graph.
    """

    basis = _surface_key(entity_type, _normalize_surface(canonical_name))
    digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]
    return f"{_CANONICAL_PREFIX}-{digest}"


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #
def cluster_entities(
    entities: Sequence[Entity],
    embedder: Embedder,
    *,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[EntityCluster]:
    """Greedy, seed-anchored, type-partitioned clustering of surface forms.

    Algorithm (deliberately simple and conservative):

    1. Collapse to *distinct* ``(type, normalized-surface)`` forms — duplicate
       mentions don't get a vote, and casing/whitespace variants merge for free.
    2. Process distinct forms in a stable sorted order (longest, then
       alphabetical) so the first-seen representative of any family is stable.
    3. For each form, compare its embedding to the *seed* (first member) of each
       existing cluster **of the same type**; join the best-matching cluster if
       its similarity meets ``threshold``, else open a new cluster. Comparing to
       the seed (single representative) rather than to every member prevents
       drift-chaining two genuinely distinct entities through a bridge form.
    4. The canonical name is the cluster seed; the ``canonical_id`` is derived
       deterministically from it.

    Different :class:`EntityType` values never merge.
    """

    # Step 1: distinct (type, normalized surface) -> a display surface form.
    # Keep the first display form we see for a normalized key (stable via sort).
    distinct: dict[tuple[EntityType, str], str] = {}
    for ent in entities:
        norm = _normalize_surface(ent.name)
        if not norm:
            continue
        key = (ent.type, norm)
        if key not in distinct:
            distinct[key] = ent.name.strip()

    # Step 2: stable processing order — longer surfaces first (more specific),
    # then alphabetical by normalized form. Partition by type.
    by_type: dict[EntityType, list[tuple[str, str]]] = {}
    for (etype, norm), display in distinct.items():
        by_type.setdefault(etype, []).append((norm, display))
    for forms in by_type.values():
        forms.sort(key=lambda pair: (-len(pair[0]), pair[0]))

    clusters: list[EntityCluster] = []
    for etype, forms in by_type.items():
        norms = [norm for norm, _ in forms]
        displays = [disp for _, disp in forms]
        vectors = _embed_all(embedder, displays)

        # Each open cluster tracks: seed vector + member surface forms.
        seed_vecs: list[list[float]] = []
        member_lists: list[list[str]] = []
        seed_displays: list[str] = []

        for i, _norm in enumerate(norms):
            vec = vectors[i]
            best_idx = -1
            best_sim = threshold  # must meet-or-exceed to merge
            for c_idx, seed_vec in enumerate(seed_vecs):
                sim = _cosine(vec, seed_vec)
                if sim >= best_sim:
                    best_sim = sim
                    best_idx = c_idx
            if best_idx >= 0:
                member_lists[best_idx].append(displays[i])
            else:
                seed_vecs.append(vec)
                member_lists.append([displays[i]])
                seed_displays.append(displays[i])

        for seed_disp, members in zip(seed_displays, member_lists):
            clusters.append(
                EntityCluster(
                    canonical_id=_canonical_id(seed_disp, etype),
                    canonical_name=seed_disp,
                    entity_type=etype,
                    members=sorted(set(members)),
                )
            )

    # Stable overall ordering of clusters by canonical id.
    clusters.sort(key=lambda c: c.canonical_id)
    return clusters


def resolve_subjects(
    claims: Sequence[ExtractedClaim],
    embedder: Embedder,
    *,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> SubjectResolution:
    """Cluster every claim's ``subject_entity`` and stamp a ``canonical_id``.

    Returns copies of the claims (inputs are never mutated) with
    ``subject_entity.canonical_id`` filled, plus the canonical-entity table and
    a ``(type, surface) -> canonical_id`` lookup the loader / consensus helper
    reuse for bucketing.
    """

    subjects = [c.subject_entity for c in claims]
    clusters = cluster_entities(subjects, embedder, threshold=threshold)

    # Build (type, normalized surface) -> canonical_id from the clusters.
    surface_to_id: dict[str, str] = {}
    for cluster in clusters:
        for member in cluster.members:
            surface_to_id[_surface_key(cluster.entity_type, _normalize_surface(member))] = (
                cluster.canonical_id
            )

    resolved: list[ExtractedClaim] = []
    for claim in claims:
        ent = claim.subject_entity
        key = _surface_key(ent.type, _normalize_surface(ent.name))
        canonical_id = surface_to_id.get(key)
        new_entity = ent.model_copy(update={"canonical_id": canonical_id})
        resolved.append(claim.model_copy(update={"subject_entity": new_entity}))

    return SubjectResolution(
        claims=resolved,
        clusters=clusters,
        surface_to_id=surface_to_id,
    )
