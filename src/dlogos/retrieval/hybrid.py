"""Hybrid retrieval over the dLogos temporal graph (spec §8).

Three complementary signals, fused with Reciprocal Rank Fusion (RRF):

- **Semantic** — cosine similarity between an injected embedder's query vector
  and each claim's embedding. Finds topically relevant claims even when the
  wording differs.
- **Lexical (BM25)** — exact term / name matching. Catches the precise speaker
  names, tickers, product names that a dense vector blurs. Uses ``rank_bm25``
  if it happens to be installed, otherwise a small pure-python BM25 over the
  core dependency set (no extra install required).
- **Graph traversal** — a breadth-first expansion from the seed claims through
  the store's edges (``about``/``mentions``/``agrees_with``/``supersedes`` …),
  pulling the connected neighborhood. Surfaces claims that share an entity or
  contradict a seed even when they are neither lexically nor semantically close
  to the query.

An optional **temporal filter** restricts results to claims whose validity
window overlaps a requested ``[as_of_start, as_of_end]`` interval — the
bitemporal "what was believed during this period" filter (§6, §8).

The retriever depends only on a small structural :class:`GraphStore` Protocol,
so it is decoupled from the concrete Graphiti/Neo4j store the graph module
builds and is unit-testable with an in-memory fake. Heavy/optional deps are
imported lazily inside functions, never at module top level.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np

from dlogos.schema import SourceSpan, Stance


class TemporalMode(str, Enum):
    """How the temporal window is matched against a claim (spec §6/§8).

    - ``event_time`` — keep claims whose *event-time* (when the claim was made,
      i.e. the episode publish/recording date) falls inside ``[start, end]``.
      This is the "who's been discussing X in the last N months" filter — the
      dominant temporal archetype — and the default.
    - ``validity_overlap`` — keep claims whose *validity interval*
      ``[valid_from or event_time, valid_to or +inf]`` overlaps ``[start, end]``.
      This is the bitemporal "what was believed as-of this period" filter: an
      un-invalidated claim stays valid into the future, so a Jan claim with no
      ``valid_to`` still matches a later window.
    """

    event_time = "event_time"
    validity_overlap = "validity_overlap"

# --------------------------------------------------------------------------- #
# The unit of retrieval and the store contract.
# --------------------------------------------------------------------------- #


@dataclass
class RetrievableClaim:
    """A claim as the retriever sees it — flattened for indexing.

    The graph store materializes these from its reified Claim nodes. ``text`` is
    the searchable surface (typically the claim's object plus subject), used for
    both lexical and (when no precomputed vector is given) semantic matching.
    ``embedding`` is the precomputed claim vector when the store has one;
    otherwise the retriever embeds ``text`` on the fly with the injected
    embedder. ``neighbor_ids`` are the claim ids reachable in one hop, used by
    graph traversal. ``event_time`` / ``valid_from`` / ``valid_to`` drive the
    temporal filter.
    """

    claim_id: str
    text: str
    speaker_id: str | None = None
    subject_id: str | None = None
    stance: Stance | None = None
    source_span: SourceSpan | None = None
    embedding: list[float] | None = None
    neighbor_ids: tuple[str, ...] = ()
    event_time: datetime | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None


@runtime_checkable
class GraphStore(Protocol):
    """Structural contract the retriever needs from a graph store.

    The concrete Graphiti/Neo4j store (and the in-memory fake used in tests)
    satisfy this without importing anything from here. Kept intentionally small:
    list claims, fetch one, and walk one hop of neighbors.
    """

    def all_claims(self) -> list[RetrievableClaim]:
        """Every retrievable claim. (At PoC scale this is fine; the firehose
        path would push filtering into the store.)"""
        ...

    def get_claim(self, claim_id: str) -> RetrievableClaim | None:
        """Fetch a single claim by id, or ``None`` if absent."""
        ...

    def neighbors(self, claim_id: str) -> list[str]:
        """Claim ids reachable from ``claim_id`` in one hop."""
        ...


class Embedder(Protocol):
    """Minimal embedder surface (the test ``FakeEmbedder`` satisfies this)."""

    def embed(self, text: str) -> list[float]:
        ...


# --------------------------------------------------------------------------- #
# Adapter: materialize RetrievableClaims from the graph module's store.
# --------------------------------------------------------------------------- #
def claims_from_graph_store(
    store: object,
    *,
    as_of: datetime | None = None,
    embedder: Embedder | None = None,
) -> list[RetrievableClaim]:
    """Build :class:`RetrievableClaim`\\ s from a :class:`dlogos.graph` store.

    The graph module owns its own richer store contract (``ClaimNode`` /
    ``QueryResult`` / bitemporal edges). Retrieval does not depend on those
    types at import time — this adapter imports them lazily so that importing
    :mod:`dlogos.retrieval` never pulls the graph subpackage, and so unit tests
    can use the in-memory fake in this module without it.

    The adapter reads live claim rows via ``store.query(as_of=...)``, flattens
    each to a ``RetrievableClaim`` (searchable text = subject name + object),
    and derives one-hop ``neighbor_ids`` from the store's ``edges`` that touch
    the claim (``agrees_with`` / ``disputes`` / ``supersedes`` / shared-entity
    via ``about``/``mentions`` are all collapsed into adjacency). ``embedding``
    is filled by ``embedder`` when provided, else left ``None`` (the retriever
    will embed on demand).
    """

    rows = store.query(as_of=as_of)  # type: ignore[attr-defined]

    # Build claim->neighbor adjacency from any claim-to-claim edges the store
    # exposes. We look for an ``edges`` mapping (the fake store has one); the
    # real store would expose its own neighbor accessor in the spike.
    adjacency: dict[str, set[str]] = {}
    edges = getattr(store, "edges", None)
    if edges:
        edge_iter = edges.values() if hasattr(edges, "values") else edges
        for edge in edge_iter:
            src = getattr(edge, "src_id", None)
            dst = getattr(edge, "dst_id", None)
            if getattr(edge, "invalidated", False):
                continue
            if src is None or dst is None:
                continue
            adjacency.setdefault(src, set()).add(dst)
            adjacency.setdefault(dst, set()).add(src)

    out: list[RetrievableClaim] = []
    for row in rows:
        claim = row.claim
        subject_name = row.subject.name if getattr(row, "subject", None) else ""
        text = f"{subject_name} {claim.object}".strip()
        out.append(
            RetrievableClaim(
                claim_id=claim.claim_id,
                text=text,
                speaker_id=claim.speaker_id,
                subject_id=claim.subject_canonical_id,
                stance=claim.stance,
                source_span=claim.source_span,
                embedding=embedder.embed(text) if embedder is not None else None,
                neighbor_ids=tuple(sorted(adjacency.get(claim.claim_id, ()))),
                event_time=getattr(row, "event_time", None),
            )
        )
    return out


class _InMemoryStore:
    """A trivial :class:`GraphStore` backed by a list of claims.

    Convenience wrapper so callers (and the adapter above) can hand a flat list
    of :class:`RetrievableClaim`\\ s to :class:`HybridRetriever` without writing
    a store class. Neighbor lookups read each claim's ``neighbor_ids``.
    """

    def __init__(self, claims: list[RetrievableClaim]) -> None:
        self._claims = list(claims)
        self._by_id = {c.claim_id: c for c in self._claims}

    def all_claims(self) -> list[RetrievableClaim]:
        return list(self._claims)

    def get_claim(self, claim_id: str) -> RetrievableClaim | None:
        return self._by_id.get(claim_id)

    def neighbors(self, claim_id: str) -> list[str]:
        c = self._by_id.get(claim_id)
        return list(c.neighbor_ids) if c else []


def in_memory_store(claims: list[RetrievableClaim]) -> GraphStore:
    """Wrap a flat list of claims as a :class:`GraphStore`.

    Handy for the eval's naive-vector-RAG arm and for tests: hand it claims and
    get a store the :class:`HybridRetriever` can run over directly.
    """

    return _InMemoryStore(claims)


# --------------------------------------------------------------------------- #
# Reciprocal Rank Fusion — the pure fuser.
# --------------------------------------------------------------------------- #
def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    *,
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse several ranked id-lists into one, by Reciprocal Rank Fusion.

    RRF scores an item by ``sum_over_lists( weight / (k + rank) )`` where
    ``rank`` is 1-based within each list it appears in. It needs no score
    calibration across the signals — only their *orderings* — which is exactly
    why it suits fusing cosine, BM25, and traversal, whose raw scores are not
    comparable.

    Parameters
    ----------
    ranked_lists:
        Each inner list is ids ordered best-first for one signal.
    k:
        The RRF damping constant (the standard default is 60). Larger ``k``
        flattens the contribution of top ranks.
    weights:
        Optional per-list weights (same length as ``ranked_lists``); default is
        equal weight. Lets a caller lean on, say, lexical over semantic.

    Returns
    -------
    list[tuple[str, float]]
        ``(id, fused_score)`` pairs, highest score first. Ties are broken
        deterministically by id so output ordering is stable.
    """

    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must match the number of ranked lists")

    scores: dict[str, float] = {}
    for lst, weight in zip(ranked_lists, weights):
        for rank, item in enumerate(lst, start=1):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)

    # Sort by score desc, then id asc for a deterministic, stable order.
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


# --------------------------------------------------------------------------- #
# Lexical scoring — BM25 with a pure-python fallback (core-dep only).
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class _PurePythonBM25:
    """A small, dependency-free Okapi BM25 over a fixed corpus.

    Deterministic and import-light — used when ``rank_bm25`` is not installed
    (it is not a core dependency). Scoring matches standard Okapi BM25 with the
    usual ``k1=1.5``, ``b=0.75`` defaults.
    """

    def __init__(
        self, corpus_tokens: list[list[str]], *, k1: float = 1.5, b: float = 0.75
    ) -> None:
        self.k1 = k1
        self.b = b
        self.corpus_tokens = corpus_tokens
        self.n_docs = len(corpus_tokens)
        self.doc_len = [len(d) for d in corpus_tokens]
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0

        # Document frequency per term.
        df: dict[str, int] = {}
        for tokens in corpus_tokens:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1
        self.df = df

        # Per-doc term frequencies.
        self.tf: list[dict[str, int]] = []
        for tokens in corpus_tokens:
            counts: dict[str, int] = {}
            for term in tokens:
                counts[term] = counts.get(term, 0) + 1
            self.tf.append(counts)

    def _idf(self, term: str) -> float:
        # BM25+-style non-negative idf (the rank_bm25 'BM25Okapi' floors at ~eps,
        # we floor at 0 to avoid negative contributions for very common terms).
        n_q = self.df.get(term, 0)
        if n_q == 0:
            return 0.0
        return max(0.0, math.log((self.n_docs - n_q + 0.5) / (n_q + 0.5) + 1.0))

    def scores(self, query_tokens: list[str]) -> list[float]:
        out = [0.0] * self.n_docs
        if not self.n_docs:
            return out
        for term in query_tokens:
            idf = self._idf(term)
            if idf == 0.0:
                continue
            for i in range(self.n_docs):
                freq = self.tf[i].get(term, 0)
                if freq == 0:
                    continue
                denom = freq + self.k1 * (
                    1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1.0)
                )
                out[i] += idf * (freq * (self.k1 + 1)) / denom
        return out


def _build_bm25(corpus_tokens: list[list[str]]):
    """Build a BM25 scorer, preferring ``rank_bm25`` if available (lazy import).

    Returns an object with a ``.get_scores(query_tokens) -> list[float]`` method
    (the rank_bm25 API) or a wrapper exposing the same, so callers don't care
    which backend ran.
    """

    try:  # pragma: no cover - exercised only when rank_bm25 is installed
        from rank_bm25 import BM25Okapi  # type: ignore

        return BM25Okapi(corpus_tokens)
    except Exception:
        bm25 = _PurePythonBM25(corpus_tokens)

        class _Adapter:
            def get_scores(self, query_tokens: list[str]) -> list[float]:
                return bm25.scores(query_tokens)

        return _Adapter()


# --------------------------------------------------------------------------- #
# Cosine helper.
# --------------------------------------------------------------------------- #
def _cosine(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(va @ vb) / (na * nb)


def _as_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


# --------------------------------------------------------------------------- #
# Result object.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RetrievalResult:
    """A fused retrieval hit with per-signal provenance.

    ``score`` is the fused RRF score; the per-signal ranks (1-based, or ``None``
    if the claim was not surfaced by that signal) let a caller see *why* a claim
    ranked where it did — useful both for debugging and for the eval's
    provenance inspection.
    """

    claim: RetrievableClaim
    score: float
    semantic_rank: int | None = None
    lexical_rank: int | None = None
    graph_rank: int | None = None


# --------------------------------------------------------------------------- #
# The retriever.
# --------------------------------------------------------------------------- #
class HybridRetriever:
    """Semantic + BM25 + graph-traversal retrieval, fused with RRF.

    Inject the :class:`GraphStore` and an :class:`Embedder`; both are small
    structural protocols so unit tests pass in in-memory fakes. No network, no
    heavy deps at construction or call time.
    """

    def __init__(
        self,
        store: GraphStore,
        embedder: Embedder,
        *,
        rrf_k: int = 60,
        semantic_weight: float = 1.0,
        lexical_weight: float = 1.0,
        graph_weight: float = 1.0,
        traversal_depth: int = 1,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self.graph_weight = graph_weight
        self.traversal_depth = traversal_depth

    # -- individual signals ------------------------------------------------- #
    def _semantic_ranking(
        self, query: str, claims: list[RetrievableClaim]
    ) -> list[str]:
        """Claim ids ordered by cosine similarity to the query, best first."""

        qvec = self.embedder.embed(query)
        scored: list[tuple[str, float]] = []
        for c in claims:
            vec = c.embedding if c.embedding is not None else self.embedder.embed(c.text)
            scored.append((c.claim_id, _cosine(qvec, vec)))
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return [cid for cid, _ in scored]

    def _lexical_ranking(
        self, query: str, claims: list[RetrievableClaim]
    ) -> list[str]:
        """Claim ids ordered by BM25 score, best first."""

        corpus_tokens = [_tokenize(c.text) for c in claims]
        bm25 = _build_bm25(corpus_tokens)
        raw = bm25.get_scores(_tokenize(query))
        scores = [float(s) for s in raw]
        order = sorted(
            range(len(claims)),
            key=lambda i: (-scores[i], claims[i].claim_id),
        )
        return [claims[i].claim_id for i in order]

    def _graph_ranking(
        self, seed_ids: list[str], universe: set[str]
    ) -> list[str]:
        """Breadth-first traversal from the seeds, ordered by hop distance.

        Closer (fewer hops) ranks higher; within a hop level ties break by id
        for determinism. Seeds themselves are included at distance 0. Only ids
        within ``universe`` (e.g. the temporally-filtered set) are emitted.
        """

        order: list[str] = []
        seen: set[str] = set()
        # frontier as a sorted list keeps traversal deterministic.
        frontier = sorted(s for s in seed_ids if s in universe)
        depth = 0
        while frontier and depth <= self.traversal_depth:
            next_frontier: list[str] = []
            for cid in frontier:
                if cid in seen:
                    continue
                seen.add(cid)
                order.append(cid)
                for nb in sorted(self.store.neighbors(cid)):
                    if nb in universe and nb not in seen:
                        next_frontier.append(nb)
            frontier = sorted(set(next_frontier))
            depth += 1
        return order

    # -- temporal filter ---------------------------------------------------- #
    @staticmethod
    def _passes_temporal(
        claim: RetrievableClaim,
        start: datetime | None,
        end: datetime | None,
        mode: TemporalMode,
    ) -> bool:
        """Whether a claim passes the temporal window under ``mode``.

        A claim with no temporal info at all is treated as always-valid under
        both modes (it is not excluded by a filter it cannot answer to).
        """

        if start is not None:
            start = _as_aware(start)
        if end is not None:
            end = _as_aware(end)

        if mode is TemporalMode.event_time:
            et = claim.event_time or claim.valid_from
            if et is None:
                return True  # cannot place it; do not exclude
            et = _as_aware(et)
            if start is not None and et < start:
                return False
            if end is not None and et > end:
                return False
            return True

        # validity_overlap: window [valid_from or event_time, valid_to or +inf].
        lower = claim.valid_from or claim.event_time
        upper = claim.valid_to
        if lower is not None:
            lower = _as_aware(lower)
        if upper is not None:
            upper = _as_aware(upper)

        if end is not None and lower is not None and lower > end:
            return False
        if start is not None and upper is not None and upper < start:
            return False
        return True

    # -- public API --------------------------------------------------------- #
    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        as_of_start: datetime | None = None,
        as_of_end: datetime | None = None,
        temporal_mode: TemporalMode = TemporalMode.event_time,
        traverse: bool = True,
        seed_k: int = 5,
    ) -> list[RetrievalResult]:
        """Run hybrid retrieval and return the top fused results.

        Parameters
        ----------
        query:
            Natural-language query string.
        top_k:
            Number of fused results to return.
        as_of_start / as_of_end:
            Optional temporal window; only claims passing the window under
            ``temporal_mode`` are considered. Either bound may be ``None``
            (open).
        temporal_mode:
            ``event_time`` (default) keeps claims *made* within the window —
            the "who discussed X in the last N months" filter.
            ``validity_overlap`` keeps claims whose validity interval overlaps
            the window — the bitemporal "what was believed as-of" filter.
        traverse:
            Whether to include the graph-traversal signal. The traversal seeds
            from the top ``seed_k`` claims of the *fused semantic+lexical* order
            so it expands around the already-relevant neighborhood.
        seed_k:
            Number of seed claims for traversal.
        """

        claims = self.store.all_claims()

        # Temporal filter first — it defines the universe every signal works on.
        if as_of_start is not None or as_of_end is not None:
            claims = [
                c
                for c in claims
                if self._passes_temporal(
                    c, as_of_start, as_of_end, temporal_mode
                )
            ]
        if not claims:
            return []

        by_id = {c.claim_id: c for c in claims}
        universe = set(by_id)

        semantic = self._semantic_ranking(query, claims)
        lexical = self._lexical_ranking(query, claims)

        ranked_lists = [semantic, lexical]
        weights = [self.semantic_weight, self.lexical_weight]

        graph: list[str] = []
        if traverse and self.traversal_depth > 0:
            # Seed traversal from the fused content order so we expand around
            # the most relevant claims, not arbitrary ones.
            seed_fused = reciprocal_rank_fusion(
                [semantic, lexical],
                k=self.rrf_k,
                weights=[self.semantic_weight, self.lexical_weight],
            )
            seeds = [cid for cid, _ in seed_fused[:seed_k]]
            graph = self._graph_ranking(seeds, universe)
            if graph:
                ranked_lists.append(graph)
                weights.append(self.graph_weight)

        fused = reciprocal_rank_fusion(ranked_lists, k=self.rrf_k, weights=weights)

        sem_rank = {cid: i + 1 for i, cid in enumerate(semantic)}
        lex_rank = {cid: i + 1 for i, cid in enumerate(lexical)}
        gph_rank = {cid: i + 1 for i, cid in enumerate(graph)}

        results: list[RetrievalResult] = []
        for cid, score in fused[:top_k]:
            results.append(
                RetrievalResult(
                    claim=by_id[cid],
                    score=score,
                    semantic_rank=sem_rank.get(cid),
                    lexical_rank=lex_rank.get(cid),
                    graph_rank=gph_rank.get(cid),
                )
            )
        return results
