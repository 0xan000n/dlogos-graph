"""The dLogos MCP server and its testable handler functions (spec §8, §9).

Five tools expose the temporal dialogue graph to a model:

- ``search_dialogue(query, top_k, since, until)`` — hybrid retrieval over the
  graph, optionally restricted to an event-time window. The general "what was
  said about this" entry point.
- ``who_discussed(topic, since)`` — which attributed speakers discussed a topic
  (optionally since a date), with the spans that anchor each.
- ``consensus_trend(subject, window_days)`` — the headline primitive: how the
  consensus on a subject moved over time, bucketed by the resolved
  ``canonical_id`` so *Apple* / *iPhone* / *Apple hardware* aggregate (§7.4a).
- ``belief_history(person, subject)`` — one person's stance on a subject over
  time: the belief-tracking view (guests, not just hosts, are subjects — §3).
- ``provenance_lookup(claim)`` — resolve a claim reference back to its source
  span (episode + timestamp + attributed speaker): "where was X discussed".

The split that keeps the tools testable **without the ``mcp`` package**:

- Each handler (``*_handler``) is a plain function over an injected
  :class:`RetrievalSurface` and returns a Pydantic result model. No ``mcp``
  import, no network, no heavy deps — unit tests call these directly.
- :func:`build_server` lazily imports ``mcp`` *inside the function* and
  registers each tool as a thin wrapper that calls the matching handler. So
  importing this module never requires ``mcp`` to be installed.

``RetrievalSurface`` is a small structural Protocol: the real adapter wraps
``dlogos.retrieval`` (``HybridRetriever`` + ``consensus_over_time``) and the
graph store; tests pass a fake. The handlers depend only on this surface, never
on the concrete retriever, so they stay decoupled and deterministic.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from dlogos.retrieval.consensus import ConsensusTrend, consensus_over_time
from dlogos.retrieval.hybrid import (
    HybridRetriever,
    RetrievalResult,
    TemporalMode,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from dlogos.retrieval.hybrid import Embedder
    from dlogos.schema import ExtractedClaim

# --------------------------------------------------------------------------- #
# The surface the handlers delegate to (injected; a fake satisfies it in tests)
# --------------------------------------------------------------------------- #


@runtime_checkable
class RetrievalSurface(Protocol):
    """What the MCP handlers need from the retrieval/consensus layer.

    Deliberately small and structural so the concrete adapter (over
    :class:`~dlogos.retrieval.hybrid.HybridRetriever` +
    :func:`~dlogos.retrieval.consensus.consensus_over_time`) and an in-memory
    fake both satisfy it without importing anything from here.
    """

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[RetrievalResult]:
        """Hybrid retrieval, optionally restricted to an event-time window."""
        ...

    def consensus(
        self,
        subject: str,
        *,
        window_days: int = 30,
    ) -> ConsensusTrend:
        """Consensus-over-time for a subject, bucketed at ``window_days``."""
        ...

    def provenance(self, claim_ref: str) -> RetrievalResult | None:
        """Resolve a claim reference (id or text) back to its source span."""
        ...


# --------------------------------------------------------------------------- #
# Result models — plain Pydantic, JSON-serializable for the MCP tool boundary
# --------------------------------------------------------------------------- #
class DialogueHit(BaseModel):
    """One retrieved claim, flattened to what a tool caller needs to read it."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    text: str
    speaker_id: str | None = None
    subject_id: str | None = None
    stance: str | None = None
    episode_id: str | None = None
    t_start: float | None = None
    t_end: float | None = None
    event_time: datetime | None = None
    score: float = 0.0


class SearchDialogueResult(BaseModel):
    """Result of ``search_dialogue``: ranked hits for a query/window."""

    model_config = ConfigDict(extra="forbid")

    query: str
    since: datetime | None = None
    until: datetime | None = None
    hits: list[DialogueHit] = Field(default_factory=list)


class SpeakerMention(BaseModel):
    """A speaker plus the span that anchors their discussion of the topic."""

    model_config = ConfigDict(extra="forbid")

    speaker_id: str
    episode_id: str | None = None
    t_start: float | None = None
    t_end: float | None = None
    snippet: str = ""
    event_time: datetime | None = None


class WhoDiscussedResult(BaseModel):
    """Result of ``who_discussed``: distinct attributed speakers + anchors."""

    model_config = ConfigDict(extra="forbid")

    topic: str
    since: datetime | None = None
    speakers: list[str] = Field(default_factory=list)
    mentions: list[SpeakerMention] = Field(default_factory=list)


class TrendPoint(BaseModel):
    """One bucket of the consensus timeline, JSON-friendly."""

    model_config = ConfigDict(extra="forbid")

    start: datetime
    end: datetime
    claim_count: int
    net_sentiment: float
    net_stance: float
    speakers: list[str] = Field(default_factory=list)


class ConsensusTrendResult(BaseModel):
    """Result of ``consensus_trend``: the shift + per-bucket timeline."""

    model_config = ConfigDict(extra="forbid")

    subject: str
    direction: str
    sentiment_delta: float
    all_speakers: list[str] = Field(default_factory=list)
    points: list[TrendPoint] = Field(default_factory=list)


class BeliefPoint(BaseModel):
    """One step in a person's stance on a subject over time."""

    model_config = ConfigDict(extra="forbid")

    start: datetime
    end: datetime
    claim_count: int
    net_sentiment: float
    net_stance: float


class BeliefHistoryResult(BaseModel):
    """Result of ``belief_history``: one person's stance on a subject in time."""

    model_config = ConfigDict(extra="forbid")

    person: str
    subject: str
    direction: str
    sentiment_delta: float
    points: list[BeliefPoint] = Field(default_factory=list)
    found: bool = True


class ProvenanceResult(BaseModel):
    """Result of ``provenance_lookup``: a claim resolved to its source span."""

    model_config = ConfigDict(extra="forbid")

    claim_ref: str
    found: bool
    claim_id: str | None = None
    text: str | None = None
    speaker_id: str | None = None
    episode_id: str | None = None
    t_start: float | None = None
    t_end: float | None = None
    event_time: datetime | None = None


# --------------------------------------------------------------------------- #
# Flattening helpers (pure)
# --------------------------------------------------------------------------- #
def _hit_from_result(result: RetrievalResult) -> DialogueHit:
    claim = result.claim
    span = claim.source_span
    return DialogueHit(
        claim_id=claim.claim_id,
        text=claim.text,
        speaker_id=claim.speaker_id,
        subject_id=claim.subject_id,
        stance=claim.stance.value if claim.stance is not None else None,
        episode_id=span.episode_id if span is not None else None,
        t_start=span.t_start if span is not None else None,
        t_end=span.t_end if span is not None else None,
        event_time=claim.event_time,
        score=result.score,
    )


# --------------------------------------------------------------------------- #
# Handlers — plain functions, no ``mcp`` import, fully unit-testable
# --------------------------------------------------------------------------- #
def search_dialogue_handler(
    surface: RetrievalSurface,
    query: str,
    *,
    top_k: int = 10,
    since: datetime | None = None,
    until: datetime | None = None,
) -> SearchDialogueResult:
    """Hybrid retrieval over the graph, optionally within an event-time window.

    The general "what was said about this" tool. ``since``/``until`` bound the
    event-time (when it was said) so a caller can ask "what was believed during
    this period" — the bitemporal filter the graph exists to serve.
    """

    results = surface.search(query, top_k=top_k, since=since, until=until)
    return SearchDialogueResult(
        query=query,
        since=since,
        until=until,
        hits=[_hit_from_result(r) for r in results],
    )


def who_discussed_handler(
    surface: RetrievalSurface,
    topic: str,
    *,
    since: datetime | None = None,
    top_k: int = 25,
) -> WhoDiscussedResult:
    """Which distinct attributed speakers discussed ``topic`` (since a date).

    De-duplicates speakers by resolved id, preserving first-appearance order
    (the cast of the story), and keeps the first anchoring span per speaker so a
    caller can jump to where each said it. Hits with no resolved speaker are
    skipped — attribution is the whole point of this tool.
    """

    results = surface.search(topic, top_k=top_k, since=since, until=None)

    speakers: list[str] = []
    seen: set[str] = set()
    mentions: list[SpeakerMention] = []
    for r in results:
        claim = r.claim
        spk = claim.speaker_id
        if not spk or spk in seen:
            continue
        seen.add(spk)
        speakers.append(spk)
        span = claim.source_span
        mentions.append(
            SpeakerMention(
                speaker_id=spk,
                episode_id=span.episode_id if span is not None else None,
                t_start=span.t_start if span is not None else None,
                t_end=span.t_end if span is not None else None,
                snippet=claim.text,
                event_time=claim.event_time,
            )
        )
    return WhoDiscussedResult(
        topic=topic, since=since, speakers=speakers, mentions=mentions
    )


def consensus_trend_handler(
    surface: RetrievalSurface,
    subject: str,
    *,
    window_days: int = 30,
) -> ConsensusTrendResult:
    """How the consensus on ``subject`` moved over time.

    Delegates to the surface's consensus primitive (bucketed by the resolved
    ``canonical_id`` so surface variants of the subject aggregate) and flattens
    the trend to a JSON-friendly timeline.
    """

    trend = surface.consensus(subject, window_days=window_days)
    return _trend_result(trend)


def _trend_result(trend: ConsensusTrend) -> ConsensusTrendResult:
    return ConsensusTrendResult(
        subject=trend.subject,
        direction=trend.direction.value,
        sentiment_delta=trend.sentiment_delta,
        all_speakers=list(trend.all_speakers),
        points=[
            TrendPoint(
                start=b.start,
                end=b.end,
                claim_count=b.claim_count,
                net_sentiment=b.net_sentiment,
                net_stance=b.net_stance,
                speakers=list(b.speakers),
            )
            for b in trend.buckets
        ],
    )


def belief_history_handler(
    surface: RetrievalSurface,
    person: str,
    subject: str,
    *,
    window_days: int = 30,
) -> BeliefHistoryResult:
    """One person's stance on a subject over time (the belief-tracking view).

    Runs the subject's consensus trend, then keeps only the buckets in which the
    person actually spoke (their per-speaker contribution). The per-person
    timeline is what makes "did this guest change their mind about X" answerable.
    """

    trend = surface.consensus(subject, window_days=window_days)

    points: list[BeliefPoint] = []
    sentiments: list[float] = []
    for b in trend.buckets:
        contribution = b.speaker_breakdown.get(person)
        if contribution is None:
            continue
        count, mean_sentiment = contribution
        points.append(
            BeliefPoint(
                start=b.start,
                end=b.end,
                claim_count=count,
                net_sentiment=mean_sentiment,
                # No per-speaker stance scalar is carried in the bucket; the
                # bucket-level net_stance is the closest signal available.
                net_stance=b.net_stance,
            )
        )
        sentiments.append(mean_sentiment)

    delta = (sentiments[-1] - sentiments[0]) if len(sentiments) >= 2 else 0.0
    if len(sentiments) < 2:
        direction = "insufficient"
    elif delta > 0.1:
        direction = "rising"
    elif delta < -0.1:
        direction = "falling"
    else:
        direction = "flat"

    return BeliefHistoryResult(
        person=person,
        subject=subject,
        direction=direction,
        sentiment_delta=delta,
        points=points,
        found=bool(points),
    )


def provenance_lookup_handler(
    surface: RetrievalSurface,
    claim: str,
) -> ProvenanceResult:
    """Resolve a claim reference back to its source span.

    ``claim`` is a claim id (or a text reference the surface can resolve). The
    point is provenance integrity (§9): every assertion traces to a real span,
    not a confident hallucination. Returns ``found=False`` when nothing resolves
    rather than fabricating a span.
    """

    result = surface.provenance(claim)
    if result is None:
        return ProvenanceResult(claim_ref=claim, found=False)

    c = result.claim
    span = c.source_span
    return ProvenanceResult(
        claim_ref=claim,
        found=True,
        claim_id=c.claim_id,
        text=c.text,
        speaker_id=c.speaker_id,
        episode_id=span.episode_id if span is not None else None,
        t_start=span.t_start if span is not None else None,
        t_end=span.t_end if span is not None else None,
        event_time=c.event_time,
    )


# --------------------------------------------------------------------------- #
# Concrete surface — wires the handlers to the REAL retrieval/consensus objects
# --------------------------------------------------------------------------- #
class GraphRetrievalSurface:
    """A real :class:`RetrievalSurface` over the dLogos retrieval stack.

    This is the adapter that makes the MCP tools delegate to the *real*
    retrieval objects rather than a fake: ``search`` runs the
    :class:`~dlogos.retrieval.hybrid.HybridRetriever` (semantic + BM25 + graph
    traversal, RRF-fused) over claims materialized from the graph store, with
    the event-time window applied; ``consensus`` runs the pure
    :func:`~dlogos.retrieval.consensus.consensus_over_time` over the resolved
    claim set bucketed by ``canonical_id``; ``provenance`` resolves a claim
    reference (id, then text) back to its source span.

    The retriever and the consensus claim set are both supplied at construction
    so this object never touches a heavy dep — :meth:`from_graph_store` builds
    one from a graph store + embedder + the resolved claims/event-times the
    pipeline already produced.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        *,
        consensus_claims: "list[ExtractedClaim]" | None = None,
        event_times: dict[str, datetime] | None = None,
    ) -> None:
        self._retriever = retriever
        self._consensus_claims = list(consensus_claims or [])
        self._event_times = dict(event_times or {})

    @classmethod
    def from_graph_store(
        cls,
        store: object,
        embedder: "Embedder",
        *,
        consensus_claims: "list[ExtractedClaim]" | None = None,
        event_times: dict[str, datetime] | None = None,
        **retriever_kwargs: object,
    ) -> "GraphRetrievalSurface":
        """Build a surface from a graph store + embedder (the wiring entry point).

        Materializes :class:`~dlogos.retrieval.hybrid.RetrievableClaim`\\ s from
        the store (live rows + one-hop adjacency) via
        :func:`~dlogos.retrieval.hybrid.claims_from_graph_store`, wraps them in an
        in-memory retrieval store, and constructs a
        :class:`~dlogos.retrieval.hybrid.HybridRetriever`. The resolved
        ``consensus_claims`` (the same ``ExtractedClaim``\\ s loaded into the
        graph) and their ``event_times`` power the consensus primitive, which
        needs stance/sentiment/confidence the flattened graph rows do not carry.
        """

        from dlogos.retrieval.hybrid import (
            claims_from_graph_store,
            in_memory_store,
        )

        retrievable = claims_from_graph_store(store, embedder=embedder)
        retriever = HybridRetriever(
            in_memory_store(retrievable),
            embedder,
            **retriever_kwargs,  # type: ignore[arg-type]
        )
        return cls(
            retriever,
            consensus_claims=consensus_claims,
            event_times=event_times,
        )

    # -- RetrievalSurface contract ----------------------------------------- #
    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[RetrievalResult]:
        """Hybrid retrieval, optionally restricted to an event-time window."""

        return self._retriever.retrieve(
            query,
            top_k=top_k,
            as_of_start=since,
            as_of_end=until,
            temporal_mode=TemporalMode.event_time,
        )

    def consensus(self, subject: str, *, window_days: int = 30) -> ConsensusTrend:
        """Consensus-over-time for a subject, bucketed at ``window_days``."""

        return consensus_over_time(
            self._consensus_claims,
            self._event_times,
            subject=subject,
            bucket=timedelta(days=window_days),
        )

    def provenance(self, claim_ref: str) -> RetrievalResult | None:
        """Resolve a claim reference (id first, then a text match) to its span."""

        claims = self._retriever.store.all_claims()
        for c in claims:
            if c.claim_id == claim_ref:
                return RetrievalResult(claim=c, score=1.0)
        ref_lower = claim_ref.strip().lower()
        if ref_lower:
            for c in claims:
                if ref_lower in c.text.lower():
                    return RetrievalResult(claim=c, score=1.0)
        return None


# --------------------------------------------------------------------------- #
# The MCP server — lazy ``mcp`` import, thin wrappers over the handlers
# --------------------------------------------------------------------------- #
def build_server(surface: RetrievalSurface, *, name: str = "dlogos"):
    """Build a ``FastMCP`` server exposing the five tools over ``surface``.

    ``mcp`` is imported **inside** this function (HARD CONSTRAINT: heavy/optional
    deps never at module top level), so importing :mod:`dlogos.mcp.server` for
    the handler unit tests never requires the ``mcp`` package to be installed.

    Each registered tool is a thin wrapper that parses optional ISO-date strings
    and delegates to the matching ``*_handler``; the result models serialize to
    JSON-able dicts at the tool boundary. Run the returned server with
    ``server.run()`` (stdio) in a deployment context.
    """

    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore
    except Exception as exc:  # pragma: no cover - only when mcp is absent
        raise RuntimeError(
            "The 'mcp' optional dependency is required to build the MCP server. "
            "Install the 'mcp' extra. (Handlers are usable without it.)"
        ) from exc

    server = FastMCP(name)

    def _parse_dt(value: str | None) -> datetime | None:
        if value is None or value == "":
            return None
        return datetime.fromisoformat(value)

    @server.tool()
    def search_dialogue(
        query: str,
        top_k: int = 10,
        since: str | None = None,
        until: str | None = None,
    ) -> dict:
        """Search the dialogue graph for what was said about a query.

        Optionally restrict to an event-time window via ISO ``since``/``until``.
        """
        return search_dialogue_handler(
            surface,
            query,
            top_k=top_k,
            since=_parse_dt(since),
            until=_parse_dt(until),
        ).model_dump(mode="json")

    @server.tool()
    def who_discussed(topic: str, since: str | None = None) -> dict:
        """List the attributed speakers who discussed a topic (since a date)."""
        return who_discussed_handler(
            surface, topic, since=_parse_dt(since)
        ).model_dump(mode="json")

    @server.tool()
    def consensus_trend(subject: str, window_days: int = 30) -> dict:
        """Show how the consensus on a subject moved over time."""
        return consensus_trend_handler(
            surface, subject, window_days=window_days
        ).model_dump(mode="json")

    @server.tool()
    def belief_history(person: str, subject: str, window_days: int = 30) -> dict:
        """Show one person's stance on a subject over time."""
        return belief_history_handler(
            surface, person, subject, window_days=window_days
        ).model_dump(mode="json")

    @server.tool()
    def provenance_lookup(claim: str) -> dict:
        """Resolve a claim reference to its source episode + timestamp."""
        return provenance_lookup_handler(surface, claim).model_dump(mode="json")

    return server
