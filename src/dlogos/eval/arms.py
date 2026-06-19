"""The four eval arms (spec §9), as async callables behind one interface.

Each arm answers a :class:`~dlogos.eval.golden.GoldenQuery` and returns an
:class:`Answer` (text + citations). The four arms (spec §9):

1. :class:`ModelAloneArm` -- the frontier model with no tools. The floor.
2. :class:`ModelWebSearchArm` -- model + live web search. The Perplexity bar:
   neutralizes naive recency/provenance so dLogos must win on *structure*.
3. :class:`ModelNaiveRagArm` -- model over a dumb top-k vector index built from
   the SAME ~200-pod transcripts. Isolates whether the graph/temporal/stance
   structure beats dumb retrieval on identical data.
4. :class:`ModelDLogosArm` -- model + the dLogos temporal graph (via the MCP /
   retriever surface).

Design constraints:
- All collaborators (the frontier client, the web-search tool, the vector
  retriever, the dLogos retriever) are **injected**, so unit tests run with a
  fake client and no network (HARD CONSTRAINT: deterministic tests).
- The frontier/openai client is used via a small *protocol*; nothing heavy is
  imported at module top level.

A :class:`Citation` is the unit the rubric's speaker-verified check consumes:
it names a speaker AND an (episode, t_start, t_end) span, so a citation can be
rejected if the named speaker is not the one speaking at that timestamp.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from dlogos.eval.golden import GoldenQuery


# --------------------------------------------------------------------------- #
# Shared answer interface
# --------------------------------------------------------------------------- #
class Citation(BaseModel):
    """A claimed source pointer carried by an answer.

    ``speaker_id`` is who the answer *attributes* the claim to; the rubric's
    speaker-verified check confirms that this speaker is the one actually
    speaking at ``[t_start, t_end]`` in the diarized transcript (spec §9).
    """

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    t_start: float = Field(ge=0.0)
    t_end: float = Field(ge=0.0)
    speaker_id: str = Field(description="Whom the answer attributes this span to.")
    snippet: str = Field(default="", description="Optional quoted text.")


class Answer(BaseModel):
    """The shared output of every arm: free text plus structured citations.

    ``arm`` records which arm produced it (used for the unblind map, never read
    by a blinded scorer). Web-search/dLogos answers carry citations; the
    model-alone arm typically carries none.
    """

    model_config = ConfigDict(extra="forbid")

    arm: str
    text: str
    citations: list[Citation] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Injected collaborator protocols (kept structural so fakes satisfy them)
# --------------------------------------------------------------------------- #
@runtime_checkable
class ChatClient(Protocol):
    """Minimal async chat surface over an OpenAI-compatible frontier model.

    Real impl wraps the ``openai`` AsyncClient (imported lazily by the caller,
    not here). ``system`` + ``user`` in, completion text out.
    """

    async def complete(self, *, system: str, user: str) -> str: ...


@runtime_checkable
class WebSearchTool(Protocol):
    """A live web-search tool returning ranked text snippets for a query."""

    async def search(self, query: str) -> list[str]: ...


@runtime_checkable
class VectorRetriever(Protocol):
    """Dumb top-k vector retrieval over the same transcripts (arm 3).

    Returns spans with enough provenance to cite; structure-free on purpose.
    """

    async def retrieve(self, query: str, *, k: int = 8) -> list[Citation]: ...


@runtime_checkable
class DLogosRetriever(Protocol):
    """The dLogos graph surface (arm 4): structured, temporal, attributed.

    Returns both an evidence context string (consensus-over-time synthesis,
    attributed) and the speaker-verified citations behind it.
    """

    async def query(self, query: str) -> tuple[str, list[Citation]]: ...


# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #
ARM_MODEL_ALONE = "model_alone"
ARM_WEB_SEARCH = "model_web_search"
ARM_NAIVE_RAG = "model_naive_rag"
ARM_DLOGOS = "model_dlogos"

ALL_ARMS = (ARM_MODEL_ALONE, ARM_WEB_SEARCH, ARM_NAIVE_RAG, ARM_DLOGOS)

_BASE_SYSTEM = (
    "You answer questions about what people have said on podcasts: who said "
    "what, when, and how the consensus moved. Attribute claims to speakers and "
    "cite episode + timestamp when you have a source."
)


def _format_citations(citations: list[Citation]) -> str:
    if not citations:
        return "(no sources provided)"
    lines = []
    for c in citations:
        snip = f" -- {c.snippet}" if c.snippet else ""
        lines.append(
            f"[{c.episode_id} @ {c.t_start:.1f}-{c.t_end:.1f}s, "
            f"speaker={c.speaker_id}]{snip}"
        )
    return "\n".join(lines)


class ModelAloneArm:
    """Arm 1: frontier model, no tools. The floor."""

    name = ARM_MODEL_ALONE

    def __init__(self, client: ChatClient) -> None:
        self._client = client

    async def __call__(self, query: GoldenQuery) -> Answer:
        text = await self._client.complete(
            system=_BASE_SYSTEM
            + " You have no tools and no live data; answer from memory only.",
            user=query.query_text,
        )
        # The floor arm cannot produce verifiable citations.
        return Answer(arm=self.name, text=text, citations=[])


class ModelWebSearchArm:
    """Arm 2: model + live web search (the Perplexity bar)."""

    name = ARM_WEB_SEARCH

    def __init__(self, client: ChatClient, search: WebSearchTool) -> None:
        self._client = client
        self._search = search

    async def __call__(self, query: GoldenQuery) -> Answer:
        snippets = await self._search.search(query.query_text)
        context = "\n".join(f"- {s}" for s in snippets) or "(no results)"
        text = await self._client.complete(
            system=_BASE_SYSTEM
            + " You have live web search. Use the results below.",
            user=f"Web results:\n{context}\n\nQuestion: {query.query_text}",
        )
        # Web snippets are not podcast spans; this arm carries no speaker-verified
        # citations into the diarized transcript (recency/provenance neutralized).
        return Answer(arm=self.name, text=text, citations=[])


class ModelNaiveRagArm:
    """Arm 3: model + dumb top-k vector RAG over the SAME transcripts.

    Carries the retrieved spans as citations so the speaker-verified check can
    bite -- naive retrieval may surface a topically-relevant span where the
    attributed speaker is wrong, which is exactly what the check should catch.
    """

    name = ARM_NAIVE_RAG

    def __init__(self, client: ChatClient, retriever: VectorRetriever, *, k: int = 8) -> None:
        self._client = client
        self._retriever = retriever
        self._k = k

    async def __call__(self, query: GoldenQuery) -> Answer:
        citations = await self._retriever.retrieve(query.query_text, k=self._k)
        context = _format_citations(citations)
        text = await self._client.complete(
            system=_BASE_SYSTEM
            + " You retrieved raw transcript spans (no graph/temporal "
            "structure). Use them.",
            user=f"Retrieved spans:\n{context}\n\nQuestion: {query.query_text}",
        )
        return Answer(arm=self.name, text=text, citations=list(citations))


class ModelDLogosArm:
    """Arm 4: model + the dLogos temporal graph (structured + attributed)."""

    name = ARM_DLOGOS

    def __init__(self, client: ChatClient, retriever: DLogosRetriever) -> None:
        self._client = client
        self._retriever = retriever

    async def __call__(self, query: GoldenQuery) -> Answer:
        context, citations = await self._retriever.query(query.query_text)
        text = await self._client.complete(
            system=_BASE_SYSTEM
            + " You have the dLogos temporal dialogue graph: stance-tagged, "
            "speaker-attributed claims with consensus-over-time. Synthesize how "
            "the position moved across attributed sources.",
            user=f"dLogos evidence:\n{context}\n\nQuestion: {query.query_text}",
        )
        return Answer(arm=self.name, text=text, citations=list(citations))


# --------------------------------------------------------------------------- #
# Concrete retriever adapters — bridge the REAL retrieval surface to the arms
# --------------------------------------------------------------------------- #
@runtime_checkable
class _SearchSurface(Protocol):
    """The slice of a retrieval surface these adapters consume.

    Matches :class:`dlogos.mcp.server.RetrievalSurface` (and its
    ``GraphRetrievalSurface`` implementation) structurally, without importing it
    — so arms stay decoupled and the import graph has no cycle.
    """

    def search(self, query: str, *, top_k: int = 10, since=None, until=None): ...

    def consensus(self, subject: str, *, window_days: int = 30): ...


def _citation_from_hit(hit: object) -> Citation | None:
    """Project one :class:`~dlogos.retrieval.hybrid.RetrievalResult` to a Citation.

    Returns ``None`` when the hit lacks a source span or a resolved speaker
    (attribution is the point of a dLogos citation — an unattributed hit cannot
    seed the speaker-verified check).
    """

    claim = getattr(hit, "claim", None)
    if claim is None:
        return None
    span = getattr(claim, "source_span", None)
    speaker_id = getattr(claim, "speaker_id", None)
    if span is None or not speaker_id:
        return None
    return Citation(
        episode_id=span.episode_id,
        t_start=span.t_start,
        t_end=span.t_end,
        speaker_id=speaker_id,
        snippet=getattr(claim, "text", "") or "",
    )


class GraphVectorRetriever:
    """Arm-3 adapter: *dumb* top-k retrieval over the SAME graph claims.

    Wraps a retrieval surface but uses ONLY its plain ``search`` (semantic +
    lexical fusion over the flattened claims) — deliberately no consensus, no
    temporal bucketing, no stance synthesis. This is the structure-free control
    (spec §9 arm 3): it returns the top-k spans as citations so the
    speaker-verified check can still bite on a topically-relevant-but-
    misattributed span.
    """

    def __init__(self, surface: _SearchSurface, *, k: int = 8) -> None:
        self._surface = surface
        self._k = k

    async def retrieve(self, query: str, *, k: int = 8) -> list[Citation]:
        hits = self._surface.search(query, top_k=k or self._k)
        out: list[Citation] = []
        for hit in hits:
            cit = _citation_from_hit(hit)
            if cit is not None:
                out.append(cit)
        return out


class DLogosGraphRetriever:
    """Arm-4 adapter: the structured, temporal, attributed dLogos surface.

    Wraps a retrieval surface and produces the two things
    :class:`ModelDLogosArm` consumes: an *evidence context string* that
    synthesizes the consensus-over-time across attributed speakers, and the
    speaker-verified citations behind it.

    The context is built by: (1) running ``search`` to find the relevant claims;
    (2) taking the dominant subject ``canonical_id`` among the hits and running
    ``consensus`` on it to get the per-bucket attributed trend; (3) rendering the
    trend (direction + delta + the speaker cast + per-bucket sentiment) plus the
    top attributed spans. This is exactly the temporal-consensus synthesis the
    rubric elevates (§9).
    """

    def __init__(
        self, surface: _SearchSurface, *, top_k: int = 8, window_days: int = 30
    ) -> None:
        self._surface = surface
        self._top_k = top_k
        self._window_days = window_days

    async def query(self, query: str) -> tuple[str, list[Citation]]:
        hits = self._surface.search(query, top_k=self._top_k)
        citations: list[Citation] = []
        for hit in hits:
            cit = _citation_from_hit(hit)
            if cit is not None:
                citations.append(cit)

        subject = self._dominant_subject(hits)
        context = self._render_context(query, subject, hits)
        return context, citations

    @staticmethod
    def _dominant_subject(hits: list) -> str | None:
        """The most frequent subject ``canonical_id`` among the hits."""

        counts: dict[str, int] = {}
        for hit in hits:
            claim = getattr(hit, "claim", None)
            sid = getattr(claim, "subject_id", None) if claim is not None else None
            if sid:
                counts[sid] = counts.get(sid, 0) + 1
        if not counts:
            return None
        # Ties broken by id for determinism.
        return max(sorted(counts), key=lambda k: counts[k])

    def _render_context(
        self, query: str, subject: str | None, hits: list
    ) -> str:
        """Render the attributed consensus-over-time evidence block."""

        lines: list[str] = []
        if subject is not None:
            trend = self._surface.consensus(
                subject, window_days=self._window_days
            )
            lines.append(
                f"Consensus on {trend.subject}: direction={trend.direction.value}, "
                f"sentiment_delta={trend.sentiment_delta:+.2f}, "
                f"attributed speakers={', '.join(trend.all_speakers) or '(none)'}."
            )
            for b in trend.buckets:
                if b.claim_count == 0:
                    continue
                spk = ", ".join(b.speakers)
                lines.append(
                    f"  [{b.start.date()}..{b.end.date()}] "
                    f"net_sentiment={b.net_sentiment:+.2f} "
                    f"net_stance={b.net_stance:+.2f} by {spk}"
                )

        if hits:
            lines.append("Attributed spans:")
            for hit in hits:
                claim = getattr(hit, "claim", None)
                if claim is None:
                    continue
                span = getattr(claim, "source_span", None)
                spk = getattr(claim, "speaker_id", None) or "(unattributed)"
                text = getattr(claim, "text", "") or ""
                if span is not None:
                    lines.append(
                        f"  {spk} @ {span.episode_id} "
                        f"[{span.t_start:.1f}-{span.t_end:.1f}s]: {text}"
                    )
        return "\n".join(lines) or "(no dLogos evidence found)"
