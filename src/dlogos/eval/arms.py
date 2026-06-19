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
