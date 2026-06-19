"""Tests for the four eval arms (spec §9).

All collaborators are fakes -- no network, fully deterministic.
"""

from __future__ import annotations

from dlogos.eval.arms import (
    ARM_DLOGOS,
    ARM_MODEL_ALONE,
    ARM_NAIVE_RAG,
    ARM_WEB_SEARCH,
    Answer,
    Citation,
    ModelAloneArm,
    ModelDLogosArm,
    ModelNaiveRagArm,
    ModelWebSearchArm,
)
from dlogos.eval.golden import AnswerShape, Archetype, Domain, GoldenQuery


def _query() -> GoldenQuery:
    return GoldenQuery(
        id="gq-test",
        archetype=Archetype.temporal_consensus,
        domain=Domain.technology,
        query_text="How has the consensus on X moved?",
        pre_registered_answer_shape=AnswerShape(min_attributed_sources=2),
    )


class FakeChatClient:
    """Echoes the prompt deterministically and records the calls it saw."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def complete(self, *, system: str, user: str) -> str:
        self.calls.append((system, user))
        return f"ANSWER<<{user[:40]}>>"


class FakeWebSearch:
    async def search(self, query: str) -> list[str]:
        return [f"web snippet about {query[:10]}", "another snippet"]


class FakeVectorRetriever:
    def __init__(self, citations: list[Citation]) -> None:
        self._citations = citations
        self.last_k: int | None = None

    async def retrieve(self, query: str, *, k: int = 8) -> list[Citation]:
        self.last_k = k
        return list(self._citations)


class FakeDLogosRetriever:
    def __init__(self, context: str, citations: list[Citation]) -> None:
        self._context = context
        self._citations = citations

    async def query(self, query: str) -> tuple[str, list[Citation]]:
        return self._context, list(self._citations)


async def test_model_alone_has_no_citations_and_no_tool_context() -> None:
    client = FakeChatClient()
    arm = ModelAloneArm(client)
    ans = await arm(_query())
    assert isinstance(ans, Answer)
    assert ans.arm == ARM_MODEL_ALONE
    assert ans.citations == []
    # Floor arm: system prompt declares no tools / no live data.
    system, _user = client.calls[0]
    assert "no tools" in system


async def test_web_search_arm_feeds_results_but_carries_no_span_citations() -> None:
    client = FakeChatClient()
    arm = ModelWebSearchArm(client, FakeWebSearch())
    ans = await arm(_query())
    assert ans.arm == ARM_WEB_SEARCH
    # Web text is not a diarized podcast span -> no speaker-verifiable citations.
    assert ans.citations == []
    _system, user = client.calls[0]
    assert "web snippet" in user


async def test_naive_rag_passes_through_retrieved_citations_and_k() -> None:
    cit = Citation(
        episode_id="ep-0001",
        t_start=4.5,
        t_end=10.0,
        speaker_id="spk-analyst",
        snippet="iPhone plateaued",
    )
    retriever = FakeVectorRetriever([cit])
    client = FakeChatClient()
    arm = ModelNaiveRagArm(client, retriever, k=5)
    ans = await arm(_query())
    assert ans.arm == ARM_NAIVE_RAG
    assert ans.citations == [cit]
    assert retriever.last_k == 5
    # The retrieved span is rendered into the prompt for the model.
    _system, user = client.calls[0]
    assert "ep-0001" in user


async def test_dlogos_arm_uses_structured_context_and_citations() -> None:
    cit = Citation(
        episode_id="ep-0003",
        t_start=600.0,
        t_end=612.0,
        speaker_id="spk-analyst",
    )
    retriever = FakeDLogosRetriever("consensus moved from neg to pos", [cit])
    client = FakeChatClient()
    arm = ModelDLogosArm(client, retriever)
    ans = await arm(_query())
    assert ans.arm == ARM_DLOGOS
    assert ans.citations == [cit]
    system, user = client.calls[0]
    assert "temporal dialogue graph" in system
    assert "consensus moved" in user


async def test_all_four_arms_share_the_answer_interface() -> None:
    client = FakeChatClient()
    arms = [
        ModelAloneArm(client),
        ModelWebSearchArm(client, FakeWebSearch()),
        ModelNaiveRagArm(client, FakeVectorRetriever([])),
        ModelDLogosArm(client, FakeDLogosRetriever("ctx", [])),
    ]
    q = _query()
    answers = [await arm(q) for arm in arms]
    assert all(isinstance(a, Answer) for a in answers)
    assert {a.arm for a in answers} == {
        ARM_MODEL_ALONE,
        ARM_WEB_SEARCH,
        ARM_NAIVE_RAG,
        ARM_DLOGOS,
    }
