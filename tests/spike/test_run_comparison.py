"""Deterministic tests for the spike orchestration (spec §7.6).

Drive :class:`SpikeRunner` over synthetic transcripts with every collaborator
faked: a canned-JSON extractor client (also tracking token usage), a fake
Graphiti-native pipeline for Approach A, the conftest ``FakeEmbedder``, a
``FakeGraphStore``, and a :class:`FakeClock`. No network, no heavy imports, and
fully deterministic timing.
"""

from __future__ import annotations

import json

import pytest

from dlogos.extraction.extractor import ClaimExtractor
from dlogos.graph.fake_store import FakeGraphStore
from dlogos.schema import (
    Entity,
    EntityType,
    ExtractedClaim,
    Predicate,
    SourceSpan,
    SpeakerRef,
    Stance,
    Transcript,
    TranscriptSegment,
)
from dlogos.spike.run_comparison import (
    FakeClock,
    NativeExtractionResult,
    Pricing,
    SpikeRunner,
    run_spike,
)
from dlogos.spike.score import score_comparison


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, client: "FakeExtractClient") -> None:
        self._client = client

    async def create(self, **kwargs: object) -> _FakeCompletion:
        self._client.calls += 1
        self._client.prompt_tokens += self._client.prompt_tokens_per_call
        self._client.completion_tokens += self._client.completion_tokens_per_call
        return _FakeCompletion(self._client.next_content())


class _FakeChat:
    def __init__(self, client: "FakeExtractClient") -> None:
        self.completions = _FakeCompletions(client)


class FakeExtractClient:
    """A canned-JSON OpenAI-compatible client tracking token usage.

    Every ``create`` call returns one claim attributed to ``SPEAKER_01`` inside
    the chunk window (so the extractor's span/speaker checks accept it) and
    bumps the running token counters the runner reads for the $/episode metric.
    """

    def __init__(
        self,
        *,
        prompt_tokens_per_call: int = 100,
        completion_tokens_per_call: int = 40,
        content: str | None = None,
    ) -> None:
        self.chat = _FakeChat(self)
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.prompt_tokens_per_call = prompt_tokens_per_call
        self.completion_tokens_per_call = completion_tokens_per_call
        self._content = content

    def next_content(self) -> str:
        if self._content is not None:
            return self._content
        # One valid claim per chunk, grounded at SPEAKER_01 within [4.5, 10.0].
        return json.dumps(
            {
                "claims": [
                    {
                        "speaker_label": "SPEAKER_01",
                        "predicate": "rates_negative",
                        "subject": "Apple",
                        "subject_type": "organization",
                        "object": "hardware has plateaued",
                        "stance": "asserts",
                        "sentiment": -0.5,
                        "confidence": 0.8,
                        "t_start": 4.5,
                        "t_end": 10.0,
                    }
                ]
            }
        )


class FakeNativePipeline:
    """Approach-A fake: returns canned claims + token/dedup counts per episode.

    ``dedup_calls_per_episode`` models whether Graphiti's per-add LLM node-dedup
    was left on (>0) or bypassed (0) on the bulk path.
    """

    def __init__(
        self,
        *,
        claims_per_episode: int = 1,
        prompt_tokens_per_episode: int = 500,
        completion_tokens_per_episode: int = 200,
        dedup_calls_per_episode: int = 0,
    ) -> None:
        self._cpe = claims_per_episode
        self._ppe = prompt_tokens_per_episode
        self._cpte = completion_tokens_per_episode
        self._dedup = dedup_calls_per_episode
        self.ingested: list[str] = []

    def ingest(self, transcript: Transcript) -> NativeExtractionResult:
        self.ingested.append(transcript.episode_id)
        claims = [
            ExtractedClaim(
                speaker=SpeakerRef(
                    label="SPEAKER_01", resolved_id="spk-a", name="Guest"
                ),
                predicate=Predicate.rates_negative,
                subject_entity=Entity(
                    name="Apple",
                    type=EntityType.organization,
                    canonical_id="ent-apple",
                ),
                object="hardware has plateaued",
                stance=Stance.asserts,
                sentiment=-0.5,
                confidence=0.8,
                source_span=SourceSpan(
                    episode_id=transcript.episode_id, t_start=4.5, t_end=10.0
                ),
            )
            for _ in range(self._cpe)
        ]
        return NativeExtractionResult(
            claims=claims,
            prompt_tokens=self._ppe,
            completion_tokens=self._cpte,
            llm_dedup_calls=self._dedup,
        )


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _transcript(episode_id: str) -> Transcript:
    return Transcript(
        episode_id=episode_id,
        language="en",
        duration_s=28.0,
        segments=[
            TranscriptSegment(
                speaker="SPEAKER_00",
                text="My guest today is an Apple watcher.",
                t_start=0.0,
                t_end=4.5,
            ),
            TranscriptSegment(
                speaker="SPEAKER_01",
                text="The iPhone has plateaued on hardware innovation.",
                t_start=4.5,
                t_end=10.0,
            ),
        ],
    )


@pytest.fixture
def transcripts() -> list[Transcript]:
    return [_transcript("ep-0001"), _transcript("ep-0002")]


def _runner(
    *,
    fake_embedder,
    native: FakeNativePipeline | None = None,
    client: FakeExtractClient | None = None,
    store: FakeGraphStore | None = None,
    pricing: Pricing | None = None,
    bypass_llm_dedup: bool = True,
) -> tuple[SpikeRunner, FakeGraphStore]:
    store = store or FakeGraphStore()
    extractor = ClaimExtractor(client or FakeExtractClient())
    runner = SpikeRunner(
        native_pipeline=native or FakeNativePipeline(),
        extractor=extractor,
        embedder=fake_embedder,
        store=store,
        pricing=pricing or Pricing(),
        clock=FakeClock(step=5.0),
        bypass_llm_dedup=bypass_llm_dedup,
    )
    return runner, store


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_runs_both_approaches_over_same_fixtures(
    transcripts, fake_embedder
) -> None:
    native = FakeNativePipeline()
    runner, store = _runner(fake_embedder=fake_embedder, native=native)
    result = await runner.run(transcripts)

    assert result.episodes == 2
    assert result.approach_a.approach == "A"
    assert result.approach_b.approach == "B"
    # Approach A ingested every transcript via the native pipeline.
    assert native.ingested == ["ep-0001", "ep-0002"]
    # Both approaches captured diarization ground-truth for both episodes.
    assert set(result.approach_a.transcript_segments) == {"ep-0001", "ep-0002"}
    assert set(result.approach_b.transcript_segments) == {"ep-0001", "ep-0002"}


@pytest.mark.asyncio
async def test_approach_b_extracts_resolves_and_bulk_loads(
    transcripts, fake_embedder
) -> None:
    store = FakeGraphStore()
    runner, store = _runner(fake_embedder=fake_embedder, store=store)
    result = await runner.run(transcripts)

    b = result.approach_b
    # One claim per chunk per episode -> at least one loadable claim.
    assert len(b.claims) >= 1
    # Every loaded claim carries a resolved speaker id + canonical entity id.
    for claim in b.claims:
        assert claim.speaker.resolved_id
        assert claim.subject_entity.canonical_id
    # The bulk load ran and bypassed per-add LLM dedup (spec §7.5/§7.6).
    assert store.bulk_load_calls == 1
    assert store.llm_dedup_invocations == 0
    assert b.llm_dedup_calls == 0
    assert store.claim_count() == len(b.claims)


@pytest.mark.asyncio
async def test_approach_b_json_parse_counts_are_captured(
    transcripts, fake_embedder
) -> None:
    runner, _ = _runner(fake_embedder=fake_embedder)
    result = await runner.run(transcripts)
    b = result.approach_b
    # One parse attempt per chunk; the canned client always parses.
    assert b.parse_attempts >= 1
    assert b.parse_successes == b.parse_attempts


@pytest.mark.asyncio
async def test_approach_b_counts_parse_failure_on_bad_json(
    transcripts, fake_embedder
) -> None:
    # A client that always returns non-JSON forces the extractor to raise after
    # its own retry; the runner records the failed attempt and drops the chunk.
    bad_client = FakeExtractClient(content="this is not json")
    runner, _ = _runner(fake_embedder=fake_embedder, client=bad_client)
    result = await runner.run(transcripts)
    b = result.approach_b
    assert b.parse_attempts >= 1
    assert b.parse_successes == 0
    assert b.claims == []


@pytest.mark.asyncio
async def test_token_usage_threaded_into_approach_b(
    transcripts, fake_embedder
) -> None:
    client = FakeExtractClient(
        prompt_tokens_per_call=100, completion_tokens_per_call=40
    )
    runner, _ = _runner(fake_embedder=fake_embedder, client=client)
    result = await runner.run(transcripts)
    b = result.approach_b
    # Token totals come off the injected client and match its running counters.
    assert b.prompt_tokens == client.prompt_tokens
    assert b.completion_tokens == client.completion_tokens
    assert b.prompt_tokens > 0


@pytest.mark.asyncio
async def test_fake_clock_makes_wall_clock_deterministic(
    transcripts, fake_embedder
) -> None:
    runner, _ = _runner(fake_embedder=fake_embedder)
    result = await runner.run(transcripts)
    # FakeClock advances by step=5.0 on each non-first call; each approach reads
    # the clock twice (t0, t1) -> a single 5.0s delta per approach.
    assert result.approach_a.wall_clock_seconds == 5.0
    assert result.approach_b.wall_clock_seconds == 5.0


@pytest.mark.asyncio
async def test_non_bypassed_dedup_recorded_for_approach_b(
    transcripts, fake_embedder
) -> None:
    store = FakeGraphStore()
    runner, store = _runner(
        fake_embedder=fake_embedder, store=store, bypass_llm_dedup=False
    )
    result = await runner.run(transcripts)
    b = result.approach_b
    # With bypass disabled the store simulates per-add dedup, and the runner
    # records the non-zero dedup-call count.
    assert b.llm_dedup_calls == len(b.claims)
    assert store.llm_dedup_invocations == len(b.claims)


@pytest.mark.asyncio
async def test_approach_a_surfaces_non_bypassed_dedup_cost(
    transcripts, fake_embedder
) -> None:
    # Approach A leaves per-add dedup on (the default): 3 dedup calls/episode.
    native = FakeNativePipeline(dedup_calls_per_episode=3)
    runner, _ = _runner(fake_embedder=fake_embedder, native=native)
    result = await runner.run(transcripts)
    a = result.approach_a
    assert a.llm_dedup_calls == 6  # 3 per episode × 2 episodes
    assert a.prompt_tokens == 1000  # 500 per episode × 2


@pytest.mark.asyncio
async def test_run_spike_convenience_matches_runner(
    transcripts, fake_embedder
) -> None:
    client = FakeExtractClient()
    result = await run_spike(
        transcripts,
        native_pipeline=FakeNativePipeline(),
        extractor=ClaimExtractor(client),
        embedder=fake_embedder,
        store=FakeGraphStore(),
        pricing=Pricing(),
        clock=FakeClock(step=5.0),
    )
    assert result.episodes == 2
    # End-to-end: the comparison scores cleanly through the scorer.
    report = score_comparison(result)
    assert report.recommended in {"A", "B"}
    assert report.approach_b.json_parse_success_rate is not None
