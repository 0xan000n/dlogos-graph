"""Tests for the end-to-end pipeline orchestrator (spec §5).

Drives :class:`~dlogos.pipeline.Pipeline` over the offline fakes and asserts the
stage composition: ASR -> prune -> speaker identity -> chunk -> extract ->
resolve -> bulk load. All collaborators are injected; no network, no heavy deps.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from dlogos.asr.mock_backend import MockASRBackend
from dlogos.extraction.extractor import ClaimExtractor
from dlogos.graph.fake_store import FakeGraphStore
from dlogos.pipeline import EpisodeInput, Pipeline, PipelineDeps, run_pipeline
from dlogos.schema import Episode


class _FakeExtractionClient:
    """Async OpenAI-compatible client returning canned, chunk-grounded JSON."""

    def __init__(self, payload: str) -> None:
        self._payload = payload
        self.create_calls = 0

    @property
    def chat(self):
        outer = self

        class _Completions:
            async def create(self, **kwargs: object):
                outer.create_calls += 1
                return {"choices": [{"message": {"content": outer._payload}}]}

        class _Chat:
            completions = _Completions()

        return _Chat()


def _payload() -> str:
    return json.dumps(
        {
            "claims": [
                {
                    "speaker_label": "SPEAKER_01",
                    "predicate": "rates_negative",
                    "subject": "Apple",
                    "subject_type": "organization",
                    "object": "hardware innovation has plateaued",
                    "stance": "asserts",
                    "sentiment": -0.6,
                    "confidence": 0.82,
                    "t_start": 4.5,
                    "t_end": 10.0,
                },
                {
                    "speaker_label": "SPEAKER_01",
                    "predicate": "expects",
                    "subject": "the iPhone",
                    "subject_type": "organization",
                    "object": "a rebound",
                    "stance": "predicts",
                    "sentiment": 0.3,
                    "confidence": 0.65,
                    "t_start": 23.0,
                    "t_end": 28.0,
                },
            ]
        }
    )


def _episode(episode_id: str = "ep-0001", *, month: int = 1) -> Episode:
    return Episode(
        episode_id=episode_id,
        show_id="show-tech",
        guid=f"guid-{episode_id}",
        title="t",
        published_at=datetime(2026, month, 10, tzinfo=timezone.utc),
        audio_url="https://example.invalid/a.mp3",
    )


def _deps(store: FakeGraphStore, fake_embedder, client=None) -> PipelineDeps:
    return PipelineDeps(
        asr=MockASRBackend(),
        extractor=ClaimExtractor(client or _FakeExtractionClient(_payload())),
        embedder=fake_embedder,
        store=store,
        fallback_speaker_id=lambda ep, label: (f"spk-{label.lower()}", None),
    )


async def test_pipeline_runs_all_stages_and_loads(fake_embedder) -> None:
    store = FakeGraphStore()
    result = await Pipeline(_deps(store, fake_embedder)).run(
        [EpisodeInput(episode=_episode())]
    )

    assert result.claims_loaded == 2
    assert store.claim_count() == 2
    assert store.bulk_load_calls == 1
    # The dedup-bypass fast path: no per-add LLM dedup invocations.
    assert store.llm_dedup_invocations == 0
    # Event-time was joined from the episode publish date.
    assert result.event_times["ep-0001"] == datetime(
        2026, 1, 10, tzinfo=timezone.utc
    )


async def test_pipeline_stamps_resolved_speaker_and_canonical_ids(
    fake_embedder,
) -> None:
    store = FakeGraphStore()
    result = await Pipeline(_deps(store, fake_embedder)).run(
        [EpisodeInput(episode=_episode())]
    )

    for claim in result.resolved_claims:
        assert claim.speaker.resolved_id == "spk-speaker_01"
        assert claim.subject_entity.canonical_id is not None

    # Apple and "the iPhone" cluster to ONE canonical entity (fake-embedder
    # geometry: ~0.99 cosine), so consensus does not fragment (§7.4a).
    canon_ids = {c.subject_entity.canonical_id for c in result.resolved_claims}
    assert len(canon_ids) == 1


async def test_pipeline_drops_unresolved_speaker_claims_without_fallback(
    fake_embedder,
) -> None:
    store = FakeGraphStore()
    deps = PipelineDeps(
        asr=MockASRBackend(),
        extractor=ClaimExtractor(_FakeExtractionClient(_payload())),
        embedder=fake_embedder,
        store=store,
        fallback_speaker_id=None,  # no fallback -> unresolved labels dropped
    )
    result = await Pipeline(deps).run([EpisodeInput(episode=_episode())])

    # No gallery, no guest resolver, no fallback -> nothing resolves -> nothing
    # loads, but the per-episode run still records the extracted claims for audit.
    assert result.claims_loaded == 0
    assert store.claim_count() == 0
    assert len(result.runs[0].claims) == 2


async def test_pipeline_multi_episode_builds_consensus_surface(
    fake_embedder,
) -> None:
    store = FakeGraphStore()
    episodes = [
        EpisodeInput(episode=_episode("ep-0001", month=1)),
        EpisodeInput(episode=_episode("ep-0002", month=3)),
    ]
    result = await run_pipeline(episodes, _deps(store, fake_embedder))

    # Two episodes loaded; one bulk load over the whole batch.
    assert result.claims_loaded == 4
    assert store.bulk_load_calls == 1

    surface = result.build_retrieval_surface(fake_embedder)
    # The consensus surface buckets by canonical id across both episodes.
    canon_id = result.resolved_claims[0].subject_entity.canonical_id
    trend = surface.consensus(canon_id, window_days=30)
    assert trend.subject == canon_id
    # Both episodes contribute claims to the timeline.
    assert sum(b.claim_count for b in trend.buckets) == 4
