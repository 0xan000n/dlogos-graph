"""Front-to-back integration test for the orchestrated pipeline (GAP 5).

Drives the FULL :class:`~dlogos.pipeline.Pipeline` from a
:class:`~dlogos.ingestion.manifest.CorpusManifest` — through the real front-half
ingestion path (Podcast Index -> idempotent job queue -> audio fetch) and the
real cross-episode identity path (host-anchored
:class:`~dlogos.speakers.identity.HostGallery` + recurring
:class:`~dlogos.speakers.guests.GuestResolver`) — entirely offline on fakes.

Two fixture episodes share one host (*Alice*, seeded from the manifest with
reference audio) and one recurring guest (*Jane Doe*, named in the host's spoken
intro of both episodes). The test asserts the orchestrator — not a per-episode
fallback hook — makes cross-episode identity real:

1. the shared host resolves to ONE canonical Speaker across both episodes;
2. the recurring guest resolves to ONE canonical Speaker across both episodes;
3. a claim from *episode 2* lands in the graph attributed to that guest.

Everything is injected: a fake Podcast Index client, an
:class:`~dlogos.ingestion.queue.InMemoryJobQueue`, an
:class:`~dlogos.ingestion.fetch.AudioFetcher` over an ``httpx`` mock transport, a
per-episode fake ASR backend, the conftest ``fake_embedder``, a fake async
extraction client, a :class:`~dlogos.resolution.wikidata.WikidataLinker` over a
fake Wikidata client, and a :class:`~dlogos.graph.fake_store.FakeGraphStore`.
No network, no heavy/optional deps.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import numpy as np

from dlogos.extraction.extractor import ClaimExtractor
from dlogos.graph.fake_store import FakeGraphStore
from dlogos.ingestion.fetch import AudioFetcher, InMemoryBlobStore
from dlogos.ingestion.manifest import CorpusManifest, ManifestRow
from dlogos.ingestion.queue import InMemoryJobQueue, JobStatus
from dlogos.pipeline import (
    IngestionDeps,
    Pipeline,
    PipelineDeps,
)
from dlogos.resolution.wikidata import WikidataLinker
from dlogos.schema import Transcript, TranscriptSegment
from dlogos.speakers.guests import GuestResolver


# --------------------------------------------------------------------------- #
# The shared show / host / guest
# --------------------------------------------------------------------------- #
SHOW_ID = "show-1"
FEED_URL = "https://feeds.example.invalid/show-1.xml"
HOST_NAME = "Alice"
HOST_REF = "ref/alice.wav"
HOST_SPEAKER_ID = "host-alice"  # host_speaker_id("Alice")
GUEST_QID_ID = "guest-Q111"  # guest-<QID>, QID from the fake Wikidata client


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakePodcastIndex:
    """A Podcast Index returning two canned recent episodes for the feed."""

    def __init__(self, items_by_feed: dict[str, list[dict[str, Any]]]) -> None:
        self._items = items_by_feed
        self.calls: list[str] = []

    def recent_episodes(
        self,
        *,
        feed_id: int | None = None,
        feed_url: str | None = None,
        max_results: int | None = None,
        since: int | None = None,
    ) -> list[dict[str, Any]]:
        self.calls.append(feed_url or "")
        return list(self._items.get(feed_url or "", []))


class FakeVoiceEmbedder:
    """Maps reference audio + per-episode host turns near one centroid.

    Alice's reference and BOTH episodes' host turns map close together (so the
    gallery resolves the host in each episode); the guest turns map orthogonally
    (so they never collide with the host).
    """

    _TABLE: dict[str, list[float]] = {
        HOST_REF: [1.0, 0.0, 0.0],
        "ep-1/SPEAKER_00": [0.99, 0.10, 0.0],
        "ep-2/SPEAKER_00": [0.98, 0.12, 0.0],
        "ep-1/SPEAKER_01": [0.0, 0.0, 1.0],
        "ep-2/SPEAKER_01": [0.0, 0.0, 1.0],
    }

    def embed(self, sample_ref: str) -> list[float]:
        if sample_ref in self._TABLE:
            v = np.asarray(self._TABLE[sample_ref], dtype=float)
            return (v / (np.linalg.norm(v) or 1.0)).tolist()
        rng = np.random.default_rng(abs(hash(sample_ref)) % (2**32))
        v = rng.standard_normal(3)
        return (v / (np.linalg.norm(v) or 1.0)).tolist()


class FakeWikidataClient:
    """Deterministic Wikidata client: 'jane doe' -> Q111, else no match."""

    _DB: dict[str, list[dict[str, Any]]] = {
        "jane doe": [{"id": "Q111", "label": "Jane Doe", "description": "economist"}],
    }

    def __init__(self) -> None:
        self.calls: list[str] = []

    def search(
        self, name: str, *, entity_type: Any = None, limit: int = 5
    ) -> list[dict[str, Any]]:
        self.calls.append(name)
        return list(self._DB.get(name.strip().lower(), []))


class PerEpisodeASR:
    """ASR backend returning a host-intro + guest-exchange transcript per feed url.

    The audio path the pipeline passes through is the enclosure URL; we key the
    transcript template on it so each episode gets its own (still deterministic)
    diarized transcript naming the guest in the host's intro.
    """

    def __init__(self, by_audio_url: dict[str, Transcript]) -> None:
        self._by_url = by_audio_url

    def transcribe(self, audio_path: str) -> Transcript:
        template = self._by_url[audio_path]
        return template.model_copy(deep=True)


class FakeExtractionClient:
    """Async OpenAI-compatible client returning canned, chunk-grounded JSON."""

    def __init__(self, payload: str) -> None:
        self._payload = payload

    @property
    def chat(self):
        outer = self

        class _Completions:
            async def create(self, **kwargs: object):
                return {"choices": [{"message": {"content": outer._payload}}]}

        class _Chat:
            completions = _Completions()

        return _Chat()


# --------------------------------------------------------------------------- #
# Transcript + extraction fixtures
# --------------------------------------------------------------------------- #
def _transcript(episode_id: str) -> Transcript:
    """Two-speaker episode: host (SPEAKER_00) intros guest Jane Doe (SPEAKER_01)."""

    return Transcript(
        episode_id=episode_id,
        language="en",
        duration_s=30.0,
        segments=[
            TranscriptSegment(
                speaker="SPEAKER_00",
                text="Welcome back. My guest today is Jane Doe, an economist.",
                t_start=0.0,
                t_end=6.0,
            ),
            TranscriptSegment(
                speaker="SPEAKER_01",
                text="Thanks. I think the iPhone has plateaued on hardware innovation.",
                t_start=6.0,
                t_end=14.0,
            ),
            TranscriptSegment(
                speaker="SPEAKER_00",
                text="So you'd rate Apple's hardware story negatively right now?",
                t_start=14.0,
                t_end=20.0,
            ),
            TranscriptSegment(
                speaker="SPEAKER_01",
                text="Yes, negatively, though I expect a rebound next cycle.",
                t_start=20.0,
                t_end=30.0,
            ),
        ],
    )


def _extraction_payload() -> str:
    """One guest claim grounded in the SPEAKER_01 windows (same for both eps)."""

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
                    "t_start": 6.0,
                    "t_end": 14.0,
                },
                {
                    "speaker_label": "SPEAKER_01",
                    "predicate": "expects",
                    "subject": "Apple hardware",
                    "subject_type": "organization",
                    "object": "a rebound next cycle",
                    "stance": "predicts",
                    "sentiment": 0.3,
                    "confidence": 0.65,
                    "t_start": 20.0,
                    "t_end": 30.0,
                },
            ]
        }
    )


# --------------------------------------------------------------------------- #
# Wiring helpers
# --------------------------------------------------------------------------- #
def _manifest() -> CorpusManifest:
    return CorpusManifest(
        rows=[
            ManifestRow(
                show_id=SHOW_ID,
                feed_url=FEED_URL,
                domains=["technology"],
                known_hosts=[HOST_NAME],
                reference_audio={HOST_NAME: HOST_REF},
                deep_backfill=True,
            )
        ]
    )


# Two episodes on the feed; distinct GUIDs + enclosure URLs.
EP1_URL = "https://cdn.example.invalid/show-1/ep-1.mp3"
EP2_URL = "https://cdn.example.invalid/show-1/ep-2.mp3"
_FEED_ITEMS = {
    FEED_URL: [
        {
            "guid": "ep-1",
            "title": "Apple hardware, part one",
            "datePublished": 1_736_467_200,  # 2025-01-10 UTC
            "enclosureUrl": EP1_URL,
        },
        {
            "guid": "ep-2",
            "title": "Apple hardware, part two",
            "datePublished": 1_739_059_200,  # 2025-02-09 UTC
            "enclosureUrl": EP2_URL,
        },
    ]
}


def _audio_transport() -> httpx.MockTransport:
    """Serves distinct bytes per enclosure so content hashes differ."""

    bodies = {EP1_URL: b"AUDIO-EP-1", EP2_URL: b"AUDIO-EP-2"}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url in bodies:
            return httpx.Response(200, content=bodies[url])
        return httpx.Response(404, text="not found")

    return httpx.MockTransport(handler)


def _build_pipeline(store: FakeGraphStore, fake_embedder) -> Pipeline:
    asr = PerEpisodeASR(
        {EP1_URL: _transcript("ep-1"), EP2_URL: _transcript("ep-2")}
    )
    extractor = ClaimExtractor(FakeExtractionClient(_extraction_payload()))
    guest_resolver = GuestResolver(
        wikidata=WikidataLinker(FakeWikidataClient()), min_appearances=2
    )

    deps = PipelineDeps(
        asr=asr,
        extractor=extractor,
        embedder=fake_embedder,
        store=store,
        guest_resolver=guest_resolver,
        # The diarization seam: derive a per-episode voiceprint sample ref so the
        # host gallery (seeded from the manifest) can voiceprint-match the host.
        voice_sample_ref_for=lambda episode_id, label: f"{episode_id}/{label}",
    )

    ingestion = IngestionDeps(
        podcast_index=FakePodcastIndex(_FEED_ITEMS),
        queue=InMemoryJobQueue(),
        fetcher=AudioFetcher(
            blob_store=InMemoryBlobStore(),
            client=httpx.Client(transport=_audio_transport()),
        ),
    )

    return Pipeline.from_manifest(
        _manifest(),
        deps,
        ingestion,
        FakeVoiceEmbedder(),
    )


# --------------------------------------------------------------------------- #
# The integration test
# --------------------------------------------------------------------------- #
async def test_full_orchestrator_resolves_shared_host_and_recurring_guest(
    fake_embedder,
) -> None:
    store = FakeGraphStore()
    pipeline = _build_pipeline(store, fake_embedder)

    result = await pipeline.run_from_manifest(_manifest())

    # --- Two episodes flowed all the way through the front half. ------------- #
    assert {run.episode_id for run in result.runs} == {"ep-1", "ep-2"}
    # Event-time was parsed from the Podcast Index datePublished (unix seconds).
    assert result.event_times["ep-1"].year == 2025

    runs = {run.episode_id: run for run in result.runs}

    # --- (1) The shared host resolves to ONE canonical Speaker across both. -- #
    host_ep1 = runs["ep-1"].speaker_resolutions["SPEAKER_00"]
    host_ep2 = runs["ep-2"].speaker_resolutions["SPEAKER_00"]
    assert host_ep1.is_resolved and host_ep2.is_resolved
    assert host_ep1.resolved.is_host and host_ep2.resolved.is_host
    assert host_ep1.resolved.speaker_id == HOST_SPEAKER_ID
    # Same canonical id in BOTH episodes — cross-episode host identity is real.
    assert host_ep1.resolved.speaker_id == host_ep2.resolved.speaker_id

    # --- (2) The recurring guest resolves to ONE canonical Speaker across both. #
    guest_ep1 = runs["ep-1"].speaker_resolutions["SPEAKER_01"]
    guest_ep2 = runs["ep-2"].speaker_resolutions["SPEAKER_01"]
    assert guest_ep1.is_resolved and guest_ep2.is_resolved
    assert guest_ep1.resolved.speaker_id == GUEST_QID_ID
    assert guest_ep1.resolved.wikidata_qid == "Q111"
    # Same canonical id in BOTH episodes — cross-episode guest identity is real.
    assert guest_ep1.resolved.speaker_id == guest_ep2.resolved.speaker_id
    # Host and guest are distinct canonical speakers.
    assert guest_ep1.resolved.speaker_id != host_ep1.resolved.speaker_id

    # --- (3) A claim from EPISODE 2 lands in the graph attributed to the guest. #
    ep2_claims = [
        c for c in result.resolved_claims if c.source_span.episode_id == "ep-2"
    ]
    assert ep2_claims, "episode 2 must contribute at least one loaded claim"
    assert all(c.speaker.resolved_id == GUEST_QID_ID for c in ep2_claims)

    # The claim is actually in the store, attributed to the guest's speaker id.
    guest_claims = store.query(speaker_id=GUEST_QID_ID)
    ep2_in_graph = [
        r for r in guest_claims if r.claim.source_span.episode_id == "ep-2"
    ]
    assert ep2_in_graph, "an ep-2 guest claim must be queryable in the graph"
    row = ep2_in_graph[0]
    assert row.speaker is not None
    assert row.speaker.speaker_id == GUEST_QID_ID
    assert row.speaker.name == "Jane Doe"
    # The guest claim carries a resolved canonical subject (resolution ran).
    assert row.claim.subject_canonical_id

    # The dedup-bypass fast path ran (single bulk load, no per-add LLM dedup).
    assert store.bulk_load_calls == 1
    assert store.llm_dedup_invocations == 0


async def test_ingest_is_guid_idempotent_across_repolls(fake_embedder) -> None:
    """Re-ingesting the same manifest enqueues each GUID exactly once (spec §7.1)."""

    store = FakeGraphStore()
    asr = PerEpisodeASR(
        {EP1_URL: _transcript("ep-1"), EP2_URL: _transcript("ep-2")}
    )
    queue = InMemoryJobQueue()
    deps = PipelineDeps(
        asr=asr,
        extractor=ClaimExtractor(FakeExtractionClient(_extraction_payload())),
        embedder=fake_embedder,
        store=store,
        voice_sample_ref_for=lambda episode_id, label: f"{episode_id}/{label}",
    )
    ingestion = IngestionDeps(
        podcast_index=FakePodcastIndex(_FEED_ITEMS),
        queue=queue,
        fetcher=AudioFetcher(
            blob_store=InMemoryBlobStore(),
            client=httpx.Client(transport=_audio_transport()),
        ),
    )
    pipeline = Pipeline.from_manifest(
        _manifest(), deps, ingestion, FakeVoiceEmbedder()
    )

    first = pipeline.ingest(_manifest())
    assert {ep.episode.episode_id for ep in first} == {"ep-1", "ep-2"}
    # Both jobs done after the first drain.
    assert queue.stats()[JobStatus.done] == 2

    # A second poll over the SAME feed enqueues nothing new (GUID idempotency),
    # so no fresh queued jobs appear and the drain yields no new episodes.
    second = pipeline.ingest(_manifest())
    assert second == []
    assert queue.stats()[JobStatus.queued] == 0
    assert queue.stats()[JobStatus.done] == 2


async def test_episode_carries_content_hash_from_fetcher(fake_embedder) -> None:
    """The audio fetch stamps the enclosure content hash onto the Episode."""

    store = FakeGraphStore()
    pipeline = _build_pipeline(store, fake_embedder)
    episodes = {ep.episode.episode_id: ep for ep in pipeline.ingest(_manifest())}

    from dlogos.ingestion.fetch import sha256_bytes

    assert episodes["ep-1"].episode.audio_sha256 == sha256_bytes(b"AUDIO-EP-1")
    assert episodes["ep-2"].episode.audio_sha256 == sha256_bytes(b"AUDIO-EP-2")
    # Distinct audio -> distinct hashes.
    assert (
        episodes["ep-1"].episode.audio_sha256
        != episodes["ep-2"].episode.audio_sha256
    )
