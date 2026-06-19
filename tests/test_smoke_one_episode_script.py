"""Tests for scripts/smoke_one_episode.py — the one-episode smoke runner.

The smoke script's REAL backends (AssemblyAI, DeepInfra, Neo4j) cannot run here
(no keys / no Neo4j) — its first real execution IS the smoke. What we CAN and do
test offline:

1. The module imports with no key and leaks no heavy/optional dependency.
2. ``run_smoke`` — the injectable core — drives the SAME pipeline path the
   production ``main`` does, but on the offline fakes (mock ASR + fake store +
   fake embedder + fake extraction client), and produces a cited answer whose
   citation snippet maps back to the real transcript span.
3. The env-validation gate fails loudly (and lists every missing var) rather
   than silently falling through to a fake.
4. The episode-construction helpers and the snippet-at-span mapping are correct.

The script is loaded by path and registered in ``sys.modules`` (so its
dataclasses resolve under Python's string-annotation rules) — mirroring how it
runs as ``__main__`` in production.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from dlogos.asr.mock_backend import MockASRBackend
from dlogos.graph.fake_store import FakeGraphStore
from dlogos.schema import Episode

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "smoke_one_episode.py"


def _load_script_module():
    """Load the script as an importable module, registered in sys.modules.

    Registration before ``exec_module`` is required so the module's dataclasses
    resolve their string annotations (``from __future__ import annotations``) —
    the same state the module is in when run as ``__main__``.
    """

    spec = importlib.util.spec_from_file_location("smoke_one_episode", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["smoke_one_episode"] = module
    spec.loader.exec_module(module)
    return module


smoke = _load_script_module()


# --------------------------------------------------------------------------- #
# Offline fakes (mirror tests/test_e2e_smoke.py)
# --------------------------------------------------------------------------- #
class _FakeExtractionClient:
    """Async OpenAI-compatible client returning canned, chunk-grounded JSON."""

    def __init__(self, payload: str) -> None:
        self._payload = payload

    @property
    def chat(self):
        outer = self

        class _Completions:
            async def create(self, **kwargs):
                return {"choices": [{"message": {"content": outer._payload}}]}

        class _Chat:
            completions = _Completions()

        return _Chat()


def _extraction_payload() -> str:
    """Two claims grounded in the MockASRBackend transcript windows."""

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
                    "subject": "Apple hardware",
                    "subject_type": "organization",
                    "object": "a rebound next cycle",
                    "stance": "predicts",
                    "sentiment": 0.3,
                    "confidence": 0.65,
                    "t_start": 23.0,
                    "t_end": 28.0,
                },
            ]
        }
    )


class _FakeFrontierClient:
    """Deterministic frontier ChatClient: echoes the evidence into the answer."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def complete(self, *, system: str, user: str) -> str:
        self.calls.append((system, user))
        return f"Synthesis:\n{user}"


def _episode() -> Episode:
    return Episode(
        episode_id="ep-smoke",
        show_id="show-tech",
        guid="guid-ep-smoke",
        title="The state of Apple hardware",
        published_at=datetime(2026, 1, 10, tzinfo=timezone.utc),
        audio_url="https://example.invalid/ep-smoke.mp3",
    )


# --------------------------------------------------------------------------- #
# 1) Import-time guarantees
# --------------------------------------------------------------------------- #
def test_script_imports_without_heavy_deps() -> None:
    heavy = [
        "torch",
        "whisperx",
        "pyannote",
        "graphiti_core",
        "neo4j",
        "gradio",
        "mcp",
        "FlagEmbedding",
        "sentence_transformers",
    ]
    leaked = [h for h in heavy if h in sys.modules]
    assert leaked == [], f"heavy deps leaked at script import: {leaked}"


# --------------------------------------------------------------------------- #
# 2) run_smoke over the offline fakes — a cited answer that maps to transcript
# --------------------------------------------------------------------------- #
async def test_run_smoke_produces_a_verifiable_cited_answer(fake_embedder) -> None:
    result = await smoke.run_smoke(
        episode=_episode(),
        audio_path="ignored-by-mock",
        asr=MockASRBackend(),
        extractor=__import__(
            "dlogos.extraction.extractor", fromlist=["ClaimExtractor"]
        ).ClaimExtractor(_FakeExtractionClient(_extraction_payload())),
        embedder=fake_embedder,
        store=FakeGraphStore(),
        question="What does the analyst claim about Apple hardware?",
        top_k=8,
    )

    # Claims flowed all the way into the graph.
    assert result.pipeline.claims_loaded >= 1
    # The dedup-bypass fast path ran (no per-add LLM dedup).
    assert result.pipeline.store.llm_dedup_invocations == 0
    # A non-empty answer and at least one citation.
    assert result.answer_text.strip()
    assert result.citations, "the smoke must surface at least one cited claim"

    # Each citation maps to a real diarized span in the SAME transcript.
    for cit in result.citations:
        assert cit.episode_id == result.transcript.episode_id
        spoken, label = smoke.snippet_at_span(
            result.transcript, cit.t_start, cit.t_end
        )
        assert spoken.strip(), "cited span must land on real transcript text"
        assert label is not None


async def test_run_smoke_frontier_side_by_side(fake_embedder) -> None:
    client = _FakeFrontierClient()
    result = await smoke.run_smoke(
        episode=_episode(),
        audio_path="x",
        asr=MockASRBackend(),
        extractor=__import__(
            "dlogos.extraction.extractor", fromlist=["ClaimExtractor"]
        ).ClaimExtractor(_FakeExtractionClient(_extraction_payload())),
        embedder=fake_embedder,
        store=FakeGraphStore(),
        question="What does the analyst claim about Apple hardware?",
        frontier_client=client,
    )
    # Both arms ran (model-alone + dLogos), so the client saw two completions.
    assert len(client.calls) == 2
    assert result.frontier_model_alone is not None
    assert result.frontier_dlogos is not None
    # The dLogos arm still carries speaker-verified citations; model-alone does
    # not (its evidence is empty). The reported answer is the dLogos one.
    assert result.citations


# --------------------------------------------------------------------------- #
# 3) Loud env-validation gate
# --------------------------------------------------------------------------- #
class _Settings:
    """Minimal settings stand-in for the validation gate."""

    assemblyai_api_key = ""
    extraction_api_key = ""
    embed_api_key = ""
    neo4j_uri = "bolt://localhost:7687"
    podcast_index_key = ""
    frontier_api_key = ""


def test_validate_env_lists_every_missing_required_var() -> None:
    args = smoke.build_arg_parser().parse_args(
        ["--audio-url", "https://x/ep.mp3"]
    )
    with pytest.raises(smoke.SmokeConfigError) as ei:
        smoke._validate_env(_Settings(), args)
    msg = str(ei.value)
    assert "ASSEMBLYAI_API_KEY" in msg
    assert "EXTRACTION_API_KEY" in msg
    assert "EMBED_API_KEY" in msg
    # NEO4J_URI is set, so it must NOT be listed.
    assert "NEO4J_URI" not in msg


def test_validate_env_requires_podcast_key_for_feed_mode() -> None:
    s = _Settings()
    s.assemblyai_api_key = "k"
    s.extraction_api_key = "k"
    s.embed_api_key = "k"
    args = smoke.build_arg_parser().parse_args(
        ["--feed-url", "https://feeds/x.xml"]
    )
    with pytest.raises(smoke.SmokeConfigError, match="PODCAST_INDEX_KEY"):
        smoke._validate_env(s, args)


def test_validate_env_requires_frontier_key_when_frontier_flag() -> None:
    s = _Settings()
    s.assemblyai_api_key = "k"
    s.extraction_api_key = "k"
    s.embed_api_key = "k"
    args = smoke.build_arg_parser().parse_args(
        ["--audio-url", "https://x/ep.mp3", "--frontier"]
    )
    with pytest.raises(smoke.SmokeConfigError, match="FRONTIER_API_KEY"):
        smoke._validate_env(s, args)


def test_validate_env_passes_when_all_present() -> None:
    s = _Settings()
    s.assemblyai_api_key = "k"
    s.extraction_api_key = "k"
    s.embed_api_key = "k"
    args = smoke.build_arg_parser().parse_args(
        ["--audio-url", "https://x/ep.mp3"]
    )
    smoke._validate_env(s, args)  # no raise


# --------------------------------------------------------------------------- #
# 4) Episode construction + snippet mapping
# --------------------------------------------------------------------------- #
def test_episode_from_audio_url_slugs_and_defaults() -> None:
    ep = smoke.episode_from_audio_url(
        audio_url="https://cdn.example.com/path/My-Episode.mp3",
        title="t",
        show_id="show",
    )
    assert ep.episode_id == "My-Episode"
    assert ep.guid == "My-Episode"
    assert ep.audio_url.endswith("My-Episode.mp3")
    assert ep.published_at.tzinfo is not None


def test_episode_from_feed_picks_the_indexed_item() -> None:
    class _FakePI:
        def recent_episodes(self, *, feed_url, max_results=None, since=None):
            return [
                {"guid": "g0", "enclosureUrl": "https://x/0.mp3", "title": "zero",
                 "datePublished": 1700000000},
                {"guid": "g1", "enclosureUrl": "https://x/1.mp3", "title": "one",
                 "datePublished": 1700100000},
            ]

    ep, audio_url = smoke.episode_from_feed(
        feed_url="https://feeds/x.xml",
        episode_index=1,
        show_id="show",
        podcast_index=_FakePI(),
    )
    assert ep.guid == "g1"
    assert audio_url == "https://x/1.mp3"
    assert ep.title == "one"


def test_episode_from_feed_out_of_range_fails_loudly() -> None:
    class _FakePI:
        def recent_episodes(self, *, feed_url, max_results=None, since=None):
            return [{"guid": "g0", "enclosureUrl": "https://x/0.mp3"}]

    with pytest.raises(smoke.SmokeConfigError, match="out of range"):
        smoke.episode_from_feed(
            feed_url="https://feeds/x.xml",
            episode_index=5,
            show_id="s",
            podcast_index=_FakePI(),
        )


def test_snippet_at_span_picks_max_overlap_segment(synthetic_transcript) -> None:
    # The SPEAKER_01 plateaued line is the segment [4.5, 10.0].
    text, label = smoke.snippet_at_span(synthetic_transcript, 5.0, 9.0)
    assert "plateaued" in text
    assert label == "SPEAKER_01"
