"""One-episode smoke using the in-memory FakeGraphStore (no Neo4j/Docker needed).

Runs the REAL hosted backends — AssemblyAI ASR + DeepInfra extraction/embeddings —
and swaps ONLY the Neo4j persistence layer for the in-memory FakeGraphStore (the
identical GraphStore Protocol the 527-test suite exercises). This isolates the
genuinely-uncertain real components (diarization quality, open-weight extraction
quality) from the infra-heavy Neo4j layer, for environments without Docker.

ASR is wrapped in a content-hash transcript cache so re-runs (e.g. tuning the
question or the extraction model) don't re-pay AssemblyAI for the same audio.
Extraction is wrapped in a parallel content-hash claims cache so re-runs to
iterate on the *grounding* pass or the *graph viewer* don't re-pay DeepInfra
either. Crucially the claims cache stores the RAW, pre-grounding extractor
output (it caches at the ``extract_many`` boundary, before the pipeline's
grounding step runs), so grounding still runs every time over the regrounded
transcript spans — only the expensive model call is skipped.

After the graph is loaded it is exported to ``out/graph.json`` (via the
standard-library ``graph.export`` projection) so the localhost graph viewer has
data to render.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# Sibling import: scripts/ is sys.path[0] when run as `python scripts/<this>.py`.
from smoke_one_episode import episode_from_audio_url, print_report, run_smoke

from dlogos.asr.hosted_backend import AssemblyAIBackend
from dlogos.config import settings
from dlogos.extraction.chunking import Chunk
from dlogos.extraction.extractor import ClaimExtractor
from dlogos.graph.export import write_graph_json
from dlogos.graph.fake_store import FakeGraphStore
from dlogos.resolution.hosted_embedder import OpenAICompatibleEmbedder
from dlogos.schema import ExtractedClaim, Transcript


class CachingASR:
    """Wrap an ASRBackend with a content-hash transcript cache (avoids re-paying)."""

    def __init__(self, inner: Any, cache_dir: str = "/tmp/dlogos_asr_cache") -> None:
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, audio_path: str) -> str:
        h = hashlib.sha256()
        p = Path(audio_path)
        h.update(p.read_bytes() if p.exists() else audio_path.encode())
        return h.hexdigest()[:32]

    def transcribe(self, audio_path: str) -> Transcript:
        cache_file = self.cache_dir / f"{self._key(audio_path)}.json"
        if cache_file.exists():
            print(f"[smoke] using cached transcript {cache_file}", file=sys.stderr)
            return Transcript.model_validate_json(cache_file.read_text())
        transcript = self.inner.transcribe(audio_path)
        cache_file.write_text(transcript.model_dump_json())
        print(f"[smoke] cached transcript -> {cache_file}", file=sys.stderr)
        return transcript


class CachingExtractor:
    """Wrap a ClaimExtractor with a content-hash RAW-claims cache.

    Mirrors :class:`CachingASR`: the cache key is the SHA-256 of the chunk
    contents (which derive deterministically from the transcript) blended with
    the extraction model name, so re-running against the same transcript + model
    reuses the cached claims instead of re-calling DeepInfra. The cache holds the
    extractor's RAW output (this wraps ``extract_many``, which the pipeline calls
    *before* its grounding step), so the grounding pass — the thing we iterate on
    — still runs every time over the cached spans.
    """

    def __init__(
        self, inner: ClaimExtractor, cache_dir: str = "/tmp/dlogos_claims_cache"
    ) -> None:
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Read the model name through the same private settings the extractor
        # calls DeepInfra with, so a model swap busts the cache.
        self._model = getattr(
            getattr(inner, "_settings", None), "extraction_model", "unknown"
        )

    def _key(self, chunks: list[Chunk]) -> str:
        h = hashlib.sha256()
        h.update(self._model.encode())
        for ch in chunks:
            h.update(ch.model_dump_json().encode())
        return h.hexdigest()[:32]

    async def extract_many(self, chunks: list[Chunk]) -> list[ExtractedClaim]:
        cache_file = self.cache_dir / f"{self._key(chunks)}.json"
        if cache_file.exists():
            print(f"[smoke] using cached raw claims {cache_file}", file=sys.stderr)
            return [
                ExtractedClaim.model_validate(c)
                for c in json.loads(cache_file.read_text())
            ]
        claims = await self.inner.extract_many(chunks)
        cache_file.write_text(
            "[" + ",".join(c.model_dump_json() for c in claims) + "]"
        )
        print(
            f"[smoke] cached {len(claims)} raw claims -> {cache_file}",
            file=sys.stderr,
        )
        return claims


async def main() -> int:
    p = argparse.ArgumentParser(prog="run_smoke_inmemory")
    p.add_argument("--audio-file", required=True, help="Local audio file (uploaded to AssemblyAI).")
    p.add_argument("--audio-url", default=None, help="Display/metadata URL (optional).")
    p.add_argument("--title", default="Smoke episode")
    p.add_argument("--show-id", default="the-indicator")
    p.add_argument(
        "--question",
        default="What is claimed in this episode, by whom, and about what?",
    )
    p.add_argument("--language", default="en")
    p.add_argument("--speakers-expected", type=int, default=2)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument(
        "--graph-out",
        default="out/graph.json",
        help="Where to write the vis-network graph JSON the viewer loads.",
    )
    args = p.parse_args()

    episode = episode_from_audio_url(
        audio_url=args.audio_url or args.audio_file,
        title=args.title,
        show_id=args.show_id,
        episode_id=None,
    )
    asr = CachingASR(
        AssemblyAIBackend(
            language_code=args.language,
            speakers_expected=args.speakers_expected,
        )
    )
    extractor = CachingExtractor(ClaimExtractor.from_settings(settings))
    embedder = OpenAICompatibleEmbedder.from_settings(settings)
    store = FakeGraphStore()

    print(f"[smoke] uploading + transcribing {args.audio_file} via AssemblyAI ...", file=sys.stderr)
    result = await run_smoke(
        episode=episode,
        audio_path=args.audio_file,
        asr=asr,
        extractor=extractor,
        embedder=embedder,
        store=store,
        question=args.question,
        top_k=args.top_k,
    )
    print_report(result, question=args.question)

    # Export the loaded graph for the localhost viewer (stdlib-only projection).
    graph_path = Path(args.graph_out)
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    write_graph_json(store, graph_path)
    print(f"[smoke] wrote graph for the viewer -> {graph_path}", file=sys.stderr)
    print(f"\nGraph JSON: {graph_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
