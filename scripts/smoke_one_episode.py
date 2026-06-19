#!/usr/bin/env python3
"""One-episode smoke run: real audio -> real graph -> a cited, attributed answer.

This is RUNBOOK step 1 (docs/RUNBOOK.md). It takes ONE real podcast episode all
the way through real, lowest-friction hosted infra and prints an attributed,
cited answer a human can verify *by ear*:

    real audio
      -> AssemblyAI ASR + diarization          (hosted, no GPU)
      -> chunk + open-weight claim extraction   (DeepInfra, OpenAI-compatible)
      -> subject-entity resolution (BGE-M3)      (DeepInfra embeddings)
      -> bulk-load reified, bitemporal claims    (direct Neo4j store)
      -> hybrid retrieval + consensus            (the dLogos arm surface)
      -> PRINT: the answer + every claim's speaker, episode, [t_start,t_end],
                and the exact transcript snippet at that span.

WHAT IT PROVES (and what it does NOT): one episode cannot show a temporal SHIFT
(that needs >= 2 episodes over time — RUNBOOK step 2). The smoke proves the
FOUNDATION: that the machine produces correct, *sourced* claims — the right
claim, attributed to the right speaker, at a checkable timestamp. The human
opens the audio at the printed timestamp and confirms the named speaker is the
one talking and is saying what the claim says.

HONESTY: the real backends (AssemblyAI, DeepInfra, Neo4j) cannot be exercised
without keys + a running Neo4j, so this script's FIRST real run is the smoke
itself. Every real client is INJECTABLE (see :func:`run_smoke`), so importing
this module needs no keys and pulls in no heavy/optional dependency at import
time — heavy imports happen lazily inside the backend factories.

Two ways to point it at an episode:

    # A) a direct audio URL + minimal metadata (simplest):
    uv run python scripts/smoke_one_episode.py \\
        --audio-url https://example.com/ep.mp3 \\
        --title "Episode title" --show-id my-show \\
        --question "What does the guest claim about AI?"

    # B) an RSS feed via the Podcast Index + an episode index:
    uv run python scripts/smoke_one_episode.py \\
        --feed-url https://feeds.example.com/show.xml --episode-index 0 \\
        --question "What does the host claim about the economy?"

Optional head-to-head preview (only when FRONTIER_* is set): add ``--frontier``
to also run the SAME question through the model-alone arm and the dLogos arm and
print them side by side — a preview of RUNBOOK step 6's scorecard.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

# Import-light: only the shared schema + pipeline wiring types at module load.
# Every REAL backend (AssemblyAI httpx, openai, neo4j) is imported lazily inside
# its factory below, so importing this module is cheap and key-free.
from dlogos.pipeline import EpisodeInput, Pipeline, PipelineDeps, PipelineResult
from dlogos.schema import Episode, Transcript


# --------------------------------------------------------------------------- #
# Loud-failure helper
# --------------------------------------------------------------------------- #
class SmokeConfigError(RuntimeError):
    """Raised when required configuration (a key / URL) is missing.

    The smoke is explicitly a REAL run; a missing key is a hard stop with a
    clear message, never a silent fall-through to a fake.
    """


def _require(settings: Any, missing: list[str]) -> None:
    if missing:
        raise SmokeConfigError(
            "Missing required configuration for the smoke run: "
            + ", ".join(missing)
            + ".\nCopy .env.example to .env and fill these in (see "
            "docs/RUNBOOK.md step 1)."
        )


# --------------------------------------------------------------------------- #
# Frontier chat adapter (optional head-to-head). Satisfies eval.arms.ChatClient.
# --------------------------------------------------------------------------- #
class FrontierChatAdapter:
    """A :class:`dlogos.eval.arms.ChatClient` over an OpenAI-compatible frontier.

    ``complete(system, user) -> str``. The ``openai`` import is lazy (in the
    factory), so this class is importable without the SDK; only the optional
    ``--frontier`` path constructs a real one.
    """

    def __init__(self, client: Any, model: str) -> None:
        self._client = client
        self._model = model

    @classmethod
    def from_settings(cls, settings: Any) -> "FrontierChatAdapter":
        from openai import AsyncOpenAI  # lazy

        client = AsyncOpenAI(
            base_url=settings.frontier_base_url,
            api_key=settings.frontier_api_key,
        )
        return cls(client, settings.frontier_model)

    async def complete(self, *, system: str, user: str) -> str:
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
        )
        choices = getattr(resp, "choices", None) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        return (getattr(message, "content", "") if message else "") or ""


# --------------------------------------------------------------------------- #
# Episode construction (two ingest modes)
# --------------------------------------------------------------------------- #
def episode_from_audio_url(
    *,
    audio_url: str,
    title: str,
    show_id: str,
    episode_id: str | None = None,
    guid: str | None = None,
    published_at: datetime | None = None,
) -> Episode:
    """Build an :class:`Episode` directly from a known audio URL + metadata.

    ``episode_id``/``guid`` default to a slug of the URL so the smoke is
    reproducible from the URL alone; ``published_at`` defaults to ``now`` (one
    episode carries no temporal SHIFT, so the exact event-time is not load-
    bearing for the smoke — only that every claim traces back to this episode).
    """

    slug = _slug_from_url(audio_url)
    eid = episode_id or slug
    return Episode(
        episode_id=eid,
        show_id=show_id,
        guid=guid or eid,
        title=title,
        published_at=published_at or datetime.now(timezone.utc),
        audio_url=audio_url,
    )


def episode_from_feed(
    *,
    feed_url: str,
    episode_index: int,
    show_id: str | None,
    podcast_index: Any | None = None,
) -> tuple[Episode, str]:
    """Resolve a feed's recent episodes and build the one at ``episode_index``.

    Returns ``(episode, audio_url)``. ``podcast_index`` is injectable (a
    :class:`~dlogos.pipeline.PodcastIndexLike`); when ``None`` a real
    :class:`~dlogos.ingestion.podcast_index.PodcastIndexClient` is built lazily
    from settings (which is why the ``httpx``-importing client is not pulled in
    at module load).
    """

    if podcast_index is None:
        from dlogos.ingestion.podcast_index import PodcastIndexClient

        podcast_index = PodcastIndexClient()

    items = podcast_index.recent_episodes(feed_url=feed_url, max_results=episode_index + 1)
    if not items:
        raise SmokeConfigError(
            f"Podcast Index returned no episodes for feed {feed_url!r}."
        )
    if episode_index >= len(items):
        raise SmokeConfigError(
            f"--episode-index {episode_index} out of range: the feed returned "
            f"only {len(items)} episode(s)."
        )
    item = items[episode_index]
    audio_url = str(item.get("enclosureUrl") or "")
    if not audio_url:
        raise SmokeConfigError(
            f"Episode {episode_index} of {feed_url!r} has no enclosureUrl."
        )
    guid = str(item.get("guid") or audio_url)
    published = item.get("datePublished")
    published_at = (
        datetime.fromtimestamp(float(published), tz=timezone.utc)
        if isinstance(published, (int, float)) and not isinstance(published, bool)
        else datetime.now(timezone.utc)
    )
    episode = Episode(
        episode_id=guid,
        show_id=show_id or _slug_from_url(feed_url),
        guid=guid,
        title=str(item.get("title") or guid),
        published_at=published_at,
        audio_url=audio_url,
    )
    return episode, audio_url


def _slug_from_url(url: str) -> str:
    from pathlib import Path
    from urllib.parse import urlparse

    stem = Path(urlparse(url).path).stem
    return stem or "smoke-episode"


# --------------------------------------------------------------------------- #
# The core run (everything injected; importable + testable without keys)
# --------------------------------------------------------------------------- #
@dataclass
class SmokeResult:
    """What the smoke produced, for printing and for assertions in tests."""

    pipeline: PipelineResult
    transcript: Transcript
    answer_text: str
    citations: list[Any]  # list[dlogos.eval.arms.Citation]
    evidence_context: str
    frontier_model_alone: str | None = None
    frontier_dlogos: str | None = None


async def run_smoke(
    *,
    episode: Episode,
    audio_path: str,
    asr: Any,
    extractor: Any,
    embedder: Any,
    store: Any,
    question: str,
    top_k: int = 8,
    fallback_speaker_id: Any | None = None,
    frontier_client: Any | None = None,
) -> SmokeResult:
    """Wire the injected real backends through the pipeline and answer one query.

    Pure orchestration over injected collaborators (no key reads, no network of
    its own), so a unit test drives it with the offline fakes exactly as the
    production ``main`` drives it with the real AssemblyAI / DeepInfra / Neo4j
    backends. Mirrors the proven wiring in ``tests/test_e2e_smoke.py``.
    """

    from dlogos.eval.arms import (
        DLogosGraphRetriever,
        ModelAloneArm,
        ModelDLogosArm,
    )
    from dlogos.eval.golden import AnswerShape, Archetype, Domain, GoldenQuery

    # One-off speakers (the long tail) get a stable per-episode id so the loader
    # accepts their claims (spec §7.3). For a single episode this is the whole
    # speaker set — no cross-episode gallery is needed for the smoke.
    if fallback_speaker_id is None:
        def fallback_speaker_id(episode_id: str, label: str) -> tuple[str, str | None]:
            return (f"spk-{episode_id}-{label.lower()}", None)

    deps = PipelineDeps(
        asr=asr,
        extractor=extractor,
        embedder=embedder,
        store=store,
        fallback_speaker_id=fallback_speaker_id,
    )

    # 1) ingest/transcribe + 2) extract -> resolve -> bulk_load into the store.
    result = await Pipeline(deps).run(
        [EpisodeInput(episode=episode, audio_path=audio_path)]
    )
    transcript = result.runs[0].transcript

    # 3) one attribution/provenance query through the real retrieval/consensus
    #    path (the dLogos arm surface), built straight off the loaded graph.
    surface = result.build_retrieval_surface(embedder)
    retriever = DLogosGraphRetriever(surface, top_k=top_k)

    query = GoldenQuery(
        id="smoke",
        archetype=Archetype.provenance,
        domain=Domain.technology,
        query_text=question,
        pre_registered_answer_shape=AnswerShape(min_attributed_sources=1),
    )

    evidence_context, citations = await retriever.query(question)

    frontier_model_alone: str | None = None
    frontier_dlogos: str | None = None
    answer_text = evidence_context

    if frontier_client is not None:
        # 4 (optional) head-to-head: same question, model-alone vs dLogos arm.
        alone = await ModelAloneArm(frontier_client)(query)
        dlogos = await ModelDLogosArm(frontier_client, retriever)(query)
        frontier_model_alone = alone.text
        frontier_dlogos = dlogos.text
        answer_text = dlogos.text
        citations = list(dlogos.citations)

    return SmokeResult(
        pipeline=result,
        transcript=transcript,
        answer_text=answer_text,
        citations=list(citations),
        evidence_context=evidence_context,
        frontier_model_alone=frontier_model_alone,
        frontier_dlogos=frontier_dlogos,
    )


# --------------------------------------------------------------------------- #
# Verification snippet: map a cited [t_start,t_end] back to transcript text
# --------------------------------------------------------------------------- #
def snippet_at_span(
    transcript: Transcript, t_start: float, t_end: float
) -> tuple[str, str | None]:
    """Return ``(spoken_text, diarization_label)`` at a transcript time span.

    Picks the diarized segment whose ``[t_start, t_end]`` overlaps the cited
    span the most — the same "who is actually speaking here" logic the rubric's
    speaker-verified check uses. This is what lets the human cross-check the
    printed timestamp against the audio by ear.
    """

    best_text = ""
    best_label: str | None = None
    best_overlap = 0.0
    for seg in transcript.segments:
        lo = max(t_start, seg.t_start)
        hi = min(t_end, seg.t_end)
        overlap = max(0.0, hi - lo)
        if overlap > best_overlap:
            best_overlap = overlap
            best_text = seg.text
            best_label = seg.speaker
    return best_text, best_label


# --------------------------------------------------------------------------- #
# Printing (the human-eyeballing surface)
# --------------------------------------------------------------------------- #
def print_report(result: SmokeResult, *, question: str) -> None:
    """Print the answer + per-claim verifiable provenance for ear-checking."""

    pr = result.pipeline
    transcript = result.transcript
    ep = pr.runs[0].episode_id

    print("=" * 78)
    print("dLogos ONE-EPISODE SMOKE — attributed, cited answer")
    print("=" * 78)
    print(f"Episode id    : {ep}")
    print(f"Audio         : {transcript.duration_s:.0f}s, language={transcript.language}")
    print(f"Diarized turns: {len(transcript.segments)}")
    print(f"Claims loaded : {pr.claims_loaded} "
          f"(extracted+resolved {len(pr.resolved_claims)})")
    n_entities = len(pr.subject_resolution.clusters)
    print(f"Canon. subjects: {n_entities}")
    print()
    print(f"QUESTION: {question}")
    print("-" * 78)
    print("ANSWER:")
    print(result.answer_text or "(empty)")
    print("-" * 78)

    # Per-claim provenance — the load-bearing, ear-verifiable output.
    print("CITED CLAIMS (verify each by ear at the printed timestamp):")
    if not result.citations:
        print("  (no citations — see failure modes in docs/RUNBOOK.md step 1)")
    for i, cit in enumerate(result.citations, start=1):
        spoken, label = snippet_at_span(transcript, cit.t_start, cit.t_end)
        print()
        print(f"  [{i}] speaker={cit.speaker_id}")
        print(f"      episode={cit.episode_id}  span=[{cit.t_start:.1f}s, {cit.t_end:.1f}s]"
              f"  diarized_label={label}")
        if cit.snippet:
            print(f"      claim text : {cit.snippet}")
        print(f"      transcript : \"{spoken.strip()}\"")
        print(f"      >> open the audio at {_fmt_ts(cit.t_start)} and confirm "
              f"{cit.speaker_id} is the one speaking, and is saying this.")

    if result.frontier_model_alone is not None:
        print()
        print("=" * 78)
        print("HEAD-TO-HEAD PREVIEW (RUNBOOK step 6 scorecard)")
        print("=" * 78)
        print("[ARM 1: model alone — no tools, no sources]")
        print(result.frontier_model_alone or "(empty)")
        print()
        print("[ARM 4: model + dLogos temporal graph — attributed + cited]")
        print(result.frontier_dlogos or "(empty)")
        print("-" * 78)
        print("Only the dLogos arm carries speaker-verified citations above. The "
              "model-alone arm cannot anchor a claim to a real episode timestamp.")
    print("=" * 78)


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:d}:{s:02d}"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="smoke_one_episode",
        description="Take ONE real podcast episode end-to-end and print an "
        "attributed, cited answer for by-ear verification.",
    )
    src = p.add_argument_group("episode source (choose one)")
    src.add_argument("--audio-url", help="Direct audio enclosure URL.")
    src.add_argument("--feed-url", help="RSS feed URL (resolved via Podcast Index).")
    src.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="0-based index into the feed's recent episodes (with --feed-url).",
    )
    meta = p.add_argument_group("episode metadata (with --audio-url)")
    meta.add_argument("--title", default="Smoke episode", help="Episode title.")
    meta.add_argument("--show-id", default=None, help="Show id (slug).")
    meta.add_argument("--episode-id", default=None, help="Override episode id.")

    q = p.add_argument_group("query")
    q.add_argument(
        "--question",
        default="What is claimed in this episode, by whom, and about what?",
        help="The attribution/provenance question to ask the graph.",
    )
    q.add_argument("--top-k", type=int, default=8, help="Hits to retrieve/cite.")

    asr = p.add_argument_group("ASR options")
    asr.add_argument(
        "--language",
        default=None,
        help="Force an ISO language code (e.g. en); default = auto-detect.",
    )
    asr.add_argument(
        "--speakers-expected",
        type=int,
        default=None,
        help="Optional hint to the diarizer of how many speakers to expect.",
    )

    p.add_argument(
        "--frontier",
        action="store_true",
        help="Also run the model-alone vs dLogos head-to-head (needs FRONTIER_*).",
    )
    p.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Skip creating Neo4j constraints/indexes (assume already created).",
    )
    return p


# The config-default placeholders (present only when there is NO .env). Treat
# them as "not configured" so a no-.env run fails loudly here rather than 401-ing
# at DeepInfra later — the RUNBOOK tells the user to `cp .env.example .env`.
_PLACEHOLDER_KEYS = {"", "sk-no-key-required"}


def _is_unset(value: str | None) -> bool:
    return (value or "").strip() in _PLACEHOLDER_KEYS


def _validate_env(settings: Any, args: argparse.Namespace) -> None:
    """Fail loudly listing EVERY missing required var before any network call."""

    missing: list[str] = []
    if not (settings.assemblyai_api_key or "").strip():
        missing.append("ASSEMBLYAI_API_KEY")
    if _is_unset(settings.extraction_api_key):
        missing.append("EXTRACTION_API_KEY")
    if _is_unset(settings.embed_api_key):
        missing.append("EMBED_API_KEY")
    if not (settings.neo4j_uri or "").strip():
        missing.append("NEO4J_URI")
    if args.feed_url and not (settings.podcast_index_key or "").strip():
        missing.append("PODCAST_INDEX_KEY (required for --feed-url)")
    if args.frontier and not (settings.frontier_api_key or "").strip():
        missing.append("FRONTIER_API_KEY (required for --frontier)")
    _require(settings, missing)


def _build_real_backends(settings: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Construct the REAL injectable backends (heavy imports happen in here)."""

    from dlogos.asr.hosted_backend import AssemblyAIBackend
    from dlogos.extraction.extractor import ClaimExtractor
    from dlogos.graph.neo4j_store import Neo4jStore
    from dlogos.resolution.hosted_embedder import OpenAICompatibleEmbedder

    asr = AssemblyAIBackend(
        language_code=args.language,
        speakers_expected=args.speakers_expected,
    )
    extractor = ClaimExtractor.from_settings(settings)
    embedder = OpenAICompatibleEmbedder.from_settings(settings)
    store = Neo4jStore.connect(settings)
    if not args.no_bootstrap:
        store.bootstrap_constraints()
    return {"asr": asr, "extractor": extractor, "embedder": embedder, "store": store}


async def _amain(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.audio_url and not args.feed_url:
        print("error: provide --audio-url OR --feed-url. See --help.", file=sys.stderr)
        return 2

    from dlogos.config import settings

    try:
        _validate_env(settings, args)
    except SmokeConfigError as exc:
        print(f"\nCONFIG ERROR:\n{exc}\n", file=sys.stderr)
        return 2

    # Resolve the episode (may hit the Podcast Index for --feed-url).
    if args.audio_url:
        episode = episode_from_audio_url(
            audio_url=args.audio_url,
            title=args.title,
            show_id=args.show_id or "smoke-show",
            episode_id=args.episode_id,
        )
        audio_path = args.audio_url
    else:
        episode, audio_path = episode_from_feed(
            feed_url=args.feed_url,
            episode_index=args.episode_index,
            show_id=args.show_id,
        )

    backends = _build_real_backends(settings, args)
    frontier_client = (
        FrontierChatAdapter.from_settings(settings) if args.frontier else None
    )

    print(f"Transcribing + processing {episode.audio_url} ...", file=sys.stderr)
    print("(AssemblyAI transcription typically takes a fraction of audio length; "
          "see RUNBOOK for expected runtime.)", file=sys.stderr)

    store = backends["store"]
    try:
        result = await run_smoke(
            episode=episode,
            audio_path=audio_path,
            asr=backends["asr"],
            extractor=backends["extractor"],
            embedder=backends["embedder"],
            store=store,
            question=args.question,
            top_k=args.top_k,
            frontier_client=frontier_client,
        )
    finally:
        # Always release the Neo4j driver.
        close = getattr(store, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass

    print_report(result, question=args.question)

    if not result.citations:
        # A smoke that loads claims but cites none is a soft failure worth a
        # non-zero exit so CI / a runner notices.
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(_amain(argv))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
