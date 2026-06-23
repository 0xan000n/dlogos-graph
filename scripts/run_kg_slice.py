"""The 20-episode knowledge-graph slice driver (plan Phase 3, Task 3.3).

Mirrors ``scripts/run_smoke_inmemory.py`` (reusing its content-hash
:class:`CachingASR` + :class:`CachingExtractor`), but scales the *correctness*
machinery from one episode to a multi-episode slice: it wires the persistent,
Wikidata-anchored :class:`~dlogos.resolution.incremental.IncrementalResolver`
(over a :class:`~dlogos.resolution.canonical_store.SqliteCanonicalStore`), the
name-driven :class:`~dlogos.speakers.speaker_store.NameSpeakerResolver` (over a
:class:`~dlogos.speakers.speaker_store.SqliteSpeakerStore`), and word-level
re-segmentation through **one** :class:`~dlogos.pipeline.Pipeline` so the same
real-world entity / host / guest becomes **one** canonical node across episodes.

After the load it exports ``out/graph.json`` for the localhost viewer and prints
a :class:`~dlogos.eval.fragmentation.FragReport` over a small built-in probe set
— the resolution-quality number ("how close to one node per real entity?").

Three run modes, all off the same injectable core (:func:`run_kg_slice`):

    # A) a real corpus-manifest slice (audio via AssemblyAI ASR):
    uv run python scripts/run_kg_slice.py --manifest manifests/slice.json

    # B) the 20 public *text* transcripts (NO ASR — fetch + parse the text,
    #    real speaker names canonicalized by the name-driven resolver):
    uv run python scripts/run_kg_slice.py \\
        --transcripts docs/corpus/ai_sensemaking_20.json

    # C) an explicit list of local audio files (dev / offline-ish):
    uv run python scripts/run_kg_slice.py \\
        --audio-files ep1.mp3 ep2.mp3 --show-id the-indicator

The persistent sqlite stores live under ``out/`` so re-runs are *incremental*:
episode N+1 (even in a separate process) resolves against episodes 1..N. This
module imports no key and pulls in no heavy/optional dependency at import time;
every real backend (AssemblyAI httpx, openai, Wikidata) is built lazily inside
its factory, exactly like the smoke. The offline test
(``tests/test_run_kg_slice_script.py``) drives :func:`run_kg_slice` over two fake
episodes with in-memory stores — no network. The live ``--manifest`` run is the
USER's next step, not this script's tests.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

# Import-light at module load: only the shared schema + pipeline wiring + the
# pure fragmentation eval. The heavy sibling wrappers (CachingASR / CachingExtractor
# from run_smoke_inmemory, which pull AssemblyAI/DeepInfra/httpx) and every real
# backend factory are imported LAZILY inside the live-wiring functions — so
# importing this module (e.g. the offline test loading run_kg_slice) costs no
# heavy/optional dependency, exactly like smoke_one_episode.
from dlogos.eval.fragmentation import FragReport, Probe, fragmentation_report
from dlogos.graph.export import write_graph_json
from dlogos.pipeline import EpisodeInput, Pipeline, PipelineDeps, PipelineResult
from dlogos.schema import Episode


# --------------------------------------------------------------------------- #
# A small, configurable built-in probe set (the resolution-quality yardstick)
# --------------------------------------------------------------------------- #
DEFAULT_PROBES: list[Probe] = [
    Probe(
        name="OpenAI",
        aliases=["openai", "open ai", "openai inc"],
        qid="Q21708200",
    ),
    Probe(
        name="Apple",
        aliases=["apple", "apple inc", "apple inc.", "the iphone maker"],
        qid="Q312",
    ),
    Probe(
        name="the Fed",
        aliases=[
            "the fed",
            "the federal reserve",
            "federal reserve",
            "the federal reserve system",
        ],
        qid="Q53536",
    ),
]


# --------------------------------------------------------------------------- #
# The injectable core (everything injected; importable + testable without keys)
# --------------------------------------------------------------------------- #
@dataclass
class KgSliceResult:
    """What the slice produced, for printing and for assertions in tests.

    ``pipeline`` is the :class:`~dlogos.pipeline.PipelineResult` of the final
    run (its ``.store`` is the shared, fully-loaded graph). ``frag_report`` is
    the per-probe canonical-node count over that store. ``graph_path`` is where
    the vis-network JSON was written (``None`` when export was skipped).
    """

    pipeline: PipelineResult
    frag_report: FragReport
    graph_path: Path | None = None


async def run_kg_slice(
    *,
    episodes: Sequence[dict[str, Any]],
    embedder: Any,
    store: Any,
    subject_resolver: Any,
    speaker_resolver: Any,
    probes: Sequence[Probe],
    asr: Any,
    extractor: Any,
    fallback_speaker_id: Any | None = None,
    resegment_words: bool = True,
    graph_out: str | Path | None = None,
) -> KgSliceResult:
    """Run a multi-episode slice through the persistent-resolution pipeline.

    Pure orchestration over injected collaborators (no key reads, no network of
    its own), so the offline test drives it with fakes exactly as production
    ``main`` drives it with real AssemblyAI / DeepInfra / sqlite backends.

    ``episodes`` is a list of dicts, one per episode, each carrying at least an
    ``"episode"`` (an :class:`~dlogos.schema.Episode`) and an ``"audio_path"``
    (plus optional ``"domains"`` / guest metadata). The single injected ``asr``
    and ``extractor`` serve *every* episode — the live slice shares one
    AssemblyAI + one DeepInfra backend; the offline test injects routing fakes
    that dispatch by ``audio_path`` / ``chunk.episode_id``.

    **All episodes flow through ONE** :class:`Pipeline.run`, which is what makes
    resolution and the cross-claim dialogue edges *cross-episode*: the injected
    persistent ``subject_resolver`` resolves every subject against the accumulated
    canonical set in one pass (one ``canonical_id`` per real entity across
    episodes), the name-driven ``speaker_resolver`` gives the same host/guest one
    ``speaker_id``, and — crucially — the loader's single batch
    ``derive_relation_edges`` pass sees *all* episodes' claims at once, so a later
    episode disputing/superseding an earlier one yields a real cross-episode
    ``disputes`` / ``supersedes`` edge (these only exist *between* claims and so
    cannot be derived one episode at a time).

    After the load it runs :func:`~dlogos.eval.fragmentation.fragmentation_report`
    over the store's entity nodes and, when ``graph_out`` is set, exports the
    vis-network graph JSON for the viewer.
    """

    # A stable per-episode fallback id so the loader accepts long-tail speakers
    # the name resolver could not name (spec §7.3) — same default as the smoke.
    if fallback_speaker_id is None:
        def fallback_speaker_id(episode_id: str, label: str) -> tuple[str, str | None]:
            return (f"spk-{episode_id}-{label.lower()}", None)

    def _ep_input(ep: dict[str, Any]) -> EpisodeInput:
        episode: Episode = ep["episode"]
        return EpisodeInput(
            episode=episode,
            audio_path=ep.get("audio_path", ""),
            metadata_guest_names=list(ep.get("metadata_guest_names", [])),
            guest_label=ep.get("guest_label"),
            domains=list(ep.get("domains", [])),
        )

    # Text-transcript labels ARE speaker names — tell the pipeline so its name
    # resolver canonicalizes them across episodes (audio diarization gives A/B).
    from dlogos.ingestion.transcript_source import TranscriptBackend

    deps = PipelineDeps(
        asr=asr,
        extractor=extractor,
        embedder=embedder,
        store=store,
        fallback_speaker_id=fallback_speaker_id,
        resegment_words=resegment_words,
        subject_resolver=subject_resolver,
        speaker_resolver=speaker_resolver,
        labels_are_names=isinstance(asr, TranscriptBackend),
    )

    # ONE Pipeline.run over the whole slice (cross-episode resolution + the
    # batch-level cross-claim dialogue edges).
    result = await Pipeline(deps).run([_ep_input(ep) for ep in episodes])

    # Fragmentation over the fully-loaded store (read the EntityNodes directly).
    entity_nodes = list(store.entities.values())
    frag_report = fragmentation_report(entity_nodes, list(probes))

    graph_path: Path | None = None
    if graph_out is not None:
        graph_path = Path(graph_out)
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        write_graph_json(store, graph_path)

    return KgSliceResult(
        pipeline=result, frag_report=frag_report, graph_path=graph_path
    )


# --------------------------------------------------------------------------- #
# Printing (the human-eyeballing surface)
# --------------------------------------------------------------------------- #
def print_frag_report(result: KgSliceResult, *, n_episodes: int) -> None:
    """Print the graph summary + per-probe fragmentation counts."""

    store = result.pipeline.store
    n_nodes = (
        len(store.speakers) + len(store.entities) + len(store.claims)
    )
    print("=" * 78)
    print("dLogos KG SLICE — multi-episode resolution + fragmentation")
    print("=" * 78)
    print(f"Episodes processed : {n_episodes}")
    print(f"Speakers / entities / claims : "
          f"{len(store.speakers)} / {len(store.entities)} / {len(store.claims)}")
    print(f"Graph nodes / edges: {n_nodes} / {len(store.edges)}")
    print(f"Claims loaded      : {result.pipeline.claims_loaded}")
    print("-" * 78)
    print("FRAGMENTATION (distinct canonical nodes per probed entity — 1 is ideal):")
    fr = result.frag_report
    for pr in fr.per_probe:
        ids = ", ".join(pr.canonical_ids) if pr.canonical_ids else "(none)"
        print(f"  {pr.probe.name:<20} {pr.fragments} node(s)   [{ids}]")
    if fr.worst is not None:
        print(f"  mean fragments = {fr.mean_fragments:.2f}   "
              f"worst = {fr.worst.probe.name} ({fr.worst.fragments})")
    print("=" * 78)
    if result.graph_path is not None:
        print(f"Graph JSON: {result.graph_path}")


# --------------------------------------------------------------------------- #
# Live wiring: build the real backends + persistent stores from settings
# --------------------------------------------------------------------------- #
def _build_episodes_from_audio_files(
    audio_files: list[str], *, show_id: str, language: str | None
) -> list[dict[str, Any]]:
    """One episode dict per local audio file (the ``--audio-files`` dev path)."""

    # Lazy sibling import (scripts/ on sys.path when run as __main__): keeps the
    # module import-light for the offline test.
    from smoke_one_episode import episode_from_audio_url

    episodes: list[dict[str, Any]] = []
    for path in audio_files:
        episode = episode_from_audio_url(
            audio_url=path,
            title=Path(path).stem,
            show_id=show_id,
            episode_id=None,
        )
        episodes.append({"episode": episode, "audio_path": path})
    return episodes


def _episodes_from_transcript_corpus(corpus_path: str) -> list[dict[str, Any]]:
    """Read the public-text-transcript corpus into ``run_kg_slice`` episode dicts.

    Builds the canonical :class:`~dlogos.pipeline.EpisodeInput` per corpus row via
    :func:`~dlogos.ingestion.transcript_source.build_episodes_from_corpus` (the
    single source of truth for the corpus→EpisodeInput mapping, also covered by
    its own offline test), then flattens each to the ``dict`` shape
    :func:`run_kg_slice`'s ``_ep_input`` consumes. ``audio_path`` is the transcript
    URL the injected :class:`TranscriptBackend` fetches + parses.
    """

    from dlogos.ingestion.transcript_source import build_episodes_from_corpus

    out: list[dict[str, Any]] = []
    for ep in build_episodes_from_corpus(corpus_path):
        out.append(
            {
                "episode": ep.episode,
                "audio_path": ep.audio_path,
                "metadata_guest_names": list(ep.metadata_guest_names),
                "domains": list(ep.domains),
            }
        )
    return out


@dataclass
class _LiveSlice:
    """The constructed live backends + episode list for a real slice run."""

    episodes: list[dict[str, Any]]
    asr: Any
    extractor: Any
    embedder: Any
    store: Any
    subject_resolver: Any
    speaker_resolver: Any
    known_hosts: list[str] = field(default_factory=list)


def _build_live_slice(settings: Any, args: argparse.Namespace) -> _LiveSlice:
    """Construct the REAL backends + persistent stores (heavy imports in here).

    Wraps AssemblyAI ASR in the content-hash :class:`CachingASR` and the
    DeepInfra extractor in :class:`CachingExtractor` (so re-running a slice does
    not re-pay), opens the persistent :class:`SqliteCanonicalStore` /
    :class:`SqliteSpeakerStore` under ``out/`` (incremental re-runs), and wires
    the Wikidata-anchored :class:`IncrementalResolver` + name-driven
    :class:`NameSpeakerResolver`.
    """

    # Lazy sibling import (the heavy CachingASR/CachingExtractor wrappers reused
    # from the in-memory smoke driver, per the plan) + the real backend factories.
    from run_smoke_inmemory import CachingASR, CachingExtractor

    from dlogos.extraction.extractor import ClaimExtractor
    from dlogos.resolution.canonical_store import SqliteCanonicalStore
    from dlogos.resolution.cascade import llm_adjudicator_from_client
    from dlogos.resolution.hosted_embedder import OpenAICompatibleEmbedder
    from dlogos.resolution.incremental import IncrementalResolver
    from dlogos.resolution.wikidata import WikidataLinker
    from dlogos.speakers.speaker_store import (
        NameSpeakerResolver,
        SqliteSpeakerStore,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The "ASR" backend depends on the episode source. For the public-text-
    # transcript slice (--transcripts) there is NO audio: the TranscriptBackend
    # fetches each transcript URL and parses it (real speaker NAMES, no
    # diarization). For audio sources (--manifest / --audio-files) it is the
    # AssemblyAI ASR wrapped in the content-hash transcript cache. Either way a
    # single backend serves the whole batch -> one Pipeline.run. The extractor is
    # the same DeepInfra extractor in the claims-cache path for every mode.
    if args.transcripts:
        from dlogos.ingestion.transcript_source import (
            TranscriptBackend,
            known_speakers_map,
        )

        # Build {url -> [host, *guests]} once from the corpus so the backend can
        # enrich each transcript's bare/ambiguous speaker labels ("Jim" ->
        # "Jim Rutt", "Nate" -> the episode's actual Nate) to full names.
        speakers_by_url = known_speakers_map(args.transcripts)
        asr = TranscriptBackend(
            cache_dir=str(out_dir / "transcript_cache"),
            language=args.language,
            known_speakers_for=lambda url: speakers_by_url.get(url, []),
        )
    else:
        from dlogos.asr.hosted_backend import AssemblyAIBackend

        asr = CachingASR(
            AssemblyAIBackend(
                language_code=args.language,
                speakers_expected=args.speakers_expected,
            )
        )
    extractor = CachingExtractor(ClaimExtractor.from_settings(settings))
    embedder = OpenAICompatibleEmbedder.from_settings(settings)

    # Persistent stores (cross-run incremental resolution lives here).
    entity_store = SqliteCanonicalStore(out_dir / "canonical.db")
    speaker_store = SqliteSpeakerStore(out_dir / "speakers.db")

    # The DeepInfra adjudicator (the extraction model doubles as the
    # ambiguous-pair adjudicator). The cascade calls it SYNCHRONOUSLY, so it
    # needs a SYNC OpenAI client — the extractor's own client is AsyncOpenAI and
    # would return un-awaited coroutines (always-False + leaky). Build a sync one.
    from openai import OpenAI

    adj_model = settings.extraction_model
    llm_adjudicator = (
        llm_adjudicator_from_client(
            OpenAI(
                base_url=settings.extraction_base_url,
                api_key=settings.extraction_api_key,
            ),
            adj_model,
        )
        if adj_model
        else None
    )

    subject_resolver = IncrementalResolver(
        entity_store,
        embedder,
        wikidata_linker=WikidataLinker(),
        llm_adjudicator=llm_adjudicator,
    )
    speaker_resolver = NameSpeakerResolver(speaker_store)

    # Episode source: the public-text-transcript corpus, OR a corpus manifest
    # (resolved to recent episodes), OR an explicit list of local audio files.
    known_hosts: list[str] = []
    if args.transcripts:
        episodes = _episodes_from_transcript_corpus(args.transcripts)
    elif args.manifest:
        episodes, known_hosts = _episodes_from_manifest(settings, args)
    else:
        episodes = _build_episodes_from_audio_files(
            args.audio_files, show_id=args.show_id, language=args.language
        )

    return _LiveSlice(
        episodes=episodes,
        asr=asr,
        extractor=extractor,
        embedder=embedder,
        store=_build_store(settings, args),
        subject_resolver=subject_resolver,
        speaker_resolver=speaker_resolver,
        known_hosts=known_hosts,
    )


def _build_store(settings: Any, args: argparse.Namespace) -> Any:
    """The graph store the slice loads into.

    Defaults to the in-memory :class:`~dlogos.graph.fake_store.FakeGraphStore`
    (the same Protocol the suite exercises, no Docker), so the slice runs without
    Neo4j; pass ``--neo4j`` to load into a real Neo4j instead.
    """

    if args.neo4j:
        from dlogos.graph.neo4j_store import Neo4jStore

        store = Neo4jStore.connect(settings)
        store.bootstrap_constraints()
        return store

    from dlogos.graph.fake_store import FakeGraphStore

    return FakeGraphStore()


def _episodes_from_manifest(
    settings: Any, args: argparse.Namespace
) -> tuple[list[dict[str, Any]], list[str]]:
    """Resolve a :class:`CorpusManifest` to episode dicts via the Podcast Index.

    Returns ``(episodes, known_hosts)``. Each manifest row's feed is resolved to
    its recent episodes (capped by ``--max-episodes-per-feed``); the row's
    domains flow onto each episode for guest disambiguation, and its known hosts
    are accumulated for the report. Heavy imports (the Podcast Index httpx
    client) happen here, not at module load.
    """

    from dlogos.ingestion.manifest import load_manifest
    from dlogos.ingestion.podcast_index import PodcastIndexClient

    manifest = load_manifest(args.manifest)
    podcast_index = PodcastIndexClient()

    episodes: list[dict[str, Any]] = []
    known_hosts: list[str] = []
    for row in manifest.rows:
        known_hosts.extend(row.known_hosts)
        items = podcast_index.recent_episodes(
            feed_url=row.feed_url, max_results=args.max_episodes_per_feed
        )
        for item in items:
            audio_url = str(item.get("enclosureUrl") or "")
            if not audio_url:
                continue
            guid = str(item.get("guid") or audio_url)
            published = item.get("datePublished")
            published_at = (
                datetime.fromtimestamp(float(published), tz=timezone.utc)
                if isinstance(published, (int, float))
                and not isinstance(published, bool)
                else datetime.now(timezone.utc)
            )
            episode = Episode(
                episode_id=guid,
                show_id=row.show_id,
                guid=guid,
                title=str(item.get("title") or guid),
                published_at=published_at,
                audio_url=audio_url,
            )
            episodes.append(
                {
                    "episode": episode,
                    "audio_path": audio_url,
                    "domains": list(row.domains),
                }
            )
    return episodes, known_hosts


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_kg_slice",
        description="Run a multi-episode slice through the persistent, "
        "Wikidata-anchored resolution pipeline and print a fragmentation report.",
    )
    src = p.add_argument_group("episode source (choose one)")
    src.add_argument(
        "--transcripts",
        default=None,
        help="A public-text-transcript corpus JSON (e.g. "
        "docs/corpus/ai_sensemaking_20.json): fetch + parse each transcript URL, "
        "no ASR. Speaker NAMES drive resolution directly.",
    )
    src.add_argument(
        "--manifest",
        default=None,
        help="A CorpusManifest JSON (rows resolved to recent episodes).",
    )
    src.add_argument(
        "--audio-files",
        nargs="+",
        default=None,
        help="Explicit list of local audio files (the offline/dev path).",
    )
    p.add_argument("--show-id", default="the-indicator",
                   help="Show id for the --audio-files path.")
    p.add_argument("--out-dir", default="out",
                   help="Directory for the persistent sqlite stores + graph.json.")
    p.add_argument("--graph-out", default="out/graph.json",
                   help="Where to write the vis-network graph JSON for the viewer.")
    p.add_argument("--max-episodes-per-feed", type=int, default=None,
                   help="Cap on episodes pulled per manifest feed.")
    p.add_argument("--language", default="en",
                   help="ASR language code (default en).")
    p.add_argument("--speakers-expected", type=int, default=2,
                   help="Diarization speaker-count hint.")
    p.add_argument("--no-resegment", action="store_true",
                   help="Disable word-level re-segmentation (keep utterance spans).")
    p.add_argument("--neo4j", action="store_true",
                   help="Load into Neo4j instead of the in-memory FakeGraphStore.")
    return p


async def _amain(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.transcripts and not args.manifest and not args.audio_files:
        print("error: provide --transcripts OR --manifest OR --audio-files. "
              "See --help.", file=sys.stderr)
        return 2

    from dlogos.config import settings

    live = _build_live_slice(settings, args)
    if not live.episodes:
        print("error: the episode source resolved to zero episodes.",
              file=sys.stderr)
        return 2

    print(f"[kg-slice] processing {len(live.episodes)} episode(s) ...",
          file=sys.stderr)
    result = await run_kg_slice(
        episodes=live.episodes,
        embedder=live.embedder,
        store=live.store,
        subject_resolver=live.subject_resolver,
        speaker_resolver=live.speaker_resolver,
        probes=DEFAULT_PROBES,
        asr=live.asr,
        extractor=live.extractor,
        resegment_words=not args.no_resegment,
        graph_out=args.graph_out,
    )

    print_frag_report(result, n_episodes=len(live.episodes))

    close = getattr(live.store, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # pragma: no cover - best-effort cleanup
            pass
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(_amain(argv))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
