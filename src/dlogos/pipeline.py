"""End-to-end pipeline orchestrator (spec §5, the architecture skeleton).

Composes the per-stage modules into one offline-runnable flow:

    ingest (manifest -> podcast index -> queue -> fetch) ->
    ASR -> diarize/prune -> cross-episode speaker identity ->
    chunk -> extract -> resolve (subject-entity clustering) -> bulk-load (graph)

Every collaborator is **injected** through :class:`PipelineDeps` (the back half)
and :class:`IngestionDeps` (the front half), so the whole pipeline runs offline
on the core dependency group: a fake Podcast Index client, an
:class:`~dlogos.ingestion.queue.InMemoryJobQueue`, an
:class:`~dlogos.ingestion.fetch.AudioFetcher` over a mock HTTP transport, a
:class:`~dlogos.asr.mock_backend.MockASRBackend`, an in-memory
:class:`~dlogos.graph.fake_store.FakeGraphStore`, the conftest ``FakeEmbedder``,
a fake async extraction client, and (optionally) a host gallery + guest
resolver. No stage imports a heavy/optional dependency at module top level.

The orchestrator is deliberately a *composition* of the module interfaces — it
adds no new domain logic, only the wiring: the front-half ingestion drive
(manifest row -> Podcast Index recent episodes -> idempotent job queue -> audio
fetch -> :class:`Episode`), the manifest-seeded cross-episode identity (a
host-anchored :class:`~dlogos.speakers.identity.HostGallery` plus a recurring
:class:`~dlogos.speakers.guests.GuestResolver` run *across* episodes), and the
small "stamp the resolved speaker id onto each extracted claim" join that sits
between the speaker-identity stage and subject resolution. The output
(:class:`PipelineResult`) carries the resolved claims, the per-episode
event-time map, and the loaded graph store, so a caller can immediately build a
retrieval surface over it (see :meth:`PipelineResult.build_retrieval_surface`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Protocol, runtime_checkable

from dlogos.asr.base import ASRBackend, drop_low_talk_time_speakers
from dlogos.asr.word_segmentation import resegment_by_words
from dlogos.extraction.chunking import (
    DEFAULT_MAX_CHARS,
    DEFAULT_OVERLAP_SEGMENTS,
    chunk_transcript,
)
from dlogos.extraction.extractor import ClaimExtractor
from dlogos.extraction.grounding import ground_claims
from dlogos.graph.loader import ClaimLoader
from dlogos.ingestion.fetch import FetchResult
from dlogos.ingestion.manifest import CorpusManifest, ManifestRow
from dlogos.ingestion.queue import JobQueue
from dlogos.resolution.subjects import (
    DEFAULT_SIMILARITY_THRESHOLD,
    Embedder,
    SubjectResolution,
    resolve_subjects,
)
from dlogos.schema import Episode, ExtractedClaim, Transcript
from dlogos.speakers.guests import GuestResolution, GuestResolver
from dlogos.speakers.identity import (
    CanonicalSpeaker,
    HostGallery,
    SpeakerResolution,
    VoiceEmbedder,
)
from dlogos.speakers.speaker_store import extract_label_names


# --------------------------------------------------------------------------- #
# Persistent-resolver seams (Phase 3)
# --------------------------------------------------------------------------- #
@runtime_checkable
class SubjectResolverLike(Protocol):
    """The slice of a subject resolver the orchestrator drives (spec Phase 3).

    Declared structurally so the default batch
    :func:`~dlogos.resolution.subjects.resolve_subjects` path and the persistent
    :class:`~dlogos.resolution.incremental.IncrementalResolver` are
    interchangeable: anything exposing ``resolve(claims) -> SubjectResolution``
    can be injected as :attr:`PipelineDeps.subject_resolver` to replace the
    in-batch clustering with cross-episode resolution against a persistent store.
    The real :class:`IncrementalResolver` (whose ``resolve`` has exactly this
    shape) satisfies it unchanged.
    """

    def resolve(
        self, claims: list[ExtractedClaim]
    ) -> SubjectResolution:  # pragma: no cover - protocol
        ...


@runtime_checkable
class SpeakerResolverLike(Protocol):
    """The slice of a name-driven speaker resolver the orchestrator drives.

    Declared structurally so the
    :class:`~dlogos.speakers.speaker_store.NameSpeakerResolver` can be injected
    as :attr:`PipelineDeps.speaker_resolver` to fill diarization labels from
    spoken/manifest names against a persistent speaker store *before* the
    existing host-gallery/guest/fallback chain. The real ``NameSpeakerResolver``
    (whose ``resolve`` has exactly this shape) satisfies it unchanged.
    """

    def resolve(
        self,
        transcript: Transcript,
        label_names: dict[str, str],
        *,
        qids: dict[str, str] | None = None,
    ) -> dict[str, SpeakerResolution]:  # pragma: no cover - protocol
        ...


# --------------------------------------------------------------------------- #
# Ingestion front-end seams (the part that produces Episodes)
# --------------------------------------------------------------------------- #
@runtime_checkable
class PodcastIndexLike(Protocol):
    """The slice of :class:`~dlogos.ingestion.podcast_index.PodcastIndexClient`
    the orchestrator drives.

    Declared structurally so a unit test can inject a tiny fake returning
    canned ``items`` (no network, no signed headers) — the real client (which
    touches ``httpx`` only inside its methods) satisfies it unchanged.
    """

    def recent_episodes(
        self,
        *,
        feed_id: int | None = None,
        feed_url: str | None = None,
        max_results: int | None = None,
        since: int | None = None,
    ) -> list[dict[str, Any]]:  # pragma: no cover - protocol
        ...


@runtime_checkable
class AudioFetcherLike(Protocol):
    """The slice of :class:`~dlogos.ingestion.fetch.AudioFetcher` we drive.

    ``fetch`` is GUID-idempotent and content-hash-deduped; the orchestrator
    only needs the resulting :class:`~dlogos.ingestion.fetch.FetchResult`'s
    ``content_hash`` (stamped onto the Episode as ``audio_sha256``) and the
    idempotency guarantee. Tests inject an ``AudioFetcher`` over a mock HTTP
    transport.
    """

    def fetch(self, guid: str, audio_url: str) -> FetchResult:  # pragma: no cover
        ...


@dataclass
class IngestionDeps:
    """The injected front-half collaborators that turn a manifest into Episodes.

    Mirrors the spec §7.1 path: resolve a feed's recent episodes via the
    Podcast Index, model the work as a queue (idempotent on episode GUID so
    re-polling never reprocesses), then fetch + content-hash the enclosure. All
    three are injectable so the integration test drives them on fakes:
    a fake :class:`PodcastIndexLike`, an
    :class:`~dlogos.ingestion.queue.InMemoryJobQueue`, and an
    :class:`~dlogos.ingestion.fetch.AudioFetcher` over a mock transport.

    Parameters
    ----------
    podcast_index:
        Resolves each manifest row's feed to recent episode items.
    queue:
        The job queue backfill fans out onto; GUID is the idempotency key, so a
        re-ingested feed enqueues each episode exactly once.
    fetcher:
        Audio-enclosure fetcher (GUID-idempotent, content-hash dedupe). When
        ``None`` the audio fetch is skipped (episodes still flow through, just
        without an ``audio_sha256`` stamp) — handy for tests that don't model a
        blob store.
    max_episodes_per_feed:
        Upper bound on episodes pulled per feed (the Podcast Index ``max``).
    since:
        Optional unix-timestamp lower bound forwarded to the incremental poller
        so only newer items are pulled.
    """

    podcast_index: PodcastIndexLike
    queue: JobQueue
    fetcher: AudioFetcherLike | None = None
    max_episodes_per_feed: int | None = None
    since: int | None = None


# --------------------------------------------------------------------------- #
# Injected dependencies
# --------------------------------------------------------------------------- #
@dataclass
class PipelineDeps:
    """The injected collaborators the pipeline composes (spec §5).

    Only ``asr``, ``extractor``, ``embedder`` and ``store`` are required; the
    speaker-identity collaborators are optional so a minimal offline run (where
    a trivial label->id mapping is enough) still works.

    Parameters
    ----------
    asr:
        ASR backend (:class:`~dlogos.asr.base.ASRBackend`). Offline runs inject
        :class:`~dlogos.asr.mock_backend.MockASRBackend`.
    extractor:
        The open-weight :class:`~dlogos.extraction.extractor.ClaimExtractor`
        (built around an injected fake async client offline).
    embedder:
        Subject-entity :class:`~dlogos.resolution.subjects.Embedder` (the
        conftest ``FakeEmbedder`` offline).
    store:
        The graph store the bulk loader targets — anything implementing
        :class:`~dlogos.graph.store.GraphStore` (``FakeGraphStore`` offline).
    host_gallery:
        Optional host-anchored voiceprint gallery for cross-episode speaker
        identity. When absent, host resolution is skipped.
    guest_resolver:
        Optional recurring-guest resolver (metadata + intro + Wikidata). When
        absent, guest resolution is skipped.
    fallback_speaker_id:
        Optional ``(episode_id, label) -> (speaker_id, name)`` callable used
        for diarization labels that neither the host gallery nor the guest
        resolver resolved. Spec §7.3 says one-off unknown guests "remain
        per-episode speakers"; this gives them a *stable per-episode* id so the
        loader (which requires a resolved id) can still ingest their claims —
        the same role the spike's ``IdentitySpeakerResolver`` plays. When
        ``None`` (the default), unresolved labels stay unresolved and their
        claims are dropped from the load (kept in the per-episode run for audit).
    voice_sample_ref_for:
        Optional ``(episode_id, label) -> sample_ref`` callable that supplies a
        per-episode voiceprint sample key for the host gallery when an
        :class:`EpisodeInput` does not carry explicit ``voice_sample_refs``. This
        is the seam a real diarization stage fills (a key into the stored
        diarization-segment audio); injecting it lets the manifest-driven path
        resolve hosts by voiceprint without the caller hand-assembling per-label
        refs. ``None`` (the default) means the gallery only runs against any
        explicit ``EpisodeInput.voice_sample_refs``.
    ground_claims:
        When ``True`` (the default), each episode's extracted claims are
        regrounded onto the transcript segment their evidence actually came from
        *after* extraction and *before* speaker-stamping — snapping the
        LLM-estimated ``source_span`` to the segment's real ``[t_start, t_end]``
        and correcting the diarization ``label`` to that segment's speaker (spec
        §7.4). Because the corrected label then drives speaker resolution, the
        canonical speaker id derives from the grounded label, not the model's
        guess. Set ``False`` to keep the raw extractor spans/labels (e.g. to
        A/B the defect the grounding pass fixes).
    resegment_words:
        When ``True``, each transcript's coarse utterance ``segments`` are
        rebuilt from its word-level stream into tight, sentence-bounded spans
        (``asr.word_segmentation.resegment_by_words``) right after the talk-time
        prune and *before* speaker resolution, so the downstream grounding pass
        snaps claims to ~sentence spans instead of multi-minute blobs. A no-op
        when the transcript carries no words (the mock/WhisperX paths), so it is
        safe to leave on. ``False`` (the default) keeps the backend's utterance
        segments, preserving today's behavior.
    subject_resolver:
        Optional persistent, cross-episode subject resolver (a
        :class:`SubjectResolverLike`, i.e. the
        :class:`~dlogos.resolution.incremental.IncrementalResolver`). When
        present, step 8 calls ``subject_resolver.resolve(all_claims)`` instead of
        the in-batch :func:`~dlogos.resolution.subjects.resolve_subjects`, so the
        same real-world entity resolves to one ``canonical_id`` *across* episodes
        (and across separate runs, against the resolver's persistent store).
        ``None`` (the default) keeps the in-batch clustering — byte-identical to
        today.
    speaker_resolver:
        Optional name-driven, cross-episode speaker resolver (a
        :class:`SpeakerResolverLike`, i.e. the
        :class:`~dlogos.speakers.speaker_store.NameSpeakerResolver`). When
        present, ``_resolve_speakers`` first mines the transcript for
        ``label->name`` (self-intros + host guest-intros via
        :func:`~dlogos.speakers.speaker_store.extract_label_names`) and lets the
        resolver fill those labels with stable cross-episode ``speaker_id``s
        *before* the existing host-gallery/guest/fallback chain — name-resolved
        labels win, everything else falls through unchanged. ``None`` (the
        default) skips it entirely, preserving today's behavior.
    """

    asr: ASRBackend
    extractor: ClaimExtractor
    embedder: Embedder
    store: object  # a dlogos.graph.store.GraphStore
    host_gallery: HostGallery | None = None
    guest_resolver: GuestResolver | None = None
    fallback_speaker_id: (
        Callable[[str, str], tuple[str, str | None] | None] | None
    ) = None
    voice_sample_ref_for: Callable[[str, str], str | None] | None = None
    ground_claims: bool = True
    resegment_words: bool = False
    subject_resolver: SubjectResolverLike | None = None
    speaker_resolver: SpeakerResolverLike | None = None
    # When True, the transcript's diarization "labels" already ARE speaker names
    # (text transcripts: "Tristan Harris" not "A"), so the name resolver maps each
    # label to itself and canonicalizes it across episodes. False for audio.
    labels_are_names: bool = False


# --------------------------------------------------------------------------- #
# Per-episode inputs + outputs
# --------------------------------------------------------------------------- #
@dataclass
class EpisodeInput:
    """One episode to run through the pipeline.

    ``audio_path`` is handed to the ASR backend (ignored by the mock).
    ``voice_sample_refs`` maps each per-episode diarization label to a
    voice-sample key the host gallery's embedder understands (used only when a
    gallery is injected). ``metadata_guest_names`` / ``guest_label`` feed the
    recurring-guest resolver. ``domains`` is the show's domain context for guest
    Wikidata disambiguation.
    """

    episode: Episode
    audio_path: str = ""
    voice_sample_refs: dict[str, str] = field(default_factory=dict)
    metadata_guest_names: list[str] = field(default_factory=list)
    guest_label: str | None = None
    domains: list[str] = field(default_factory=list)


@dataclass
class EpisodeRun:
    """The per-episode intermediate artifacts (for inspection / tests)."""

    episode_id: str
    transcript: Transcript
    speaker_resolutions: dict[str, SpeakerResolution]
    claims: list[ExtractedClaim]  # resolved (speaker id + canonical id stamped)


@dataclass
class PipelineResult:
    """The full pipeline output across all processed episodes.

    ``resolved_claims`` are the claims as loaded into the graph (speaker ids +
    subject ``canonical_id`` stamped). ``event_times`` maps each episode id to
    its publish date — the validity anchor the consensus helper and the graph
    loader join against. ``store`` is the loaded graph store. ``subject_resolution``
    carries the canonical-entity table for inspection.
    """

    runs: list[EpisodeRun]
    resolved_claims: list[ExtractedClaim]
    event_times: dict[str, datetime]
    store: object
    subject_resolution: SubjectResolution
    claims_loaded: int

    def build_retrieval_surface(self, embedder: Embedder, **retriever_kwargs: object):
        """Build a :class:`~dlogos.mcp.server.GraphRetrievalSurface` over the result.

        Convenience so a caller can go straight from a pipeline run to a queryable
        surface (the MCP tools / the dLogos eval arm) without re-wiring. Imports
        the MCP adapter lazily to keep the pipeline module import-light.
        """

        from dlogos.mcp.server import GraphRetrievalSurface

        return GraphRetrievalSurface.from_graph_store(
            self.store,
            embedder,
            consensus_claims=self.resolved_claims,
            event_times=self.event_times,
            **retriever_kwargs,
        )


# --------------------------------------------------------------------------- #
# The orchestrator
# --------------------------------------------------------------------------- #
class Pipeline:
    """Composes the dLogos stages end to end (spec §5).

    Construct with injected :class:`PipelineDeps`; optionally pass
    :class:`IngestionDeps` to drive the front half from a
    :class:`~dlogos.ingestion.manifest.CorpusManifest`. Call :meth:`run` with
    episodes you already have, or :meth:`run_from_manifest` to ingest them from a
    manifest first. The orchestrator runs each episode through ASR -> talk-time
    prune -> cross-episode speaker identity (host gallery + recurring guests) ->
    chunk -> extract -> stamp resolved speaker ids, accumulates all claims, then
    runs subject-entity resolution and a single bulk graph load over the whole
    batch (so the load bypasses Graphiti's per-add LLM dedup — spec §7.5/§7.6).
    """

    def __init__(
        self,
        deps: PipelineDeps,
        *,
        ingestion: IngestionDeps | None = None,
        min_talk_time_fraction: float = 0.05,
        max_chunk_chars: int = DEFAULT_MAX_CHARS,
        overlap_segments: int = DEFAULT_OVERLAP_SEGMENTS,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        ingestion_time: datetime | None = None,
        bypass_llm_dedup: bool = True,
    ) -> None:
        self._deps = deps
        self._ingestion = ingestion
        self._min_talk_time_fraction = min_talk_time_fraction
        self._max_chunk_chars = max_chunk_chars
        self._overlap_segments = overlap_segments
        self._similarity_threshold = similarity_threshold
        self._ingestion_time = ingestion_time
        self._bypass = bypass_llm_dedup

    async def run(self, episodes: list[EpisodeInput]) -> PipelineResult:
        """Run every episode through the pipeline and load the batch into the graph.

        Stages per episode (1-7) accumulate extracted, speaker-resolved claims;
        stages 8-9 (subject resolution + bulk load) run once over the whole batch.
        """

        event_times: dict[str, datetime] = {}
        runs: list[EpisodeRun] = []
        all_claims: list[ExtractedClaim] = []

        # Pre-resolve recurring guests across the whole batch first (the resolver
        # needs every episode's candidates before it can decide who recurs).
        guest_resolutions = self._resolve_guests(episodes)

        for ep_input in episodes:
            episode = ep_input.episode
            event_times[episode.episode_id] = episode.published_at

            # 1-2) ingest is the caller-supplied Episode; ASR -> Transcript.
            # The pipeline owns episode identity: bind the transcript (and thus
            # every downstream source_span) to the Episode being processed,
            # rather than trusting the ASR backend to know which episode this is.
            transcript = self._deps.asr.transcribe(ep_input.audio_path)
            if transcript.episode_id != episode.episode_id:
                transcript = transcript.model_copy(
                    update={"episode_id": episode.episode_id}
                )

            # 3) diarize/prune: drop sub-threshold diarization labels (spec §7.2).
            transcript = drop_low_talk_time_speakers(
                transcript, min_fraction=self._min_talk_time_fraction
            )

            # 3b) optional word-level re-segmentation: rebuild coarse utterance
            # segments into tight, sentence-bounded spans from the word stream so
            # grounded citations snap to ~sentences rather than multi-minute
            # blobs (Phase 0). No-op when the transcript carries no words (the
            # mock/WhisperX utterance-only paths), so it is safe before speaker
            # resolution. Off by default — opt in via PipelineDeps.
            if self._deps.resegment_words:
                transcript = resegment_by_words(transcript)

            # 4) cross-episode speaker identity: host gallery + recurring guests.
            label_resolution = self._resolve_speakers(
                transcript, ep_input, guest_resolutions
            )

            # 5) chunk with overlap, carrying speaker labels into the prompt.
            chunks = chunk_transcript(
                transcript,
                max_chars=self._max_chunk_chars,
                overlap_segments=self._overlap_segments,
            )

            # 6) extract stance-tagged claims (speaker refs carry only labels).
            ep_claims = await self._deps.extractor.extract_many(chunks)

            # 6b) ground each claim to the transcript segment its evidence came
            # from (spec §7.4): the extractor's source_span is LLM-estimated on a
            # coarse grid and its label can disagree with the diarization at the
            # cited time. Regrounding snaps the span to the segment's REAL
            # [t_start, t_end] and corrects the diarization label — and it runs
            # BEFORE speaker-stamping so the corrected label is what speaker
            # resolution reads, i.e. the canonical id derives from the grounded
            # label, not the model's guess. On-by-default; flag-guarded for A/B.
            if self._deps.ground_claims:
                ep_claims = ground_claims(ep_claims, transcript)

            # 7) stamp the resolved speaker id/name onto each claim.
            for claim in ep_claims:
                self._stamp_speaker(claim, label_resolution)

            runs.append(
                EpisodeRun(
                    episode_id=episode.episode_id,
                    transcript=transcript,
                    speaker_resolutions=label_resolution,
                    claims=ep_claims,
                )
            )
            all_claims.extend(ep_claims)

        # 8) subject-entity resolution over the WHOLE batch -> canonical_id.
        # An injected persistent resolver (IncrementalResolver) resolves each
        # subject against the ACCUMULATED canonical set so the same real-world
        # entity gets one id across episodes/runs; with no resolver injected we
        # fall back to the in-batch clustering — byte-identical to today.
        if self._deps.subject_resolver is not None:
            subject_resolution = self._deps.subject_resolver.resolve(all_claims)
        else:
            subject_resolution = resolve_subjects(
                all_claims,
                self._deps.embedder,
                threshold=self._similarity_threshold,
            )
        resolved_claims = subject_resolution.claims

        # Reflect the canonical_id stamping back onto the per-episode runs so the
        # EpisodeRun.claims and the batch agree.
        self._refresh_run_claims(runs, resolved_claims)

        # 9) one bulk graph load over the resolved batch (dedup bypassed).
        loader = ClaimLoader(
            event_times=event_times, ingestion_time=self._ingestion_time
        )
        loadable = [c for c in resolved_claims if self._is_loadable(c)]
        claims_loaded = loader.bulk_load(
            self._deps.store, loadable, bypass_llm_dedup=self._bypass
        )

        return PipelineResult(
            runs=runs,
            resolved_claims=loadable,
            event_times=event_times,
            store=self._deps.store,
            subject_resolution=subject_resolution,
            claims_loaded=claims_loaded,
        )

    # ------------------------------------------------------------------ #
    # Front-half: ingestion entry path (manifest -> Episodes)
    # ------------------------------------------------------------------ #
    @classmethod
    def from_manifest(
        cls,
        manifest: CorpusManifest,
        deps: PipelineDeps,
        ingestion: IngestionDeps,
        voice_embedder: VoiceEmbedder,
        **kwargs: Any,
    ) -> "Pipeline":
        """Build a pipeline whose host gallery is *seeded from the manifest*.

        The corpus manifest is the spec's reviewed input artifact (§4): it names
        each show's ``known_hosts`` (with optional reference audio). This seeds a
        host-anchored :class:`~dlogos.speakers.identity.HostGallery` (§7.3) so
        cross-episode host identity runs in the orchestrator rather than via the
        per-episode fallback hook. The seeded gallery is injected into a *copy*
        of ``deps`` (the caller's ``deps`` is left untouched) only when no
        gallery was supplied — an explicitly injected one always wins.

        ``voice_embedder`` is the (injected) :class:`VoiceEmbedder` the gallery
        embeds reference audio with; tests pass a deterministic fake.
        """

        gallery = deps.host_gallery
        if gallery is None:
            gallery = HostGallery.from_manifest_rows(
                list(manifest.rows), voice_embedder
            )
        seeded = PipelineDeps(
            asr=deps.asr,
            extractor=deps.extractor,
            embedder=deps.embedder,
            store=deps.store,
            host_gallery=gallery,
            guest_resolver=deps.guest_resolver,
            fallback_speaker_id=deps.fallback_speaker_id,
            voice_sample_ref_for=deps.voice_sample_ref_for,
            ground_claims=deps.ground_claims,
            resegment_words=deps.resegment_words,
            subject_resolver=deps.subject_resolver,
            speaker_resolver=deps.speaker_resolver,
        )
        return cls(seeded, ingestion=ingestion, **kwargs)

    def ingest(self, manifest: CorpusManifest) -> list[EpisodeInput]:
        """Drive the front half: a manifest -> queued, fetched :class:`Episode`\\s.

        For every manifest row this resolves the feed's recent episodes via the
        injected Podcast Index, enqueues each onto the job queue keyed by GUID
        (so re-ingesting a feed never double-processes — spec §7.1), then leases
        each job, fetches + content-hashes its enclosure (GUID-idempotent), and
        builds an :class:`EpisodeInput`. The row's ``domains`` flow onto each
        episode for guest-Wikidata disambiguation; the row's ``known_hosts`` are
        attached as the host display name so the gallery + downstream attribution
        agree on the show's host.

        Returns the :class:`EpisodeInput` list in feed/queue order. Requires
        :class:`IngestionDeps` to have been injected.
        """

        if self._ingestion is None:
            raise ValueError(
                "Pipeline.ingest requires IngestionDeps; construct the pipeline "
                "with ingestion=IngestionDeps(...) (or use run_from_manifest)."
            )
        ing = self._ingestion

        # Row lookup so a leased job can recover its show context (domains/hosts).
        row_by_show: dict[str, ManifestRow] = {r.show_id: r for r in manifest.rows}

        # 1) Resolve + enqueue every feed's recent episodes (idempotent on GUID).
        for row in manifest.rows:
            items = ing.podcast_index.recent_episodes(
                feed_url=row.feed_url,
                max_results=ing.max_episodes_per_feed,
                since=ing.since,
            )
            for item in items:
                guid = str(item.get("guid") or "").strip()
                if not guid:
                    # No GUID -> no idempotency key; skip rather than risk a
                    # duplicate-work or a crash (spec §7.1 keys on GUID).
                    continue
                payload = {
                    "guid": guid,
                    "show_id": row.show_id,
                    "audio_url": str(item.get("enclosureUrl") or ""),
                    "title": str(item.get("title") or ""),
                    "date_published": item.get("datePublished"),
                }
                ing.queue.enqueue(payload, idempotency_key=guid)

        # 2) Lease each queued job, fetch the enclosure, build an EpisodeInput.
        episodes: list[EpisodeInput] = []
        while True:
            job = ing.queue.lease()
            if job is None:
                break
            try:
                ep_input = self._episode_input_from_job(job.payload, row_by_show)
            except Exception:
                # A bad payload should not poison the whole backfill: mark this
                # job failed and keep draining the queue.
                ing.queue.nack(job.id, requeue=False)
                continue
            episodes.append(ep_input)
            ing.queue.ack(job.id)
        return episodes

    def _episode_input_from_job(
        self, payload: dict[str, Any], row_by_show: dict[str, ManifestRow]
    ) -> EpisodeInput:
        """Map one queued ingestion job into an :class:`EpisodeInput`.

        Fetches the audio enclosure (GUID-idempotent) when a fetcher is
        injected, stamping the content hash onto the Episode as ``audio_sha256``.
        Attaches the show's domains for guest disambiguation.
        """

        guid = str(payload["guid"])
        show_id = str(payload.get("show_id") or "")
        audio_url = str(payload.get("audio_url") or "")
        row = row_by_show.get(show_id)

        audio_sha256: str | None = None
        ingestion = self._ingestion
        if ingestion is not None and ingestion.fetcher is not None and audio_url:
            result = ingestion.fetcher.fetch(guid, audio_url)
            audio_sha256 = result.content_hash

        episode = Episode(
            episode_id=guid,
            show_id=show_id,
            guid=guid,
            title=str(payload.get("title") or guid),
            published_at=_published_at(payload.get("date_published")),
            audio_url=audio_url,
            audio_sha256=audio_sha256,
        )
        return EpisodeInput(
            episode=episode,
            audio_path=audio_url,
            domains=list(row.domains) if row is not None else [],
        )

    async def run_from_manifest(self, manifest: CorpusManifest) -> PipelineResult:
        """Ingest a manifest's episodes and run them through the full pipeline.

        The single front-to-back entry point: :meth:`ingest` drives the queue +
        fetch front half, then :meth:`run` carries the resulting episodes through
        ASR -> identity -> extract -> resolve -> load. Identity is orchestrated
        (host gallery + guest resolver), not hand-fed.
        """

        return await self.run(self.ingest(manifest))

    # ------------------------------------------------------------------ #
    # Stage helpers
    # ------------------------------------------------------------------ #
    def _resolve_guests(
        self, episodes: list[EpisodeInput]
    ) -> dict[str, GuestResolution]:
        """Run recurring-guest resolution across the batch, keyed by episode id.

        Returns a map ``episode_id -> GuestResolution`` for guests resolved to a
        stable id in that episode. Empty when no guest resolver is injected.
        """

        resolver = self._deps.guest_resolver
        if resolver is None:
            return {}

        for ep_input in episodes:
            transcript = self._deps.asr.transcribe(ep_input.audio_path)
            # Bind to the input episode id (see the note in ``run``) so guest
            # candidates are keyed under the correct episode.
            if transcript.episode_id != ep_input.episode.episode_id:
                transcript = transcript.model_copy(
                    update={"episode_id": ep_input.episode.episode_id}
                )
            guest_label = ep_input.guest_label or self._infer_guest_label(
                transcript, ep_input
            )
            resolver.add_episode(
                transcript,
                show_id=ep_input.episode.show_id,
                metadata_names=ep_input.metadata_guest_names or None,
                guest_label=guest_label,
                context=ep_input.domains or None,
            )

        by_episode: dict[str, GuestResolution] = {}
        for resolution in resolver.resolve():
            if not resolution.is_resolved:
                continue
            for episode_id in resolution.episode_ids:
                by_episode[episode_id] = resolution
        return by_episode

    def _infer_guest_label(
        self, transcript: Transcript, ep_input: EpisodeInput
    ) -> str | None:
        """Best-effort guest diarization label when the caller didn't supply one.

        A real feed's metadata names the guest but not *which* diarization label
        is theirs. When a host gallery is injected we infer it: resolve each
        label, then pick the most-talkative label the gallery did **not** match
        to a host. That ties the resolved recurring-guest id to the right
        per-episode turns without the caller hand-labelling them. Returns
        ``None`` when there's no gallery or every label resolved to a host.
        """

        gallery = self._deps.host_gallery
        labels = list(dict.fromkeys(seg.speaker for seg in transcript.segments))
        voice_sample_refs = self._voice_sample_refs(ep_input, labels)
        host_labels: set[str] = set()
        if gallery is not None and voice_sample_refs:
            for label, res in gallery.resolve_transcript(
                transcript, voice_sample_refs
            ).items():
                resolved = res.resolved
                if res.is_resolved and resolved is not None and resolved.is_host:
                    host_labels.add(label)

        talk_time: dict[str, float] = {}
        for seg in transcript.segments:
            if seg.speaker in host_labels:
                continue
            talk_time[seg.speaker] = talk_time.get(seg.speaker, 0.0) + max(
                0.0, seg.t_end - seg.t_start
            )
        if not talk_time:
            return None
        # Deterministic: most talk time, ties broken by label order.
        return max(sorted(talk_time), key=lambda lbl: talk_time[lbl])

    def _voice_sample_refs(
        self, ep_input: EpisodeInput, labels: list[str]
    ) -> dict[str, str]:
        """The per-label voiceprint sample refs for this episode.

        Explicit ``EpisodeInput.voice_sample_refs`` win. Otherwise, when the
        injected ``voice_sample_ref_for`` hook is present (the diarization seam),
        derive a ref per diarization label so the manifest-driven path can
        voiceprint-match hosts without the caller hand-assembling refs.
        """

        if ep_input.voice_sample_refs:
            return ep_input.voice_sample_refs
        hook = self._deps.voice_sample_ref_for
        if hook is None:
            return {}
        episode_id = ep_input.episode.episode_id
        refs: dict[str, str] = {}
        for label in labels:
            ref = hook(episode_id, label)
            if ref:
                refs[label] = ref
        return refs

    def _resolve_speakers(
        self,
        transcript: Transcript,
        ep_input: EpisodeInput,
        guest_resolutions: dict[str, GuestResolution],
    ) -> dict[str, SpeakerResolution]:
        """Resolve each per-episode label by name, then host gallery, then guest.

        When a name-driven ``speaker_resolver`` is injected it runs *first*:
        spoken/manifest names give the host and recurring guests stable
        cross-episode ``speaker_id``s, and those name-resolved labels **win** over
        everything downstream. Host voiceprint resolution runs next (when a
        gallery is injected); any label still unresolved that matches the
        episode's recurring guest is filled from the batch-level guest
        resolution. Labels with none of these stay per-episode (an unresolved
        :class:`SpeakerResolution`, or the per-episode fallback id).
        """

        labels = list(dict.fromkeys(seg.speaker for seg in transcript.segments))
        resolution: dict[str, SpeakerResolution] = {}

        # Name-driven cross-episode resolution wins (runs before the gallery so a
        # spoken/manifest name anchors the canonical speaker id even when a
        # voiceprint is unavailable). Only its *resolved* labels are kept; its
        # unresolved labels fall through to the gallery/guest/fallback chain.
        name_resolution = self._resolve_speakers_by_name(transcript)

        gallery = self._deps.host_gallery
        voice_sample_refs = self._voice_sample_refs(ep_input, labels)
        if gallery is not None and voice_sample_refs:
            resolution = gallery.resolve_transcript(transcript, voice_sample_refs)

        for label, res in name_resolution.items():
            if res.is_resolved:
                resolution[label] = res

        guest = guest_resolutions.get(ep_input.episode.episode_id)
        guest_per_ep = (
            guest.resolution_for(ep_input.episode.episode_id)
            if guest is not None
            else None
        )

        fallback = self._deps.fallback_speaker_id
        episode_id = ep_input.episode.episode_id

        for label in labels:
            current = resolution.get(label)
            if current is not None and current.is_resolved:
                continue
            if (
                guest_per_ep is not None
                and guest_per_ep.resolved is not None
                and guest_per_ep.label == label
            ):
                resolution[label] = guest_per_ep
                continue

            fallback_res = self._fallback_resolution(fallback, episode_id, label)
            if fallback_res is not None:
                resolution[label] = fallback_res
            elif label not in resolution:
                resolution[label] = SpeakerResolution(
                    label=label, resolved=None, score=0.0
                )
        return resolution

    def _resolve_speakers_by_name(
        self, transcript: Transcript
    ) -> dict[str, SpeakerResolution]:
        """Name-driven cross-episode label resolution (Phase 3, opt-in).

        Returns an empty map when no ``speaker_resolver`` is injected — so the
        default speaker path is byte-identical to today. When one is present, it
        mines the transcript for ``label->name`` (self-intros + host guest-intros
        via :func:`~dlogos.speakers.speaker_store.extract_label_names`) and lets
        the injected resolver canonicalize each named label to a stable
        cross-episode ``speaker_id`` against its persistent store. Labels with no
        detected name come back unresolved (and fall through to the
        gallery/guest/fallback chain).
        """

        resolver = self._deps.speaker_resolver
        if resolver is None:
            return {}
        label_names = dict(extract_label_names(transcript, known_hosts=[]))
        if self._deps.labels_are_names:
            # Text transcripts: the diarization "label" already IS the speaker's
            # name ("Tristan Harris"), so map each label to itself. This is what
            # lets the resolver unify the same person across episodes — audio
            # diarization gives "A"/"B" and must mine names from spoken intros.
            for seg in transcript.segments:
                label_names.setdefault(seg.speaker, seg.speaker)
        if not label_names:
            return {}
        return resolver.resolve(transcript, label_names)

    @staticmethod
    def _fallback_resolution(
        fallback: Callable[[str, str], tuple[str, str | None] | None] | None,
        episode_id: str,
        label: str,
    ) -> SpeakerResolution | None:
        """Build a per-episode :class:`SpeakerResolution` from the fallback hook."""

        if fallback is None:
            return None
        assigned = fallback(episode_id, label)
        if assigned is None:
            return None
        speaker_id, name = assigned
        return SpeakerResolution(
            label=label,
            resolved=CanonicalSpeaker(
                speaker_id=speaker_id, name=name, is_host=False
            ),
            score=0.0,
        )

    @staticmethod
    def _stamp_speaker(
        claim: ExtractedClaim, label_resolution: dict[str, SpeakerResolution]
    ) -> None:
        """Write a resolved speaker id/name onto a claim's SpeakerRef in place."""

        res = label_resolution.get(claim.speaker.label)
        if res is None or res.resolved is None:
            return
        claim.speaker.resolved_id = res.resolved.speaker_id
        claim.speaker.name = res.resolved.name

    @staticmethod
    def _refresh_run_claims(
        runs: list[EpisodeRun], resolved_claims: list[ExtractedClaim]
    ) -> None:
        """Replace each run's claims with the canonical-id-stamped copies.

        ``resolve_subjects`` returns *copies* (it does not mutate inputs), so the
        per-episode runs would otherwise still hold the pre-resolution claims.
        We re-slice the resolved batch back into per-episode runs in order.
        """

        idx = 0
        for run in runs:
            n = len(run.claims)
            run.claims = resolved_claims[idx : idx + n]
            idx += n

    @staticmethod
    def _is_loadable(claim: ExtractedClaim) -> bool:
        """A claim is loadable iff it carries a resolved speaker + canonical id.

        The bulk loader requires both (it raises otherwise — spec §7.5). An
        unresolved one-off speaker (long tail) is dropped from the load rather
        than crashing the batch; it remains in the per-episode run for audit.
        """

        return bool(
            claim.speaker.resolved_id and claim.subject_entity.canonical_id
        )


# --------------------------------------------------------------------------- #
# Convenience entry point
# --------------------------------------------------------------------------- #
async def run_pipeline(
    episodes: list[EpisodeInput],
    deps: PipelineDeps,
    *,
    ingestion_time: datetime | None = None,
    **pipeline_kwargs: object,
) -> PipelineResult:
    """Construct a :class:`Pipeline` and run it over ``episodes`` in one call."""

    pipeline = Pipeline(
        deps,
        ingestion_time=ingestion_time,
        **pipeline_kwargs,  # type: ignore[arg-type]
    )
    return await pipeline.run(episodes)


def _published_at(date_published: Any) -> datetime:
    """Coerce a Podcast Index ``datePublished`` (unix seconds) to a UTC datetime.

    The Podcast Index reports publish times as integer unix seconds. We parse
    that into a timezone-aware UTC :class:`datetime` (the event-time anchor every
    bitemporal edge joins against — spec §6). A missing/unparseable value falls
    back to ingestion ``now`` so a malformed feed item never crashes the load,
    only loses its precise event-time.
    """

    if isinstance(date_published, bool):  # guard: bool is an int subclass
        return utc_now()
    if isinstance(date_published, (int, float)):
        try:
            return datetime.fromtimestamp(float(date_published), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):  # pragma: no cover - defensive
            return utc_now()
    if isinstance(date_published, str) and date_published.strip():
        try:
            return datetime.fromtimestamp(
                float(date_published.strip()), tz=timezone.utc
            )
        except ValueError:
            pass
        try:
            parsed = datetime.fromisoformat(date_published.strip())
        except ValueError:
            return utc_now()
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return utc_now()


def utc_now() -> datetime:
    """UTC ``now`` helper (kept here so callers don't reimport timezone)."""

    return datetime.now(timezone.utc)
