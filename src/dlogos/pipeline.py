"""End-to-end pipeline orchestrator (spec §5, the architecture skeleton).

Composes the per-stage modules into one offline-runnable flow:

    ingest -> ASR -> diarize/prune -> cross-episode speaker identity ->
    chunk -> extract -> resolve (subject-entity clustering) -> bulk-load (graph)

Every collaborator is **injected** through :class:`PipelineDeps`, so the whole
pipeline runs offline on the core dependency group: a
:class:`~dlogos.asr.mock_backend.MockASRBackend`, an in-memory
:class:`~dlogos.graph.fake_store.FakeGraphStore`, the conftest ``FakeEmbedder``,
a fake async extraction client, and (optionally) a host gallery + guest
resolver. No stage imports a heavy/optional dependency at module top level.

The orchestrator is deliberately a *composition* of the module interfaces — it
adds no new domain logic, only the wiring and the small "stamp the resolved
speaker id onto each extracted claim" join that sits between the speaker-identity
stage and subject resolution. The output (:class:`PipelineResult`) carries the
resolved claims, the per-episode event-time map, and the loaded graph store, so a
caller can immediately build a retrieval surface over it (see
:meth:`PipelineResult.build_retrieval_surface`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from dlogos.asr.base import ASRBackend, drop_low_talk_time_speakers
from dlogos.extraction.chunking import (
    DEFAULT_MAX_CHARS,
    DEFAULT_OVERLAP_SEGMENTS,
    chunk_transcript,
)
from dlogos.extraction.extractor import ClaimExtractor
from dlogos.graph.loader import ClaimLoader
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
)


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

    Construct with injected :class:`PipelineDeps`; call :meth:`run` with the
    episodes to process. The orchestrator runs each episode through ASR ->
    talk-time prune -> speaker identity -> chunk -> extract -> stamp resolved
    speaker ids, accumulates all claims, then runs subject-entity resolution and
    a single bulk graph load over the whole batch (so the load bypasses
    Graphiti's per-add LLM dedup — spec §7.5/§7.6).
    """

    def __init__(
        self,
        deps: PipelineDeps,
        *,
        min_talk_time_fraction: float = 0.05,
        max_chunk_chars: int = DEFAULT_MAX_CHARS,
        overlap_segments: int = DEFAULT_OVERLAP_SEGMENTS,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        ingestion_time: datetime | None = None,
        bypass_llm_dedup: bool = True,
    ) -> None:
        self._deps = deps
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
        subject_resolution = resolve_subjects(
            all_claims, self._deps.embedder, threshold=self._similarity_threshold
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
            resolver.add_episode(
                transcript,
                show_id=ep_input.episode.show_id,
                metadata_names=ep_input.metadata_guest_names or None,
                guest_label=ep_input.guest_label,
                context=ep_input.domains or None,
            )

        by_episode: dict[str, GuestResolution] = {}
        for resolution in resolver.resolve():
            if not resolution.is_resolved:
                continue
            for episode_id in resolution.episode_ids:
                by_episode[episode_id] = resolution
        return by_episode

    def _resolve_speakers(
        self,
        transcript: Transcript,
        ep_input: EpisodeInput,
        guest_resolutions: dict[str, GuestResolution],
    ) -> dict[str, SpeakerResolution]:
        """Resolve each per-episode label via host gallery, then recurring guest.

        Host voiceprint resolution runs first (when a gallery is injected); any
        label still unresolved that matches the episode's recurring guest is
        filled from the batch-level guest resolution. Labels with neither stay
        per-episode (an unresolved :class:`SpeakerResolution`).
        """

        labels = list(dict.fromkeys(seg.speaker for seg in transcript.segments))
        resolution: dict[str, SpeakerResolution] = {}

        gallery = self._deps.host_gallery
        if gallery is not None and ep_input.voice_sample_refs:
            resolution = gallery.resolve_transcript(
                transcript, ep_input.voice_sample_refs
            )

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


def utc_now() -> datetime:
    """UTC ``now`` helper (kept here so callers don't reimport timezone)."""

    return datetime.now(timezone.utc)
