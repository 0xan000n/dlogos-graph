"""Spike orchestration (spec §7.6): Approach A vs Approach B over N fixtures.

Runs both candidate integration shapes over the *same* N transcript fixtures
and captures, per approach, the artifacts :mod:`dlogos.spike.score` needs:

- **Approach A** — Graphiti *native* extraction pointed at an open-weight
  OpenAI-compatible endpoint. Graphiti owns extraction; on the bulk path its
  per-add LLM node-dedup must be disabled, but if it is *not* (the default),
  the per-add dedup calls show up as a hidden $/episode line item the spike
  surfaces. Modeled here behind an injected ``GraphitiNativePipeline`` protocol
  so the spike runs offline against a fake.
- **Approach B** — *our* extractor -> resolution -> bulk load. Chunk each
  transcript, run :class:`~dlogos.extraction.extractor.ClaimExtractor`
  (injected fake client), resolve subject entities
  (:func:`~dlogos.resolution.subjects.resolve_subjects`), stamp resolved
  speakers, then :meth:`~dlogos.graph.loader.ClaimLoader.bulk_load` into an
  injected store with the per-add LLM dedup **bypassed**.

Everything that would touch a network, a GPU, or a wall clock is **injected**:
the extractor client, the embedder, the graph store, the native-A pipeline, and
a :class:`Clock`. With the test doubles this whole module runs deterministically
on the core dependency group alone — no heavy imports at module top level.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from dlogos.extraction.chunking import chunk_transcript
from dlogos.extraction.extractor import ClaimExtractor, ExtractionError
from dlogos.graph.loader import ClaimLoader
from dlogos.resolution.subjects import Embedder, resolve_subjects
from dlogos.schema import ExtractedClaim, SpeakerRef, Transcript, TranscriptSegment


# --------------------------------------------------------------------------- #
# Injected timing — a Clock so wall-clock is deterministic in tests
# --------------------------------------------------------------------------- #
@runtime_checkable
class Clock(Protocol):
    """A monotonic seconds source. Production passes ``time.monotonic``."""

    def __call__(self) -> float:  # pragma: no cover - protocol
        ...


class FakeClock:
    """Deterministic clock: each call advances by a fixed step.

    Inject this in tests so an approach's ``wall_clock_seconds`` is exactly
    ``step × (calls - 1)`` regardless of the real machine — the throughput
    metric is then a pure function of the fixtures.
    """

    def __init__(self, *, start: float = 0.0, step: float = 1.0) -> None:
        self._t = start
        self._step = step
        self._first = True

    def __call__(self) -> float:
        if self._first:
            self._first = False
            return self._t
        self._t += self._step
        return self._t


# --------------------------------------------------------------------------- #
# Cost model inputs (injected unit prices)
# --------------------------------------------------------------------------- #
class Pricing(BaseModel):
    """Injected unit prices for the $/episode estimate (spec §7.6 axis 3).

    Token prices are per *single* token (not per-1K) to keep the arithmetic in
    :mod:`dlogos.spike.score` a plain multiply. ``dedup_call_usd`` is the flat
    cost of one Graphiti per-add LLM node-dedup call — the hidden line item the
    bulk path bypasses.
    """

    model_config = ConfigDict(extra="forbid")

    extraction_prompt_usd_per_token: float = Field(default=0.0, ge=0.0)
    extraction_completion_usd_per_token: float = Field(default=0.0, ge=0.0)
    dedup_call_usd: float = Field(default=0.0, ge=0.0)


# --------------------------------------------------------------------------- #
# Captured artifacts per approach
# --------------------------------------------------------------------------- #
class ApproachArtifacts(BaseModel):
    """Everything :mod:`dlogos.spike.score` reads about one approach's run.

    Deliberately a flat, serializable record so a run can be persisted and
    re-scored. ``transcript_segments`` is the diarization ground-truth (one
    list per episode) the span / attribution checks validate against.
    """

    model_config = ConfigDict(extra="forbid")

    approach: str = Field(description="'A' or 'B'.")
    label: str
    episodes: int = Field(ge=0)
    claims: list[ExtractedClaim] = Field(default_factory=list)

    wall_clock_seconds: float = Field(default=0.0, ge=0.0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    llm_dedup_calls: int = Field(
        default=0, ge=0, description="Per-add LLM node-dedup calls NOT bypassed."
    )

    # Approach B JSON reliability (open-weight structured output wobble).
    parse_attempts: int = Field(default=0, ge=0)
    parse_successes: int = Field(default=0, ge=0)

    # Diarization ground-truth per episode for the validity checks.
    transcript_segments: dict[str, list[TranscriptSegment]] = Field(
        default_factory=dict
    )

    pricing: Pricing = Field(default_factory=Pricing)


class ComparisonResult(BaseModel):
    """The paired output of a spike run: both approaches over the same fixtures."""

    model_config = ConfigDict(extra="forbid")

    episodes: int = Field(ge=0)
    approach_a: ApproachArtifacts
    approach_b: ApproachArtifacts


# --------------------------------------------------------------------------- #
# Approach A: Graphiti-native extraction (injected pipeline)
# --------------------------------------------------------------------------- #
class NativeExtractionResult(BaseModel):
    """What an Approach-A native pipeline reports for one transcript.

    The native pipeline owns extraction + load, so it reports the claims it
    produced plus its token usage and how many per-add LLM dedup calls it made
    (zero iff it was configured to bypass dedup on the bulk path).
    """

    model_config = ConfigDict(extra="forbid")

    claims: list[ExtractedClaim] = Field(default_factory=list)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    llm_dedup_calls: int = Field(default=0, ge=0)


@runtime_checkable
class GraphitiNativePipeline(Protocol):
    """Approach-A seam: Graphiti extracts + loads one transcript.

    Production wraps a real Graphiti client pointed at an open-weight endpoint
    (heavy imports stay inside that wrapper, never here). Tests inject a fake
    returning canned :class:`NativeExtractionResult`s, so the spike runs offline.
    """

    def ingest(self, transcript: Transcript) -> NativeExtractionResult:  # pragma: no cover
        ...


# --------------------------------------------------------------------------- #
# Approach B: speaker-resolution input
# --------------------------------------------------------------------------- #
@runtime_checkable
class SpeakerResolver(Protocol):
    """Maps a per-episode diarization label to a resolved speaker id/name.

    Approach B stamps resolved speaker ids onto extracted claims before the
    bulk load (the loader requires them). In the spike this is injected; the
    real cross-episode identity (host gallery + guest resolution) lives in
    :mod:`dlogos.speakers`. Returning ``None`` leaves the label unresolved.
    """

    def resolve(self, episode_id: str, label: str) -> tuple[str, str | None] | None:  # pragma: no cover
        ...


class IdentitySpeakerResolver:
    """Trivial resolver: deterministic ``spk-<episode>-<label>`` ids.

    Good enough for the spike's load path (every claim gets a stable resolved
    id so :meth:`ClaimLoader.bulk_load` accepts it) without pulling in the real
    voiceprint gallery. Inject a richer resolver to exercise real identity.
    """

    def resolve(
        self, episode_id: str, label: str
    ) -> tuple[str, str | None] | None:
        return (f"spk-{label.lower()}", None)


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #
class SpikeRunner:
    """Orchestrates Approach A vs Approach B over a list of transcripts.

    Parameters
    ----------
    native_pipeline:
        The Approach-A :class:`GraphitiNativePipeline` (injected; a fake in
        tests).
    extractor:
        The Approach-B :class:`~dlogos.extraction.extractor.ClaimExtractor`
        (constructed around an injected fake client).
    embedder:
        Subject-entity :class:`~dlogos.resolution.subjects.Embedder` for
        Approach-B resolution (inject the conftest ``FakeEmbedder``).
    store:
        The graph store for Approach B's bulk load (inject ``FakeGraphStore``).
    pricing:
        Unit prices for the $/episode estimate.
    clock:
        Monotonic seconds source; inject :class:`FakeClock` for determinism.
    speaker_resolver:
        Maps diarization labels to resolved speaker ids before the bulk load.
    bypass_llm_dedup:
        Whether Approach B's bulk load bypasses Graphiti per-add LLM dedup
        (the spec's required default — ``True``).
    """

    def __init__(
        self,
        *,
        native_pipeline: GraphitiNativePipeline,
        extractor: ClaimExtractor,
        embedder: Embedder,
        store: object,
        pricing: Pricing | None = None,
        clock: Clock | None = None,
        speaker_resolver: SpeakerResolver | None = None,
        bypass_llm_dedup: bool = True,
    ) -> None:
        self._native = native_pipeline
        self._extractor = extractor
        self._embedder = embedder
        self._store = store
        self._pricing = pricing or Pricing()
        import time

        self._clock: Clock = clock or time.monotonic
        self._resolver: SpeakerResolver = (
            speaker_resolver or IdentitySpeakerResolver()
        )
        self._bypass = bypass_llm_dedup

    # -- public API --------------------------------------------------------- #
    async def run(self, transcripts: list[Transcript]) -> ComparisonResult:
        """Run both approaches over ``transcripts`` and return the comparison."""

        a = await self._run_approach_a(transcripts)
        b = await self._run_approach_b(transcripts)
        return ComparisonResult(
            episodes=len(transcripts), approach_a=a, approach_b=b
        )

    # -- Approach A --------------------------------------------------------- #
    async def _run_approach_a(
        self, transcripts: list[Transcript]
    ) -> ApproachArtifacts:
        t0 = self._clock()
        claims: list[ExtractedClaim] = []
        prompt_tokens = 0
        completion_tokens = 0
        dedup_calls = 0
        seg_map: dict[str, list[TranscriptSegment]] = {}

        for tr in transcripts:
            seg_map[tr.episode_id] = list(tr.segments)
            result = self._native.ingest(tr)
            claims.extend(result.claims)
            prompt_tokens += result.prompt_tokens
            completion_tokens += result.completion_tokens
            dedup_calls += result.llm_dedup_calls

        t1 = self._clock()
        return ApproachArtifacts(
            approach="A",
            label="Graphiti-native extraction (open-weight endpoint)",
            episodes=len(transcripts),
            claims=claims,
            wall_clock_seconds=max(0.0, t1 - t0),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            llm_dedup_calls=dedup_calls,
            transcript_segments=seg_map,
            pricing=self._pricing,
        )

    # -- Approach B --------------------------------------------------------- #
    async def _run_approach_b(
        self, transcripts: list[Transcript]
    ) -> ApproachArtifacts:
        t0 = self._clock()
        raw_claims: list[ExtractedClaim] = []
        parse_attempts = 0
        parse_successes = 0
        seg_map: dict[str, list[TranscriptSegment]] = {}

        for tr in transcripts:
            seg_map[tr.episode_id] = list(tr.segments)
            chunks = chunk_transcript(tr)
            for chunk in chunks:
                parse_attempts += 1
                try:
                    chunk_claims = await self._extractor.extract(chunk)
                except ExtractionError:
                    # A parse failure even after the extractor's own retry:
                    # count the failed attempt and drop the chunk's claims.
                    continue
                parse_successes += 1
                raw_claims.extend(chunk_claims)

        # Resolution: subject-entity clustering -> canonical_id, then stamp
        # resolved speaker ids so the loader accepts the batch.
        resolution = resolve_subjects(raw_claims, self._embedder)
        resolved = [
            self._stamp_speaker(c) for c in resolution.claims
        ]
        loadable = [c for c in resolved if self._is_loadable(c)]

        loaded = self._bulk_load(loadable)

        t1 = self._clock()
        return ApproachArtifacts(
            approach="B",
            label="Our extractor -> resolution -> bulk load",
            episodes=len(transcripts),
            claims=loadable,
            wall_clock_seconds=max(0.0, t1 - t0),
            prompt_tokens=self._extractor_prompt_tokens(),
            completion_tokens=self._extractor_completion_tokens(),
            llm_dedup_calls=0 if self._bypass else loaded,
            parse_attempts=parse_attempts,
            parse_successes=parse_successes,
            transcript_segments=seg_map,
            pricing=self._pricing,
        )

    # -- Approach B helpers ------------------------------------------------- #
    def _stamp_speaker(self, claim: ExtractedClaim) -> ExtractedClaim:
        episode_id = claim.source_span.episode_id
        resolved = self._resolver.resolve(episode_id, claim.speaker.label)
        if resolved is None:
            return claim
        speaker_id, name = resolved
        new_ref = SpeakerRef(
            label=claim.speaker.label, resolved_id=speaker_id, name=name
        )
        return claim.model_copy(update={"speaker": new_ref})

    @staticmethod
    def _is_loadable(claim: ExtractedClaim) -> bool:
        """A claim is loadable iff resolution stamped speaker + entity ids."""

        return bool(
            claim.speaker.resolved_id and claim.subject_entity.canonical_id
        )

    def _bulk_load(self, claims: list[ExtractedClaim]) -> int:
        loader = ClaimLoader()
        return loader.bulk_load(
            self._store, claims, bypass_llm_dedup=self._bypass
        )

    def _extractor_prompt_tokens(self) -> int:
        """Read prompt-token usage off the injected client if it tracks it.

        A real ``AsyncOpenAI`` does not expose a running total, so production
        threads token usage through a wrapping client; the fake test client
        exposes ``prompt_tokens``. Absent the attribute, return 0 (cost for
        Approach B then reflects only any non-bypassed dedup calls).
        """

        client = getattr(self._extractor, "_client", None)
        return int(getattr(client, "prompt_tokens", 0) or 0)

    def _extractor_completion_tokens(self) -> int:
        client = getattr(self._extractor, "_client", None)
        return int(getattr(client, "completion_tokens", 0) or 0)


# --------------------------------------------------------------------------- #
# Convenience entry point
# --------------------------------------------------------------------------- #
async def run_spike(
    transcripts: list[Transcript],
    *,
    native_pipeline: GraphitiNativePipeline,
    extractor: ClaimExtractor,
    embedder: Embedder,
    store: object,
    pricing: Pricing | None = None,
    clock: Clock | None = None,
    speaker_resolver: SpeakerResolver | None = None,
    bypass_llm_dedup: bool = True,
) -> ComparisonResult:
    """Construct a :class:`SpikeRunner` and run it over ``transcripts``.

    Thin wrapper so a caller (or the pipeline layer) can run the spike in one
    call while still injecting every collaborator.
    """

    runner = SpikeRunner(
        native_pipeline=native_pipeline,
        extractor=extractor,
        embedder=embedder,
        store=store,
        pricing=pricing,
        clock=clock,
        speaker_resolver=speaker_resolver,
        bypass_llm_dedup=bypass_llm_dedup,
    )
    return await runner.run(transcripts)
