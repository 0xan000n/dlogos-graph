"""End-to-end smoke test: the whole tree wired together, offline (spec §5, §9).

Runs the real :class:`~dlogos.pipeline.Pipeline` on ONE fixture episode through
the offline fakes (mock ASR backend + fake graph store + fake embedder + a fake
async extraction client), then answers ONE ``temporal_consensus`` golden query
through the real dLogos arm (:class:`~dlogos.eval.arms.ModelDLogosArm` +
:class:`~dlogos.eval.arms.DLogosGraphRetriever` over the pipeline's loaded
graph). Asserts a non-empty, *cited* answer whose citation passes the
speaker-verified check against the same transcript the pipeline diarized.

This is the integration proof: ingest -> ASR -> diarize/prune -> speaker
identity -> chunk -> extract -> resolve -> bulk-load -> retrieval surface ->
dLogos arm, all on the core dependency group with no network and no heavy deps.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from dlogos.asr.mock_backend import MockASRBackend
from dlogos.eval.arms import (
    ARM_DLOGOS,
    Answer,
    DLogosGraphRetriever,
    ModelDLogosArm,
)
from dlogos.eval.golden import AnswerShape, Archetype, Domain, GoldenQuery
from dlogos.eval.rubric import verify_citation
from dlogos.extraction.extractor import ClaimExtractor
from dlogos.graph.fake_store import FakeGraphStore
from dlogos.pipeline import EpisodeInput, Pipeline, PipelineDeps
from dlogos.schema import Episode


# --------------------------------------------------------------------------- #
# Offline fakes
# --------------------------------------------------------------------------- #
class _FakeExtractionClient:
    """An async OpenAI-compatible client returning canned, chunk-grounded JSON.

    Satisfies :class:`~dlogos.extraction.extractor.AsyncChatClient`. The claims
    are grounded in the mock transcript's segment windows so the extractor's
    span-in-window validation accepts them, and attributed to the speaker labels
    present in the chunk.
    """

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


class _FakeFrontierClient:
    """A deterministic frontier chat client for the dLogos arm.

    Echoes the dLogos evidence into the answer so the test can assert the
    structured, attributed context actually reached the model.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def complete(self, *, system: str, user: str) -> str:
        self.calls.append((system, user))
        return f"Synthesis based on dLogos evidence:\n{user}"


def _extraction_payload() -> str:
    """Two claims grounded in the canonical synthetic transcript windows."""

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


def _episode() -> Episode:
    return Episode(
        episode_id="ep-0001",
        show_id="show-tech",
        guid="guid-ep-0001",
        title="The state of Apple hardware",
        published_at=datetime(2026, 1, 10, tzinfo=timezone.utc),
        audio_url="https://example.invalid/ep-0001.mp3",
    )


def _temporal_consensus_query() -> GoldenQuery:
    return GoldenQuery(
        id="gq-smoke",
        archetype=Archetype.temporal_consensus,
        domain=Domain.technology,
        query_text=(
            "How has the consensus on Apple hardware moved across the shows?"
        ),
        pre_registered_answer_shape=AnswerShape(
            expected_subjects=["Apple", "Apple hardware"],
            expected_stance_shift=True,
            min_attributed_sources=1,
        ),
        deep_tier=True,
    )


# --------------------------------------------------------------------------- #
# The smoke test
# --------------------------------------------------------------------------- #
async def test_e2e_pipeline_then_dlogos_arm(fake_embedder) -> None:
    store = FakeGraphStore()
    extractor = ClaimExtractor(_FakeExtractionClient(_extraction_payload()))

    deps = PipelineDeps(
        asr=MockASRBackend(),
        extractor=extractor,
        embedder=fake_embedder,
        store=store,
        # One-off speakers get a stable per-episode id so the loader accepts them
        # (spec §7.3: the long tail "remains a per-episode speaker").
        fallback_speaker_id=lambda episode_id, label: (
            f"spk-{episode_id}-{label.lower()}",
            None,
        ),
    )

    # --- Run the pipeline on ONE fixture episode through the fakes. ---------- #
    result = await Pipeline(deps).run([EpisodeInput(episode=_episode())])

    # Claims were extracted, resolved (speaker id + canonical id), and loaded.
    assert result.claims_loaded >= 1
    assert store.claim_count() == result.claims_loaded
    # The dedup-bypass fast path ran — no per-add LLM dedup invocations.
    assert store.llm_dedup_invocations == 0
    # Every loaded claim is fully resolved.
    for claim in result.resolved_claims:
        assert claim.speaker.resolved_id
        assert claim.subject_entity.canonical_id

    # --- Answer ONE temporal_consensus golden query via the dLogos arm. ------ #
    surface = result.build_retrieval_surface(fake_embedder)
    retriever = DLogosGraphRetriever(surface, top_k=8)
    arm = ModelDLogosArm(_FakeFrontierClient(), retriever)

    answer = await arm(_temporal_consensus_query())

    # A non-empty, cited answer.
    assert isinstance(answer, Answer)
    assert answer.arm == ARM_DLOGOS
    assert answer.text.strip()
    assert answer.citations, "the dLogos arm must return at least one citation"

    # The structured, attributed evidence reached the model (not just raw text).
    assert "Attributed spans" in answer.text or "Consensus on" in answer.text

    # --- The citation is speaker-verified against the pipeline's transcript. - #
    transcript = result.runs[0].transcript
    # Map each transcript segment index -> its resolved speaker id (the same
    # ids the pipeline stamped onto claims).
    label_to_id = {
        label: res.resolved.speaker_id
        for label, res in result.runs[0].speaker_resolutions.items()
        if res.resolved is not None
    }
    segment_speaker_ids = {
        idx: label_to_id[seg.speaker]
        for idx, seg in enumerate(transcript.segments)
        if seg.speaker in label_to_id
    }

    cit = answer.citations[0]
    verdict = verify_citation(cit, transcript, segment_speaker_ids)
    assert verdict.passed, verdict.reason
    assert verdict.actual_speaker_id == cit.speaker_id


def test_e2e_imports_use_no_heavy_deps() -> None:
    """Guard: importing the wiring path pulls no heavy/optional dependency."""

    import sys

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
    assert leaked == [], f"heavy deps leaked at import time: {leaked}"
