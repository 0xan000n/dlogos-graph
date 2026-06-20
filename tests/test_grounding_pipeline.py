"""End-to-end grounding through the pipeline (spec §7.4).

The unit tests in ``tests/extraction/test_grounding.py`` pin the grounding
*function* down in isolation. These tests pin the *wiring*: that
:class:`~dlogos.pipeline.Pipeline` runs the grounding pass on each episode's
extracted claims AFTER extraction and BEFORE speaker-stamping, so a claim whose
extractor span is a coarse LLM estimate gets snapped to the real transcript
segment — and the corrected diarization label is what speaker resolution then
reads, so the canonical speaker id derives from the GROUNDED label, not the
model's guess.

Everything is injected on fakes (a :class:`MockASRBackend` carrying a bespoke
transcript, a canned async extraction client, the deterministic fake embedder,
and a :class:`FakeGraphStore`); no network, no heavy deps.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from dlogos.asr.mock_backend import MockASRBackend
from dlogos.extraction.extractor import ClaimExtractor
from dlogos.graph.fake_store import FakeGraphStore
from dlogos.pipeline import EpisodeInput, Pipeline, PipelineDeps
from dlogos.schema import Episode, Transcript, TranscriptSegment

EPISODE_ID = "ep-ground-pipe"

# The real diarized segment the claim's evidence came from: SPEAKER_01 at a
# precise [6.0, 12.5]s. The extractor will instead emit a coarse, whole-window
# estimate and a WRONG (but valid) speaker label; grounding must fix both.
TRUE_SEGMENT_START = 6.0
TRUE_SEGMENT_END = 12.5
TRUE_SEGMENT_LABEL = "SPEAKER_01"
EVIDENCE = "sourdough bread needs a long cold fermentation to develop flavor"

# The defective span the extractor "estimates" (it spans the whole chunk window,
# the failure the smoke surfaced: a cited [t_start,t_end] that is a multi-segment
# blob rather than the one utterance the claim came from).
ESTIMATED_START = 0.0
ESTIMATED_END = 19.0


def _transcript() -> Transcript:
    """Three distinctly-worded segments by three different speakers.

    The wordings share no salient content words, so a claim's evidence can only
    match the one segment it was actually drawn from.
    """

    segments = [
        TranscriptSegment(
            speaker="SPEAKER_00",
            text="The new electric pickup truck has incredible towing capacity.",
            t_start=0.0,
            t_end=TRUE_SEGMENT_START,
        ),
        TranscriptSegment(
            speaker=TRUE_SEGMENT_LABEL,
            text="Honestly, sourdough bread needs a long cold fermentation to "
            "develop flavor.",
            t_start=TRUE_SEGMENT_START,
            t_end=TRUE_SEGMENT_END,
        ),
        TranscriptSegment(
            speaker="SPEAKER_02",
            text="Quantum computers will eventually break most public-key "
            "cryptography schemes.",
            t_start=TRUE_SEGMENT_END,
            t_end=ESTIMATED_END,
        ),
    ]
    return Transcript(
        episode_id=EPISODE_ID,
        language="en",
        segments=segments,
        duration_s=ESTIMATED_END,
    )


class _FakeExtractionClient:
    """Async OpenAI-compatible client returning one claim with a DEFECTIVE span.

    The claim's ``object`` is segment-2's evidence, but its ``t_start/t_end`` is
    the whole-chunk estimate and its ``speaker_label`` is SPEAKER_00 — the wrong
    (but valid) diarization label. That is exactly the pair of defects the
    grounding pass exists to correct.
    """

    def __init__(self) -> None:
        self.create_calls = 0

    @property
    def chat(self):
        outer = self

        class _Completions:
            async def create(self, **kwargs: object):
                outer.create_calls += 1
                payload = json.dumps(
                    {
                        "claims": [
                            {
                                # WRONG label: the audio at the evidence is
                                # SPEAKER_01, but the model attributes SPEAKER_00.
                                "speaker_label": "SPEAKER_00",
                                "predicate": "rates_positive",
                                "subject": "sourdough",
                                "subject_type": "concept",
                                "object": EVIDENCE,
                                "stance": "asserts",
                                "sentiment": 0.2,
                                "confidence": 0.8,
                                # COARSE span: the whole chunk window, not the
                                # real segment [6.0, 12.5].
                                "t_start": ESTIMATED_START,
                                "t_end": ESTIMATED_END,
                            }
                        ]
                    }
                )
                return {"choices": [{"message": {"content": payload}}]}

        class _Chat:
            completions = _Completions()

        return _Chat()


def _episode() -> Episode:
    return Episode(
        episode_id=EPISODE_ID,
        show_id="show-ground",
        guid=f"guid-{EPISODE_ID}",
        title="grounding",
        published_at=datetime(2026, 1, 10, tzinfo=timezone.utc),
        audio_url="https://example.invalid/a.mp3",
    )


def _deps(store: FakeGraphStore, fake_embedder, *, ground: bool = True) -> PipelineDeps:
    return PipelineDeps(
        asr=MockASRBackend(transcript=_transcript()),
        extractor=ClaimExtractor(_FakeExtractionClient()),
        embedder=fake_embedder,
        store=store,
        # A per-label fallback so the (corrected) diarization label resolves to a
        # stable speaker id — this is what proves the canonical id derives from
        # the GROUNDED label rather than the model's guess.
        fallback_speaker_id=lambda ep, label: (f"spk-{label.lower()}", None),
        ground_claims=ground,
    )


async def test_pipeline_regrounds_estimated_span_to_real_segment(
    fake_embedder,
) -> None:
    """Through Pipeline.run, the coarse estimated span snaps to the real one."""

    store = FakeGraphStore()
    result = await Pipeline(_deps(store, fake_embedder)).run(
        [EpisodeInput(episode=_episode())]
    )

    # Exactly one claim survived to the load, carrying the GROUNDED span.
    assert len(result.resolved_claims) == 1
    claim = result.resolved_claims[0]
    assert claim.source_span.t_start == TRUE_SEGMENT_START
    assert claim.source_span.t_end == TRUE_SEGMENT_END
    # The episode id on the span is preserved by the regrounding copy.
    assert claim.source_span.episode_id == EPISODE_ID

    # The diarization label was corrected from the model's wrong SPEAKER_00 to
    # the segment's real SPEAKER_01, and the canonical speaker id derives from
    # THAT corrected label (via the fallback) — the whole point of grounding
    # before speaker-stamping.
    assert claim.speaker.label == TRUE_SEGMENT_LABEL
    assert claim.speaker.resolved_id == f"spk-{TRUE_SEGMENT_LABEL.lower()}"

    # And it actually loaded into the graph.
    assert result.claims_loaded == 1
    assert store.claim_count() == 1


async def test_grounded_span_is_loaded_into_the_graph_node(fake_embedder) -> None:
    """The regrounded span flows all the way onto the loaded ClaimNode."""

    store = FakeGraphStore()
    await Pipeline(_deps(store, fake_embedder)).run(
        [EpisodeInput(episode=_episode())]
    )

    [node] = list(store.claims.values())
    assert node.source_span.t_start == TRUE_SEGMENT_START
    assert node.source_span.t_end == TRUE_SEGMENT_END
    # The graph speaker node is keyed by the GROUNDED label's id, so the loaded
    # ASSERTS edge attributes the claim to the corrected speaker.
    assert node.speaker_id == f"spk-{TRUE_SEGMENT_LABEL.lower()}"


async def test_grounding_flag_off_keeps_raw_estimated_span(fake_embedder) -> None:
    """With ground_claims=False the raw (coarse) extractor span survives.

    This is the A/B control: it both proves the flag actually gates the pass and
    pins down the defect the on-by-default path corrects.
    """

    store = FakeGraphStore()
    result = await Pipeline(
        _deps(store, fake_embedder, ground=False)
    ).run([EpisodeInput(episode=_episode())])

    assert len(result.resolved_claims) == 1
    claim = result.resolved_claims[0]
    # Raw coarse span survives, unsnapped.
    assert claim.source_span.t_start == ESTIMATED_START
    assert claim.source_span.t_end == ESTIMATED_END
    # And the model's wrong label survives -> a wrong canonical attribution.
    assert claim.speaker.label == "SPEAKER_00"
    assert claim.speaker.resolved_id == "spk-speaker_00"
