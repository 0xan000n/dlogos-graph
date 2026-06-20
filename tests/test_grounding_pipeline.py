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
from dlogos.schema import Episode, Transcript, TranscriptSegment, Word

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


# --------------------------------------------------------------------------- #
# Word-level re-segmentation feeding grounding (Phase 0, Task 0.4)
# --------------------------------------------------------------------------- #
# A transcript whose ONLY utterance segment is a coarse whole-episode blob, but
# whose word stream resegments into tight ~sentence spans. With
# resegment_words=True grounding snaps the claim to the fine sentence (a small
# span); off (default) it can only snap to the coarse blob.
_WS_EPISODE_ID = "ep-resegment"
_WS_EVIDENCE = "Inflation is finally cooling across the economy."
_WS_DURATION = 30.0
# The fine sentence the evidence belongs to lives late in the episode.
_WS_FINE_START = 20.0
_WS_FINE_END = 23.5


def _resegment_transcript() -> Transcript:
    """One coarse 0..30s segment + a word stream of three ~sentence runs.

    Both speakers clear the 0.05 talk-time floor; the evidence sentence is the
    third run (speaker A) at [20.0, 23.5].
    """

    coarse = TranscriptSegment(
        speaker="A",
        text=(
            "Markets opened higher today on tech earnings. "
            "I disagree, the rally looks fragile and overextended. "
            + _WS_EVIDENCE
        ),
        t_start=0.0,
        t_end=_WS_DURATION,
    )

    def _run(words_text, start, gap, speaker):
        words, t = [], start
        for tok in words_text.split():
            words.append(Word(text=tok, t_start=t, t_end=t + gap, speaker=speaker))
            t += gap
        return words

    words = (
        _run("Markets opened higher today on tech earnings.", 0.0, 0.5, "A")
        + _run("I disagree, the rally looks fragile and overextended.", 8.0, 0.5, "B")
        + _run(_WS_EVIDENCE, _WS_FINE_START, 0.5, "A")
    )
    return Transcript(
        episode_id=_WS_EPISODE_ID,
        language="en",
        segments=[coarse],
        words=words,
        duration_s=_WS_DURATION,
    )


class _WSExtractionClient:
    """Emits one claim whose object is the evidence sentence, with a COARSE
    span (the whole chunk window) and an arbitrary (valid) label.

    The emitted span must fit the chunk window the extractor validates against;
    it spans [0, last-segment-end], which is the *coarse* blob the grounding
    pass then snaps to the fine sentence (with resegment on) or leaves coarse
    (off). ``_WS_FINE_END`` is the last resegmented segment's end and also the
    coarse single segment's end is the full duration — so [0, _WS_FINE_END]
    lies inside both chunk windows.
    """

    @property
    def chat(self):
        class _Completions:
            async def create(self, **kwargs: object):
                payload = json.dumps(
                    {
                        "claims": [
                            {
                                "speaker_label": "A",
                                "predicate": "explains",
                                "subject": "inflation",
                                "subject_type": "concept",
                                "object": _WS_EVIDENCE,
                                "stance": "asserts",
                                "sentiment": 0.1,
                                "confidence": 0.8,
                                "t_start": 0.0,
                                "t_end": _WS_FINE_END,
                            }
                        ]
                    }
                )
                return {"choices": [{"message": {"content": payload}}]}

        class _Chat:
            completions = _Completions()

        return _Chat()


def _ws_episode() -> Episode:
    return Episode(
        episode_id=_WS_EPISODE_ID,
        show_id="show-reseg",
        guid=f"guid-{_WS_EPISODE_ID}",
        title="resegment",
        published_at=datetime(2026, 1, 12, tzinfo=timezone.utc),
        audio_url="https://example.invalid/r.mp3",
    )


def _ws_deps(store, fake_embedder, *, resegment: bool) -> PipelineDeps:
    return PipelineDeps(
        asr=MockASRBackend(transcript=_resegment_transcript()),
        extractor=ClaimExtractor(_WSExtractionClient()),
        embedder=fake_embedder,
        store=store,
        fallback_speaker_id=lambda ep, label: (f"spk-{label.lower()}", None),
        resegment_words=resegment,
    )


async def test_resegment_words_grounds_to_fine_segment(fake_embedder) -> None:
    """With resegment_words=True the claim grounds to the tight sentence span."""

    store = FakeGraphStore()
    result = await Pipeline(_ws_deps(store, fake_embedder, resegment=True)).run(
        [EpisodeInput(episode=_ws_episode())]
    )

    assert len(result.resolved_claims) == 1
    claim = result.resolved_claims[0]
    # Snapped to the fine sentence, not the 30s blob.
    assert claim.source_span.t_start == _WS_FINE_START
    assert claim.source_span.t_end == _WS_FINE_END
    # The grounded span is short — well under the default re-segmentation cap.
    assert (claim.source_span.t_end - claim.source_span.t_start) <= 15.0


async def test_resegment_words_off_keeps_coarse_segment(fake_embedder) -> None:
    """Default (off): only the coarse blob exists, so the span stays coarse.

    The A/B control for Task 0.4: with no fine segments to snap to, the claim
    keeps a coarse span that begins at the top of the episode (0.0) and is far
    wider than the fine sentence — the exact defect resegmentation fixes.
    """

    store = FakeGraphStore()
    result = await Pipeline(_ws_deps(store, fake_embedder, resegment=False)).run(
        [EpisodeInput(episode=_ws_episode())]
    )

    assert len(result.resolved_claims) == 1
    claim = result.resolved_claims[0]
    # Coarse span: starts at the episode top, NOT the fine sentence start (20.0).
    assert claim.source_span.t_start == 0.0
    assert claim.source_span.t_start != _WS_FINE_START
    # And the span is much wider than the tight ~sentence the on-path produces.
    span = claim.source_span.t_end - claim.source_span.t_start
    fine_span = _WS_FINE_END - _WS_FINE_START
    assert span > fine_span
