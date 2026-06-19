"""Shared, deterministic test fixtures.

These back unit tests across every subpackage and require ONLY the core
dependency group. No real randomness, no real network: the fake embedder maps
known strings to fixed vectors, and the synthetic transcript/claims are
hand-authored so consensus/temporal logic is exercised deterministically.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from dlogos.schema import (
    Entity,
    EntityType,
    ExtractedClaim,
    Predicate,
    SourceSpan,
    SpeakerRef,
    Stance,
    Transcript,
    TranscriptSegment,
)


def _dt(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


@pytest.fixture
def synthetic_transcript() -> Transcript:
    """A tiny 2-speaker transcript mentioning Apple/the iPhone and OpenAI.

    Speaker labels are raw diarization labels (unresolved), and timestamps are
    monotonic so the speaker-verified citation check has something to bite on.
    """

    segments = [
        TranscriptSegment(
            speaker="SPEAKER_00",
            text="Welcome back. My guest today is a longtime Apple watcher.",
            t_start=0.0,
            t_end=4.5,
        ),
        TranscriptSegment(
            speaker="SPEAKER_01",
            text="Thanks. I think the iPhone has plateaued on hardware innovation.",
            t_start=4.5,
            t_end=10.0,
        ),
        TranscriptSegment(
            speaker="SPEAKER_00",
            text="Interesting. And what about OpenAI's pace this year?",
            t_start=10.0,
            t_end=14.0,
        ),
        TranscriptSegment(
            speaker="SPEAKER_01",
            text="OpenAI is moving fast, maybe too fast on safety, frankly.",
            t_start=14.0,
            t_end=19.5,
        ),
        TranscriptSegment(
            speaker="SPEAKER_00",
            text="So you'd rate Apple's hardware story negatively right now?",
            t_start=19.5,
            t_end=23.0,
        ),
        TranscriptSegment(
            speaker="SPEAKER_01",
            text="Yes, negatively, though I expect a rebound next cycle.",
            t_start=23.0,
            t_end=28.0,
        ),
    ]
    return Transcript(
        episode_id="ep-0001",
        language="en",
        segments=segments,
        duration_s=28.0,
    )


@pytest.fixture
def synthetic_claims() -> list[ExtractedClaim]:
    """Synthetic claims spanning time for consensus / temporal tests.

    Two speakers express stances on Apple across several dates so that a
    consensus-over-time helper can bucket and detect a shift, and so a
    contradiction (asserts vs disputes) exists at a single point.
    """

    def claim(
        *,
        speaker_label: str,
        speaker_id: str,
        predicate: Predicate,
        subject: str,
        subject_type: EntityType,
        obj: str,
        stance: Stance,
        sentiment: float,
        confidence: float,
        episode_id: str,
        t_start: float,
        t_end: float,
    ) -> ExtractedClaim:
        return ExtractedClaim(
            speaker=SpeakerRef(label=speaker_label, resolved_id=speaker_id),
            predicate=predicate,
            subject_entity=Entity(name=subject, type=subject_type),
            object=obj,
            stance=stance,
            sentiment=sentiment,
            confidence=confidence,
            source_span=SourceSpan(
                episode_id=episode_id, t_start=t_start, t_end=t_end
            ),
        )

    return [
        claim(
            speaker_label="SPEAKER_01",
            speaker_id="spk-analyst",
            predicate=Predicate.rates_negative,
            subject="Apple",
            subject_type=EntityType.organization,
            obj="hardware innovation has plateaued",
            stance=Stance.asserts,
            sentiment=-0.6,
            confidence=0.82,
            episode_id="ep-0001",
            t_start=4.5,
            t_end=10.0,
        ),
        claim(
            speaker_label="SPEAKER_00",
            speaker_id="spk-host",
            predicate=Predicate.rates_positive,
            subject="Apple",
            subject_type=EntityType.organization,
            obj="services growth offsets hardware",
            stance=Stance.disputes,
            sentiment=0.4,
            confidence=0.7,
            episode_id="ep-0002",
            t_start=120.0,
            t_end=130.0,
        ),
        claim(
            speaker_label="SPEAKER_01",
            speaker_id="spk-analyst",
            predicate=Predicate.expects,
            subject="Apple",
            subject_type=EntityType.organization,
            obj="a hardware rebound next cycle",
            stance=Stance.predicts,
            sentiment=0.3,
            confidence=0.65,
            episode_id="ep-0003",
            t_start=600.0,
            t_end=612.0,
        ),
        claim(
            speaker_label="SPEAKER_02",
            speaker_id="spk-guest-b",
            predicate=Predicate.rates_positive,
            subject="Apple",
            subject_type=EntityType.organization,
            obj="strongest product cycle in years",
            stance=Stance.asserts,
            sentiment=0.8,
            confidence=0.78,
            episode_id="ep-0004",
            t_start=45.0,
            t_end=58.0,
        ),
    ]


@pytest.fixture
def claim_event_times() -> dict[str, datetime]:
    """Event-time (publish date) per episode for the synthetic claims.

    Kept separate from the claims themselves because event-time is an episode
    property; temporal tests join the two.
    """

    return {
        "ep-0001": _dt(2026, 1, 10),
        "ep-0002": _dt(2026, 2, 15),
        "ep-0003": _dt(2026, 4, 1),
        "ep-0004": _dt(2026, 5, 20),
    }


class FakeEmbedder:
    """Deterministic embedder: known strings map to fixed unit vectors.

    Unknown strings hash to a stable pseudo-vector so behaviour is repeatable
    without any model or network. Inject this anywhere an embedder is needed.
    """

    DIM = 8

    _TABLE: dict[str, list[float]] = {
        "Apple": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "the iPhone": [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "iPhone": [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "Apple hardware": [0.85, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0],
        "OpenAI": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }

    def _hash_vector(self, text: str) -> list[float]:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        vec = rng.standard_normal(self.DIM)
        norm = float(np.linalg.norm(vec)) or 1.0
        return (vec / norm).tolist()

    def embed(self, text: str) -> list[float]:
        if text in self._TABLE:
            vec = np.asarray(self._TABLE[text], dtype=float)
            norm = float(np.linalg.norm(vec)) or 1.0
            return (vec / norm).tolist()
        return self._hash_vector(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    """A deterministic embedder mapping known strings to fixed vectors."""

    return FakeEmbedder()
