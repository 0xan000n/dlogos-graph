"""A deterministic, offline ASR backend.

:class:`MockASRBackend` satisfies the :class:`~dlogos.asr.base.ASRBackend`
Protocol and returns a fixed, hand-authored :class:`~dlogos.schema.Transcript`
without touching audio, GPUs, or the network. It exists so the full pipeline
(extraction → resolution → graph → retrieval → eval) can run *offline* on core
deps only, and so every unit test downstream of ASR has a stable input.

The default transcript mirrors the synthetic fixture in ``tests/conftest.py``
(a 2-speaker Apple/iPhone/OpenAI exchange) so behaviour is identical whether a
test injects the fixture directly or drives the pipeline through this backend.
The backend is intentionally *pure*: ``transcribe`` ignores the audio path's
*contents* and returns a deep copy of its template, so repeated calls — and
calls across processes — are byte-for-byte identical.
"""

from __future__ import annotations

from dlogos.schema import Transcript, TranscriptSegment


def _default_transcript_segments() -> list[TranscriptSegment]:
    """The canonical synthetic segments (kept in lock-step with conftest)."""

    return [
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


def default_synthetic_transcript(episode_id: str = "ep-0001") -> Transcript:
    """Build the canonical synthetic transcript for a given episode id."""

    return Transcript(
        episode_id=episode_id,
        language="en",
        segments=_default_transcript_segments(),
        duration_s=28.0,
    )


class MockASRBackend:
    """Deterministic ASR backend returning a fixed synthetic transcript.

    Parameters
    ----------
    transcript:
        Optional template to return. Defaults to the canonical synthetic
        transcript (matching ``tests/conftest.py``). The template is stored and
        every :meth:`transcribe` call returns a *deep copy*, so callers may
        mutate the result without affecting later calls.
    episode_id:
        Episode id stamped onto the *default* transcript. Ignored when an
        explicit ``transcript`` template is supplied (its id wins).

    The backend implements the :class:`~dlogos.asr.base.ASRBackend` Protocol;
    inject it anywhere a real backend would go to run the pipeline offline.
    """

    def __init__(
        self,
        transcript: Transcript | None = None,
        *,
        episode_id: str = "ep-0001",
    ) -> None:
        self._template: Transcript = (
            transcript
            if transcript is not None
            else default_synthetic_transcript(episode_id=episode_id)
        )

    def transcribe(self, audio_path: str) -> Transcript:
        """Return a deep copy of the template, ignoring the audio contents.

        ``audio_path`` is accepted to satisfy the Protocol but is not read —
        the mock is offline and deterministic by construction.
        """

        return self._template.model_copy(deep=True)
