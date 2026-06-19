"""Tests for the deterministic offline ASR backend (``MockASRBackend``).

Confirms the mock backend is byte-for-byte deterministic, never reads audio,
returns independent copies, matches the conftest synthetic fixture, and never
imports a heavy dep. Core deps only.
"""

from __future__ import annotations

import sys

from dlogos.asr.base import ASRBackend, talk_time_by_speaker
from dlogos.asr.mock_backend import MockASRBackend, default_synthetic_transcript
from dlogos.schema import Transcript, TranscriptSegment


def test_transcribe_is_deterministic_across_calls() -> None:
    backend = MockASRBackend()
    a = backend.transcribe("/does/not/exist.mp3")
    b = backend.transcribe("/some/other/path.wav")
    # Independent of the audio path; identical content.
    assert a == b
    assert a.model_dump() == b.model_dump()


def test_transcribe_ignores_audio_path_contents() -> None:
    backend = MockASRBackend()
    # A path that does not exist must still produce a transcript (offline).
    t = backend.transcribe("/nonexistent/episode-9999.flac")
    assert isinstance(t, Transcript)
    assert len(t.segments) == 6


def test_returns_independent_copies_not_shared_state() -> None:
    backend = MockASRBackend()
    first = backend.transcribe("a.mp3")
    first.segments.clear()  # mutate the returned copy
    second = backend.transcribe("b.mp3")
    # The second call is unaffected by mutating the first.
    assert len(second.segments) == 6
    assert second.segments[0].speaker == "SPEAKER_00"


def test_default_matches_conftest_fixture(synthetic_transcript) -> None:
    # The mock's default transcript mirrors the shared synthetic fixture so
    # driving the pipeline through the backend matches injecting the fixture.
    produced = MockASRBackend().transcribe("anything.mp3")
    assert produced == synthetic_transcript


def test_episode_id_is_configurable() -> None:
    backend = MockASRBackend(episode_id="ep-custom")
    assert backend.transcribe("x.mp3").episode_id == "ep-custom"


def test_custom_template_is_returned_and_copied() -> None:
    custom = Transcript(
        episode_id="ep-panel",
        language="en",
        segments=[
            TranscriptSegment(speaker="SPEAKER_00", text="hi", t_start=0.0, t_end=1.0),
        ],
        duration_s=1.0,
    )
    backend = MockASRBackend(custom)
    out = backend.transcribe("ignored.mp3")
    assert out == custom
    # Explicit template's episode id wins over the episode_id kwarg default.
    assert out.episode_id == "ep-panel"
    # Mutating the output must not corrupt the template for the next call.
    out.segments.clear()
    assert len(backend.transcribe("again.mp3").segments) == 1


def test_default_synthetic_transcript_helper() -> None:
    t = default_synthetic_transcript(episode_id="ep-7")
    assert t.episode_id == "ep-7"
    assert t.duration_s == 28.0
    # The two speakers split the talk time; both well above the 5% threshold.
    stats = talk_time_by_speaker(t)
    assert set(stats.by_speaker) == {"SPEAKER_00", "SPEAKER_01"}
    assert min(stats.fractions.values()) > 0.05


def test_backend_is_asr_backend_protocol_instance() -> None:
    assert isinstance(MockASRBackend(), ASRBackend)


def test_no_heavy_deps_imported_by_mock_path() -> None:
    # Importing + using the mock backend must never pull in torch/whisperx/
    # pyannote. (Other tests in the session may import them, so we only assert
    # the modules under test do not.)
    import dlogos.asr  # noqa: F401  (import triggers package __init__)
    import dlogos.asr.base  # noqa: F401
    import dlogos.asr.mock_backend  # noqa: F401

    MockASRBackend().transcribe("x.mp3")
    for heavy in ("torch", "whisperx", "pyannote", "pyannote.audio"):
        assert heavy not in sys.modules, f"{heavy} must not be imported by the mock path"
