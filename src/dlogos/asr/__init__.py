"""ASR subpackage: transcription + diarization + alignment.

Turns raw episode audio into a speaker-labeled, word-timestamped
:class:`~dlogos.schema.Transcript` (the unit every downstream stage consumes).

Public surface:

- :class:`~dlogos.asr.base.ASRBackend` — the Protocol every backend satisfies.
- :func:`~dlogos.asr.base.drop_low_talk_time_speakers` — the talk-time
  threshold helper (SPoRC precedent: drop speakers under a small % of total
  talk time) that prunes spurious diarization labels before resolution.
- :class:`~dlogos.asr.mock_backend.MockASRBackend` — a deterministic, offline
  backend that returns a fixed synthetic transcript so the pipeline runs with
  core deps only (no torch / whisperx / pyannote).

The real backends (:mod:`dlogos.asr.whisperx_backend`,
:mod:`dlogos.asr.diarization`) import their heavy dependencies *lazily* inside
methods, so importing this package — and running the unit tests — never pulls
in torch / whisperx / pyannote.
"""

from __future__ import annotations

from dlogos.asr.base import (
    ASRBackend,
    TalkTimeStats,
    drop_low_talk_time_speakers,
    talk_time_by_speaker,
)
from dlogos.asr.mock_backend import MockASRBackend

__all__ = [
    "ASRBackend",
    "TalkTimeStats",
    "drop_low_talk_time_speakers",
    "talk_time_by_speaker",
    "MockASRBackend",
]
