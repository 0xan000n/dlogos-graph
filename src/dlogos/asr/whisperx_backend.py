"""Real WhisperX ASR backend (whisper-large-v3, word timestamps).

This is the production transcription path (spec §7.2): WhisperX runs
whisper-large-v3 for ASR with word-level timestamps, then forced-alignment
sharpens those word boundaries. Diarization (pyannote) and the
token→speaker-turn mapping live in :mod:`dlogos.asr.diarization`; this backend
orchestrates ASR → alignment → diarization → :class:`Transcript`.

HARD CONSTRAINT — *every* heavy dependency (``torch``, ``whisperx``) is
imported **lazily inside methods**, never at module top level. Importing this
module therefore costs nothing and never requires the ``asr`` optional extras,
so the unit-test suite (core deps only) can import it freely. Construction is
also cheap: nothing loads until :meth:`transcribe` is called.
"""

from __future__ import annotations

from dlogos.schema import Transcript, TranscriptSegment


class WhisperXBackend:
    """WhisperX + pyannote ASR backend (satisfies :class:`ASRBackend`).

    Parameters
    ----------
    model_name:
        Whisper model to load. Defaults to ``whisper-large-v3`` per spec §7.2.
    device:
        ``"cuda"`` or ``"cpu"``. ``None`` auto-detects at transcribe time.
    compute_type:
        faster-whisper compute type (e.g. ``"float16"`` on GPU,
        ``"int8"`` on CPU). ``None`` picks a sane default for the device.
    language:
        Optional forced language code; ``None`` lets WhisperX detect it.
    batch_size:
        Inference batch size for the transcription pass.
    hf_token:
        HuggingFace token for the gated pyannote diarization pipeline. When
        ``None`` the backend reads it from configuration lazily.
    diarize:
        When ``True`` (default) run pyannote diarization and assign speakers;
        when ``False`` emit a single-speaker transcript (``SPEAKER_00``).
    min_speakers / max_speakers:
        Optional hints passed through to the diarizer.

    Nothing in ``__init__`` imports a heavy dep — models are loaded inside
    :meth:`transcribe`.
    """

    def __init__(
        self,
        model_name: str = "whisper-large-v3",
        *,
        device: str | None = None,
        compute_type: str | None = None,
        language: str | None = None,
        batch_size: int = 16,
        hf_token: str | None = None,
        diarize: bool = True,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.batch_size = batch_size
        self.hf_token = hf_token
        self.diarize = diarize
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

    # ------------------------------------------------------------------ #
    # Lazy device / dependency resolution
    # ------------------------------------------------------------------ #
    def _resolve_device(self) -> str:
        """Pick a device, lazily importing torch only to probe for CUDA."""

        if self.device is not None:
            return self.device
        import torch  # lazy: heavy dep, imported only when transcribing

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _resolve_compute_type(self, device: str) -> str:
        if self.compute_type is not None:
            return self.compute_type
        return "float16" if device == "cuda" else "int8"

    def _resolve_hf_token(self) -> str | None:
        if self.hf_token is not None:
            return self.hf_token
        # Lazy config read so importing this module needs no settings either.
        try:
            from dlogos.config import settings

            # Reuse the extraction key slot is wrong; pyannote needs a HF token.
            # We deliberately do not invent a config field here — callers pass
            # hf_token explicitly when diarizing. Returning None lets pyannote
            # fall back to its own env (HUGGINGFACE_TOKEN) if present.
            _ = settings  # touch to show intent; no dedicated field today
        except Exception:  # pragma: no cover - config import is cheap & safe
            pass
        return None

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def transcribe(self, audio_path: str) -> Transcript:
        """Transcribe + align + diarize ``audio_path`` into a Transcript.

        Pipeline (all heavy imports are local to this method):

        1. Load whisper-large-v3 and transcribe with word timestamps.
        2. Forced-align the words to sharpen timestamps.
        3. (Optional) pyannote diarization, then map words → speaker turns by
           timing via :mod:`dlogos.asr.diarization`.
        4. Collapse aligned, speaker-tagged words into
           :class:`~dlogos.schema.TranscriptSegment` turns.
        """

        import whisperx  # lazy: heavy dep

        device = self._resolve_device()
        compute_type = self._resolve_compute_type(device)

        audio = whisperx.load_audio(audio_path)

        # 1) ASR (word-level via faster-whisper under the hood).
        model = whisperx.load_model(
            self.model_name,
            device,
            compute_type=compute_type,
            language=self.language,
        )
        asr_result = model.transcribe(audio, batch_size=self.batch_size)
        detected_language: str = asr_result.get("language", self.language or "en")

        # 2) Forced alignment → word-level timestamps.
        align_model, align_meta = whisperx.load_align_model(
            language_code=detected_language, device=device
        )
        aligned = whisperx.align(
            asr_result["segments"],
            align_model,
            align_meta,
            audio,
            device,
            return_char_alignments=False,
        )

        # 3) Diarization + token→speaker mapping.
        if self.diarize:
            from dlogos.asr.diarization import (
                assign_word_speakers,
                run_pyannote_diarization,
            )

            diarize_segments = run_pyannote_diarization(
                audio_path,
                hf_token=self._resolve_hf_token(),
                device=device,
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers,
            )
            aligned = assign_word_speakers(diarize_segments, aligned)

        # 4) Collapse into speaker-turn segments.
        segments = self._aligned_to_segments(aligned)
        duration_s = self._estimate_duration(audio, segments)

        # episode_id is the source-of-truth caller key; the loader stamps the
        # canonical id, so here we mirror it from the audio path stem.
        episode_id = self._episode_id_from_path(audio_path)

        return Transcript(
            episode_id=episode_id,
            language=detected_language,
            segments=segments,
            duration_s=duration_s,
        )

    # ------------------------------------------------------------------ #
    # Pure helpers (no heavy deps — unit-testable in isolation)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _episode_id_from_path(audio_path: str) -> str:
        """Derive a provisional episode id from the audio file name."""

        from pathlib import Path

        return Path(audio_path).stem or "unknown-episode"

    @staticmethod
    def _aligned_to_segments(aligned: dict) -> list[TranscriptSegment]:
        """Collapse WhisperX aligned segments into speaker-turn segments.

        WhisperX returns a ``{"segments": [...]}`` dict where each segment may
        carry a ``"speaker"`` (after diarization assignment) plus ``"start"``,
        ``"end"`` and ``"text"``. Adjacent segments by the *same* speaker are
        merged into a single turn so the output reads as turns, not words.
        """

        raw = aligned.get("segments", []) if isinstance(aligned, dict) else []
        turns: list[TranscriptSegment] = []

        for seg in raw:
            speaker = str(seg.get("speaker", "SPEAKER_00"))
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            start = float(seg.get("start", 0.0) or 0.0)
            end = float(seg.get("end", start) or start)

            if turns and turns[-1].speaker == speaker:
                prev = turns[-1]
                turns[-1] = TranscriptSegment(
                    speaker=speaker,
                    text=f"{prev.text} {text}".strip(),
                    t_start=prev.t_start,
                    t_end=max(prev.t_end, end),
                )
            else:
                turns.append(
                    TranscriptSegment(
                        speaker=speaker,
                        text=text,
                        t_start=start,
                        t_end=max(start, end),
                    )
                )

        return turns

    @staticmethod
    def _estimate_duration(audio, segments: list[TranscriptSegment]) -> float:
        """Best-effort episode duration in seconds.

        Prefer the raw audio length (samples / 16 kHz, WhisperX's sample rate);
        fall back to the last segment's ``t_end`` if the audio object is opaque.
        """

        try:
            import numpy as np  # core dep

            if audio is not None and hasattr(audio, "__len__"):
                return float(len(np.asarray(audio))) / 16000.0
        except Exception:  # pragma: no cover - defensive only
            pass
        return segments[-1].t_end if segments else 0.0
