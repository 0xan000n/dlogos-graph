"""Hosted ASR backend: AssemblyAI (diarization + word timestamps, no GPU).

This is the *lowest-friction* real transcription path for the one-episode smoke
run (spec: smoke targets hosted infra, not GPU/self-hosting). A single
``ASSEMBLYAI_API_KEY`` buys speaker diarization, word-level timestamps and
speaker labels over plain HTTPS — no torch, no whisperx, no pyannote, no CUDA.

It satisfies the same :class:`~dlogos.asr.base.ASRBackend` Protocol as the
WhisperX backend, so it drops into the pipeline interchangeably and its output —
a fully populated :class:`~dlogos.schema.Transcript` — is byte-compatible with
every downstream stage.

AssemblyAI REST contract used here (verifiable against current API docs at
https://www.assemblyai.com/docs/api-reference) — base URL ``https://api.assemblyai.com``:

1. **Upload** (only when given a local file): ``POST /v2/upload``
   - Headers: ``authorization: <API_KEY>``, ``content-type: application/octet-stream``
   - Body: the raw audio bytes (streamed).
   - Response JSON: ``{"upload_url": "https://cdn.assemblyai.com/upload/..."}``.
   We read the ``upload_url`` field and feed it as ``audio_url`` below. When the
   caller already passes an ``http(s)://`` URL we skip upload entirely.

2. **Submit transcription**: ``POST /v2/transcript``
   - Headers: ``authorization: <API_KEY>``, ``content-type: application/json``
   - Body fields we set:
       - ``audio_url`` — the uploaded/remote audio URL.
       - ``speaker_labels: true`` — enables diarization (utterances + speaker tags).
       - ``punctuate: true`` / ``format_text: true`` — readable segment text.
       - ``language_detection: true`` (default) OR ``language_code`` if pinned.
       - optional ``speakers_expected`` when the caller hints a speaker count.
   - Response JSON: a transcript object with an ``id`` and ``status`` field.

3. **Poll**: ``GET /v2/transcript/{id}`` until ``status`` is terminal.
   - ``status`` ∈ {``queued``, ``processing``, ``completed``, ``error``}.
   - On ``completed`` we read:
       - ``utterances``: ``[{"speaker": "A", "text": "...", "start": <ms>,
         "end": <ms>, "confidence": <float>}, ...]`` — one per diarized turn.
       - ``audio_duration``: episode length in **seconds** (int/float).
       - ``language_code``: the detected/forced ISO language (e.g. ``"en"``).
   - On ``error`` we raise with the ``error`` message.

Unit conversion: AssemblyAI emits ``start``/``end`` in **milliseconds**; the
schema (:class:`~dlogos.schema.TranscriptSegment`) is in **seconds**, so every
utterance offset is divided by 1000. ``audio_duration`` is already in seconds.

HARD CONSTRAINT — ``httpx`` is imported **lazily inside methods**, never at
module top level. Importing this module costs nothing and pulls in no optional
dep, so the core-only unit-test suite imports it freely. The API key is read
lazily from :data:`dlogos.config.settings` (``ASSEMBLYAI_API_KEY``).

HONESTY: this backend cannot be exercised in CI (no key / no network). The unit
tests below drive the full upload→submit→poll→map flow against an
``httpx.MockTransport`` returning canned AssemblyAI JSON; its first REAL run is
the smoke itself.
"""

from __future__ import annotations

import time
from typing import Any

from dlogos.schema import Transcript, TranscriptSegment

# AssemblyAI REST surface (see module docstring for field-level details).
DEFAULT_BASE_URL = "https://api.assemblyai.com"
_UPLOAD_PATH = "/v2/upload"
_TRANSCRIPT_PATH = "/v2/transcript"

# Terminal poll states.
_STATUS_COMPLETED = "completed"
_STATUS_ERROR = "error"


class AssemblyAITranscriptionError(RuntimeError):
    """Raised when AssemblyAI reports ``status == "error"`` or an HTTP failure."""


class AssemblyAIBackend:
    """Hosted AssemblyAI ASR backend (satisfies :class:`ASRBackend`).

    Parameters
    ----------
    api_key:
        AssemblyAI API key. When ``None`` (default) it is read lazily from
        ``settings.assemblyai_api_key`` (``ASSEMBLYAI_API_KEY``) on first use.
    base_url:
        API base URL. Defaults to ``https://api.assemblyai.com``; override for
        the EU endpoint (``https://api.eu.assemblyai.com``) or tests.
    language_code:
        Optional forced ISO language code (e.g. ``"en"``). When ``None`` the
        backend lets AssemblyAI auto-detect (``language_detection: true``).
    speakers_expected:
        Optional hint passed through as ``speakers_expected`` to nudge the
        diarizer toward a known number of speakers.
    poll_interval_s:
        Seconds to wait between status polls (default ``3.0``).
    poll_timeout_s:
        Give up polling after this many seconds (default ``1800`` = 30 min).
    transport:
        Optional ``httpx`` transport (e.g. an ``httpx.MockTransport`` in tests).
        Injected straight into the lazily-built ``httpx.Client`` so the full
        upload→submit→poll flow can be exercised without a network.

    Nothing in ``__init__`` imports ``httpx`` or reads the key — both are
    deferred to :meth:`transcribe`.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        language_code: str | None = None,
        speakers_expected: int | None = None,
        poll_interval_s: float = 3.0,
        poll_timeout_s: float = 1800.0,
        transport: Any | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.language_code = language_code
        self.speakers_expected = speakers_expected
        self.poll_interval_s = poll_interval_s
        self.poll_timeout_s = poll_timeout_s
        self._transport = transport

    # ------------------------------------------------------------------ #
    # Lazy key / client resolution
    # ------------------------------------------------------------------ #
    def _resolve_api_key(self) -> str:
        """Resolve the API key: explicit constructor arg wins, else settings."""

        if self.api_key:
            return self.api_key
        # Lazy config read so importing this module needs no settings either.
        from dlogos.config import settings

        key = (settings.assemblyai_api_key or "").strip()
        if not key:
            raise AssemblyAITranscriptionError(
                "No AssemblyAI API key: set ASSEMBLYAI_API_KEY or pass api_key=."
            )
        return key

    def _build_client(self) -> Any:
        """Build a configured ``httpx.Client`` (httpx imported lazily here)."""

        import httpx  # lazy: optional dep, imported only when transcribing

        kwargs: dict[str, Any] = {
            "base_url": self.base_url,
            "headers": {"authorization": self._resolve_api_key()},
            # Uploads stream the whole file; give the request room to finish.
            "timeout": httpx.Timeout(60.0, read=300.0),
        }
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return httpx.Client(**kwargs)

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def transcribe(self, audio_path: str) -> Transcript:
        """Transcribe + diarize ``audio_path`` into a :class:`Transcript`.

        Flow (all HTTP local to this method):

        1. If ``audio_path`` is an ``http(s)://`` URL, use it directly;
           otherwise upload the local file and use the returned ``upload_url``.
        2. Submit a ``/v2/transcript`` job with ``speaker_labels: true``.
        3. Poll ``/v2/transcript/{id}`` until ``completed`` (or ``error``).
        4. Map ``utterances`` → ordered :class:`TranscriptSegment` turns
           (milliseconds → seconds), stamping language + duration.
        """

        client = self._build_client()
        try:
            audio_url = self._resolve_audio_url(client, audio_path)
            transcript_id = self._submit(client, audio_url)
            result = self._poll(client, transcript_id)
        finally:
            client.close()

        episode_id = self._episode_id_from_path(audio_path)
        return self._result_to_transcript(result, episode_id)

    # ------------------------------------------------------------------ #
    # HTTP steps (httpx objects in, plain data out)
    # ------------------------------------------------------------------ #
    def _resolve_audio_url(self, client: Any, audio_path: str) -> str:
        """Return a remote audio URL, uploading the local file if needed."""

        if _is_remote_url(audio_path):
            return audio_path
        return self._upload(client, audio_path)

    def _upload(self, client: Any, audio_path: str) -> str:
        """Stream a local file to ``POST /v2/upload``; return its ``upload_url``."""

        with open(audio_path, "rb") as fh:
            resp = client.post(
                _UPLOAD_PATH,
                content=fh,
                headers={"content-type": "application/octet-stream"},
            )
        _raise_for_status(resp, "upload")
        upload_url = resp.json().get("upload_url")
        if not isinstance(upload_url, str) or not upload_url:
            raise AssemblyAITranscriptionError(
                "AssemblyAI upload response missing 'upload_url'."
            )
        return upload_url

    def _submit(self, client: Any, audio_url: str) -> str:
        """Submit a transcription job; return the transcript ``id``."""

        body = self._build_submit_body(audio_url)
        resp = client.post(_TRANSCRIPT_PATH, json=body)
        _raise_for_status(resp, "submit")
        data = resp.json()
        transcript_id = data.get("id")
        if not isinstance(transcript_id, str) or not transcript_id:
            raise AssemblyAITranscriptionError(
                "AssemblyAI submit response missing transcript 'id'."
            )
        return transcript_id

    def _build_submit_body(self, audio_url: str) -> dict[str, Any]:
        """Assemble the ``POST /v2/transcript`` request body (pure, testable)."""

        body: dict[str, Any] = {
            "audio_url": audio_url,
            "speaker_labels": True,
            "punctuate": True,
            "format_text": True,
        }
        if self.language_code:
            body["language_code"] = self.language_code
        else:
            body["language_detection"] = True
        if self.speakers_expected is not None:
            body["speakers_expected"] = self.speakers_expected
        return body

    def _poll(self, client: Any, transcript_id: str) -> dict[str, Any]:
        """Poll ``GET /v2/transcript/{id}`` until a terminal status.

        Returns the completed transcript JSON. Raises on ``status == "error"``
        or when ``poll_timeout_s`` elapses. ``time.sleep`` is used between polls;
        tests set ``poll_interval_s=0`` to avoid real waiting.
        """

        deadline = time.monotonic() + self.poll_timeout_s
        path = f"{_TRANSCRIPT_PATH}/{transcript_id}"
        while True:
            resp = client.get(path)
            _raise_for_status(resp, "poll")
            data = resp.json()
            status = data.get("status")
            if status == _STATUS_COMPLETED:
                return data
            if status == _STATUS_ERROR:
                raise AssemblyAITranscriptionError(
                    f"AssemblyAI transcription failed: "
                    f"{data.get('error', 'unknown error')}"
                )
            if time.monotonic() >= deadline:
                raise AssemblyAITranscriptionError(
                    f"AssemblyAI polling timed out after {self.poll_timeout_s}s "
                    f"(last status: {status!r})."
                )
            time.sleep(self.poll_interval_s)

    # ------------------------------------------------------------------ #
    # Pure mapping helpers (no httpx — unit-testable in isolation)
    # ------------------------------------------------------------------ #
    def _result_to_transcript(
        self, result: dict[str, Any], episode_id: str
    ) -> Transcript:
        """Map a completed AssemblyAI result into a schema :class:`Transcript`."""

        segments = self._utterances_to_segments(result.get("utterances"))
        language = str(result.get("language_code") or self.language_code or "en")
        duration_s = self._resolve_duration(result, segments)

        return Transcript(
            episode_id=episode_id,
            language=language,
            segments=segments,
            duration_s=duration_s,
        )

    @staticmethod
    def _utterances_to_segments(
        utterances: Any,
    ) -> list[TranscriptSegment]:
        """Map AssemblyAI ``utterances`` to ordered :class:`TranscriptSegment`s.

        Each utterance is already one diarized speaker turn. We convert the
        ``start``/``end`` from **milliseconds → seconds**, preserve the diarized
        ``speaker`` label verbatim (AssemblyAI uses ``"A"``, ``"B"``, ...), and
        sort by start time so segments are strictly time-ordered regardless of
        the order the API returned them in. Empty-text turns are skipped.
        """

        if not isinstance(utterances, list):
            return []

        segments: list[TranscriptSegment] = []
        for utt in utterances:
            if not isinstance(utt, dict):
                continue
            text = str(utt.get("text", "")).strip()
            if not text:
                continue
            speaker = str(utt.get("speaker", "A"))
            t_start = _ms_to_s(utt.get("start", 0))
            t_end = _ms_to_s(utt.get("end", 0))
            # Guard against boundary glitches where end < start.
            t_end = max(t_start, t_end)
            segments.append(
                TranscriptSegment(
                    speaker=speaker,
                    text=text,
                    t_start=t_start,
                    t_end=t_end,
                )
            )

        segments.sort(key=lambda s: (s.t_start, s.t_end))
        return segments

    @staticmethod
    def _resolve_duration(
        result: dict[str, Any], segments: list[TranscriptSegment]
    ) -> float:
        """Episode duration in seconds.

        Prefer AssemblyAI's ``audio_duration`` (already seconds); fall back to
        the last segment's ``t_end`` when the field is absent.
        """

        raw = result.get("audio_duration")
        if isinstance(raw, (int, float)) and not isinstance(raw, bool) and raw >= 0:
            return float(raw)
        return segments[-1].t_end if segments else 0.0

    @staticmethod
    def _episode_id_from_path(audio_path: str) -> str:
        """Derive a provisional episode id from the audio file name / URL.

        Mirrors :class:`WhisperXBackend` so both real backends stamp a
        consistent provisional id; the loader assigns the canonical id later.
        """

        from pathlib import Path
        from urllib.parse import urlparse

        if _is_remote_url(audio_path):
            path = urlparse(audio_path).path
            stem = Path(path).stem
        else:
            stem = Path(audio_path).stem
        return stem or "unknown-episode"


# --------------------------------------------------------------------------- #
# Module-level pure helpers
# --------------------------------------------------------------------------- #
def _is_remote_url(audio_path: str) -> bool:
    """True when ``audio_path`` is an ``http://`` or ``https://`` URL."""

    lowered = audio_path.strip().lower()
    return lowered.startswith("http://") or lowered.startswith("https://")


def _ms_to_s(value: Any) -> float:
    """Convert an AssemblyAI millisecond offset to seconds (clamped ≥ 0)."""

    try:
        secs = float(value) / 1000.0
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, secs)


def _raise_for_status(resp: Any, stage: str) -> None:
    """Raise :class:`AssemblyAITranscriptionError` on a non-2xx HTTP response."""

    status = getattr(resp, "status_code", 0)
    if 200 <= status < 300:
        return
    body = ""
    try:
        body = resp.text
    except Exception:  # pragma: no cover - defensive only
        pass
    raise AssemblyAITranscriptionError(
        f"AssemblyAI {stage} failed: HTTP {status}. {body[:500]}"
    )
