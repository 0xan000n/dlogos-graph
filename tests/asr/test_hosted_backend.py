"""Tests for the hosted AssemblyAI ASR backend (``AssemblyAIBackend``).

These drive the *full* upload→submit→poll→map flow against an
``httpx.MockTransport`` that returns canned AssemblyAI JSON — an upload
response, a submit response, then polling that goes ``processing`` →
``completed`` with diarized ``utterances``. No network, fully deterministic.

What we assert:

- the request sequence is correct (upload, submit-with-``speaker_labels``, poll
  the returned id) and the authorization header is attached;
- the canned ``utterances`` map into a :class:`~dlogos.schema.Transcript` with
  speaker labels preserved verbatim, **milliseconds → seconds** conversion, and
  segments returned in time order;
- a remote ``http(s)://`` audio URL skips the upload step entirely;
- ``status == "error"`` raises;
- importing/constructing the backend pulls in no heavy dep.

``httpx`` is an optional dep; these tests are skipped if it is not installed so
the core-only suite stays green either way.
"""

from __future__ import annotations

import json
import sys

import pytest

from dlogos.asr.base import ASRBackend
from dlogos.asr.hosted_backend import (
    AssemblyAIBackend,
    AssemblyAITranscriptionError,
    _is_remote_url,
    _ms_to_s,
)
from dlogos.schema import Transcript

httpx = pytest.importorskip("httpx")


# --------------------------------------------------------------------------- #
# Canned AssemblyAI payloads (shapes mirror the live API; see backend docstring)
# --------------------------------------------------------------------------- #
_UPLOAD_URL = "https://cdn.assemblyai.com/upload/canned-file-id"
_TRANSCRIPT_ID = "transcript_canned_123"

# Two diarized turns, intentionally returned OUT of time order to prove the
# backend sorts by start time. start/end are in MILLISECONDS.
_UTTERANCES = [
    {
        "speaker": "B",
        "text": "I think the iPhone has plateaued on hardware.",
        "start": 4500,
        "end": 10000,
        "confidence": 0.94,
    },
    {
        "speaker": "A",
        "text": "Welcome back. My guest watches Apple closely.",
        "start": 0,
        "end": 4500,
        "confidence": 0.97,
    },
    # Empty-text turn must be skipped.
    {"speaker": "A", "text": "   ", "start": 10000, "end": 10100},
]

_COMPLETED = {
    "id": _TRANSCRIPT_ID,
    "status": "completed",
    "language_code": "en",
    "audio_duration": 10.0,  # seconds (already)
    "utterances": _UTTERANCES,
}


class _Recorder:
    """Captures each request so the flow/headers can be asserted."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str]] = []  # (method, path)
        self.auth_headers: list[str | None] = []
        self.submit_bodies: list[dict] = []
        self._poll_calls = 0

    def handler(self, poll_processing_first: bool = True):
        """Build an ``httpx.MockTransport`` handler closing over this recorder."""

        def _handle(request: "httpx.Request") -> "httpx.Response":
            path = request.url.path
            self.requests.append((request.method, path))
            self.auth_headers.append(request.headers.get("authorization"))

            if request.method == "POST" and path == "/v2/upload":
                return httpx.Response(200, json={"upload_url": _UPLOAD_URL})

            if request.method == "POST" and path == "/v2/transcript":
                self.submit_bodies.append(json.loads(request.content.decode()))
                return httpx.Response(
                    200, json={"id": _TRANSCRIPT_ID, "status": "queued"}
                )

            if request.method == "GET" and path == f"/v2/transcript/{_TRANSCRIPT_ID}":
                self._poll_calls += 1
                if poll_processing_first and self._poll_calls == 1:
                    return httpx.Response(
                        200, json={"id": _TRANSCRIPT_ID, "status": "processing"}
                    )
                return httpx.Response(200, json=_COMPLETED)

            return httpx.Response(404, json={"error": f"unexpected {path}"})

        return _handle


def _backend(handler, **kwargs) -> AssemblyAIBackend:
    transport = httpx.MockTransport(handler)
    return AssemblyAIBackend(
        api_key="test-key",
        transport=transport,
        poll_interval_s=0.0,  # do not actually sleep between polls
        **kwargs,
    )


@pytest.fixture
def local_audio(tmp_path) -> str:
    """A real (tiny) on-disk file so the upload step streams real bytes.

    Named ``ep-0001.mp3`` so the derived episode_id is deterministic; contents
    are irrelevant because the MockTransport never inspects the uploaded bytes.
    """

    p = tmp_path / "ep-0001.mp3"
    p.write_bytes(b"\x00\x01fake-audio-bytes")
    return str(p)


# --------------------------------------------------------------------------- #
# Full local-file flow: upload → submit → poll(processing→completed) → map
# --------------------------------------------------------------------------- #
def test_full_local_file_flow_maps_transcript(local_audio: str) -> None:
    rec = _Recorder()
    backend = _backend(rec.handler(poll_processing_first=True))

    transcript = backend.transcribe(local_audio)

    assert isinstance(transcript, Transcript)
    assert transcript.episode_id == "ep-0001"
    assert transcript.language == "en"
    assert transcript.duration_s == 10.0

    # Empty-text utterance dropped; two real turns remain.
    assert len(transcript.segments) == 2

    # Sorted by start time even though the API returned them out of order.
    first, second = transcript.segments
    assert first.t_start == 0.0 and first.t_end == 4.5  # ms → s
    assert second.t_start == 4.5 and second.t_end == 10.0

    # Speaker labels preserved verbatim (AssemblyAI "A"/"B").
    assert first.speaker == "A"
    assert second.speaker == "B"
    assert "Welcome back" in first.text
    assert "plateaued" in second.text


def test_request_sequence_and_auth_header(local_audio: str) -> None:
    rec = _Recorder()
    backend = _backend(rec.handler(poll_processing_first=True))
    backend.transcribe(local_audio)

    methods_paths = rec.requests
    # 1 upload, 1 submit, 2 polls (processing then completed).
    assert methods_paths[0] == ("POST", "/v2/upload")
    assert methods_paths[1] == ("POST", "/v2/transcript")
    assert methods_paths[2] == ("GET", f"/v2/transcript/{_TRANSCRIPT_ID}")
    assert methods_paths[3] == ("GET", f"/v2/transcript/{_TRANSCRIPT_ID}")
    assert len(methods_paths) == 4

    # Authorization header carried on every request (raw key, no "Bearer").
    assert all(h == "test-key" for h in rec.auth_headers)


def test_submit_body_enables_speaker_labels(local_audio: str) -> None:
    rec = _Recorder()
    backend = _backend(rec.handler(poll_processing_first=False))
    backend.transcribe(local_audio)

    assert len(rec.submit_bodies) == 1
    body = rec.submit_bodies[0]
    assert body["audio_url"] == _UPLOAD_URL  # uploaded file URL fed back in
    assert body["speaker_labels"] is True
    # No forced language → auto-detect requested.
    assert body.get("language_detection") is True
    assert "language_code" not in body


def test_forced_language_and_speakers_expected_in_body(local_audio: str) -> None:
    rec = _Recorder()
    backend = _backend(
        rec.handler(poll_processing_first=False),
        language_code="en",
        speakers_expected=2,
    )
    backend.transcribe(local_audio)

    body = rec.submit_bodies[0]
    assert body["language_code"] == "en"
    assert "language_detection" not in body
    assert body["speakers_expected"] == 2


# --------------------------------------------------------------------------- #
# Remote URL: upload step is skipped
# --------------------------------------------------------------------------- #
def test_remote_url_skips_upload() -> None:
    rec = _Recorder()
    backend = _backend(rec.handler(poll_processing_first=False))

    remote = "https://example.com/podcasts/ep-0042.mp3"
    transcript = backend.transcribe(remote)

    # No upload call; submit uses the remote URL directly.
    assert ("POST", "/v2/upload") not in rec.requests
    assert rec.submit_bodies[0]["audio_url"] == remote
    # episode_id derived from the URL path stem.
    assert transcript.episode_id == "ep-0042"


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #
def test_status_error_raises() -> None:
    def handler(request: "httpx.Request") -> "httpx.Response":
        path = request.url.path
        if request.method == "POST" and path == "/v2/transcript":
            return httpx.Response(200, json={"id": _TRANSCRIPT_ID, "status": "queued"})
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "id": _TRANSCRIPT_ID,
                    "status": "error",
                    "error": "decoding failed",
                },
            )
        return httpx.Response(404)

    backend = _backend(handler)
    with pytest.raises(AssemblyAITranscriptionError, match="decoding failed"):
        backend.transcribe("https://example.com/bad.mp3")


def test_http_error_on_submit_raises() -> None:
    def handler(request: "httpx.Request") -> "httpx.Response":
        if request.url.path == "/v2/transcript":
            return httpx.Response(401, text="invalid api key")
        return httpx.Response(404)

    backend = _backend(handler)
    with pytest.raises(AssemblyAITranscriptionError, match="HTTP 401"):
        backend.transcribe("https://example.com/x.mp3")


def test_missing_api_key_raises(monkeypatch) -> None:
    # No explicit key and settings resolves to empty → clear error.
    from dlogos.config import settings as cfg

    monkeypatch.setattr(cfg, "assemblyai_api_key", "", raising=False)
    backend = AssemblyAIBackend(
        transport=httpx.MockTransport(lambda r: httpx.Response(200))
    )
    with pytest.raises(AssemblyAITranscriptionError, match="No AssemblyAI API key"):
        backend.transcribe("https://example.com/x.mp3")


def test_poll_timeout_raises() -> None:
    def handler(request: "httpx.Request") -> "httpx.Response":
        if request.url.path == "/v2/transcript":
            return httpx.Response(200, json={"id": _TRANSCRIPT_ID, "status": "queued"})
        # Never completes.
        return httpx.Response(200, json={"id": _TRANSCRIPT_ID, "status": "processing"})

    backend = _backend(handler, poll_timeout_s=0.0)
    with pytest.raises(AssemblyAITranscriptionError, match="timed out"):
        backend.transcribe("https://example.com/x.mp3")


# --------------------------------------------------------------------------- #
# Pure helpers + Protocol conformance + no heavy deps
# --------------------------------------------------------------------------- #
def test_ms_to_s_conversion_and_clamp() -> None:
    assert _ms_to_s(4500) == 4.5
    assert _ms_to_s(0) == 0.0
    assert _ms_to_s(-100) == 0.0  # clamped
    assert _ms_to_s("not-a-number") == 0.0


def test_is_remote_url() -> None:
    assert _is_remote_url("https://example.com/a.mp3")
    assert _is_remote_url("HTTP://example.com/a.mp3")
    assert not _is_remote_url("/local/path/a.mp3")
    assert not _is_remote_url("a.mp3")


def test_utterances_to_segments_handles_non_list() -> None:
    # Defensive: a missing/garbled utterances field yields no segments.
    assert AssemblyAIBackend._utterances_to_segments(None) == []
    assert AssemblyAIBackend._utterances_to_segments("oops") == []


def test_duration_falls_back_to_last_segment_end() -> None:
    backend = AssemblyAIBackend(api_key="k")
    result = {"utterances": _UTTERANCES}  # no audio_duration
    transcript = backend._result_to_transcript(result, "ep-x")
    # Last segment end is 10000 ms → 10.0 s.
    assert transcript.duration_s == 10.0


def test_backend_is_asr_backend_protocol_instance() -> None:
    assert isinstance(AssemblyAIBackend(api_key="k"), ASRBackend)


def test_no_heavy_deps_imported_by_hosted_path() -> None:
    # Importing + using the hosted backend must never pull in torch/whisperx/
    # pyannote (it talks plain HTTPS, no GPU stack).
    import dlogos.asr.hosted_backend  # noqa: F401

    rec = _Recorder()
    _backend(rec.handler(poll_processing_first=False)).transcribe(
        "https://example.com/ep.mp3"
    )
    for heavy in ("torch", "whisperx", "pyannote", "pyannote.audio"):
        assert heavy not in sys.modules, f"{heavy} must not be imported by the hosted path"
