"""Tests for the claim extractor with a MOCK async client (no network)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from dlogos.extraction.chunking import chunk_transcript
from dlogos.extraction.extractor import (
    ClaimExtractor,
    ExtractionError,
    build_user_prompt,
)
from dlogos.schema import EntityType, Predicate, Stance, Transcript


# --------------------------------------------------------------------------- #
# Fake OpenAI-compatible async client
# --------------------------------------------------------------------------- #
class _FakeCompletions:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> dict[str, Any]:
        # Snapshot the messages list — the extractor mutates it in place
        # across the retry, so recording the live reference would alias.
        snapshot = dict(kwargs)
        if "messages" in snapshot:
            snapshot["messages"] = [dict(m) for m in snapshot["messages"]]
        self.calls.append(snapshot)
        if not self._responses:
            raise AssertionError("fake client ran out of canned responses")
        content = self._responses.pop(0)
        # Dict-shaped response (the extractor tolerates dict or object form).
        return {"choices": [{"message": {"content": content}}]}


class _FakeChat:
    def __init__(self, responses: list[str]) -> None:
        self.completions = _FakeCompletions(responses)


class FakeClient:
    """Returns canned JSON strings in order, recording the call kwargs."""

    def __init__(self, responses: list[str]) -> None:
        self.chat = _FakeChat(responses)

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.chat.completions.calls


def _one_chunk(transcript: Transcript):
    chunks = chunk_transcript(transcript)
    assert len(chunks) == 1
    return chunks[0]


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #
async def test_extracts_valid_claims(synthetic_transcript: Transcript) -> None:
    chunk = _one_chunk(synthetic_transcript)
    payload = {
        "claims": [
            {
                "speaker_label": "SPEAKER_01",
                "predicate": "rates_negative",
                "subject": "iPhone",
                "subject_type": "organization",
                "object": "hardware innovation has plateaued",
                "stance": "asserts",
                "sentiment": -0.6,
                "confidence": 0.8,
                "t_start": 4.5,
                "t_end": 10.0,
            }
        ]
    }
    client = FakeClient([json.dumps(payload)])
    extractor = ClaimExtractor(client)
    claims = await extractor.extract(chunk)

    assert len(claims) == 1
    claim = claims[0]
    assert claim.speaker.label == "SPEAKER_01"
    assert claim.predicate == Predicate.rates_negative
    assert claim.subject_entity.name == "iPhone"
    assert claim.subject_entity.type == EntityType.organization
    assert claim.stance == Stance.asserts
    assert claim.sentiment == pytest.approx(-0.6)
    assert claim.confidence == pytest.approx(0.8)
    assert claim.source_span.episode_id == chunk.episode_id
    assert chunk.t_start <= claim.source_span.t_start <= claim.source_span.t_end <= chunk.t_end
    # Only one model call on the happy path.
    assert len(client.calls) == 1


async def test_request_uses_json_object_and_settings_model(
    synthetic_transcript: Transcript,
) -> None:
    chunk = _one_chunk(synthetic_transcript)
    client = FakeClient([json.dumps({"claims": []})])
    extractor = ClaimExtractor(client)
    await extractor.extract(chunk)

    kwargs = client.calls[0]
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["model"] == extractor._settings.extraction_model
    # System + user message present.
    roles = [m["role"] for m in kwargs["messages"]]
    assert roles[0] == "system"
    assert "user" in roles


def test_prompt_carries_labels_vocab_and_bounds(
    synthetic_transcript: Transcript,
) -> None:
    chunk = _one_chunk(synthetic_transcript)
    vocab = [p.value for p in Predicate]
    prompt = build_user_prompt(chunk, vocab)
    # Speaker labels present.
    assert "SPEAKER_00" in prompt
    assert "SPEAKER_01" in prompt
    # Controlled vocabulary listed.
    for p in vocab:
        assert p in prompt
    # Chunk time bounds present.
    assert f"[{chunk.t_start:.2f}, {chunk.t_end:.2f}]" in prompt


# --------------------------------------------------------------------------- #
# Predicate mapping inside extraction
# --------------------------------------------------------------------------- #
async def test_free_predicate_mapped_to_vocabulary(
    synthetic_transcript: Transcript,
) -> None:
    chunk = _one_chunk(synthetic_transcript)
    payload = {
        "claims": [
            {
                "speaker_label": "SPEAKER_01",
                "predicate": "bearish on",  # synonym -> rates_negative
                "subject": "Apple",
                "subject_type": "organization",
                "object": "weak hardware",
                "stance": "asserts",
                "sentiment": -0.5,
                "confidence": 0.7,
                "t_start": 4.5,
                "t_end": 10.0,
            }
        ]
    }
    extractor = ClaimExtractor(FakeClient([json.dumps(payload)]))
    claims = await extractor.extract(chunk)
    assert len(claims) == 1
    assert claims[0].predicate == Predicate.rates_negative


async def test_unmappable_predicate_drops_claim(
    synthetic_transcript: Transcript,
) -> None:
    chunk = _one_chunk(synthetic_transcript)
    payload = {
        "claims": [
            {
                "speaker_label": "SPEAKER_01",
                "predicate": "vibes_with",  # unmappable
                "subject": "Apple",
                "subject_type": "organization",
                "object": "x",
                "stance": "asserts",
                "sentiment": 0.0,
                "confidence": 0.5,
                "t_start": 4.5,
                "t_end": 10.0,
            }
        ]
    }
    extractor = ClaimExtractor(FakeClient([json.dumps(payload)]))
    claims = await extractor.extract(chunk)
    assert claims == []


# --------------------------------------------------------------------------- #
# Validation: span bounds, speaker, stance
# --------------------------------------------------------------------------- #
async def test_span_outside_chunk_window_drops_claim(
    synthetic_transcript: Transcript,
) -> None:
    chunk = _one_chunk(synthetic_transcript)
    payload = {
        "claims": [
            {
                "speaker_label": "SPEAKER_01",
                "predicate": "rates_negative",
                "subject": "Apple",
                "subject_type": "organization",
                "object": "x",
                "stance": "asserts",
                "sentiment": -0.5,
                "confidence": 0.7,
                "t_start": 5000.0,  # well past chunk.t_end
                "t_end": 5010.0,
            }
        ]
    }
    extractor = ClaimExtractor(FakeClient([json.dumps(payload)]))
    assert await extractor.extract(chunk) == []


async def test_unknown_speaker_label_drops_claim(
    synthetic_transcript: Transcript,
) -> None:
    chunk = _one_chunk(synthetic_transcript)
    payload = {
        "claims": [
            {
                "speaker_label": "SPEAKER_99",  # not in chunk
                "predicate": "rates_negative",
                "subject": "Apple",
                "subject_type": "organization",
                "object": "x",
                "stance": "asserts",
                "sentiment": -0.5,
                "confidence": 0.7,
                "t_start": 4.5,
                "t_end": 10.0,
            }
        ]
    }
    extractor = ClaimExtractor(FakeClient([json.dumps(payload)]))
    assert await extractor.extract(chunk) == []


async def test_invalid_stance_drops_claim(synthetic_transcript: Transcript) -> None:
    chunk = _one_chunk(synthetic_transcript)
    payload = {
        "claims": [
            {
                "speaker_label": "SPEAKER_01",
                "predicate": "rates_negative",
                "subject": "Apple",
                "subject_type": "organization",
                "object": "x",
                "stance": "shrugs",  # not a Stance
                "sentiment": -0.5,
                "confidence": 0.7,
                "t_start": 4.5,
                "t_end": 10.0,
            }
        ]
    }
    extractor = ClaimExtractor(FakeClient([json.dumps(payload)]))
    assert await extractor.extract(chunk) == []


async def test_out_of_range_sentiment_confidence_clamped(
    synthetic_transcript: Transcript,
) -> None:
    chunk = _one_chunk(synthetic_transcript)
    payload = {
        "claims": [
            {
                "speaker_label": "SPEAKER_01",
                "predicate": "rates_negative",
                "subject": "Apple",
                "subject_type": "organization",
                "object": "x",
                "stance": "asserts",
                "sentiment": -5.0,  # clamps to -1.0
                "confidence": 9.0,  # clamps to 1.0
                "t_start": 4.5,
                "t_end": 10.0,
            }
        ]
    }
    extractor = ClaimExtractor(FakeClient([json.dumps(payload)]))
    claims = await extractor.extract(chunk)
    assert len(claims) == 1
    assert claims[0].sentiment == pytest.approx(-1.0)
    assert claims[0].confidence == pytest.approx(1.0)


async def test_unknown_subject_type_defaults_to_concept(
    synthetic_transcript: Transcript,
) -> None:
    chunk = _one_chunk(synthetic_transcript)
    payload = {
        "claims": [
            {
                "speaker_label": "SPEAKER_01",
                "predicate": "explains",
                "subject": "the macro picture",
                "subject_type": "weather",  # not an EntityType
                "object": "rates will fall",
                "stance": "asserts",
                "sentiment": 0.1,
                "confidence": 0.6,
                "t_start": 4.5,
                "t_end": 10.0,
            }
        ]
    }
    extractor = ClaimExtractor(FakeClient([json.dumps(payload)]))
    claims = await extractor.extract(chunk)
    assert len(claims) == 1
    assert claims[0].subject_entity.type == EntityType.concept


# --------------------------------------------------------------------------- #
# Retry-once on invalid JSON
# --------------------------------------------------------------------------- #
async def test_retries_once_on_invalid_json(
    synthetic_transcript: Transcript,
) -> None:
    chunk = _one_chunk(synthetic_transcript)
    good = {
        "claims": [
            {
                "speaker_label": "SPEAKER_00",
                "predicate": "questions",
                "subject": "OpenAI",
                "subject_type": "organization",
                "object": "pace this year",
                "stance": "hedges",
                "sentiment": 0.0,
                "confidence": 0.5,
                "t_start": 10.0,
                "t_end": 14.0,
            }
        ]
    }
    client = FakeClient(["this is not json {{{", json.dumps(good)])
    extractor = ClaimExtractor(client)
    claims = await extractor.extract(chunk)
    assert len(claims) == 1
    assert claims[0].subject_entity.name == "OpenAI"
    # Exactly two calls: the bad one and the retry.
    assert len(client.calls) == 2
    # The retry carries a corrective nudge as an extra user message.
    assert len(client.calls[1]["messages"]) > len(client.calls[0]["messages"])


async def test_raises_after_two_invalid_json(
    synthetic_transcript: Transcript,
) -> None:
    chunk = _one_chunk(synthetic_transcript)
    client = FakeClient(["nope", "still nope"])
    extractor = ClaimExtractor(client)
    with pytest.raises(ExtractionError):
        await extractor.extract(chunk)
    assert len(client.calls) == 2


async def test_claims_not_a_list_raises(synthetic_transcript: Transcript) -> None:
    chunk = _one_chunk(synthetic_transcript)
    client = FakeClient([json.dumps({"claims": {"oops": 1}})])
    extractor = ClaimExtractor(client)
    with pytest.raises(ExtractionError):
        await extractor.extract(chunk)


async def test_empty_claims_yields_empty_list(
    synthetic_transcript: Transcript,
) -> None:
    chunk = _one_chunk(synthetic_transcript)
    client = FakeClient([json.dumps({"claims": []})])
    extractor = ClaimExtractor(client)
    assert await extractor.extract(chunk) == []


async def test_object_style_response_supported(
    synthetic_transcript: Transcript,
) -> None:
    """The extractor also reads object-shaped responses (real SDK form)."""

    class _Msg:
        content = json.dumps(
            {
                "claims": [
                    {
                        "speaker_label": "SPEAKER_01",
                        "predicate": "expects",
                        "subject": "Apple",
                        "subject_type": "organization",
                        "object": "a rebound next cycle",
                        "stance": "predicts",
                        "sentiment": 0.3,
                        "confidence": 0.65,
                        "t_start": 23.0,
                        "t_end": 28.0,
                    }
                ]
            }
        )

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    class _ObjCompletions:
        async def create(self, **kwargs: Any) -> Any:
            return _Resp()

    class _ObjChat:
        completions = _ObjCompletions()

    class _ObjClient:
        chat = _ObjChat()

    extractor = ClaimExtractor(_ObjClient())
    chunk = _one_chunk(synthetic_transcript)
    claims = await extractor.extract(chunk)
    assert len(claims) == 1
    assert claims[0].predicate == Predicate.expects


async def test_extract_many_aggregates(synthetic_transcript: Transcript) -> None:
    # Force two chunks, one canned response per chunk.
    chunks = chunk_transcript(synthetic_transcript, max_chars=80, overlap_segments=0)
    assert len(chunks) >= 2
    responses = []
    for ch in chunks:
        label = ch.segments[0].speaker
        responses.append(
            json.dumps(
                {
                    "claims": [
                        {
                            "speaker_label": label,
                            "predicate": "explains",
                            "subject": "Apple",
                            "subject_type": "organization",
                            "object": "context",
                            "stance": "asserts",
                            "sentiment": 0.0,
                            "confidence": 0.5,
                            "t_start": ch.t_start,
                            "t_end": ch.t_end,
                        }
                    ]
                }
            )
        )
    extractor = ClaimExtractor(FakeClient(responses))
    claims = await extractor.extract_many(chunks)
    assert len(claims) == len(chunks)
