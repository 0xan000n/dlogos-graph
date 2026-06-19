"""Open-weight claim extractor (§7.4) — first-class, swappable, async.

Turns a :class:`~dlogos.extraction.chunking.Chunk` into a list of validated
:class:`~dlogos.schema.ExtractedClaim` records using an OpenAI-compatible
endpoint (an open-weight model served via vLLM/SGLang behind the same wire
protocol as the OpenAI API). The client's ``base_url`` and model id come from
:class:`~dlogos.config.Settings`.

Design points the spec pins down:

- **Carry speaker labels + per-segment timestamps into the prompt** so claims
  attribute to the right person and can anchor a real ``source_span`` (§7.4).
- **List the controlled predicate vocabulary in the prompt** and constrain the
  response to it; map any drift back via :mod:`dlogos.extraction.predicates`,
  dropping unmappable predicates rather than coercing them.
- **``response_format={"type": "json_object"}``** so the endpoint emits a JSON
  document we parse and validate into the closed schema.
- **Enforce ``source_span`` within the chunk window** — a claim whose span
  falls outside the chunk's ``[t_start, t_end]`` is a fabricated citation and
  is dropped (the eval's speaker-verified citation check depends on spans being
  real).
- **Retry once on invalid JSON.** Open-weight structured output occasionally
  wobbles; one reparse attempt with a corrective nudge is cheap insurance. A
  second failure raises.

The OpenAI client is imported **lazily** inside the factory so importing this
module (e.g. for tests) never constructs a network client, and tests inject a
fake async client that returns canned JSON — no network, fully deterministic.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from pydantic import ValidationError

from dlogos.config import Settings, settings as default_settings
from dlogos.extraction.chunking import Chunk
from dlogos.extraction.predicates import PredicateMappingError, map_predicate
from dlogos.schema import (
    Entity,
    EntityType,
    ExtractedClaim,
    SourceSpan,
    SpeakerRef,
    Stance,
)

# Small tolerance (seconds) when checking a span against the chunk window, so
# floating-point timestamp rounding does not reject an otherwise-valid span.
_SPAN_EPS = 0.05


class ExtractionError(RuntimeError):
    """Raised when the model output cannot be parsed even after one retry."""


class PredicateVocabularyError(ValueError):
    """Raised when a claim's predicate cannot be mapped to the vocabulary.

    Wraps :class:`~dlogos.extraction.predicates.PredicateMappingError` at the
    claim level; the extractor catches it and drops the offending claim.
    """


class _ChatCompletions(Protocol):
    async def create(self, **kwargs: Any) -> Any: ...


class _Chat(Protocol):
    completions: _ChatCompletions


class AsyncChatClient(Protocol):
    """Structural type for the slice of ``AsyncOpenAI`` we use.

    Tests supply a fake satisfying this protocol; production supplies a real
    ``AsyncOpenAI``. Keeping the dependency structural means no heavy import is
    needed to type or test the extractor.
    """

    chat: _Chat


_SYSTEM_PROMPT = (
    "You extract reified, stance-tagged claims from a diarized podcast "
    "transcript chunk. Attribute every claim to the speaker label that uttered "
    "it. Use ONLY the controlled predicate vocabulary provided. For each claim "
    "set a source_span whose [t_start, t_end] lies inside the chunk's time "
    "window and matches the segment(s) the claim came from. Respond with a "
    "single JSON object and nothing else."
)


def _render_schema_hint(predicates: list[str]) -> str:
    return (
        "Return JSON of the form:\n"
        '{"claims": [\n'
        "  {\n"
        '    "speaker_label": "<one of the SPEAKER_xx labels in the chunk>",\n'
        '    "predicate": "<one of the controlled predicates>",\n'
        '    "subject": "<the entity the claim is about>",\n'
        '    "subject_type": "person|organization|concept|work",\n'
        '    "object": "<free text / value the claim asserts>",\n'
        '    "stance": "asserts|disputes|hedges|predicts|retracts",\n'
        '    "sentiment": <number in [-1, 1]>,\n'
        '    "confidence": <number in [0, 1]>,\n'
        '    "t_start": <seconds, within the chunk window>,\n'
        '    "t_end": <seconds, within the chunk window>\n'
        "  }\n"
        "]}\n"
        "Controlled predicate vocabulary (use EXACTLY these): "
        + ", ".join(predicates)
        + ".\n"
        "Emit no claim you cannot ground in the chunk. If there are no claims, "
        'return {"claims": []}.'
    )


def build_user_prompt(chunk: Chunk, predicates: list[str]) -> str:
    """Assemble the per-chunk user prompt.

    Carries the chunk's time bounds, its speaker-labelled/timestamped body, and
    the controlled-vocabulary predicate list.
    """

    speaker_labels = sorted({s.speaker for s in chunk.segments})
    return (
        f"Episode: {chunk.episode_id}\n"
        f"Chunk window: [{chunk.t_start:.2f}, {chunk.t_end:.2f}] seconds.\n"
        f"Speaker labels present: {', '.join(speaker_labels)}.\n\n"
        f"Transcript chunk (one line per segment, "
        f"[start-end] SPEAKER: text):\n"
        f"{chunk.render()}\n\n"
        f"{_render_schema_hint(predicates)}"
    )


class ClaimExtractor:
    """Extract claims from chunks via an OpenAI-compatible async client.

    Parameters
    ----------
    client:
        An object satisfying :class:`AsyncChatClient` (a real ``AsyncOpenAI`` in
        production, a fake in tests). If ``None``, a real client is built lazily
        from ``settings`` on first use via :meth:`from_settings` semantics.
    settings:
        Configuration providing ``extraction_model`` (and, when building a real
        client, ``extraction_base_url`` / ``extraction_api_key``). Defaults to
        the shared singleton.
    temperature:
        Decoding temperature; ``0.0`` for deterministic-as-possible extraction.
    """

    def __init__(
        self,
        client: AsyncChatClient,
        *,
        settings: Settings | None = None,
        temperature: float = 0.0,
    ) -> None:
        self._client = client
        self._settings = settings or default_settings
        self._temperature = temperature

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "ClaimExtractor":
        """Build an extractor backed by a real ``AsyncOpenAI`` client.

        The ``openai`` import is performed here, lazily, so importing this
        module never requires constructing a network client.
        """

        cfg = settings or default_settings
        from openai import AsyncOpenAI  # lazy: avoid import at module load

        client = AsyncOpenAI(
            base_url=cfg.extraction_base_url,
            api_key=cfg.extraction_api_key,
        )
        return cls(client, settings=cfg)

    @property
    def predicate_vocabulary(self) -> list[str]:
        """The controlled predicate enum values, in declaration order."""

        from dlogos.schema import Predicate

        return [p.value for p in Predicate]

    async def extract(self, chunk: Chunk) -> list[ExtractedClaim]:
        """Extract validated claims from a single chunk.

        Calls the model, parses the JSON document (retrying once on invalid
        JSON with a corrective nudge), and validates each candidate claim:
        predicate mapped to the vocabulary, span bounded to the chunk window,
        speaker label echoed from the chunk. Invalid individual claims are
        dropped; total parse failure (after retry) raises
        :class:`ExtractionError`.
        """

        user_prompt = build_user_prompt(chunk, self.predicate_vocabulary)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        raw = await self._call(messages)
        payload = self._parse_json(raw)
        if payload is None:
            # Retry once with a corrective nudge appended.
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous reply was not valid JSON. Reply with a "
                        'single valid JSON object of the form {"claims": [...]} '
                        "and nothing else."
                    ),
                }
            )
            raw = await self._call(messages)
            payload = self._parse_json(raw)
            if payload is None:
                raise ExtractionError(
                    f"model returned invalid JSON twice for chunk "
                    f"{chunk.episode_id}#{chunk.chunk_index}"
                )

        candidates = payload.get("claims", [])
        if not isinstance(candidates, list):
            raise ExtractionError(
                f"'claims' was not a list for chunk "
                f"{chunk.episode_id}#{chunk.chunk_index}"
            )

        valid_labels = {s.speaker for s in chunk.segments}
        out: list[ExtractedClaim] = []
        for item in candidates:
            claim = self._build_claim(item, chunk, valid_labels)
            if claim is not None:
                out.append(claim)
        return out

    async def extract_many(self, chunks: list[Chunk]) -> list[ExtractedClaim]:
        """Extract claims from several chunks sequentially.

        Sequential (not concurrent) by default so a fake client's call order is
        deterministic in tests; concurrency is an orchestration concern handled
        in the pipeline layer.
        """

        claims: list[ExtractedClaim] = []
        for ch in chunks:
            claims.extend(await self.extract(ch))
        return claims

    # ----------------------------------------------------------------- #
    # Internals
    # ----------------------------------------------------------------- #
    async def _call(self, messages: list[dict[str, str]]) -> str:
        resp = await self._client.chat.completions.create(
            model=self._settings.extraction_model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=self._temperature,
        )
        return _content_of(resp)

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any] | None:
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed

    def _build_claim(
        self,
        item: Any,
        chunk: Chunk,
        valid_labels: set[str],
    ) -> ExtractedClaim | None:
        """Validate one candidate dict into an ExtractedClaim, or drop it."""

        if not isinstance(item, dict):
            return None

        # Speaker must be one of the chunk's labels (no inventing attributions).
        label = item.get("speaker_label") or item.get("speaker")
        if not isinstance(label, str) or label not in valid_labels:
            return None

        # Predicate: map onto the controlled vocabulary or drop.
        raw_pred = item.get("predicate")
        if not isinstance(raw_pred, str):
            return None
        try:
            predicate = map_predicate(raw_pred)
        except PredicateMappingError:
            return None

        # Span: must lie within the chunk window.
        t_start = item.get("t_start")
        t_end = item.get("t_end")
        if not _is_number(t_start) or not _is_number(t_end):
            return None
        t_start = float(t_start)
        t_end = float(t_end)
        if t_end < t_start:
            return None
        if (
            t_start < chunk.t_start - _SPAN_EPS
            or t_end > chunk.t_end + _SPAN_EPS
        ):
            return None
        # Clamp tiny epsilon overflow back into the window.
        t_start = max(t_start, chunk.t_start)
        t_end = min(t_end, chunk.t_end)

        subject = item.get("subject")
        if not isinstance(subject, str) or not subject.strip():
            return None
        subject_type = _coerce_entity_type(item.get("subject_type"))
        stance = _coerce_stance(item.get("stance"))
        if stance is None:
            return None

        obj = item.get("object")
        if obj is None:
            return None
        obj = str(obj)

        sentiment = item.get("sentiment", 0.0)
        confidence = item.get("confidence", 0.5)
        if not _is_number(sentiment) or not _is_number(confidence):
            return None
        sentiment = _clamp(float(sentiment), -1.0, 1.0)
        confidence = _clamp(float(confidence), 0.0, 1.0)

        try:
            return ExtractedClaim(
                speaker=SpeakerRef(label=label),
                predicate=predicate,
                subject_entity=Entity(name=subject.strip(), type=subject_type),
                object=obj,
                stance=stance,
                sentiment=sentiment,
                confidence=confidence,
                source_span=SourceSpan(
                    episode_id=chunk.episode_id,
                    t_start=t_start,
                    t_end=t_end,
                ),
            )
        except ValidationError:
            return None


def _content_of(resp: Any) -> str:
    """Pull the assistant message content out of a chat completion response.

    Tolerates both object-style (``resp.choices[0].message.content``) and
    dict-style responses so fakes can use whichever is convenient.
    """

    # Dict-shaped (common in lightweight fakes).
    if isinstance(resp, dict):
        choices = resp.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        return msg.get("content") or ""

    # Object-shaped (real openai SDK).
    choices = getattr(resp, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    if message is None:
        return ""
    return getattr(message, "content", "") or ""


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _coerce_entity_type(value: Any) -> EntityType:
    if isinstance(value, str):
        norm = value.strip().lower()
        try:
            return EntityType(norm)
        except ValueError:
            pass
    # Default: an organization/person/work that didn't resolve becomes a
    # concept rather than dropping the whole claim over a bad type label.
    return EntityType.concept


def _coerce_stance(value: Any) -> Stance | None:
    if isinstance(value, str):
        norm = value.strip().lower()
        try:
            return Stance(norm)
        except ValueError:
            return None
    return None
