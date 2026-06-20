"""Rules -> embedding -> LLM-adjudication cascade for entity matching.

When a freshly extracted subject entity arrives, we must decide: is this the
*same* real-world thing as one of the canonical entities we already hold, or a
genuinely new one? :func:`match_entity` makes that decision against a store's
candidate set, in three escalating tiers — cheapest and most certain first:

1. **Rules (Tier 1).** A Wikidata ``qid`` match, or an exact normalized-name
   match, is decisive and free — no embedding compare needed. The QID anchor
   wins even when the surface form ("the iPhone maker") embeds far from the
   stored name ("Apple").
2. **Embedding (Tier 2).** Otherwise compare against the store's top candidate
   by cosine. Per-:class:`~dlogos.schema.EntityType` thresholds gate the call:
   ``>= high`` is a confident match; ``< low`` is a confident NEW. Concepts use
   a stricter band because they paraphrase more freely than named orgs/people.
3. **LLM adjudication (Tier 3).** Only the *ambiguous middle* (between ``low``
   and ``high``) escalates to an injected yes/no adjudicator — the one place a
   model call earns its cost. With **no** adjudicator, the ambiguous case is
   resolved conservatively to NEW (prefer leaving distinct over a wrong merge),
   matching the resolution policy in :mod:`dlogos.resolution.subjects`.

The store is consumed through a small :class:`CanonicalEntityStore` Protocol
(``by_qid`` / ``by_exact_name`` / ``candidates``) so this module is decoupled
from the concrete in-memory / SQLite stores. The adjudicator is an injected
callable; :func:`llm_adjudicator_from_client` builds the real one over an
OpenAI-compatible client (DeepInfra), lazily and with no network in tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from dlogos.resolution.subjects import _normalize_surface
from dlogos.schema import Entity, EntityType

# --------------------------------------------------------------------------- #
# Per-type similarity thresholds
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TypeThresholds:
    """The confident-match (``high``) and confident-NEW (``low``) cosine bars.

    A score ``>= high`` is an automatic match; ``< low`` is an automatic NEW;
    anything in between is *ambiguous* and escalates to LLM adjudication.
    """

    high: float
    low: float


# Tuned to be conservative. Named entities (org/person) cluster tightly when
# they are the same thing, so a 0.86 high bar is safe; concepts paraphrase more
# loosely, so they get a stricter 0.90 high / 0.70 low band to avoid merging
# two distinct ideas that merely share vocabulary.
DEFAULT_THRESHOLDS: dict[EntityType, TypeThresholds] = {
    EntityType.organization: TypeThresholds(high=0.86, low=0.62),
    EntityType.person: TypeThresholds(high=0.86, low=0.62),
    EntityType.concept: TypeThresholds(high=0.90, low=0.70),
    EntityType.work: TypeThresholds(high=0.90, low=0.70),
}


@dataclass
class MatchDecision:
    """The cascade's verdict.

    ``canonical_id`` is the id of the matched canonical entity, or ``None`` to
    signal "mint a NEW canonical entity". ``reason`` is a short, auditable tag
    naming the tier that decided (``qid`` / ``exact-name`` / ``embed-0.95`` /
    ``new-0.40`` / ``new-empty`` / ``llm-yes`` / ``ambiguous-conservative``).
    """

    canonical_id: str | None
    reason: str


# --------------------------------------------------------------------------- #
# Store contract (the cascade codes to this; concrete stores satisfy it)
# --------------------------------------------------------------------------- #
@runtime_checkable
class CanonicalEntityStore(Protocol):
    """The slice of the canonical-entity store the cascade reads.

    Concrete stores (in-memory / SQLite, Task 1.1/1.2) implement a superset of
    this; the cascade depends only on the three lookup methods below. Each
    returns a record exposing at least ``.canonical_id``; ``candidates`` returns
    already-scored ``(entity, cosine)`` pairs sorted descending — the single
    similarity entry point (the ANN seam).
    """

    def by_qid(self, qid: str, entity_type: EntityType) -> Any | None: ...

    def by_exact_name(self, norm_name: str, entity_type: EntityType) -> Any | None: ...

    def candidates(
        self, embedding: list[float], entity_type: EntityType, k: int = 5
    ) -> list[tuple[Any, float]]: ...


# --------------------------------------------------------------------------- #
# The cascade
# --------------------------------------------------------------------------- #
def match_entity(
    entity: Entity,
    embedding: list[float],
    store: CanonicalEntityStore,
    *,
    qid: str | None = None,
    thresholds: dict[EntityType, TypeThresholds] = DEFAULT_THRESHOLDS,
    llm_adjudicator: Callable[[Any, Any], bool] | None = None,
) -> MatchDecision:
    """Decide whether ``entity`` matches a stored canonical entity, or is NEW.

    Runs rules -> embedding -> (optional) LLM adjudication, short-circuiting at
    the first decisive tier. ``qid`` (when present) and the exact normalized
    name are checked before any embedding compare. The ambiguous middle band is
    resolved by ``llm_adjudicator`` if injected, else conservatively to NEW.

    Returns a :class:`MatchDecision` whose ``canonical_id`` is the matched id or
    ``None`` (mint a new canonical entity).
    """

    # --- Tier 1: rules (free, decisive) ------------------------------------- #
    if qid:
        hit = store.by_qid(qid, entity.type)
        if hit is not None:
            return MatchDecision(hit.canonical_id, "qid")

    exact = store.by_exact_name(_normalize_surface(entity.name), entity.type)
    if exact is not None:
        return MatchDecision(exact.canonical_id, "exact-name")

    # --- Tier 2: embedding (per-type thresholds) ---------------------------- #
    cands = store.candidates(embedding, entity.type, k=1)
    if not cands:
        return MatchDecision(None, "new-empty")

    top, score = cands[0]
    t = thresholds[entity.type]
    if score >= t.high:
        return MatchDecision(top.canonical_id, f"embed-{score:.2f}")
    if score < t.low:
        return MatchDecision(None, f"new-{score:.2f}")

    # --- Tier 3: ambiguous middle -> LLM adjudication (only here; cheap) ----- #
    if llm_adjudicator is not None and llm_adjudicator(entity, top):
        return MatchDecision(top.canonical_id, "llm-yes")
    return MatchDecision(None, "ambiguous-conservative")


# --------------------------------------------------------------------------- #
# Real LLM adjudicator (injected OpenAI-compatible client; lazy, offline tests)
# --------------------------------------------------------------------------- #
_ADJUDICATOR_SYSTEM = (
    "You are an entity-resolution adjudicator. You decide whether two "
    "descriptions refer to the SAME real-world entity. Answer ONLY with a JSON "
    'object: {"same": true} or {"same": false}.'
)


def _names_of(obj: Any) -> str:
    """Best-effort 'name + aliases' string for either an Entity or a store record.

    Tolerates both the extractor's :class:`~dlogos.schema.Entity` (``name``) and
    a canonical-entity record (``canonical_name`` + ``aliases``) so the same
    adjudicator works on both sides of a candidate pair.
    """

    name = getattr(obj, "canonical_name", None) or getattr(obj, "name", "") or ""
    aliases = list(getattr(obj, "aliases", []) or [])
    extra = [a for a in aliases if a and a != name]
    return f"{name} (aka {', '.join(extra)})" if extra else str(name)


def _type_of(obj: Any) -> str:
    etype = getattr(obj, "entity_type", None) or getattr(obj, "type", None)
    if isinstance(etype, EntityType):
        return etype.value
    return str(etype) if etype is not None else "entity"


def llm_adjudicator_from_client(
    client: Any, model: str
) -> Callable[[Any, Any], bool]:
    """Build a yes/no adjudicator over an injected OpenAI-compatible client.

    The returned ``adj(entity, candidate) -> bool`` sends a tight JSON prompt
    via ``client.chat.completions.create(...)`` (same shape as the extractor's
    ``_call``, but synchronous because the cascade calls the adjudicator
    synchronously) and parses ``{"same": bool}``. It returns ``False`` on ANY
    parse error, missing key, or HTTP/transport failure — conservative by
    construction, so a flaky model never causes a wrong merge.

    The client is injected; unit tests pass a fake and no network is touched.
    """

    def adj(entity: Any, candidate: Any) -> bool:
        etype = _type_of(entity)
        user = (
            f"Are these the same real-world {etype}?\n"
            f"A: {_names_of(entity)}\n"
            f"B: {_names_of(candidate)}\n"
            'Answer JSON {"same": bool}.'
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _ADJUDICATOR_SYSTEM},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
        except Exception:  # noqa: BLE001 — any transport/HTTP failure -> conservative NEW
            return False

        content = _content_of(resp)
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return False
        if not isinstance(parsed, dict):
            return False
        return parsed.get("same") is True

    return adj


def _content_of(resp: Any) -> str:
    """Pull assistant content from a chat completion (dict- or object-shaped).

    Mirrors :func:`dlogos.extraction.extractor._content_of` so fakes may return
    a plain dict while the real OpenAI SDK returns objects.
    """

    if isinstance(resp, dict):
        choices = resp.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        return msg.get("content") or ""

    choices = getattr(resp, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    if message is None:
        return ""
    return getattr(message, "content", "") or ""
