"""Lightweight Wikidata linking (spec §7.4a lever ii / §7.3).

Recurring guests (and well-known orgs/people that subjects refer to) are
belief-tracking subjects: "what does *[guest]* believe about X, and has it
changed?" only works if the same person merges across shows under one stable
id. The cheap, deterministic signal for that stable id is a **Wikidata QID**.

This module canonicalizes a name (+ optional type/domain context) to a QID via
Wikidata's public ``wbsearchentities`` API. The HTTP transport is fully
INJECTABLE and the real network client is built LAZILY, so:

- unit tests pass a fake client and never touch the network;
- importing this module never opens a connection or requires credentials.

The matcher is intentionally conservative — it returns the top candidate whose
type is compatible (people/orgs only; concepts/works are skipped because the
PoC tracks beliefs of *speakers* and consensus about *named entities*, and
Wikidata disambiguation for generic concepts is noisy). No match yields
``None`` rather than a wrong QID — a wrong canonical id is worse than none.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from dlogos.config import settings
from dlogos.schema import Entity, EntityType

# Wikidata's wbsearchentities accepts a free-text search; we keep the surface
# of the call small and dependency-free so the fake client is trivial.
_SEARCH_ACTION = "wbsearchentities"
_DEFAULT_LANGUAGE = "en"
# Only these entity types are linked; concepts/works are intentionally skipped.
_LINKABLE_TYPES = frozenset({EntityType.person, EntityType.organization})


@runtime_checkable
class WikidataClient(Protocol):
    """Transport contract: name (+ optional type) -> list of candidate dicts.

    A candidate is a mapping with at least ``id`` (the QID, e.g. ``"Q312"``)
    and ``label``; ``description`` is used for light disambiguation when
    present. The real implementation hits the Wikidata API; tests inject a
    fake returning canned candidates. Keeping the contract at the
    "list of candidates" level (not raw HTTP) means the fake stays tiny.
    """

    def search(
        self,
        name: str,
        *,
        entity_type: EntityType | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:  # pragma: no cover - protocol
        ...


class _HttpxWikidataClient:
    """Real client over the Wikidata public API — httpx imported LAZILY.

    Not used in unit tests (they inject a fake). The ``httpx`` import lives
    inside ``__init__`` so importing this module costs nothing and needs no
    network; ``httpx`` is a core dep but we still keep the boundary clean.
    """

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        timeout: float = 10.0,
        language: str = _DEFAULT_LANGUAGE,
    ) -> None:
        import httpx  # lazy: keep import side-effect-free at module load

        # Default to the wbsearchentities REST entry derived from the SPARQL
        # endpoint host, or a sane public default.
        base = endpoint or _default_api_endpoint()
        self._language = language
        self._base = base
        # NB: do NOT use base_url + an empty-path GET — httpx normalizes that to
        # "api.php/?..." (trailing slash) which Wikidata 301-redirects to
        # "api.php?...". GET the full endpoint directly, and follow redirects
        # defensively so any future redirect doesn't surface as a raise.
        self._client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "dLogos-PoC/0.1 (resolution; +local)"},
        )
        # Resolution queries Wikidata once per DISTINCT entity name; many recur
        # (Anthropic/OpenAI/AGI appear hundreds of times) and the public API
        # rate-limits bursts (HTTP 429). Cache by query, throttle politely, and
        # retry-with-backoff on 429/5xx — and NEVER raise out of search(): a
        # Wikidata hiccup must degrade to "no QID", not crash the whole load.
        self._cache: dict[tuple[str, Any, int], list[dict[str, Any]]] = {}
        self._min_interval = 0.05
        self._last_call_at: float | None = None

    def search(
        self, name: str, *, entity_type: EntityType | None = None, limit: int = 5
    ) -> list[dict[str, Any]]:  # pragma: no cover - requires network
        import time

        key = (name, entity_type, limit)
        if key in self._cache:
            return self._cache[key]
        params = {
            "action": _SEARCH_ACTION,
            "search": name,
            "language": self._language,
            "uselang": self._language,
            "format": "json",
            "limit": str(limit),
            "type": "item",
        }
        result: list[dict[str, Any]] = []
        for attempt in range(4):
            if self._last_call_at is not None:
                wait = self._min_interval - (time.monotonic() - self._last_call_at)
                if wait > 0:
                    time.sleep(wait)
            try:
                resp = self._client.get(self._base, params=params)
            except Exception:  # noqa: BLE001 — transport hiccup: back off + retry
                time.sleep(0.5 * (2**attempt))
                self._last_call_at = time.monotonic()
                continue
            self._last_call_at = time.monotonic()
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(0.5 * (2**attempt))  # rate-limited / 5xx: back off
                continue
            if resp.status_code >= 400:
                break  # other client error: treat as no match, never crash
            try:
                result = list(resp.json().get("search", []))
            except Exception:  # noqa: BLE001 — malformed JSON: no match
                result = []
            break
        self._cache[key] = result
        return result

    def close(self) -> None:  # pragma: no cover - requires network
        self._client.close()


def _default_api_endpoint() -> str:
    """Derive the Wikidata ``api.php`` URL from configured SPARQL host."""

    sparql = settings.wikidata_endpoint or "https://query.wikidata.org/sparql"
    try:
        from urllib.parse import urlsplit

        parts = urlsplit(sparql)
        host = parts.netloc or "query.wikidata.org"
    except Exception:  # pragma: no cover - defensive
        host = "query.wikidata.org"
    # The public action API lives on www.wikidata.org regardless of the SPARQL host.
    _ = host
    return "https://www.wikidata.org/w/api.php"


def default_client(**kwargs: Any) -> WikidataClient:
    """Build the real Wikidata client lazily. Tests inject a fake instead."""

    return _HttpxWikidataClient(**kwargs)


# --------------------------------------------------------------------------- #
# Result model + matcher
# --------------------------------------------------------------------------- #
class WikidataMatch(BaseModel):
    """A resolved Wikidata link for one entity surface form."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="The queried surface form.")
    qid: str | None = Field(
        default=None, description="Wikidata QID (e.g. 'Q312'); None if no match."
    )
    label: str | None = Field(default=None, description="Matched Wikidata label.")
    description: str | None = Field(
        default=None, description="Matched Wikidata description, when available."
    )


class WikidataLinker:
    """Conservative name -> QID linker over an injected client.

    Caches by ``(normalized name, type)`` so repeated guests across episodes
    cost one lookup. Concepts/works are not linked (return ``None``).
    """

    def __init__(
        self,
        client: WikidataClient | None = None,
        *,
        linkable_types: Iterable[EntityType] = _LINKABLE_TYPES,
    ) -> None:
        # Lazily build a real client only if a lookup is actually attempted and
        # none was injected — so constructing a linker never opens the network.
        self._client = client
        self._linkable = frozenset(linkable_types)
        self._cache: dict[tuple[str, EntityType], WikidataMatch] = {}

    def _ensure_client(self) -> WikidataClient:
        if self._client is None:
            self._client = default_client()
        return self._client

    @staticmethod
    def _norm(name: str) -> str:
        return " ".join(name.strip().split()).casefold()

    def link(
        self,
        name: str,
        entity_type: EntityType,
        *,
        context: Sequence[str] | None = None,
    ) -> WikidataMatch:
        """Resolve one name to a :class:`WikidataMatch` (qid may be ``None``).

        ``context`` is an optional set of domain hints (e.g. a show's domain
        tags). When provided, a candidate whose description overlaps the
        context is preferred over the raw relevance-ranked top hit — this is the
        recurring-guest disambiguation lever (spec §7.3). Without context the
        matcher trusts ``wbsearchentities``' relevance ordering. Caching keys on
        ``(name, type)`` only; the first context seen for a name wins, which is
        fine because a recurring guest's domain is stable across episodes.
        """

        clean = name.strip()
        if not clean or entity_type not in self._linkable:
            return WikidataMatch(name=clean, qid=None)

        cache_key = (self._norm(clean), entity_type)
        if cache_key in self._cache:
            return self._cache[cache_key]

        candidates = self._ensure_client().search(
            clean, entity_type=entity_type, limit=5
        )
        match = self._pick(clean, candidates, context=context)
        self._cache[cache_key] = match
        return match

    def link_entity(self, entity: Entity) -> Entity:
        """Return a copy of ``entity`` whose ``canonical_id`` is set to QID.

        Only fills ``canonical_id`` when it's empty AND a QID is found, so a
        prior subject-clustering id (§7.4a lever i) is never clobbered. The
        clustering id and the QID serve different roles; here we only seed an
        id when nothing has resolved the entity yet.
        """

        match = self.link(entity.name, entity.type)
        if match.qid and entity.canonical_id is None:
            return entity.model_copy(update={"canonical_id": match.qid})
        return entity.model_copy()

    @staticmethod
    def _pick(
        name: str,
        candidates: Sequence[dict[str, Any]],
        *,
        context: Sequence[str] | None = None,
    ) -> WikidataMatch:
        """Choose the best candidate conservatively.

        Default to the first candidate — ``wbsearchentities`` already ranks by
        relevance, so the top hit is the intended sense (e.g. "Apple" ->
        *Apple Inc.*, not the fruit). We deliberately do NOT prefer an exact
        case-insensitive *label* match: a lower-relevance homonym whose label
        happens to equal the query (the fruit "apple") would otherwise hijack
        the link, which is exactly the wrong-canonical-id failure we want to
        avoid.

        When ``context`` (domain hints) is supplied, prefer the first candidate
        whose ``description`` overlaps a hint — this disambiguates a recurring
        guest by the show's domain without trusting label equality. Require a
        usable QID; return ``None`` qid if the list is empty or malformed.
        """

        if not candidates:
            return WikidataMatch(name=name, qid=None)

        chosen = candidates[0]
        if context:
            hints = {h.lower() for h in context if h}
            for cand in candidates:
                desc = str(cand.get("description") or "").lower()
                if desc and any(hint in desc for hint in hints):
                    chosen = cand
                    break
        qid = chosen.get("id")
        if not isinstance(qid, str) or not qid:
            return WikidataMatch(name=name, qid=None)
        return WikidataMatch(
            name=name,
            qid=qid,
            label=(str(chosen["label"]) if chosen.get("label") else None),
            description=(
                str(chosen["description"]) if chosen.get("description") else None
            ),
        )


def anchor_entity(entity: Entity, linker: WikidataLinker) -> str | None:
    """Resolve a *subject* entity to a Wikidata QID, or ``None``.

    The incremental resolver uses the QID as the strongest cross-episode anchor:
    two surface forms ("Apple", "the iPhone maker") that map to the same QID are
    the same canonical node regardless of how far apart they embed. Only
    person/organization subjects are anchored — concepts/works paraphrase too
    freely for ``wbsearchentities`` to disambiguate reliably, so we never even
    consult the client for them (matching :data:`_LINKABLE_TYPES` and the
    linker's own conservatism). Delegates to the injected
    :class:`WikidataLinker` (no reimplementation); returns the QID or ``None``.
    """

    if entity.type not in _LINKABLE_TYPES:
        return None
    return linker.link(entity.name, entity.type).qid


def link_entities(
    entities: Sequence[Entity],
    client: WikidataClient | None = None,
) -> list[WikidataMatch]:
    """Batch-link a set of entities, deduplicating by (name, type).

    Convenience wrapper for callers that want the match table rather than
    rewritten entities. Deterministic: results follow input order.
    """

    linker = WikidataLinker(client)
    return [linker.link(ent.name, ent.type) for ent in entities]
