"""A thin, typed client for the Podcast Index API (spec §7.1).

The Podcast Index resolves feeds to clean feed URLs and episode GUIDs, which
the rest of ingestion treats as the idempotency key. The client only touches
the network inside its methods — constructing it (and importing this module)
does no IO, so tests can inject a mocked ``httpx`` transport.

Auth: the Podcast Index uses a per-request signed header scheme. Each request
carries the API key, a unix timestamp, and a ``sha1(key + secret + timestamp)``
authorization digest plus a ``User-Agent``. See
https://podcastindex-org.github.io/docs-api/ for the contract.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

import httpx

from dlogos.config import settings

DEFAULT_BASE_URL = "https://api.podcastindex.org/api/1.0"
DEFAULT_USER_AGENT = "dLogos-PoC/0.1 (+ingestion)"


class PodcastIndexError(RuntimeError):
    """Raised when the Podcast Index API returns a non-OK response."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"Podcast Index API error {status_code}: {message}")
        self.status_code = status_code
        self.message = message


def _auth_headers(
    api_key: str,
    api_secret: str,
    *,
    user_agent: str,
    now: int | None = None,
) -> dict[str, str]:
    """Build the signed auth headers for a single request.

    ``now`` is injectable so tests are deterministic. The digest is
    ``sha1(api_key + api_secret + unix_seconds)`` per the Podcast Index spec.
    """

    ts = int(time.time()) if now is None else int(now)
    digest = hashlib.sha1(
        f"{api_key}{api_secret}{ts}".encode("utf-8")
    ).hexdigest()
    return {
        "User-Agent": user_agent,
        "X-Auth-Key": api_key,
        "X-Auth-Date": str(ts),
        "Authorization": digest,
    }


class PodcastIndexClient:
    """Synchronous Podcast Index client.

    Parameters are injectable so unit tests can pass a mocked
    :class:`httpx.Client` (e.g. wrapping a :class:`httpx.MockTransport`) and a
    fixed ``now`` for deterministic auth headers — no real network, no real
    clock.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        user_agent: str = DEFAULT_USER_AGENT,
        client: httpx.Client | None = None,
        now: int | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.podcast_index_key
        self.api_secret = (
            api_secret if api_secret is not None else settings.podcast_index_secret
        )
        self.base_url = base_url.rstrip("/")
        self.user_agent = user_agent
        self._now = now
        # Injected client wins; otherwise we lazily build one on first use so
        # that merely constructing the object opens no sockets.
        self._client = client
        self._owns_client = client is None

    # -- lifecycle ---------------------------------------------------------- #
    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=30.0)
        return self._client

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None

    def __enter__(self) -> "PodcastIndexClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- internals ---------------------------------------------------------- #
    def _request(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers = _auth_headers(
            self.api_key,
            self.api_secret,
            user_agent=self.user_agent,
            now=self._now,
        )
        # Drop None-valued params so callers can pass optional filters cleanly.
        clean = {k: v for k, v in params.items() if v is not None}
        resp = self._get_client().get(url, params=clean, headers=headers)
        if resp.status_code != 200:
            raise PodcastIndexError(resp.status_code, resp.text)
        return resp.json()

    # -- public API --------------------------------------------------------- #
    def search_feeds(
        self,
        query: str,
        *,
        max_results: int | None = None,
        clean: bool = True,
    ) -> list[dict[str, Any]]:
        """Search podcast feeds by term; returns the raw ``feeds`` list."""

        data = self._request(
            "/search/byterm",
            {"q": query, "max": max_results, "clean": "true" if clean else None},
        )
        return list(data.get("feeds", []))

    def feeds_by_category(
        self,
        category: str,
        *,
        max_results: int | None = None,
    ) -> list[dict[str, Any]]:
        """List feeds within a Podcast Index category (a chart-like view)."""

        data = self._request(
            "/podcasts/bytag",
            {"cat": category, "max": max_results},
        )
        return list(data.get("feeds", []))

    def recent_episodes(
        self,
        *,
        feed_id: int | None = None,
        feed_url: str | None = None,
        max_results: int | None = None,
        since: int | None = None,
    ) -> list[dict[str, Any]]:
        """Recent episodes for a feed (by id or URL); returns ``items``.

        ``since`` is a unix timestamp lower bound, used by the incremental
        poller to fetch only items newer than the last seen publish time.
        """

        if feed_id is not None:
            path = "/episodes/byfeedid"
            params: dict[str, Any] = {"id": feed_id, "max": max_results, "since": since}
        elif feed_url is not None:
            path = "/episodes/byfeedurl"
            params = {"url": feed_url, "max": max_results, "since": since}
        else:
            raise ValueError("recent_episodes requires feed_id or feed_url")
        data = self._request(path, params)
        return list(data.get("items", []))
