"""Tests for the Podcast Index client — all network mocked via MockTransport."""

from __future__ import annotations

import hashlib

import httpx
import pytest

from dlogos.ingestion.podcast_index import (
    PodcastIndexClient,
    PodcastIndexError,
    _auth_headers,
)


def _mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_auth_headers_are_deterministic_with_injected_now() -> None:
    headers = _auth_headers(
        "key123", "secret456", user_agent="UA/1.0", now=1_700_000_000
    )
    expected_digest = hashlib.sha1(b"key123secret4561700000000").hexdigest()
    assert headers["X-Auth-Key"] == "key123"
    assert headers["X-Auth-Date"] == "1700000000"
    assert headers["Authorization"] == expected_digest
    assert headers["User-Agent"] == "UA/1.0"


def test_search_feeds_sends_auth_and_parses_feeds() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["q"] = request.url.params.get("q")
        captured["auth"] = request.headers.get("Authorization")
        captured["key"] = request.headers.get("X-Auth-Key")
        return httpx.Response(
            200,
            json={"status": "true", "feeds": [{"id": 1, "url": "https://f/1.xml"}]},
        )

    client = PodcastIndexClient(
        api_key="k",
        api_secret="s",
        client=_mock_client(handler),
        now=1_700_000_000,
    )
    feeds = client.search_feeds("ai podcast", max_results=5)

    assert feeds == [{"id": 1, "url": "https://f/1.xml"}]
    assert captured["path"] == "/api/1.0/search/byterm"
    assert captured["q"] == "ai podcast"
    assert captured["key"] == "k"
    assert captured["auth"] == hashlib.sha1(b"ks1700000000").hexdigest()


def test_feeds_by_category_parses_feeds() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/1.0/podcasts/bytag"
        assert request.url.params.get("cat") == "Technology"
        return httpx.Response(200, json={"feeds": [{"id": 7}, {"id": 9}]})

    client = PodcastIndexClient(api_key="k", api_secret="s", client=_mock_client(handler))
    feeds = client.feeds_by_category("Technology", max_results=2)
    assert [f["id"] for f in feeds] == [7, 9]


def test_recent_episodes_by_feed_url_passes_since() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/1.0/episodes/byfeedurl"
        assert request.url.params.get("url") == "https://f/1.xml"
        assert request.url.params.get("since") == "123"
        return httpx.Response(200, json={"items": [{"guid": "g1"}]})

    client = PodcastIndexClient(api_key="k", api_secret="s", client=_mock_client(handler))
    items = client.recent_episodes(feed_url="https://f/1.xml", since=123)
    assert items == [{"guid": "g1"}]


def test_recent_episodes_requires_feed_identifier() -> None:
    client = PodcastIndexClient(api_key="k", api_secret="s", client=_mock_client(lambda r: httpx.Response(200, json={})))
    with pytest.raises(ValueError):
        client.recent_episodes()


def test_non_200_raises_podcast_index_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    client = PodcastIndexClient(api_key="bad", api_secret="bad", client=_mock_client(handler))
    with pytest.raises(PodcastIndexError) as ei:
        client.search_feeds("x")
    assert ei.value.status_code == 401
