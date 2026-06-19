"""Tests for audio fetch: content-hash dedupe and GUID idempotency."""

from __future__ import annotations

import httpx
import pytest

from dlogos.ingestion.fetch import (
    AudioFetcher,
    InMemoryBlobStore,
    sha256_bytes,
)


def _counting_handler(responses: dict[str, bytes]):
    """A handler that serves fixed bytes per URL and counts hits."""

    calls: dict[str, int] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        calls[url] = calls.get(url, 0) + 1
        if url not in responses:
            return httpx.Response(404, text="not found")
        return httpx.Response(200, content=responses[url])

    handler.calls = calls  # type: ignore[attr-defined]
    return handler


def _fetcher(handler, store=None) -> AudioFetcher:
    # NB: use `is None`, not `or` — an empty InMemoryBlobStore is falsy
    # (it defines __len__), so `store or InMemoryBlobStore()` would discard it.
    return AudioFetcher(
        blob_store=store if store is not None else InMemoryBlobStore(),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_sha256_helper_matches_hashlib() -> None:
    import hashlib

    assert sha256_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()


def test_fetch_downloads_and_stores_blob() -> None:
    handler = _counting_handler({"https://a/ep1.mp3": b"AUDIO-ONE"})
    store = InMemoryBlobStore()
    f = _fetcher(handler, store)

    res = f.fetch("guid-1", "https://a/ep1.mp3")
    assert res.downloaded is True
    assert res.deduped is False
    assert res.content_hash == sha256_bytes(b"AUDIO-ONE")
    assert res.size_bytes == len(b"AUDIO-ONE")
    assert store.has(res.blob_key)
    assert len(store) == 1


def test_guid_idempotency_returns_cached_without_redownload() -> None:
    handler = _counting_handler({"https://a/ep1.mp3": b"AUDIO-ONE"})
    f = _fetcher(handler)

    first = f.fetch("guid-1", "https://a/ep1.mp3")
    assert f.seen_guid("guid-1")
    second = f.fetch("guid-1", "https://a/ep1.mp3")

    assert second == first
    # The enclosure URL was hit exactly once despite two fetch() calls.
    assert handler.calls["https://a/ep1.mp3"] == 1


def test_content_hash_dedupe_across_distinct_guids() -> None:
    # Two different episodes (GUIDs) whose audio bytes are identical
    # (re-release / mirrored enclosure) must share one stored blob.
    same_bytes = b"IDENTICAL-AUDIO"
    handler = _counting_handler(
        {"https://a/ep1.mp3": same_bytes, "https://b/ep1-mirror.mp3": same_bytes}
    )
    store = InMemoryBlobStore()
    f = _fetcher(handler, store)

    r1 = f.fetch("guid-1", "https://a/ep1.mp3")
    r2 = f.fetch("guid-2", "https://b/ep1-mirror.mp3")

    assert r1.deduped is False
    assert r2.deduped is True
    assert r1.content_hash == r2.content_hash
    assert r1.blob_key == r2.blob_key
    # Only one blob stored despite two episodes.
    assert len(store) == 1


def test_distinct_audio_stores_two_blobs() -> None:
    handler = _counting_handler(
        {"https://a/1.mp3": b"ONE", "https://a/2.mp3": b"TWO"}
    )
    store = InMemoryBlobStore()
    f = _fetcher(handler, store)
    f.fetch("g1", "https://a/1.mp3")
    f.fetch("g2", "https://a/2.mp3")
    assert len(store) == 2


def test_empty_guid_rejected() -> None:
    handler = _counting_handler({})
    f = _fetcher(handler)
    with pytest.raises(ValueError):
        f.fetch("", "https://a/1.mp3")


def test_non_200_enclosure_raises() -> None:
    handler = _counting_handler({})  # everything 404s
    f = _fetcher(handler)
    with pytest.raises(httpx.HTTPStatusError):
        f.fetch("g1", "https://a/missing.mp3")


def test_dedupe_against_prepopulated_blob_store() -> None:
    # A blob already in the store (from a prior run) is recognized as dedupe.
    data = b"PRELOADED"
    store = InMemoryBlobStore()
    store.put(f"audio/{sha256_bytes(data)}", data)
    handler = _counting_handler({"https://a/1.mp3": data})
    f = _fetcher(handler, store)

    res = f.fetch("g1", "https://a/1.mp3")
    assert res.deduped is True
    assert len(store) == 1
