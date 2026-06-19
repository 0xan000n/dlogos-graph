"""Audio enclosure fetch with content-hash dedupe + GUID idempotency (§7.1).

Spec §7.1: "Fetch audio enclosure → content-hash → object storage. Idempotent
on episode GUID so re-polling never reprocesses." This module implements that
contract against a pluggable blob store and an injectable HTTP client:

- **GUID idempotency** — if an episode GUID has been fetched before, return the
  cached :class:`FetchResult` without re-downloading.
- **Content-hash dedupe** — two episodes whose audio bytes hash to the same
  ``sha256`` share one stored blob (re-releases / mirrored enclosures), so the
  blob store never holds the same bytes twice.

The blob store is an in-memory dict by default (no real object storage needed
for tests); the real backend (S3/R2/MinIO) plugs in behind the same tiny
:class:`BlobStore` protocol.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Protocol

import httpx


def sha256_bytes(data: bytes) -> str:
    """Hex sha256 of ``data`` — the content hash used for dedupe."""

    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class FetchResult:
    """The outcome of fetching one episode enclosure."""

    guid: str
    audio_url: str
    content_hash: str
    blob_key: str
    size_bytes: int
    # True when this call actually downloaded bytes; False when satisfied from
    # the GUID cache. (Content-hash dedupe still counts as a download of bytes
    # that turned out to be duplicate — see ``deduped``.)
    downloaded: bool
    # True when the downloaded bytes matched an existing blob (content dedupe).
    deduped: bool = False


class BlobStore(Protocol):
    """Minimal content-addressed blob store interface."""

    def has(self, key: str) -> bool: ...

    def put(self, key: str, data: bytes) -> None: ...

    def get(self, key: str) -> bytes: ...


@dataclass
class InMemoryBlobStore:
    """Dict-backed blob store for tests and local dev (no real object storage)."""

    _blobs: dict[str, bytes] = field(default_factory=dict)

    def has(self, key: str) -> bool:
        return key in self._blobs

    def put(self, key: str, data: bytes) -> None:
        self._blobs[key] = data

    def get(self, key: str) -> bytes:
        return self._blobs[key]

    def __len__(self) -> int:  # convenience for assertions
        return len(self._blobs)


def _blob_key(content_hash: str) -> str:
    """Content-addressed key: the blob is named by its own hash."""

    return f"audio/{content_hash}"


class AudioFetcher:
    """Fetches episode audio with GUID idempotency and content dedupe.

    State (the GUID → result cache) lives on the instance, so a single fetcher
    is the unit of idempotency for a backfill run. Inject an ``httpx.Client``
    (wrapping a mock transport in tests) and a :class:`BlobStore`; the only
    network call is inside :meth:`fetch`.
    """

    def __init__(
        self,
        *,
        blob_store: BlobStore | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.blob_store: BlobStore = (
            blob_store if blob_store is not None else InMemoryBlobStore()
        )
        self._client = client
        self._owns_client = client is None
        # GUID → cached result (idempotency); content_hash → blob_key (dedupe).
        self._by_guid: dict[str, FetchResult] = {}
        self._by_hash: dict[str, str] = {}

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=60.0, follow_redirects=True)
        return self._client

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None

    def __enter__(self) -> "AudioFetcher":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- introspection ------------------------------------------------------ #
    def seen_guid(self, guid: str) -> bool:
        """True if ``guid`` has already been fetched in this run."""

        return guid in self._by_guid

    def cached(self, guid: str) -> FetchResult | None:
        """The cached result for ``guid``, if any."""

        return self._by_guid.get(guid)

    # -- main entry point --------------------------------------------------- #
    def fetch(self, guid: str, audio_url: str) -> FetchResult:
        """Fetch the enclosure for ``guid``, returning a :class:`FetchResult`.

        Idempotent on ``guid``: a repeat call returns the cached result and
        performs no download. New GUIDs are downloaded, content-hashed, and
        deduped against previously stored blobs.
        """

        if not guid:
            raise ValueError("episode GUID is required for idempotency")

        cached = self._by_guid.get(guid)
        if cached is not None:
            # GUID idempotency: never reprocess a known episode.
            return cached

        resp = self._get_client().get(audio_url)
        if resp.status_code != 200:
            raise httpx.HTTPStatusError(
                f"enclosure fetch failed: {resp.status_code}",
                request=resp.request,
                response=resp,
            )
        data = resp.content
        content_hash = sha256_bytes(data)

        existing_key = self._by_hash.get(content_hash)
        if existing_key is not None or self.blob_store.has(_blob_key(content_hash)):
            # Content-hash dedupe: identical bytes already stored.
            blob_key = existing_key or _blob_key(content_hash)
            self._by_hash.setdefault(content_hash, blob_key)
            result = FetchResult(
                guid=guid,
                audio_url=audio_url,
                content_hash=content_hash,
                blob_key=blob_key,
                size_bytes=len(data),
                downloaded=True,
                deduped=True,
            )
        else:
            blob_key = _blob_key(content_hash)
            self.blob_store.put(blob_key, data)
            self._by_hash[content_hash] = blob_key
            result = FetchResult(
                guid=guid,
                audio_url=audio_url,
                content_hash=content_hash,
                blob_key=blob_key,
                size_bytes=len(data),
                downloaded=True,
                deduped=False,
            )

        self._by_guid[guid] = result
        return result
