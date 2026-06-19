"""Ingestion front-end for the dLogos pipeline (spec §5, §7.1).

The corpus enters here: feeds resolved via the Podcast Index, a versioned
corpus manifest assembled from public category charts, audio enclosures
fetched + content-hashed + deduped, and a queue → workers job abstraction
that backfill fan-out and incremental polling both ride on.

All network/IO is confined to method bodies; importing this package (and any
submodule) pulls in only the core dependency group so the unit tests run
without heavy/optional extras.
"""

from __future__ import annotations

from dlogos.ingestion.charts import (
    DOMAINS,
    ChartShow,
    build_manifest_from_charts,
    flag_high_velocity_subset,
)
from dlogos.ingestion.fetch import (
    AudioFetcher,
    FetchResult,
    sha256_bytes,
)
from dlogos.ingestion.manifest import (
    CorpusManifest,
    ManifestRow,
    load_manifest,
    save_manifest,
)
from dlogos.ingestion.podcast_index import (
    PodcastIndexClient,
    PodcastIndexError,
)
from dlogos.ingestion.queue import (
    InMemoryJobQueue,
    Job,
    JobQueue,
    JobStatus,
    SQLiteJobQueue,
)

__all__ = [
    # charts
    "DOMAINS",
    "ChartShow",
    "build_manifest_from_charts",
    "flag_high_velocity_subset",
    # fetch
    "AudioFetcher",
    "FetchResult",
    "sha256_bytes",
    # manifest
    "CorpusManifest",
    "ManifestRow",
    "load_manifest",
    "save_manifest",
    # podcast index
    "PodcastIndexClient",
    "PodcastIndexError",
    # queue
    "InMemoryJobQueue",
    "Job",
    "JobQueue",
    "JobStatus",
    "SQLiteJobQueue",
]
