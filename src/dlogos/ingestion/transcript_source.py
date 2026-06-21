"""Text-transcript ingestion backend: fetch a URL → parse → :class:`Transcript`.

This is the integration seam that lets the 20 public *text* transcripts in
``docs/corpus/ai_sensemaking_20.json`` flow through the existing pipeline with no
ASR/diarization at all. The transcripts already carry real speaker **names**
(``"Jim Rutt:"``, ``"Tristan Harris:"``), so the name-driven resolver
(:mod:`dlogos.speakers.speaker_store`) canonicalizes them directly — strictly
better than diarization labels.

Three pieces live here:

- :data:`REGISTRY` — a ``host → parser`` map covering every netloc in the 20
  episodes (``www.`` is normalized away, so ``www.dwarkesh.com`` and
  ``dwarkesh.com`` both resolve). Each value is one of the pure stdlib
  ``parse(text) -> list[TranscriptSegment]`` functions from
  :mod:`dlogos.ingestion.parsers`.
- :class:`TranscriptBackend` — an :class:`~dlogos.asr.base.ASRBackend` (it
  implements ``transcribe(audio_path) -> Transcript``) where ``audio_path`` is
  the transcript **URL**. It lazily GETs the URL with httpx (browser UA, follow
  redirects), runs the bytes through ``html_to_text`` / ``pdf_to_text`` by
  format, dispatches to the registered parser, and wraps the result in a
  :class:`~dlogos.schema.Transcript`. A content-hash cache (mirroring
  ``scripts/run_smoke_inmemory.CachingASR``) keyed on the URL stores the parsed
  Transcript JSON so re-runs do not re-fetch/re-parse.
- :func:`build_episodes_from_corpus` — read the corpus JSON into one
  :class:`~dlogos.pipeline.EpisodeInput` per episode (``episode_id`` = corpus
  ``id``, ``audio_path`` = transcript URL, guests + a domain tag attached).

Only :meth:`TranscriptBackend.transcribe`'s cache-miss path touches the network
(via a lazily-built httpx client), and HTML/PDF extraction lazy-imports the
``transcripts`` extra inside :mod:`dlogos.ingestion.transcript_extract`. Importing
this module therefore costs no heavy/optional dependency, and the offline tests
monkeypatch the fetch to a saved fixture — no network.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dlogos.ingestion.parsers import (
    dwarkesh,
    eighty_k,
    jimrutt,
    lexfridman,
    singjupost,
    substack,
    tgs,
)
from dlogos.ingestion.transcript_extract import html_to_text, pdf_to_text
from dlogos.pipeline import EpisodeInput
from dlogos.schema import Episode, Transcript, TranscriptSegment

__all__ = [
    "REGISTRY",
    "Parser",
    "TranscriptBackend",
    "parser_for_url",
    "build_episodes_from_corpus",
]

#: A pure transcript parser: readable text → ordered segments.
Parser = Callable[[str], list[TranscriptSegment]]

# Default domain tag for this slice (every episode is AI-safety / sensemaking);
# it flows onto each EpisodeInput for guest Wikidata disambiguation downstream.
_DOMAINS = ["ai-safety"]

# A browser-like UA: several of these hosts (Substack, Cloudflare-fronted sites)
# 403 a default httpx/python UA. Only used on the cache-miss network path.
_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)


# --------------------------------------------------------------------------- #
# Host → parser registry
# --------------------------------------------------------------------------- #
#: Every netloc in the 20-episode corpus maps to its parser. Keys are stored
#: **without** a leading ``www.`` (see :func:`_normalize_host`), so both the
#: ``www.`` and bare forms resolve to the same parser.
REGISTRY: dict[str, Parser] = {
    "jimruttshow.com": jimrutt.parse,
    "centerforhumanetechnology.substack.com": substack.parse,
    "lironshapira.substack.com": substack.parse,
    "dwarkesh.com": dwarkesh.parse,
    "lexfridman.com": lexfridman.parse,
    "80000hours.org": eighty_k.parse,
    "singjupost.com": singjupost.parse,
    "thegreatsimplification.com": tgs.parse,
}


def _normalize_host(netloc: str) -> str:
    """Lowercase a URL netloc and strip a leading ``www.`` (port dropped too).

    ``www.dwarkesh.com:443`` and ``dwarkesh.com`` both normalize to
    ``dwarkesh.com`` so the registry has one entry per real host.
    """

    host = netloc.lower().split("@")[-1].split(":")[0]
    return host[4:] if host.startswith("www.") else host


def parser_for_url(url: str) -> Parser:
    """Return the registered parser for ``url``'s host, or raise ``KeyError``.

    The lookup is by normalized netloc (``www.`` stripped). A host not in
    :data:`REGISTRY` is a programming error for this corpus — it raises a
    ``KeyError`` naming the host so a new source is added explicitly rather than
    silently mis-parsed.
    """

    host = _normalize_host(urlparse(url).netloc)
    try:
        return REGISTRY[host]
    except KeyError:
        raise KeyError(
            f"no transcript parser registered for host {host!r} (url={url!r})"
        ) from None


# --------------------------------------------------------------------------- #
# URL → format + episode-id helpers
# --------------------------------------------------------------------------- #
def _format_for_url(url: str) -> str:
    """``"pdf"`` when the URL path ends in ``.pdf`` (case-insensitive), else ``"html"``."""

    path = urlparse(url).path.lower()
    return "pdf" if path.endswith(".pdf") else "html"


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _episode_id_from_url(url: str) -> str:
    """A stable episode id from the URL's last path segment (its "stem").

    Mirrors the smoke driver's URL-slug fallback: take the final non-empty path
    component, drop a trailing ``.pdf``/``.html`` extension, and slugify. Used
    only when the backend is asked to transcribe a bare URL with no corpus row.
    """

    path = urlparse(url).path.rstrip("/")
    stem = path.rsplit("/", 1)[-1] if path else urlparse(url).netloc
    stem = re.sub(r"\.(pdf|html?)$", "", stem, flags=re.IGNORECASE)
    slug = _SLUG_RE.sub("-", stem.lower()).strip("-")
    return slug or "transcript"


# --------------------------------------------------------------------------- #
# The ASRBackend: URL → fetch → text → parse → Transcript (+ content cache)
# --------------------------------------------------------------------------- #
class TranscriptBackend:
    """An :class:`~dlogos.asr.base.ASRBackend` over public text transcripts.

    ``transcribe(audio_path)`` treats ``audio_path`` as a transcript **URL**:
    it fetches the URL, extracts readable text (HTML or PDF), dispatches to the
    host's registered parser, and returns a :class:`~dlogos.schema.Transcript`
    whose ``segments`` carry real human-name speakers and whose ``duration_s`` is
    the last segment's ``t_end``. This is the seam that lets the existing
    Pipeline ingest text instead of audio with no other change.

    A content-hash cache (URL-keyed, mirroring
    ``scripts/run_smoke_inmemory.CachingASR``) stores the parsed Transcript JSON
    on disk, so re-running a slice neither re-fetches nor re-parses. Pass
    ``cache_dir=None`` to disable the cache (tests that monkeypatch the fetch use
    a tmp dir or disable it).

    The httpx client is built lazily on the first real fetch, so constructing the
    backend — and importing this module — pulls in no network machinery. Tests
    monkeypatch :meth:`_fetch_text` (or inject ``fetch_text=``) to feed a saved
    fixture, exercising the parse/dispatch/cache logic fully offline.
    """

    def __init__(
        self,
        *,
        cache_dir: str | Path | None = "/tmp/dlogos_transcript_cache",
        language: str = "en",
        timeout: float = 60.0,
        fetch_text: Callable[[str], tuple[str, str]] | None = None,
    ) -> None:
        self._language = language
        self._timeout = timeout
        self._client: Any = None
        # Injection seam for tests: (url) -> (readable_text, format). When set it
        # fully replaces the network + extract path, so no fixture ever hits bs4/
        # pypdf or httpx unless the real path runs.
        self._fetch_text = fetch_text
        self._cache_dir: Path | None = (
            Path(cache_dir) if cache_dir is not None else None
        )
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    # -- caching ------------------------------------------------------------ #
    def _cache_key(self, url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()[:32]

    def _cached(self, url: str) -> Transcript | None:
        if self._cache_dir is None:
            return None
        f = self._cache_dir / f"{self._cache_key(url)}.json"
        if f.exists():
            print(f"[transcript] using cached transcript {f}", file=sys.stderr)
            return Transcript.model_validate_json(f.read_text())
        return None

    def _store(self, url: str, transcript: Transcript) -> None:
        if self._cache_dir is None:
            return
        f = self._cache_dir / f"{self._cache_key(url)}.json"
        f.write_text(transcript.model_dump_json())
        print(f"[transcript] cached transcript -> {f}", file=sys.stderr)

    # -- fetch + extract (the only network path; monkeypatched in tests) ---- #
    def _get_client(self) -> Any:
        if self._client is None:
            import httpx  # lazy: only a real fetch needs the client

            self._client = httpx.Client(
                timeout=self._timeout,
                follow_redirects=True,
                headers={"User-Agent": _BROWSER_UA},
            )
        return self._client

    def _fetch_and_extract(self, url: str) -> tuple[str, str]:
        """GET ``url`` and return ``(readable_text, format)``.

        ``format`` is ``"pdf"`` or ``"html"`` (decided by URL suffix). PDFs are
        decoded from raw bytes via :func:`pdf_to_text`; HTML pages via
        :func:`html_to_text`. The browser UA + redirect-follow handle the
        Substack/Cloudflare hosts that 403 a default client. Only reached on a
        cache miss with no injected ``fetch_text``.
        """

        fmt = _format_for_url(url)
        resp = self._get_client().get(url)
        resp.raise_for_status()
        if fmt == "pdf":
            return pdf_to_text(resp.content), "pdf"
        return html_to_text(resp.text), "html"

    def _read_text(self, url: str) -> tuple[str, str]:
        """Resolve ``(readable_text, format)`` via the injected hook or the network."""

        if self._fetch_text is not None:
            return self._fetch_text(url)
        return self._fetch_and_extract(url)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # -- the ASRBackend contract -------------------------------------------- #
    def transcribe(self, audio_path: str) -> Transcript:
        """Fetch + parse the transcript at URL ``audio_path`` into a Transcript.

        Returns a cached Transcript when one exists for this URL. Otherwise reads
        the readable text (network or injected hook), dispatches to the host's
        registered parser, builds the Transcript (``episode_id`` from the URL
        stem, ``duration_s`` = last segment ``t_end``), caches it, and returns it.
        """

        cached = self._cached(audio_path)
        if cached is not None:
            return cached

        parser = parser_for_url(audio_path)
        text, _fmt = self._read_text(audio_path)
        segments = parser(text)
        duration_s = segments[-1].t_end if segments else 0.0
        transcript = Transcript(
            episode_id=_episode_id_from_url(audio_path),
            language=self._language,
            segments=segments,
            duration_s=duration_s,
        )
        self._store(audio_path, transcript)
        return transcript


# --------------------------------------------------------------------------- #
# Corpus → EpisodeInputs
# --------------------------------------------------------------------------- #
def _show_slug(show: str) -> str:
    """Slugify a human show name (``"The Jim Rutt Show"`` → ``"the-jim-rutt-show"``)."""

    return _SLUG_RE.sub("-", show.lower()).strip("-") or "show"


def _published_at(date_str: str) -> datetime:
    """Parse a corpus ``"YYYY-MM-DD"`` date to a UTC-aware datetime.

    Falls back to ``now`` for a missing/blank date (the corpus always carries
    one, but the pipeline only needs *an* event-time, so this never raises).
    """

    s = (date_str or "").strip()
    if not s:
        return datetime.now(timezone.utc)
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def build_episodes_from_corpus(
    corpus_path: str | Path,
) -> list[EpisodeInput]:
    """Read the corpus JSON into one :class:`EpisodeInput` per episode.

    Each episode row becomes an :class:`~dlogos.pipeline.EpisodeInput` whose
    :class:`~dlogos.schema.Episode` has ``episode_id`` = the corpus ``id``,
    ``show_id`` = a slug of the show name, ``published_at`` parsed from the row's
    ``date``, and ``audio_url`` = the transcript URL. The ``audio_path`` is that
    same transcript URL (what :class:`TranscriptBackend` fetches),
    ``metadata_guest_names`` = the row's ``guests``, and ``domains`` = the slice's
    AI-safety tag for guest Wikidata disambiguation.

    Pure given the file contents: reads the JSON and builds dataclasses, no
    network. Raises on a malformed/missing file (a corpus problem the caller
    wants to see, not swallow).
    """

    data = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    episodes: list[EpisodeInput] = []
    for row in data["episodes"]:
        url = row["url"]
        episode_id = row["id"]
        episode = Episode(
            episode_id=episode_id,
            show_id=_show_slug(row.get("show", "")),
            guid=episode_id,
            title=row.get("title", episode_id),
            published_at=_published_at(row.get("date", "")),
            audio_url=url,
        )
        episodes.append(
            EpisodeInput(
                episode=episode,
                audio_path=url,
                metadata_guest_names=list(row.get("guests", [])),
                domains=list(_DOMAINS),
            )
        )
    return episodes
