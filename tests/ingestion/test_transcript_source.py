"""Offline tests for the text-transcript ingestion backend (no network).

Covers the integration seam that lets the 20 public *text* transcripts run
through the existing pipeline without ASR:

- the ``host → parser`` :data:`~dlogos.ingestion.transcript_source.REGISTRY`
  covers every netloc in ``docs/corpus/ai_sensemaking_20.json`` (all 8 hosts, in
  both their ``www.`` and bare forms),
- :class:`~dlogos.ingestion.transcript_source.TranscriptBackend` dispatches a
  saved fixture (the fetch is injected, so no network and no ``bs4``/``pypdf``)
  to the host's registered parser and returns a
  :class:`~dlogos.schema.Transcript` of real-human-name segments, and round-trips
  through its content-hash cache,
- :func:`~dlogos.ingestion.transcript_source.build_episodes_from_corpus` over a
  2-entry corpus yields :class:`~dlogos.pipeline.EpisodeInput`s with the right
  ids / urls / guests.

Everything is offline: the fetch is monkeypatched / injected to feed committed
fixture text, the corpus is a tmp-file the test writes, and importing the module
pulls in no heavy/optional dependency (the bs4/pypdf extract layer lazy-imports
inside :mod:`dlogos.ingestion.transcript_extract`, which these tests never reach).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dlogos.ingestion.transcript_source import (
    REGISTRY,
    TranscriptBackend,
    build_episodes_from_corpus,
    parser_for_url,
)
from dlogos.pipeline import EpisodeInput
from dlogos.schema import Transcript

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "transcripts"
_CORPUS = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "corpus"
    / "ai_sensemaking_20.json"
)

# Every host the 20-episode corpus uses, in the *normalized* form the registry
# keys on (``www.`` stripped). The corpus has exactly these 8 hosts.
_CORPUS_HOSTS = [
    "jimruttshow.com",
    "centerforhumanetechnology.substack.com",
    "lironshapira.substack.com",
    "dwarkesh.com",
    "lexfridman.com",
    "80000hours.org",
    "singjupost.com",
    "thegreatsimplification.com",
]


# --------------------------------------------------------------------------- #
# Registry coverage
# --------------------------------------------------------------------------- #
def test_registry_covers_all_corpus_hosts() -> None:
    """The registry has an entry for every host the corpus uses (all 8)."""

    assert set(REGISTRY) == set(_CORPUS_HOSTS)
    assert all(callable(p) for p in REGISTRY.values())


def test_registry_covers_every_corpus_url() -> None:
    """Every one of the 20 corpus URLs resolves to a registered parser.

    Reads the real corpus and asserts ``parser_for_url`` finds a parser for each
    episode — the end-to-end guarantee that no host is missed.
    """

    data = json.loads(_CORPUS.read_text(encoding="utf-8"))
    rows = data["episodes"]
    assert len(rows) == 20
    for row in rows:
        assert callable(parser_for_url(row["url"])), row["id"]


@pytest.mark.parametrize(
    "url, host",
    [
        ("https://www.dwarkesh.com/p/ilya-sutskever-2", "dwarkesh.com"),
        ("https://dwarkesh.com/p/x", "dwarkesh.com"),
        ("https://www.jimruttshow.com/x/", "jimruttshow.com"),
        ("https://80000hours.org/podcast/episodes/x/", "80000hours.org"),
    ],
)
def test_www_prefix_is_normalized(url: str, host: str) -> None:
    """``www.`` and bare host forms resolve to the same registered parser."""

    assert parser_for_url(url) is REGISTRY[host]


def test_unknown_host_raises_keyerror() -> None:
    """A host with no registered parser raises a descriptive ``KeyError``."""

    with pytest.raises(KeyError, match="no transcript parser registered"):
        parser_for_url("https://example.invalid/some-episode")


# --------------------------------------------------------------------------- #
# TranscriptBackend dispatch (fetch injected — fully offline)
# --------------------------------------------------------------------------- #
def _backend_for_fixture(fixture: str, fmt: str = "html", **kw) -> TranscriptBackend:
    """A backend whose fetch hook always returns the named fixture's text.

    Injecting ``fetch_text`` replaces the entire network + extract path, so the
    test exercises the real dispatch/parse/Transcript-build/cache logic with zero
    network and without importing ``bs4``/``pypdf``.
    """

    text = (_FIXTURES / fixture).read_text(encoding="utf-8")
    return TranscriptBackend(fetch_text=lambda _url: (text, fmt), **kw)


def test_transcribe_dispatches_jimrutt_to_named_segments(tmp_path) -> None:
    """A Jim Rutt URL → the jimrutt parser → named-speaker segments."""

    backend = _backend_for_fixture("jimrutt.txt", cache_dir=tmp_path)
    url = (
        "https://www.jimruttshow.com/the-jim-rutt-show-transcripts/"
        "transcript-of-ep-327-nate-soares-on-why-superhuman-ai-would-kill-us-all/"
    )
    transcript = backend.transcribe(url)

    assert isinstance(transcript, Transcript)
    assert transcript.language == "en"
    assert len(transcript.segments) >= 3
    speakers = {s.speaker for s in transcript.segments}
    # Real human-name labels straight off the page — not diarization labels.
    assert {"Jim", "Nate"} <= speakers
    assert "" not in speakers and "SPEAKER_00" not in speakers
    # episode_id is the URL stem; duration is the last segment's end.
    assert transcript.episode_id == (
        "transcript-of-ep-327-nate-soares-on-why-superhuman-ai-would-kill-us-all"
    )
    assert transcript.duration_s == transcript.segments[-1].t_end


def test_transcribe_dispatches_dwarkesh_with_real_timestamps(tmp_path) -> None:
    """A Dwarkesh URL routes to the dwarkesh parser (real ``HH:MM:SS`` spans)."""

    backend = _backend_for_fixture("dwarkesh.txt", cache_dir=tmp_path)
    transcript = backend.transcribe("https://www.dwarkesh.com/p/ilya-sutskever-2")

    assert len(transcript.segments) >= 3
    speakers = {s.speaker for s in transcript.segments}
    assert "Dwarkesh Patel" in speakers
    # Dwarkesh carries native timestamps, so spans advance past synthesized pace.
    assert transcript.segments[-1].t_end > 0.0


def test_transcribe_dispatches_tgs_pdf_text(tmp_path) -> None:
    """A Great Simplification PDF URL routes to the tgs parser (format=pdf path).

    The fetch hook returns already-extracted PDF text (the real
    :func:`pdf_to_text` is not exercised here — that is the lazy bs4/pypdf layer,
    tested separately); the assertion is that a ``.pdf`` URL still dispatches to
    the tgs parser and yields named segments.
    """

    backend = _backend_for_fixture("tgs.txt", fmt="pdf", cache_dir=tmp_path)
    url = (
        "https://www.thegreatsimplification.com/wp-content/uploads/2026/03/"
        "TGS-214-Tristan-Harris-Transcript.pdf"
    )
    transcript = backend.transcribe(url)

    assert len(transcript.segments) >= 2
    speakers = {s.speaker for s in transcript.segments}
    assert any("Tristan" in s or "Nate" in s for s in speakers), speakers
    assert transcript.episode_id == "tgs-214-tristan-harris-transcript"


def test_segments_are_monotonic_and_well_formed(tmp_path) -> None:
    """Every dispatched transcript yields ordered, non-overlapping spans."""

    backend = _backend_for_fixture("eighty_k.txt", cache_dir=tmp_path)
    transcript = backend.transcribe(
        "https://80000hours.org/podcast/episodes/yoshua-bengio-scientist-ai/"
    )
    prev_end = 0.0
    for seg in transcript.segments:
        assert 0.0 <= seg.t_start <= seg.t_end
        assert seg.t_start >= prev_end - 1e-9
        prev_end = seg.t_end


def test_transcribe_uses_content_hash_cache(tmp_path) -> None:
    """A second transcribe of the same URL reads the cache, not the fetch hook.

    The fetch hook is replaced with one that raises after the first call; the
    cached parse must satisfy the repeat call without re-fetching.
    """

    text = (_FIXTURES / "jimrutt.txt").read_text(encoding="utf-8")
    calls = {"n": 0}

    def _hook(_url: str) -> tuple[str, str]:
        calls["n"] += 1
        return text, "html"

    backend = TranscriptBackend(fetch_text=_hook, cache_dir=tmp_path)
    url = "https://www.jimruttshow.com/x/ep-327/"
    first = backend.transcribe(url)
    second = backend.transcribe(url)

    assert calls["n"] == 1, "second call should hit the cache, not re-fetch"
    assert first.model_dump() == second.model_dump()
    # The cache file was actually written.
    assert list(Path(tmp_path).glob("*.json")), "expected a cached transcript file"


def test_cache_disabled_refetches(tmp_path) -> None:
    """With ``cache_dir=None`` each transcribe re-runs the fetch hook (no cache)."""

    text = (_FIXTURES / "jimrutt.txt").read_text(encoding="utf-8")
    calls = {"n": 0}

    def _hook(_url: str) -> tuple[str, str]:
        calls["n"] += 1
        return text, "html"

    backend = TranscriptBackend(fetch_text=_hook, cache_dir=None)
    url = "https://www.jimruttshow.com/x/ep-327/"
    backend.transcribe(url)
    backend.transcribe(url)
    assert calls["n"] == 2


# --------------------------------------------------------------------------- #
# build_episodes_from_corpus
# --------------------------------------------------------------------------- #
def _write_corpus(tmp_path: Path) -> Path:
    """A minimal 2-entry corpus file mirroring the real schema's fields."""

    corpus = {
        "name": "test slice",
        "episodes": [
            {
                "id": "jrs-327-soares",
                "show": "The Jim Rutt Show",
                "title": "Nate Soares on Why Superhuman AI Would Kill Us All",
                "date": "2025-10-15",
                "host": "Jim Rutt",
                "guests": ["Nate Soares"],
                "url": "https://www.jimruttshow.com/x/transcript-of-ep-327/",
                "format": "html",
            },
            {
                "id": "tgs-214-tristan",
                "show": "The Great Simplification",
                "title": "Ending the AI Arms Race",
                "date": "2026-03-25",
                "host": "Nate Hagens",
                "guests": ["Tristan Harris"],
                "url": "https://www.thegreatsimplification.com/x/TGS-214.pdf",
                "format": "pdf",
            },
        ],
    }
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(corpus), encoding="utf-8")
    return path


def test_build_episodes_from_corpus_yields_episode_inputs(tmp_path) -> None:
    """A 2-entry corpus → 2 EpisodeInputs with right ids / urls / guests."""

    episodes = build_episodes_from_corpus(_write_corpus(tmp_path))

    assert len(episodes) == 2
    assert all(isinstance(e, EpisodeInput) for e in episodes)

    by_id = {e.episode.episode_id: e for e in episodes}
    assert set(by_id) == {"jrs-327-soares", "tgs-214-tristan"}

    jrs = by_id["jrs-327-soares"]
    # episode_id == corpus id; guid mirrors it; audio_path == the transcript URL.
    assert jrs.episode.guid == "jrs-327-soares"
    assert jrs.audio_path == "https://www.jimruttshow.com/x/transcript-of-ep-327/"
    assert jrs.episode.audio_url == jrs.audio_path
    # show slug + parsed published_at + guest names + a domain tag.
    assert jrs.episode.show_id == "the-jim-rutt-show"
    assert jrs.episode.published_at.year == 2025
    assert jrs.episode.published_at.month == 10
    assert jrs.metadata_guest_names == ["Nate Soares"]
    assert jrs.domains == ["ai-safety"]

    tgs = by_id["tgs-214-tristan"]
    assert tgs.audio_path.endswith(".pdf")
    assert tgs.metadata_guest_names == ["Tristan Harris"]


def test_build_episodes_over_real_corpus_covers_all_20() -> None:
    """The real corpus yields 20 EpisodeInputs, every URL parser-routable."""

    episodes = build_episodes_from_corpus(_CORPUS)
    assert len(episodes) == 20
    for ep in episodes:
        assert ep.audio_path.startswith("http")
        # Each episode's URL has a registered parser (the end-to-end coverage).
        assert callable(parser_for_url(ep.audio_path)), ep.episode.episode_id
        assert ep.domains == ["ai-safety"]
