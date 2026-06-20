"""Phase 3 Task 3.3 — offline test for the 20-episode KG slice driver.

Drives the ``run_kg_slice`` core (the injectable seam mirroring
``smoke_one_episode.run_smoke``) over **two** fake episodes that share a subject
("Apple" / "Apple Inc.", which the conftest fake embedder places at cosine
~1.0) and a host. With an :class:`~dlogos.resolution.incremental.IncrementalResolver`
over an :class:`~dlogos.resolution.canonical_store.InMemoryCanonicalStore`, an
in-memory speaker store + :class:`~dlogos.speakers.speaker_store.NameSpeakerResolver`,
word-level re-segmentation, and a :class:`~dlogos.graph.fake_store.FakeGraphStore`,
all injected through **one** :class:`~dlogos.pipeline.Pipeline.run`, the slice:

- resolves the shared subject to **one** canonical Apple node across episodes,
- gives the self-introducing host **one** ``speaker_id`` across episodes,
- derives a cross-episode ``disputes`` (or ``supersedes``) edge (two speakers on
  the one canonical subject, opposite stance), and
- yields a non-empty :class:`~dlogos.eval.fragmentation.FragReport` over a small
  built-in probe set.

Everything is injected: an in-memory speaker store (a stand-in for the sqlite
store, satisfying the same :class:`~dlogos.speakers.speaker_store.CanonicalSpeakerStore`
Protocol), a :class:`~dlogos.asr.mock_backend.MockASRBackend` carrying
word-level timings, a fake async extraction client, and the conftest fake
embedder. No network, no heavy deps — the live ``--manifest`` run is the user's
next step, not this test's.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from dlogos.eval.fragmentation import FragReport, Probe
from dlogos.extraction.extractor import ClaimExtractor
from dlogos.graph.fake_store import FakeGraphStore
from dlogos.graph.store import EdgeType
from dlogos.resolution.canonical_store import InMemoryCanonicalStore
from dlogos.resolution.incremental import IncrementalResolver
from dlogos.schema import (
    Episode,
    Transcript,
    TranscriptSegment,
    Word,
)
from dlogos.speakers.identity import CanonicalSpeaker
from dlogos.speakers.speaker_store import NameSpeakerResolver

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
_SCRIPT_PATH = _SCRIPTS_DIR / "run_kg_slice.py"


def _load_script_module():
    """Load ``scripts/run_kg_slice.py`` as an importable module.

    The script does sibling imports (``run_smoke_inmemory`` / ``smoke_one_episode``)
    that resolve from ``scripts/`` being on ``sys.path`` — exactly as when it is
    run as ``python scripts/run_kg_slice.py`` (``scripts/`` is then ``sys.path[0]``).
    We prepend it here so those imports succeed under pytest. Registration in
    ``sys.modules`` before ``exec_module`` lets the module's dataclasses resolve
    their string annotations (``from __future__ import annotations``).
    """

    if str(_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location("run_kg_slice", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_kg_slice"] = module
    spec.loader.exec_module(module)
    return module


_kg = _load_script_module()
run_kg_slice = _kg.run_kg_slice


# --------------------------------------------------------------------------- #
# In-memory speaker store (a CanonicalSpeakerStore for the offline test)
# --------------------------------------------------------------------------- #
class _InMemorySpeakerStore:
    """Process-local :class:`CanonicalSpeakerStore` for the offline driver test.

    Mirrors :class:`~dlogos.speakers.speaker_store.SqliteSpeakerStore`'s
    resolve-or-mint contract (QID first, then normalized name) without touching
    a file, so the test stays fully in memory while exercising the same
    cross-episode name->id stability the sqlite store provides at runtime.
    """

    def __init__(self) -> None:
        from dlogos.speakers.speaker_store import _mint_speaker_id, _normalize_name

        self._mint = _mint_speaker_id
        self._norm = _normalize_name
        self._by_id: dict[str, CanonicalSpeaker] = {}

    def canonical_for(
        self, *, name: str | None = None, qid: str | None = None
    ) -> CanonicalSpeaker:
        if not name and not qid:
            raise ValueError("canonical_for requires a name or a qid")
        norm = self._norm(name) if name else None
        speaker_id = self._mint(norm_name=norm or "", qid=qid)
        existing = self._by_id.get(speaker_id)
        if existing is not None:
            return existing
        speaker = CanonicalSpeaker(
            speaker_id=speaker_id, name=name or (qid or speaker_id), wikidata_qid=qid
        )
        self._by_id[speaker_id] = speaker
        return speaker

    def get(self, speaker_id: str) -> CanonicalSpeaker | None:
        return self._by_id.get(speaker_id)

    def all(self) -> list[CanonicalSpeaker]:
        return list(self._by_id.values())


# --------------------------------------------------------------------------- #
# Fakes / builders
# --------------------------------------------------------------------------- #
class _FakeExtractionClient:
    """Async OpenAI-compatible client returning a canned, chunk-grounded claim."""

    def __init__(self, payload: str) -> None:
        self._payload = payload

    @property
    def chat(self):
        outer = self

        class _Completions:
            async def create(self, **kwargs: object):
                return {"choices": [{"message": {"content": outer._payload}}]}

        class _Chat:
            completions = _Completions()

        return _Chat()


class _RoutingASR:
    """ASR that returns a different transcript per ``audio_path`` (one backend).

    The driver runs ONE ``Pipeline.run`` over the whole slice, so a single ASR
    must serve every episode. The pipeline binds each transcript to its
    episode id by audio path, so dispatching on ``audio_path`` is enough.
    """

    def __init__(self, by_path: dict[str, Transcript]) -> None:
        self._by_path = by_path

    def transcribe(self, audio_path: str) -> Transcript:
        return self._by_path[audio_path]


class _RoutingExtractor:
    """Extractor that returns a different canned claim per ``chunk.episode_id``.

    A single extractor serves the whole batch (one ``Pipeline.run``); chunks
    carry ``episode_id``, so we dispatch on the first chunk's episode to give
    each episode its own opposing claim — the two-speaker, opposite-polarity
    condition the cross-episode ``disputes`` derivation needs.
    """

    def __init__(self, by_episode: dict[str, ClaimExtractor]) -> None:
        self._by_episode = by_episode

    async def extract_many(self, chunks):
        if not chunks:
            return []
        episode_id = chunks[0].episode_id
        return await self._by_episode[episode_id].extract_many(chunks)


def _payload(
    *,
    speaker_label: str,
    subject: str,
    stance: str,
    sentiment: float,
    obj: str,
    t_start: float,
    t_end: float,
) -> str:
    """One claim about ``subject`` spoken by ``speaker_label`` with a stance.

    The two episodes carry opposite-polarity claims on the SAME canonical subject
    by DIFFERENT speakers, which is exactly the condition that derives a
    cross-episode ``disputes`` edge in the loader's relation pass. ``obj`` echoes
    the speaker's transcript line so grounding snaps the claim to that segment,
    and ``[t_start, t_end]`` lies inside the chunk window (an out-of-window span
    is dropped by the extractor as a fabricated citation).
    """

    return json.dumps(
        {
            "claims": [
                {
                    "speaker_label": speaker_label,
                    "predicate": (
                        "rates_negative" if sentiment < 0 else "rates_positive"
                    ),
                    "subject": subject,
                    "subject_type": "organization",
                    "object": obj,
                    "stance": stance,
                    "sentiment": sentiment,
                    "confidence": 0.82,
                    "t_start": t_start,
                    "t_end": t_end,
                }
            ]
        }
    )


def _words(speaker: str, text: str, t0: float) -> list[Word]:
    """Evenly-spaced word stream for ``text`` on ``speaker`` starting at ``t0``."""

    toks = text.split()
    step = 0.4
    return [
        Word(
            text=tok,
            t_start=t0 + i * step,
            t_end=t0 + (i + 1) * step,
            speaker=speaker,
        )
        for i, tok in enumerate(toks)
    ]


def _transcript(episode_id: str, *, host_line: str, other_line: str) -> Transcript:
    """Two-speaker transcript carrying word-level timings (so resegment fires)."""

    host_words = _words("SPEAKER_00", host_line, 0.0)
    other_words = _words("SPEAKER_01", other_line, host_words[-1].t_end + 0.1)
    return Transcript(
        episode_id=episode_id,
        language="en",
        segments=[
            TranscriptSegment(
                speaker="SPEAKER_00",
                text=host_line,
                t_start=host_words[0].t_start,
                t_end=host_words[-1].t_end,
            ),
            TranscriptSegment(
                speaker="SPEAKER_01",
                text=other_line,
                t_start=other_words[0].t_start,
                t_end=other_words[-1].t_end,
            ),
        ],
        words=[*host_words, *other_words],
        duration_s=other_words[-1].t_end,
    )


def _episode(episode_id: str, *, month: int) -> Episode:
    return Episode(
        episode_id=episode_id,
        show_id="show-tech",
        guid=f"guid-{episode_id}",
        title="t",
        published_at=datetime(2026, month, 10, tzinfo=timezone.utc),
        audio_url="https://example.invalid/a.mp3",
    )


# --------------------------------------------------------------------------- #
# The offline driver test
# --------------------------------------------------------------------------- #
async def test_run_kg_slice_unifies_entity_and_speaker_with_dispute_edge(
    fake_embedder, tmp_path
):
    """Two fake episodes through ONE Pipeline.run over shared persistent stores.

    ep1: the host (SPEAKER_00, "Darian Woods") asserts POSITIVELY about "Apple".
    ep2: the host re-intros, and a guest (SPEAKER_01, "Cardiff Garcia") disputes
    "Apple Inc." NEGATIVELY. The incremental resolver collapses Apple/Apple Inc.
    to one canonical id; the two opposing claims sit in the SAME load batch (one
    Pipeline.run), are by DIFFERENT resolved speakers on the one canonical
    subject with opposite polarity -> a cross-episode ``disputes`` edge.
    """

    entity_store = InMemoryCanonicalStore()
    speaker_store = _InMemorySpeakerStore()
    subject_resolver = IncrementalResolver(entity_store, fake_embedder)
    speaker_resolver = NameSpeakerResolver(speaker_store)
    store = FakeGraphStore()

    ep1_transcript = _transcript(
        "ep-0001",
        host_line="I'm Darian Woods. Apple makes the best hardware around.",
        other_line="Sure thing, glad to hear it.",
    )
    ep2_transcript = _transcript(
        "ep-0002",
        host_line="I'm Darian Woods, welcome back to the show today.",
        other_line="I'm Cardiff Garcia. the iPhone is failing on hardware now.",
    )

    # ONE ASR + ONE extractor serve the whole slice (the driver does a single
    # Pipeline.run); they route by audio_path / episode_id to give each episode
    # its own transcript and opposing claim. The object echoes the speaker's line
    # so grounding snaps the claim to that segment; the span stays inside the
    # chunk window so the extractor does not drop it as a fabricated citation.
    asr = _RoutingASR(
        {"ep-0001.mp3": ep1_transcript, "ep-0002.mp3": ep2_transcript}
    )
    extractor = _RoutingExtractor(
        {
            "ep-0001": ClaimExtractor(
                _FakeExtractionClient(
                    _payload(
                        speaker_label="SPEAKER_00",
                        subject="Apple",
                        stance="asserts",
                        sentiment=0.7,
                        obj="Apple makes the best hardware around",
                        t_start=1.2,
                        t_end=3.6,
                    )
                )
            ),
            "ep-0002": ClaimExtractor(
                _FakeExtractionClient(
                    _payload(
                        speaker_label="SPEAKER_01",
                        subject="the iPhone",
                        stance="disputes",
                        sentiment=-0.7,
                        obj="the iPhone is failing on hardware now",
                        t_start=3.6,
                        t_end=7.0,
                    )
                )
            ),
        }
    )

    episodes = [
        {"episode": _episode("ep-0001", month=1), "audio_path": "ep-0001.mp3"},
        {"episode": _episode("ep-0002", month=3), "audio_path": "ep-0002.mp3"},
    ]

    probes = [
        Probe(name="Apple", aliases=["apple", "apple inc", "the iphone maker"]),
        Probe(name="OpenAI", aliases=["openai"]),
    ]

    result = await run_kg_slice(
        episodes=episodes,
        embedder=fake_embedder,
        store=store,
        subject_resolver=subject_resolver,
        speaker_resolver=speaker_resolver,
        probes=probes,
        asr=asr,
        extractor=extractor,
        graph_out=tmp_path / "graph.json",
    )

    # One canonical Apple node across both episodes.
    apple_probe = next(
        r for r in result.frag_report.per_probe if r.probe.name == "Apple"
    )
    assert apple_probe.fragments == 1, apple_probe.canonical_ids

    # One host speaker_id across episodes (the self-introducing host).
    host_res = [
        run.speaker_resolutions.get("SPEAKER_00") for run in result.pipeline.runs
    ]
    assert all(res is not None and res.is_resolved for res in host_res)
    host_ids = {res.resolved.speaker_id for res in host_res}
    assert len(host_ids) == 1, host_ids

    # A cross-episode disputes (or supersedes) edge is present.
    derived_types = {e.type for e in store.edges.values()}
    assert (
        EdgeType.disputes in derived_types or EdgeType.supersedes in derived_types
    ), derived_types

    # Non-empty fragmentation report.
    assert isinstance(result.frag_report, FragReport)
    assert result.frag_report.per_probe, "expected a per-probe fragmentation report"

    # Both episodes loaded their one claim each.
    assert result.pipeline.claims_loaded == 2

    # The graph was exported for the viewer.
    assert result.graph_path is not None and result.graph_path.exists()
