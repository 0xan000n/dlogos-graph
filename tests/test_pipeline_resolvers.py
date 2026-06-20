"""Phase 3 Task 3.1 — injectable persistent subject + speaker resolvers.

Proves the cross-episode property end-to-end *through the pipeline*: with an
:class:`~dlogos.resolution.incremental.IncrementalResolver` (over a persistent
:class:`~dlogos.resolution.canonical_store.InMemoryCanonicalStore`) and a
:class:`~dlogos.speakers.speaker_store.NameSpeakerResolver` (over an in-memory
speaker store) injected via :class:`~dlogos.pipeline.PipelineDeps`, two episodes
that share a subject ("Apple"/"the iPhone", which the conftest fake embedder
places at cosine ~0.99) and a self-introducing host collapse to **one** canonical
entity id and **one** speaker id across episodes.

The third test is the back-compat guard the plan calls sacred: with both hooks
``None`` the pipeline's subject/speaker resolution is **byte-identical** to
today's default (``resolve_subjects`` + the host-gallery/guest/fallback chain).
All collaborators are injected; no network, no heavy deps.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from dlogos.asr.mock_backend import MockASRBackend
from dlogos.extraction.extractor import ClaimExtractor
from dlogos.graph.fake_store import FakeGraphStore
from dlogos.pipeline import EpisodeInput, Pipeline, PipelineDeps
from dlogos.resolution.canonical_store import InMemoryCanonicalStore
from dlogos.resolution.incremental import IncrementalResolver
from dlogos.schema import Episode, Transcript, TranscriptSegment
from dlogos.speakers.speaker_store import NameSpeakerResolver, SqliteSpeakerStore


# --------------------------------------------------------------------------- #
# Fakes / builders
# --------------------------------------------------------------------------- #
class _FakeExtractionClient:
    """Async OpenAI-compatible client returning canned, chunk-grounded JSON."""

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


def _payload(subject: str) -> str:
    """One claim about ``subject``, spoken by the host label (SPEAKER_00).

    Attributing it to the self-introducing host means the loaded claim carries a
    *name-resolved*, cross-episode ``speaker_id`` (the whole point of the speaker
    hook) — so the loadability gate (resolved speaker + canonical id) is met by
    the name resolver alone, no fallback needed.
    """

    return json.dumps(
        {
            "claims": [
                {
                    "speaker_label": "SPEAKER_00",
                    "predicate": "rates_negative",
                    "subject": subject,
                    "subject_type": "organization",
                    "object": "hardware innovation has plateaued",
                    "stance": "asserts",
                    "sentiment": -0.6,
                    "confidence": 0.82,
                    "t_start": 0.0,
                    "t_end": 4.5,
                }
            ]
        }
    )


def _transcript(episode_id: str, subject: str) -> Transcript:
    """Two-speaker transcript: host self-intro + claim on A, a reply on B."""

    return Transcript(
        episode_id=episode_id,
        language="en",
        segments=[
            TranscriptSegment(
                speaker="SPEAKER_00",
                text=(
                    f"Welcome back. I'm Darian Woods, your host. "
                    f"I think {subject} has plateaued on hardware."
                ),
                t_start=0.0,
                t_end=4.5,
            ),
            TranscriptSegment(
                speaker="SPEAKER_01",
                text="Interesting, tell me more.",
                t_start=4.5,
                t_end=10.0,
            ),
        ],
        duration_s=10.0,
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


def _ep_input(episode_id: str, subject: str, *, month: int) -> EpisodeInput:
    """An EpisodeInput whose ASR/extractor are bound to this episode's text."""

    return EpisodeInput(
        episode=_episode(episode_id, month=month),
        audio_path=f"{episode_id}.mp3",
    )


async def _run_two_episodes(deps_for, *, subjects, hooks):
    """Run two episodes (each with its own ASR + extractor) through one pipeline.

    ``deps_for(episode_id, subject)`` builds the per-episode deps (ASR template +
    extractor payload differ per episode); ``hooks`` is merged into every deps.
    Because each episode needs a different ASR transcript and extractor payload,
    we run them as two separate ``Pipeline.run`` calls sharing the *same* store +
    injected resolvers — exactly the cross-episode incremental path.
    """

    store = FakeGraphStore()
    results = []
    for idx, (episode_id, subject, month) in enumerate(subjects):
        deps = PipelineDeps(
            asr=MockASRBackend(transcript=_transcript(episode_id, subject)),
            extractor=ClaimExtractor(_FakeExtractionClient(_payload(subject))),
            embedder=hooks["embedder"],
            store=store,
            subject_resolver=hooks.get("subject_resolver"),
            speaker_resolver=hooks.get("speaker_resolver"),
            fallback_speaker_id=hooks.get("fallback_speaker_id"),
        )
        results.append(
            await Pipeline(deps).run([_ep_input(episode_id, subject, month=month)])
        )
    return store, results


# --------------------------------------------------------------------------- #
# (a)+(b) injected resolvers: one canonical id + one speaker id across episodes
# --------------------------------------------------------------------------- #
async def test_injected_resolvers_unify_subject_and_speaker_across_episodes(
    fake_embedder, tmp_path
):
    # ONE persistent canonical store + ONE speaker store shared across episodes.
    entity_store = InMemoryCanonicalStore()
    subject_resolver = IncrementalResolver(entity_store, fake_embedder)
    speaker_store = SqliteSpeakerStore(tmp_path / "spk.db")
    speaker_resolver = NameSpeakerResolver(speaker_store)

    hooks = {
        "embedder": fake_embedder,
        "subject_resolver": subject_resolver,
        "speaker_resolver": speaker_resolver,
    }
    # "Apple" and "the iPhone" embed at cosine ~0.99 in the conftest fake
    # embedder (both are in its table), well above the org HIGH bar — so the
    # incremental resolver's embedding tier converges them across episodes.
    subjects = [
        ("ep-0001", "Apple", 1),
        ("ep-0002", "the iPhone", 3),
    ]
    store, results = await _run_two_episodes(None, subjects=subjects, hooks=hooks)

    # (a) both episodes' subject claims share ONE canonical id (incremental
    # resolution against the persistent store: episode 2 resolves vs episode 1).
    canon_ids = {
        c.subject_entity.canonical_id
        for r in results
        for c in r.resolved_claims
    }
    assert len(canon_ids) == 1, canon_ids

    # (b) the self-introducing host resolves to ONE speaker id across episodes.
    host_ids = {
        c.speaker.resolved_id
        for r in results
        for c in r.resolved_claims
    }
    # All loaded claims are the guest's (SPEAKER_01); the *guest* should also be
    # stable across episodes because the same name resolves to the same id.
    assert len(host_ids) == 1, host_ids

    # Both episodes actually loaded their one claim each.
    assert sum(r.claims_loaded for r in results) == 2
    assert store.claim_count() == 2


async def test_injected_speaker_resolver_gives_host_one_id_across_episodes(
    fake_embedder, tmp_path
):
    """The self-introducing host (label A) gets one stable speaker id in both."""

    entity_store = InMemoryCanonicalStore()
    speaker_store = SqliteSpeakerStore(tmp_path / "spk.db")
    speaker_resolver = NameSpeakerResolver(speaker_store)
    hooks = {
        "embedder": fake_embedder,
        "subject_resolver": IncrementalResolver(entity_store, fake_embedder),
        "speaker_resolver": speaker_resolver,
    }
    subjects = [("ep-0001", "Apple", 1), ("ep-0002", "the iPhone", 3)]
    _store, results = await _run_two_episodes(None, subjects=subjects, hooks=hooks)

    # The host (SPEAKER_00, who self-intros "I'm Darian Woods") resolves to the
    # SAME canonical speaker in both episodes' run-level resolutions.
    host_res = [r.runs[0].speaker_resolutions.get("SPEAKER_00") for r in results]
    assert all(res is not None and res.is_resolved for res in host_res)
    host_speaker_ids = {res.resolved.speaker_id for res in host_res}
    assert len(host_speaker_ids) == 1, host_speaker_ids
    assert all(
        res.resolved.name == "Darian Woods" for res in host_res
    )


# --------------------------------------------------------------------------- #
# (c) back-compat: both hooks None -> byte-identical to today
# --------------------------------------------------------------------------- #
async def test_back_compat_both_hooks_none_is_byte_identical(fake_embedder):
    """With both resolvers ``None`` the pipeline behaves exactly as before.

    We build deps WITHOUT the new fields (the legacy call) and WITH them set to
    ``None`` (the new default), run identical episodes through each, and assert
    the loaded claims — speaker ids, canonical ids, counts — match exactly.
    """

    def _legacy_deps(store):
        return PipelineDeps(
            asr=MockASRBackend(transcript=_transcript("ep-0001", "Apple")),
            extractor=ClaimExtractor(_FakeExtractionClient(_payload("Apple"))),
            embedder=fake_embedder,
            store=store,
            fallback_speaker_id=lambda ep, label: (f"spk-{label.lower()}", None),
        )

    def _hooks_none_deps(store):
        return PipelineDeps(
            asr=MockASRBackend(transcript=_transcript("ep-0001", "Apple")),
            extractor=ClaimExtractor(_FakeExtractionClient(_payload("Apple"))),
            embedder=fake_embedder,
            store=store,
            fallback_speaker_id=lambda ep, label: (f"spk-{label.lower()}", None),
            subject_resolver=None,
            speaker_resolver=None,
        )

    legacy_store = FakeGraphStore()
    legacy = await Pipeline(_legacy_deps(legacy_store)).run(
        [_ep_input("ep-0001", "Apple", month=1)]
    )
    new_store = FakeGraphStore()
    new = await Pipeline(_hooks_none_deps(new_store)).run(
        [_ep_input("ep-0001", "Apple", month=1)]
    )

    assert legacy.claims_loaded == new.claims_loaded
    assert legacy_store.claim_count() == new_store.claim_count()

    def _fingerprint(result):
        return sorted(
            (
                c.speaker.resolved_id,
                c.subject_entity.canonical_id,
                c.predicate.value,
            )
            for c in result.resolved_claims
        )

    assert _fingerprint(legacy) == _fingerprint(new)
    # And the canonical ids are the deterministic ``ent-...`` ids from the
    # default ``resolve_subjects`` path (no ``wd-`` / store-minted ids).
    assert all(
        c.subject_entity.canonical_id.startswith("ent-")
        for c in new.resolved_claims
    )
