"""Tests for host-anchored cross-episode speaker identity.

Deterministic throughout: a tiny in-test :class:`FakeVoiceEmbedder` maps
voice-sample keys to fixed unit vectors, so the gallery's cosine matching is
exercised with no model and no network. The headline case — the host resolves,
an unknown speaker stays per-episode — is covered directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from dlogos.schema import SpeakerRef, Transcript, TranscriptSegment
from dlogos.speakers.identity import (
    CanonicalSpeaker,
    HostGallery,
    HostSeed,
    SpeakerResolution,
    VoiceEmbedder,
)


class FakeVoiceEmbedder:
    """Maps voice-sample keys to fixed 4-D unit vectors.

    Two well-separated directions stand in for two distinct voices; an
    "unknown" voice points in a third orthogonal direction so it matches
    neither host. Unknown keys fall back to a stable hashed vector.
    """

    _TABLE: dict[str, list[float]] = {
        # Host A reference + that host's turns in two episodes (slight jitter).
        "ref/hostA": [1.0, 0.0, 0.0, 0.0],
        "ep1/SPEAKER_00": [0.99, 0.10, 0.0, 0.0],
        "ep2/SPEAKER_01": [0.98, 0.0, 0.10, 0.0],
        # Host B reference.
        "ref/hostB": [0.0, 1.0, 0.0, 0.0],
        "ep1/SPEAKER_02": [0.10, 0.97, 0.0, 0.0],
        # An unknown guest voice — orthogonal to both hosts.
        "ep1/SPEAKER_01": [0.0, 0.0, 0.0, 1.0],
    }

    def embed(self, sample_ref: str) -> list[float]:
        if sample_ref in self._TABLE:
            v = np.asarray(self._TABLE[sample_ref], dtype=float)
            return (v / (np.linalg.norm(v) or 1.0)).tolist()
        rng = np.random.default_rng(abs(hash(sample_ref)) % (2**32))
        v = rng.standard_normal(4)
        return (v / (np.linalg.norm(v) or 1.0)).tolist()


@pytest.fixture
def voice_embedder() -> FakeVoiceEmbedder:
    return FakeVoiceEmbedder()


@pytest.fixture
def gallery(voice_embedder: FakeVoiceEmbedder) -> HostGallery:
    hosts = [
        HostSeed(
            speaker_id="spk-hostA",
            name="Host A",
            show_id="show-1",
            sample_refs=("ref/hostA",),
        ),
        HostSeed(
            speaker_id="spk-hostB",
            name="Host B",
            show_id="show-2",
            sample_refs=("ref/hostB",),
        ),
    ]
    return HostGallery.from_hosts(hosts, voice_embedder, threshold=0.75, margin=0.05)


# --------------------------------------------------------------------------- #
# Protocol / construction
# --------------------------------------------------------------------------- #
def test_fake_embedder_satisfies_protocol(voice_embedder: FakeVoiceEmbedder) -> None:
    assert isinstance(voice_embedder, VoiceEmbedder)


def test_gallery_seeds_one_centroid_per_host(gallery: HostGallery) -> None:
    assert set(gallery.speakers) == {"spk-hostA", "spk-hostB"}
    assert gallery.speaker("spk-hostA").is_host is True
    assert gallery.speaker("spk-hostA").name == "Host A"


# --------------------------------------------------------------------------- #
# Headline: host resolves; unknown stays per-episode
# --------------------------------------------------------------------------- #
def test_host_label_resolves_to_canonical_host(gallery: HostGallery) -> None:
    res = gallery.resolve_label("SPEAKER_00", "ep1/SPEAKER_00")
    assert res.is_resolved
    assert res.resolved.speaker_id == "spk-hostA"
    assert res.score >= 0.75


def test_unknown_speaker_stays_per_episode(gallery: HostGallery) -> None:
    res = gallery.resolve_label("SPEAKER_01", "ep1/SPEAKER_01")
    assert not res.is_resolved
    assert res.resolved is None


def test_resolution_projects_to_speaker_ref(gallery: HostGallery) -> None:
    resolved = gallery.resolve_label("SPEAKER_00", "ep1/SPEAKER_00").to_speaker_ref()
    assert isinstance(resolved, SpeakerRef)
    assert resolved.resolved_id == "spk-hostA"
    assert resolved.name == "Host A"

    bare = gallery.resolve_label("SPEAKER_01", "ep1/SPEAKER_01").to_speaker_ref()
    assert bare.label == "SPEAKER_01"
    assert bare.resolved_id is None
    assert bare.name is None


# --------------------------------------------------------------------------- #
# Cross-episode: the same host resolves in two different episodes
# --------------------------------------------------------------------------- #
def test_host_resolves_across_two_episodes(gallery: HostGallery) -> None:
    ep1 = gallery.resolve_label("SPEAKER_00", "ep1/SPEAKER_00")
    ep2 = gallery.resolve_label("SPEAKER_01", "ep2/SPEAKER_01")
    # Same canonical host id despite different per-episode diarization labels.
    assert ep1.resolved.speaker_id == ep2.resolved.speaker_id == "spk-hostA"


# --------------------------------------------------------------------------- #
# Whole-transcript resolution
# --------------------------------------------------------------------------- #
def test_resolve_transcript_resolves_host_only() -> None:
    embedder = FakeVoiceEmbedder()
    gallery = HostGallery.from_hosts(
        [
            HostSeed(
                speaker_id="spk-hostA",
                name="Host A",
                show_id="show-1",
                sample_refs=("ref/hostA",),
            )
        ],
        embedder,
    )
    transcript = Transcript(
        episode_id="ep1",
        language="en",
        duration_s=20.0,
        segments=[
            TranscriptSegment(speaker="SPEAKER_00", text="Welcome.", t_start=0.0, t_end=4.0),
            TranscriptSegment(speaker="SPEAKER_01", text="Hi.", t_start=4.0, t_end=8.0),
        ],
    )
    out = gallery.resolve_transcript(
        transcript,
        sample_refs={"SPEAKER_00": "ep1/SPEAKER_00", "SPEAKER_01": "ep1/SPEAKER_01"},
    )
    assert out["SPEAKER_00"].resolved.speaker_id == "spk-hostA"
    assert out["SPEAKER_01"].resolved is None


def test_label_without_sample_is_unresolved(gallery: HostGallery) -> None:
    transcript = Transcript(
        episode_id="ep1",
        language="en",
        duration_s=4.0,
        segments=[
            TranscriptSegment(speaker="SPEAKER_09", text="Mystery.", t_start=0.0, t_end=4.0)
        ],
    )
    out = gallery.resolve_transcript(transcript, sample_refs={})
    assert out["SPEAKER_09"].resolved is None
    assert out["SPEAKER_09"].score == 0.0


# --------------------------------------------------------------------------- #
# Conservative matching: ambiguity and empty gallery
# --------------------------------------------------------------------------- #
def test_ambiguous_match_within_margin_left_unresolved() -> None:
    """Two near-identical host centroids → no confident pick."""

    class TwinVoices:
        def embed(self, ref: str) -> list[float]:
            table = {
                "ref/a": [1.0, 0.0, 0.0],
                "ref/b": [0.999, 0.0447, 0.0],  # almost the same direction
                "probe": [1.0, 0.0, 0.0],
            }
            v = np.asarray(table[ref], dtype=float)
            return (v / np.linalg.norm(v)).tolist()

    gallery = HostGallery.from_hosts(
        [
            HostSeed(speaker_id="a", name="A", show_id="s", sample_refs=("ref/a",)),
            HostSeed(speaker_id="b", name="B", show_id="s", sample_refs=("ref/b",)),
        ],
        TwinVoices(),
        threshold=0.5,
        margin=0.05,
    )
    res = gallery.resolve_label("SPEAKER_00", "probe")
    # Best score clears threshold but runner-up is within margin → unresolved.
    assert res.resolved is None
    assert res.score >= 0.5


def test_empty_gallery_resolves_nothing() -> None:
    gallery = HostGallery(embedder=FakeVoiceEmbedder())
    res = gallery.resolve_voiceprint([1.0, 0.0, 0.0, 0.0], label="SPEAKER_00")
    assert res.resolved is None
    assert res.score == 0.0


def test_below_threshold_left_unresolved(gallery: HostGallery) -> None:
    # A vector orthogonal to every host centroid scores ~0.
    res = gallery.resolve_voiceprint([0.0, 0.0, 0.0, 1.0], label="SPEAKER_X")
    assert res.resolved is None
    assert res.score < 0.75


# --------------------------------------------------------------------------- #
# Same host across two shows merges into one centroid + unioned shows
# --------------------------------------------------------------------------- #
def test_same_host_two_shows_merges() -> None:
    embedder = FakeVoiceEmbedder()
    gallery = HostGallery.from_hosts(
        [
            HostSeed(speaker_id="spk-hostA", name="Host A", show_id="show-1", sample_refs=("ref/hostA",)),
            HostSeed(speaker_id="spk-hostA", name="Host A", show_id="show-9", sample_refs=("ref/hostA",)),
        ],
        embedder,
    )
    speaker = gallery.speaker("spk-hostA")
    assert speaker is not None
    assert set(speaker.show_ids) == {"show-1", "show-9"}
    # Still resolves.
    assert gallery.resolve_label("SPEAKER_00", "ep1/SPEAKER_00").resolved.speaker_id == "spk-hostA"


def test_canonical_speaker_is_immutable() -> None:
    spk = CanonicalSpeaker(speaker_id="x", name="X")
    with pytest.raises(Exception):
        spk.name = "Y"  # type: ignore[misc]


def test_speaker_resolution_dataclass_fields() -> None:
    res = SpeakerResolution(label="SPEAKER_00", resolved=None, score=0.3)
    assert res.label == "SPEAKER_00"
    assert res.is_resolved is False
