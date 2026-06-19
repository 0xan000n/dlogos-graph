"""Tests for the corpus-manifest -> HostGallery seeding adapter.

Deterministic: a tiny in-test ``FakeVoiceEmbedder`` maps reference-audio refs
to fixed unit vectors, so seeding + cosine matching run with no model and no
network. The headline cases — the adapter seeds the expected hosts, and a
host's ``reference_audio`` flows into the gallery as a matchable voiceprint —
are covered directly. The real ``ManifestRow`` is used end-to-end to prove the
duck-typed adapter accepts it.
"""

from __future__ import annotations

import numpy as np

from dlogos.ingestion.manifest import ManifestRow
from dlogos.speakers.identity import (
    HostGallery,
    HostSeed,
    host_speaker_id,
    seeds_from_manifest_rows,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeVoiceEmbedder:
    """Maps reference-audio refs (and episode turns) to fixed 3-D vectors."""

    _TABLE: dict[str, list[float]] = {
        "ref/alice.wav": [1.0, 0.0, 0.0],
        "ep1/SPEAKER_00": [0.99, 0.10, 0.0],  # Alice in an episode (jittered)
        "ref/bob.wav": [0.0, 1.0, 0.0],
        "ep1/SPEAKER_01": [0.0, 0.0, 1.0],  # unknown voice, orthogonal
    }

    def embed(self, sample_ref: str) -> list[float]:
        if sample_ref in self._TABLE:
            v = np.asarray(self._TABLE[sample_ref], dtype=float)
            return (v / (np.linalg.norm(v) or 1.0)).tolist()
        rng = np.random.default_rng(abs(hash(sample_ref)) % (2**32))
        v = rng.standard_normal(3)
        return (v / (np.linalg.norm(v) or 1.0)).tolist()


def _manifest_rows() -> list[ManifestRow]:
    return [
        ManifestRow(
            show_id="show-1",
            feed_url="https://f/1.xml",
            known_hosts=["Alice", "Bob"],
            reference_audio={"Alice": "ref/alice.wav"},  # Bob has no ref
        ),
        ManifestRow(
            show_id="show-2",
            feed_url="https://f/2.xml",
            known_hosts=["Carol"],
        ),
    ]


# --------------------------------------------------------------------------- #
# Stable id derivation
# --------------------------------------------------------------------------- #
def test_host_speaker_id_is_stable_and_slugged() -> None:
    assert host_speaker_id("Alice") == "host-alice"
    assert host_speaker_id("Jane Doe") == "host-jane-doe"
    # Same name -> same id regardless of which show seeded it.
    assert host_speaker_id("Alice") == host_speaker_id("Alice")


# --------------------------------------------------------------------------- #
# Adapter: rows -> HostSeeds
# --------------------------------------------------------------------------- #
def test_adapter_seeds_one_seed_per_named_host() -> None:
    seeds = seeds_from_manifest_rows(_manifest_rows())
    by_name = {s.name: s for s in seeds}
    assert set(by_name) == {"Alice", "Bob", "Carol"}


def test_adapter_pulls_sample_refs_from_reference_audio() -> None:
    seeds = {s.name: s for s in seeds_from_manifest_rows(_manifest_rows())}
    # Alice has a reference -> seeded as a sample ref.
    assert seeds["Alice"].sample_refs == ("ref/alice.wav",)
    # Bob and Carol have no reference -> empty (still a canonical speaker).
    assert seeds["Bob"].sample_refs == ()
    assert seeds["Carol"].sample_refs == ()


def test_adapter_stamps_show_id_and_stable_speaker_id() -> None:
    seeds = {s.name: s for s in seeds_from_manifest_rows(_manifest_rows())}
    assert seeds["Alice"].show_id == "show-1"
    assert seeds["Alice"].speaker_id == "host-alice"
    assert seeds["Carol"].show_id == "show-2"


# --------------------------------------------------------------------------- #
# Headline: manifest -> gallery seeds the expected hosts
# --------------------------------------------------------------------------- #
def test_from_manifest_rows_seeds_expected_hosts() -> None:
    gallery = HostGallery.from_manifest_rows(_manifest_rows(), FakeVoiceEmbedder())
    assert set(gallery.speakers) == {"host-alice", "host-bob", "host-carol"}
    assert gallery.speaker("host-alice").is_host is True
    assert gallery.speaker("host-alice").name == "Alice"


def test_seeded_host_with_reference_is_voiceprint_matchable() -> None:
    gallery = HostGallery.from_manifest_rows(_manifest_rows(), FakeVoiceEmbedder())
    # Alice was seeded with a reference, so her episode turn resolves to her.
    res = gallery.resolve_label("SPEAKER_00", "ep1/SPEAKER_00")
    assert res.is_resolved
    assert res.resolved.speaker_id == "host-alice"


def test_host_without_reference_defines_speaker_but_no_centroid() -> None:
    gallery = HostGallery.from_manifest_rows(_manifest_rows(), FakeVoiceEmbedder())
    # Bob is a canonical speaker...
    assert gallery.speaker("host-bob") is not None
    # ...but with no reference there is no centroid, so an unknown voice that
    # happens to be near nothing stays unresolved.
    res = gallery.resolve_label("SPEAKER_01", "ep1/SPEAKER_01")
    assert not res.is_resolved


# --------------------------------------------------------------------------- #
# A host recurring across shows merges into one gallery entry
# --------------------------------------------------------------------------- #
def test_host_across_two_shows_merges_into_one_entry() -> None:
    rows = [
        ManifestRow(
            show_id="show-1",
            feed_url="https://f/1.xml",
            known_hosts=["Alice"],
            reference_audio={"Alice": "ref/alice.wav"},
        ),
        ManifestRow(
            show_id="show-9",
            feed_url="https://f/9.xml",
            known_hosts=["Alice"],
            reference_audio={"Alice": "ref/alice.wav"},
        ),
    ]
    seeds = seeds_from_manifest_rows(rows)
    # Two seeds, same speaker_id -> gallery merges them.
    assert [s.speaker_id for s in seeds] == ["host-alice", "host-alice"]

    gallery = HostGallery.from_manifest_rows(rows, FakeVoiceEmbedder())
    assert set(gallery.speakers) == {"host-alice"}
    assert set(gallery.speaker("host-alice").show_ids) == {"show-1", "show-9"}


# --------------------------------------------------------------------------- #
# Empty / hostless inputs
# --------------------------------------------------------------------------- #
def test_rows_without_hosts_seed_nothing() -> None:
    rows = [ManifestRow(show_id="s", feed_url="https://f")]
    assert seeds_from_manifest_rows(rows) == []
    gallery = HostGallery.from_manifest_rows(rows, FakeVoiceEmbedder())
    assert gallery.speakers == {}


def test_adapter_accepts_plain_host_seed_consumers() -> None:
    # The adapter output is exactly what from_hosts already consumes.
    seeds = seeds_from_manifest_rows(_manifest_rows())
    assert all(isinstance(s, HostSeed) for s in seeds)
