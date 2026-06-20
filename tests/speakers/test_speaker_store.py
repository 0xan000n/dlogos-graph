"""Phase 2 — persistent cross-episode speaker identity (name-driven).

Covers the three Phase-2 tasks in :mod:`dlogos.speakers.speaker_store`:

- Task 2.1 :class:`SqliteSpeakerStore` — stable, persistent ``speaker_id``s
  keyed on QID-first then normalized name.
- Task 2.2 :class:`NameSpeakerResolver` — label->name map to per-label
  :class:`SpeakerResolution`s against the store.
- Task 2.3 :func:`extract_label_names` — names from spoken intros + manifest.
"""

from __future__ import annotations

from dlogos.schema import Transcript, TranscriptSegment
from dlogos.speakers.identity import CanonicalSpeaker, SpeakerResolution
from dlogos.speakers.speaker_store import (
    NameSpeakerResolver,
    SqliteSpeakerStore,
    extract_label_names,
)


# --------------------------------------------------------------------------- #
# Task 2.1 — SqliteSpeakerStore
# --------------------------------------------------------------------------- #
def test_canonical_for_name_mints_stable_id(tmp_path):
    store = SqliteSpeakerStore(tmp_path / "spk.db")
    spk = store.canonical_for(name="Darian Woods")
    assert isinstance(spk, CanonicalSpeaker)
    assert spk.speaker_id.startswith("spk-")
    assert spk.name == "Darian Woods"
    # Same name again -> same id (within the same store).
    again = store.canonical_for(name="Darian Woods")
    assert again.speaker_id == spk.speaker_id


def test_canonical_for_persists_across_store_instances(tmp_path):
    path = tmp_path / "spk.db"
    first = SqliteSpeakerStore(path)
    spk = first.canonical_for(name="Darian Woods")
    first.close()

    # A SECOND store instance on the same path returns the SAME id.
    second = SqliteSpeakerStore(path)
    reopened = second.canonical_for(name="Darian Woods")
    assert reopened.speaker_id == spk.speaker_id


def test_canonical_for_keys_on_qid_regardless_of_surface(tmp_path):
    store = SqliteSpeakerStore(tmp_path / "spk.db")
    spk = store.canonical_for(name="Cardiff Garcia", qid="Q123")
    assert spk.speaker_id == "spk-wd-Q123"
    assert spk.wikidata_qid == "Q123"
    # A different spoken surface but the same QID resolves to the same id.
    other = store.canonical_for(name="C. Garcia", qid="Q123")
    assert other.speaker_id == spk.speaker_id


def test_canonical_for_normalizes_name(tmp_path):
    store = SqliteSpeakerStore(tmp_path / "spk.db")
    a = store.canonical_for(name="Darian Woods")
    b = store.canonical_for(name="darian woods")
    c = store.canonical_for(name="Darian  Woods")
    assert a.speaker_id == b.speaker_id == c.speaker_id


# --------------------------------------------------------------------------- #
# Task 2.2 — NameSpeakerResolver
# --------------------------------------------------------------------------- #
def _two_label_transcript(episode_id: str = "ep1") -> Transcript:
    return Transcript(
        episode_id=episode_id,
        language="en",
        segments=[
            TranscriptSegment(speaker="A", text="Hello.", t_start=0.0, t_end=2.0),
            TranscriptSegment(speaker="B", text="Hi there.", t_start=2.0, t_end=4.0),
        ],
        duration_s=4.0,
    )


def test_name_resolver_resolves_known_labels(tmp_path):
    store = SqliteSpeakerStore(tmp_path / "spk.db")
    resolver = NameSpeakerResolver(store)
    transcript = _two_label_transcript()
    label_names = {"A": "Darian Woods", "B": "Cardiff Garcia"}

    out = resolver.resolve(
        transcript, label_names, qids={"Cardiff Garcia": "Q123"}
    )
    assert set(out) == {"A", "B"}
    assert isinstance(out["A"], SpeakerResolution)
    assert out["A"].is_resolved
    assert out["A"].resolved.name == "Darian Woods"
    assert out["A"].score == 1.0
    # Guest carried a QID -> id keys on it.
    assert out["B"].resolved.speaker_id == "spk-wd-Q123"


def test_name_resolver_label_without_name_is_unresolved(tmp_path):
    store = SqliteSpeakerStore(tmp_path / "spk.db")
    resolver = NameSpeakerResolver(store)
    transcript = _two_label_transcript()
    label_names = {"A": "Darian Woods"}  # B has no name

    out = resolver.resolve(transcript, label_names)
    assert out["A"].is_resolved
    assert "B" in out
    assert not out["B"].is_resolved
    assert out["B"].resolved is None


def test_name_resolver_same_host_across_episodes_same_id(tmp_path):
    path = tmp_path / "spk.db"
    store1 = SqliteSpeakerStore(path)
    r1 = NameSpeakerResolver(store1)
    out1 = r1.resolve(_two_label_transcript("ep1"), {"A": "Darian Woods"})
    store1.close()

    # Episode 2: a fresh store instance on the same path, host reused.
    store2 = SqliteSpeakerStore(path)
    r2 = NameSpeakerResolver(store2)
    out2 = r2.resolve(_two_label_transcript("ep2"), {"A": "Darian Woods"})

    assert out1["A"].resolved.speaker_id == out2["A"].resolved.speaker_id


# --------------------------------------------------------------------------- #
# Task 2.3 — extract_label_names
# --------------------------------------------------------------------------- #
def test_extract_label_names_self_intro_and_guest():
    transcript = Transcript(
        episode_id="ep1",
        language="en",
        segments=[
            TranscriptSegment(
                speaker="A",
                text="I'm Darian Woods, and my guest today is Cardiff Garcia.",
                t_start=0.0,
                t_end=6.0,
            ),
            TranscriptSegment(
                speaker="B",
                text="Thanks for having me.",
                t_start=6.0,
                t_end=8.0,
            ),
        ],
        duration_s=8.0,
    )
    out = extract_label_names(transcript, known_hosts=["Darian Woods"])
    assert out["A"] == "Darian Woods"
    # The guest name surfaces on the other (non-host) label.
    assert "Cardiff Garcia" in out.values()
    guest_label = next(lbl for lbl, name in out.items() if name == "Cardiff Garcia")
    assert guest_label == "B"


def test_extract_label_names_empty_when_no_names():
    transcript = Transcript(
        episode_id="ep1",
        language="en",
        segments=[
            TranscriptSegment(
                speaker="A", text="So inflation is cooling.", t_start=0.0, t_end=3.0
            ),
            TranscriptSegment(
                speaker="B", text="I disagree strongly.", t_start=3.0, t_end=5.0
            ),
        ],
        duration_s=5.0,
    )
    assert extract_label_names(transcript, known_hosts=[]) == {}
