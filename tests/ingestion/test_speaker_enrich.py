"""Offline tests for speaker-label enrichment (pure, stdlib, no network).

:func:`~dlogos.ingestion.speaker_enrich.enrich_speakers` maps raw transcript
speaker *labels* onto the episode's known speaker FULL names — but only when the
match is unambiguous. These tests pin the four match rules (exact / first-name /
surname / unique-substring), the conservative "leave unchanged when ambiguous or
unknown" behaviour that keeps stray non-speaker labels intact, and that the
input segments are never mutated.
"""

from __future__ import annotations

from dlogos.ingestion.speaker_enrich import enrich_speakers, resolve_label
from dlogos.schema import TranscriptSegment


def _seg(speaker: str, t_start: float = 0.0, t_end: float = 1.0) -> TranscriptSegment:
    return TranscriptSegment(
        speaker=speaker, text="hello world", t_start=t_start, t_end=t_end
    )


# --------------------------------------------------------------------------- #
# resolve_label — the per-label rules
# --------------------------------------------------------------------------- #
def test_first_name_maps_to_full_name() -> None:
    """A bare first name resolves when exactly one known speaker has it."""

    known = ["Jim Rutt", "Nate Soares"]
    assert resolve_label("Jim", known) == "Jim Rutt"
    assert resolve_label("Nate", known) == "Nate Soares"


def test_surname_maps_to_full_name() -> None:
    """A bare surname resolves when exactly one known speaker has it."""

    known = ["Jim Rutt", "Nate Soares"]
    assert resolve_label("Rutt", known) == "Jim Rutt"
    assert resolve_label("Soares", known) == "Nate Soares"


def test_exact_casefold_normalizes_to_roster_spelling() -> None:
    """An all-caps or mixed-case full label snaps to the roster's spelling."""

    known = ["Tristan Harris", "Nate Hagens"]
    assert resolve_label("TRISTAN HARRIS", known) == "Tristan Harris"
    assert resolve_label("tristan harris", known) == "Tristan Harris"


def test_unique_substring_maps_to_full_name() -> None:
    """A label that is a substring of exactly one known name resolves."""

    known = ["Dwarkesh Patel", "Ilya Sutskever"]
    # "Sutskever" first-name/surname both unique; test a partial substring too.
    assert resolve_label("Ilya Sutskever", known) == "Ilya Sutskever"
    assert resolve_label("Patel", known) == "Dwarkesh Patel"


def test_ambiguous_first_name_left_unchanged() -> None:
    """A first name shared by two known speakers ('Nate') is never guessed."""

    known = ["Nate Soares", "Nate Hagens"]
    assert resolve_label("Nate", known) == "Nate"


def test_unknown_label_left_unchanged() -> None:
    """Stray non-speaker / archival labels match nobody and stay verbatim."""

    known = ["Tristan Harris", "Nate Hagens"]
    for stray in ["Researchers", "Companies", "Governments", "Robert Oppenheimer"]:
        assert resolve_label(stray, known) == stray


def test_empty_known_speakers_returns_label() -> None:
    """With no roster, the label is returned unchanged."""

    assert resolve_label("Jim", []) == "Jim"


def test_duplicate_roster_entry_does_not_count_as_ambiguous() -> None:
    """The same full name reached twice is one match, not an ambiguous pair."""

    known = ["Jim Rutt", "Jim Rutt"]
    assert resolve_label("Jim", known) == "Jim Rutt"


def test_first_name_wins_over_surname_collision() -> None:
    """First-name rule is tried before surname; an unrelated surname collision

    on a *different* token does not block a clean first-name match."""

    known = ["Jim Rutt", "Nate Soares"]
    # "Jim" is a unique first name -> resolves even though "Rutt"/"Soares" exist.
    assert resolve_label("Jim", known) == "Jim Rutt"


# --------------------------------------------------------------------------- #
# enrich_speakers — over a list of segments
# --------------------------------------------------------------------------- #
def test_enrich_maps_first_names_across_segments() -> None:
    """Each segment's bare label is rewritten to the episode's full name."""

    segs = [_seg("Jim"), _seg("Nate"), _seg("Jim")]
    out = enrich_speakers(segs, ["Jim Rutt", "Nate Soares"])
    assert [s.speaker for s in out] == ["Jim Rutt", "Nate Soares", "Jim Rutt"]


def test_enrich_leaves_ambiguous_and_unknown_unchanged() -> None:
    """Ambiguous first names and stray labels are preserved verbatim."""

    segs = [_seg("Nate"), _seg("Researchers"), _seg("Tristan")]
    out = enrich_speakers(segs, ["Nate Soares", "Nate Hagens", "Tristan Harris"])
    assert [s.speaker for s in out] == ["Nate", "Researchers", "Tristan Harris"]


def test_enrich_preserves_text_and_spans_and_does_not_mutate_input() -> None:
    """Non-speaker fields are preserved and the input segments are untouched."""

    segs = [_seg("Jim", t_start=0.0, t_end=2.5)]
    out = enrich_speakers(segs, ["Jim Rutt"])
    assert out[0].speaker == "Jim Rutt"
    assert out[0].text == "hello world"
    assert (out[0].t_start, out[0].t_end) == (0.0, 2.5)
    # Input not mutated.
    assert segs[0].speaker == "Jim"


def test_enrich_with_empty_roster_returns_labels_verbatim() -> None:
    """An empty known-speaker list passes every label through unchanged."""

    segs = [_seg("Jim"), _seg("Nate")]
    out = enrich_speakers(segs, [])
    assert [s.speaker for s in out] == ["Jim", "Nate"]


def test_enrich_idempotent_on_already_full_names() -> None:
    """Re-running enrichment on already-full names is a no-op (exact match)."""

    segs = [_seg("Jim Rutt"), _seg("Nate Soares")]
    out = enrich_speakers(segs, ["Jim Rutt", "Nate Soares"])
    assert [s.speaker for s in out] == ["Jim Rutt", "Nate Soares"]
