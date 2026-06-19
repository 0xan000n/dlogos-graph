"""Tests for the pure token→speaker-turn mapping (``dlogos.asr.diarization``).

The pyannote run itself needs a GPU + gated weights and is not tested here; the
*timing-overlap mapping* that joins words to diarization turns is pure and is
the part that can silently misattribute, so it is tested directly. Also asserts
the diarization + whisperx modules import without any heavy dep.
"""

from __future__ import annotations

import sys

from dlogos.asr.diarization import (
    DiarizationTurn,
    assign_word_speakers,
    map_words_to_speakers,
)


def _w(start: float, end: float, word: str = "x") -> dict:
    return {"start": start, "end": end, "word": word}


def test_word_assigned_to_overlapping_turn() -> None:
    turns = [
        DiarizationTurn("SPEAKER_00", 0.0, 5.0),
        DiarizationTurn("SPEAKER_01", 5.0, 10.0),
    ]
    words = [_w(1.0, 2.0), _w(6.0, 7.0)]
    out = map_words_to_speakers(words, turns)
    assert [w["speaker"] for w in out] == ["SPEAKER_00", "SPEAKER_01"]
    # Original keys preserved.
    assert out[0]["word"] == "x"


def test_word_assigned_to_max_overlap_turn() -> None:
    # Word straddles the boundary but overlaps SPEAKER_01 more (3s vs 1s).
    turns = [
        DiarizationTurn("SPEAKER_00", 0.0, 5.0),
        DiarizationTurn("SPEAKER_01", 5.0, 12.0),
    ]
    out = map_words_to_speakers([_w(4.0, 8.0)], turns)
    assert out[0]["speaker"] == "SPEAKER_01"


def test_no_turns_falls_back_to_default_speaker() -> None:
    out = map_words_to_speakers([_w(1.0, 2.0)], [], default_speaker="SPEAKER_00")
    assert out[0]["speaker"] == "SPEAKER_00"


def test_word_in_silence_gap_uses_nearest_turn() -> None:
    # Word at 5.4-5.6 overlaps nothing; nearest turn is SPEAKER_00 (gap 0.5)
    # vs SPEAKER_01 (gap 0.4) -> SPEAKER_01 is nearer.
    turns = [
        DiarizationTurn("SPEAKER_00", 0.0, 5.0),
        DiarizationTurn("SPEAKER_01", 5.9, 10.0),
    ]
    out = map_words_to_speakers([_w(5.4, 5.6)], turns)
    assert out[0]["speaker"] == "SPEAKER_01"


def test_mapping_is_deterministic() -> None:
    turns = [
        DiarizationTurn("SPEAKER_01", 0.0, 5.0),
        DiarizationTurn("SPEAKER_00", 0.0, 5.0),  # same interval, different label
    ]
    words = [_w(1.0, 2.0)]
    runs = [map_words_to_speakers(words, turns)[0]["speaker"] for _ in range(5)]
    # Equal overlap → tie broken deterministically (earlier start, then label).
    assert len(set(runs)) == 1


def test_equal_overlap_tie_breaks_to_smaller_label() -> None:
    # Identical intervals → tie on overlap → smaller label wins after sort.
    turns = [
        DiarizationTurn("SPEAKER_01", 0.0, 5.0),
        DiarizationTurn("SPEAKER_00", 0.0, 5.0),
    ]
    out = map_words_to_speakers([_w(1.0, 2.0)], turns)
    assert out[0]["speaker"] == "SPEAKER_00"


def test_negative_duration_word_clamped() -> None:
    turns = [DiarizationTurn("SPEAKER_00", 0.0, 5.0)]
    out = map_words_to_speakers([_w(2.0, 1.0)], turns)  # end < start
    assert out[0]["speaker"] == "SPEAKER_00"


def test_assign_word_speakers_sets_segment_majority_speaker() -> None:
    # Segment with words mostly from SPEAKER_01 → segment speaker is SPEAKER_01.
    turns = [
        DiarizationTurn("SPEAKER_00", 0.0, 1.0),
        DiarizationTurn("SPEAKER_01", 1.0, 10.0),
    ]
    aligned = {
        "segments": [
            {
                "start": 0.0,
                "end": 10.0,
                "text": "a b c",
                "words": [_w(0.0, 0.5, "a"), _w(2.0, 5.0, "b"), _w(6.0, 9.0, "c")],
            }
        ]
    }
    result = assign_word_speakers(turns, aligned)
    seg = result["segments"][0]
    assert seg["speaker"] == "SPEAKER_01"
    # Per-word speakers also populated.
    assert {w["speaker"] for w in seg["words"]} == {"SPEAKER_00", "SPEAKER_01"}


def test_assign_word_speakers_segment_without_words_uses_overlap() -> None:
    turns = [
        DiarizationTurn("SPEAKER_00", 0.0, 5.0),
        DiarizationTurn("SPEAKER_01", 5.0, 10.0),
    ]
    aligned = {"segments": [{"start": 6.0, "end": 9.0, "text": "no words"}]}
    result = assign_word_speakers(turns, aligned)
    assert result["segments"][0]["speaker"] == "SPEAKER_01"


def test_diarization_module_imports_without_heavy_deps() -> None:
    # Importing the diarization + whisperx backend modules must not pull torch/
    # pyannote/whisperx (those import lazily inside functions only).
    import dlogos.asr.diarization  # noqa: F401
    import dlogos.asr.whisperx_backend  # noqa: F401
    from dlogos.asr.whisperx_backend import WhisperXBackend

    # Construction is also heavy-dep-free (no model load until transcribe()).
    WhisperXBackend(diarize=False)
    for heavy in ("torch", "whisperx", "pyannote", "pyannote.audio"):
        assert heavy not in sys.modules, f"{heavy} imported eagerly by diarization/whisperx"


def test_whisperx_aligned_to_segments_merges_same_speaker_turns() -> None:
    # The pure aligned→segments collapser is testable without WhisperX.
    from dlogos.asr.whisperx_backend import WhisperXBackend

    aligned = {
        "segments": [
            {"speaker": "SPEAKER_00", "text": "Hello", "start": 0.0, "end": 1.0},
            {"speaker": "SPEAKER_00", "text": "there", "start": 1.0, "end": 2.0},
            {"speaker": "SPEAKER_01", "text": "Hi", "start": 2.0, "end": 3.0},
            {"speaker": "SPEAKER_00", "text": "", "start": 3.0, "end": 3.1},  # empty dropped
        ]
    }
    segs = WhisperXBackend._aligned_to_segments(aligned)
    assert len(segs) == 2
    assert segs[0].speaker == "SPEAKER_00"
    assert segs[0].text == "Hello there"
    assert segs[0].t_start == 0.0 and segs[0].t_end == 2.0
    assert segs[1].speaker == "SPEAKER_01" and segs[1].text == "Hi"
