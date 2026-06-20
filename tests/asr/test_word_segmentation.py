"""Tests for word-level transcript re-segmentation (``resegment_by_words``).

The hosted AssemblyAI path emits coarse utterance ``segments`` (multi-sentence,
sometimes multi-minute) plus a fine-grained ``words`` stream. This pass rebuilds
the segments from the words so grounded citations snap to ~sentence spans:
splitting on a speaker change, a sentence end, or a duration cap.

Pure stdlib; no network, no heavy deps. When ``words`` is empty the transcript
is returned unchanged (the WhisperX/mock utterance-only paths are unaffected).
"""

from __future__ import annotations

from dlogos.asr.word_segmentation import resegment_by_words
from dlogos.schema import Transcript, TranscriptSegment, Word


def _t(start: float, end: float, text: str, speaker: str) -> Word:
    return Word(text=text, t_start=start, t_end=end, speaker=speaker)


def _transcript_with_words() -> Transcript:
    """Speaker A "Apple is great." / B "I disagree strongly." / A "Inflation is
    cooling." — a single coarse input segment, realistic word spans."""

    words = [
        _t(0.0, 0.4, "Apple", "A"),
        _t(0.4, 0.7, "is", "A"),
        _t(0.7, 1.2, "great.", "A"),
        _t(1.5, 1.7, "I", "B"),
        _t(1.7, 2.3, "disagree", "B"),
        _t(2.3, 3.0, "strongly.", "B"),
        _t(3.2, 3.9, "Inflation", "A"),
        _t(3.9, 4.1, "is", "A"),
        _t(4.1, 4.8, "cooling.", "A"),
    ]
    coarse = TranscriptSegment(
        speaker="A",
        text="Apple is great. I disagree strongly. Inflation is cooling.",
        t_start=0.0,
        t_end=4.8,
    )
    return Transcript(
        episode_id="ep-words",
        language="en",
        segments=[coarse],
        words=words,
        duration_s=4.8,
    )


def test_resegments_on_speaker_change_and_sentence_end() -> None:
    transcript = _transcript_with_words()
    result = resegment_by_words(transcript)

    # At least the three logical utterances are split apart.
    assert len(result.segments) >= 3
    # Finer than the single coarse input segment.
    assert len(result.segments) > len(transcript.segments)

    s0, s1, s2 = result.segments[0], result.segments[1], result.segments[2]

    # Segment 0: speaker A, "Apple is great.", span = [first word start, last
    # word end].
    assert s0.speaker == "A"
    assert s0.text == "Apple is great."
    assert s0.t_start == 0.0 and s0.t_end == 1.2

    # Segment 1: speaker B, split on the speaker change.
    assert s1.speaker == "B"
    assert s1.text == "I disagree strongly."
    assert s1.t_start == 1.5 and s1.t_end == 3.0

    # Segment 2: back to A on the next speaker change.
    assert s2.speaker == "A"
    assert s2.text == "Inflation is cooling."
    assert s2.t_start == 3.2 and s2.t_end == 4.8


def test_duration_cap_splits_a_long_same_speaker_run() -> None:
    """A long monologue with no sentence end still splits at the duration cap."""

    words = [
        Word(text=f"w{i}", t_start=float(i), t_end=float(i) + 1.0, speaker="A")
        for i in range(10)  # 10s of words, no sentence punctuation
    ]
    transcript = Transcript(
        episode_id="ep-long",
        language="en",
        segments=[
            TranscriptSegment(speaker="A", text="long blob", t_start=0.0, t_end=10.0)
        ],
        words=words,
        duration_s=10.0,
    )

    result = resegment_by_words(transcript, max_seg_seconds=4.0)
    # No 10s blob survives: every refined segment respects the cap.
    assert all((s.t_end - s.t_start) <= 4.0 + 1e-6 for s in result.segments)
    assert len(result.segments) >= 2


def test_empty_words_returns_transcript_unchanged() -> None:
    """No word stream → keep the utterance segments (mock/WhisperX paths)."""

    transcript = Transcript(
        episode_id="ep-nowords",
        language="en",
        segments=[
            TranscriptSegment(speaker="A", text="hello", t_start=0.0, t_end=2.0)
        ],
        words=[],
        duration_s=2.0,
    )
    result = resegment_by_words(transcript)
    assert result is transcript
    assert result.segments == transcript.segments


def test_missing_word_speaker_defaults_to_A() -> None:
    """Words with no speaker label coalesce under the default 'A' label."""

    words = [
        Word(text="hello.", t_start=0.0, t_end=0.5, speaker=None),
        Word(text="world.", t_start=0.5, t_end=1.0, speaker=None),
    ]
    transcript = Transcript(
        episode_id="ep-nospk",
        language="en",
        segments=[
            TranscriptSegment(speaker="A", text="hello. world.", t_start=0.0, t_end=1.0)
        ],
        words=words,
        duration_s=1.0,
    )
    result = resegment_by_words(transcript)
    assert all(s.speaker == "A" for s in result.segments)
