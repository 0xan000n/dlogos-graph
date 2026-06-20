"""Word-level transcript re-segmentation for tight grounded spans.

The hosted AssemblyAI path (``asr.hosted_backend``) emits coarse utterance
``segments`` — often multi-sentence and occasionally multi-minute — alongside a
fine-grained ``words`` stream (one token, one diarized label, one ms-precise
span each). Grounding a claim to a coarse segment cites a blob; grounding it to
a ~sentence span cites the actual utterance.

:func:`resegment_by_words` rebuilds a :class:`~dlogos.schema.Transcript`'s
``segments`` from its ``words``, flushing a segment on any of:

- a **speaker change** (the next word's diarization label differs);
- a **sentence end** (the current word ends with ``.``/``!``/``?``);
- a **duration cap** (the accumulating segment would exceed ``max_seg_seconds``).

Pure stdlib — no network, no heavy deps, no model. When ``words`` is empty the
transcript is returned **unchanged** (identity), so utterance-only backends
(WhisperX, the mock) are unaffected; the pass is opt-in at the pipeline level.
"""

from __future__ import annotations

from dlogos.schema import Transcript, TranscriptSegment, Word

# Default per-word label when a word carries no diarization label, so unlabeled
# words coalesce instead of each forming a singleton segment.
_DEFAULT_LABEL = "A"

# Sentence-terminating punctuation that closes a segment.
_SENTENCE_ENDERS = (".", "!", "?")


def resegment_by_words(
    transcript: Transcript, *, max_seg_seconds: float = 15.0
) -> Transcript:
    """Rebuild ``transcript.segments`` from its word-level stream.

    Returns a copy of ``transcript`` whose ``segments`` are speaker-contiguous,
    sentence-bounded, and duration-capped (``max_seg_seconds``). Each rebuilt
    :class:`~dlogos.schema.TranscriptSegment` carries its words' speaker and a
    span equal to ``[first_word.t_start, last_word.t_end]``.

    When ``transcript.words`` is empty the transcript is returned unchanged
    (the same object), so backends that only produce utterance segments are
    untouched. ``words`` and ``duration_s`` are preserved on the copy.
    """

    if not transcript.words:
        return transcript  # nothing to refine; keep utterance segments

    segs: list[TranscriptSegment] = []
    cur: list[Word] = []

    def _label(word: Word) -> str:
        return word.speaker or _DEFAULT_LABEL

    def flush() -> None:
        if cur:
            segs.append(
                TranscriptSegment(
                    speaker=_label(cur[0]),
                    text=" ".join(w.text for w in cur).strip(),
                    t_start=cur[0].t_start,
                    t_end=cur[-1].t_end,
                )
            )

    for w in transcript.words:
        if cur and (
            _label(w) != _label(cur[0])
            or (w.t_end - cur[0].t_start) > max_seg_seconds
            or cur[-1].text.endswith(_SENTENCE_ENDERS)
        ):
            flush()
            cur = []
        cur.append(w)
    flush()

    return transcript.model_copy(update={"segments": segs})
