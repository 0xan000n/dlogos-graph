"""pyannote diarization + token→speaker-turn mapping by timing.

Diarization assigns *who spoke when* (a set of time intervals, each tagged with
an anonymous label ``SPEAKER_00``…). WhisperX gives *what was said when* (words
with timestamps). This module joins the two by timing — each word is attributed
to the diarization turn whose interval it falls inside / overlaps most — and
exposes the pyannote run itself.

HARD CONSTRAINT — ``pyannote.audio`` and ``torch`` are imported **lazily inside
functions**, never at module top level, so importing this module costs nothing
and needs none of the ``asr`` optional extras. The timing-overlap mapping
(:func:`map_words_to_speakers`) is *pure* and heavy-dep-free, so it is unit
tested directly without a GPU.

ADVERSARIAL FAILURE MODES (spec §7.3 / §9 / §11 — diarization is the TOP
correctness risk, because a swapped label yields a *confident, sourced*
misattribution, the worst failure of all):

- **Crosstalk / overlapping speech.** Two people talk at once; the diarizer
  collapses both into one turn or flips the label mid-overlap, so a word lands
  on the wrong speaker. The talk-time helper cannot catch this (both speakers
  are present); only the eval's speaker-verified citation check does.
- **Remote guests / variable audio quality.** A phone-quality or
  echo-cancelled remote leg produces unstable voiceprints; the diarizer splits
  one remote guest into several labels or merges a remote guest into the host.
- **3+ speakers (panel shows).** With many voices the label count is
  unstable and short interjections get absorbed into a neighbour's turn,
  fragmenting one panelist's claims across labels or stealing another's.
- **Ad reads / host-read sponsorships.** A host reading an ad in a different
  register (or a stitched-in dynamic ad with a third voice) spawns a spurious
  label or shifts the host's label, so ad copy is attributed as a "claim" —
  and a real claim near the ad boundary may inherit the ad's label.

Mitigations layered elsewhere: host-anchored gallery + recurring-guest
resolution (``dlogos.speakers``); dropping sub-threshold labels
(``dlogos.asr.base.drop_low_talk_time_speakers``); the adversarial validation
slice (panel + remote-heavy + ad-saturated show); and the eval's
speaker-verified citation check that fails any citation where the person
speaking at the timestamp is not the attributed speaker.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiarizationTurn:
    """One diarization interval: a speaker label over ``[start, end]`` seconds.

    The heavy-dep-free intermediate produced by :func:`run_pyannote_diarization`
    and consumed by :func:`map_words_to_speakers`, so the mapping logic can be
    tested without pyannote.
    """

    speaker: str
    start: float
    end: float


def run_pyannote_diarization(
    audio_path: str,
    *,
    hf_token: str | None = None,
    device: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[DiarizationTurn]:
    """Run pyannote speaker diarization over an audio file.

    Returns a list of :class:`DiarizationTurn`, sorted by start time. The
    pyannote pipeline is *gated* on HuggingFace and requires a token; pass it
    explicitly via ``hf_token`` (or have it available in pyannote's own env).

    HARD CONSTRAINT: ``pyannote.audio`` / ``torch`` are imported here, lazily.
    """

    from pyannote.audio import Pipeline  # lazy: heavy, gated dep

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )

    if device is not None:
        import torch  # lazy: heavy dep

        pipeline.to(torch.device(device))

    diarize_kwargs: dict[str, int] = {}
    if min_speakers is not None:
        diarize_kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        diarize_kwargs["max_speakers"] = max_speakers

    annotation = pipeline(audio_path, **diarize_kwargs)

    turns = [
        DiarizationTurn(speaker=str(label), start=float(segment.start), end=float(segment.end))
        for segment, _track, label in annotation.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda t: (t.start, t.end))
    return turns


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Length of the overlap between intervals ``[a_start, a_end]`` and ``b``."""

    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def crosstalk_regions(turns: list[DiarizationTurn]) -> list[tuple[float, float]]:
    """Return the time regions where two or more *distinct* speakers overlap.

    Crosstalk — two people talking at once — is the failure the talk-time helper
    *cannot* catch (both speakers are genuinely present), so we surface it so the
    mapping can flag a low-confidence attribution. A region is reported only when
    the overlapping turns carry **different** labels; two turns with the same
    label (a diarizer emitting a split for one speaker) is not crosstalk.

    Returns a list of ``(start, end)`` intervals, sorted and non-overlapping
    (merged where adjacent), so a word's overlap with "any crosstalk" is a
    single membership test. Pure / heavy-dep-free.
    """

    raw: list[tuple[float, float]] = []
    for i, a in enumerate(turns):
        for b in turns[i + 1 :]:
            if a.speaker == b.speaker:
                continue
            lo = max(a.start, b.start)
            hi = min(a.end, b.end)
            if hi > lo:
                raw.append((lo, hi))

    if not raw:
        return []

    raw.sort()
    merged: list[tuple[float, float]] = [raw[0]]
    for lo, hi in raw[1:]:
        last_lo, last_hi = merged[-1]
        if lo <= last_hi:
            merged[-1] = (last_lo, max(last_hi, hi))
        else:
            merged.append((lo, hi))
    return merged


def _in_any_region(
    start: float, end: float, regions: list[tuple[float, float]]
) -> bool:
    """Does ``[start, end]`` overlap any region in ``regions`` (positive width)?"""

    for r_lo, r_hi in regions:
        if min(end, r_hi) > max(start, r_lo):
            return True
    return False


def map_words_to_speakers(
    words: list[dict],
    turns: list[DiarizationTurn],
    *,
    default_speaker: str = "SPEAKER_00",
    flag_crosstalk: bool = False,
) -> list[dict]:
    """Attribute each word to a diarization turn by timing overlap.

    Parameters
    ----------
    words:
        Word dicts with ``"start"`` / ``"end"`` (seconds) and arbitrary other
        keys (e.g. ``"word"``). Returned copies gain a ``"speaker"`` key.
    turns:
        Diarization turns to assign against. May be empty.
    default_speaker:
        Label used when a word overlaps no turn (e.g. words inside a silence
        gap, or when diarization produced nothing). Choosing a deterministic
        fallback — rather than dropping the word — keeps the transcript
        complete; downstream talk-time pruning can still remove a spurious
        catch-all label if it stays tiny.

    flag_crosstalk:
        When set, each returned word also gains a boolean ``"crosstalk"`` key:
        ``True`` if the word's interval overlaps a region where two *distinct*
        diarization labels are simultaneously active (see
        :func:`crosstalk_regions`). This is the one failure the talk-time helper
        cannot catch — under crosstalk the winning label can be the wrong
        speaker even though the overlap math is correct — so the flag marks the
        attribution as low-confidence for the downstream speaker-verified check.
        The chosen ``"speaker"`` is unchanged whether or not the flag is set.

    Assignment rule: pick the turn with the **largest temporal overlap** with
    the word. Ties (equal overlap) break toward the **earlier-starting** turn,
    then the lexicographically smaller label, so the mapping is fully
    deterministic. A word with no positive overlap falls back to the turn whose
    interval is *nearest* (smallest gap to the word's midpoint); if there are no
    turns at all it gets ``default_speaker``.

    Pure and deterministic — no heavy deps, directly unit tested.
    """

    regions = crosstalk_regions(turns) if flag_crosstalk else []

    if not turns:
        if flag_crosstalk:
            return [
                {**w, "speaker": default_speaker, "crosstalk": False}
                for w in words
            ]
        return [
            {**w, "speaker": default_speaker}
            for w in words
        ]

    ordered = sorted(turns, key=lambda t: (t.start, t.end, t.speaker))
    out: list[dict] = []

    for word in words:
        w_start = float(word.get("start", 0.0) or 0.0)
        w_end = float(word.get("end", w_start) or w_start)
        if w_end < w_start:
            w_end = w_start
        midpoint = (w_start + w_end) / 2.0

        best_label = default_speaker
        best_overlap = 0.0
        # Track nearest-by-gap as the fallback when nothing overlaps.
        best_gap = float("inf")
        nearest_label = default_speaker

        for turn in ordered:
            ov = _overlap(w_start, w_end, turn.start, turn.end)
            if ov > best_overlap:
                best_overlap = ov
                best_label = turn.speaker

            # Gap from the word midpoint to this turn's interval (0 if inside).
            if midpoint < turn.start:
                gap = turn.start - midpoint
            elif midpoint > turn.end:
                gap = midpoint - turn.end
            else:
                gap = 0.0
            if gap < best_gap:
                best_gap = gap
                nearest_label = turn.speaker

        speaker = best_label if best_overlap > 0.0 else nearest_label
        mapped = {**word, "speaker": speaker}
        if flag_crosstalk:
            mapped["crosstalk"] = _in_any_region(w_start, w_end, regions)
        out.append(mapped)

    return out


def assign_word_speakers(turns: list[DiarizationTurn], aligned: dict) -> dict:
    """Attach speaker labels to WhisperX-aligned segments by timing.

    Mirrors WhisperX's own ``assign_word_speakers`` but routes through the pure
    :func:`map_words_to_speakers` so the timing logic is shared and testable.
    For each aligned segment we attribute its words, then set the segment's
    ``"speaker"`` to the label that holds the most word-time in that segment
    (majority by spoken duration). Returns the same ``aligned`` dict shape with
    ``"speaker"`` populated on each segment (and each word, when present).

    Heavy-dep-free: operates on plain dicts WhisperX already produced.
    """

    segments = aligned.get("segments", []) if isinstance(aligned, dict) else []

    for segment in segments:
        seg_words = segment.get("words") or []
        if seg_words:
            mapped = map_words_to_speakers(seg_words, turns)
            segment["words"] = mapped
            # Majority speaker by word duration within the segment.
            by_dur: dict[str, float] = {}
            for w in mapped:
                w_start = float(w.get("start", 0.0) or 0.0)
                w_end = float(w.get("end", w_start) or w_start)
                dur = max(0.0, w_end - w_start)
                spk = str(w.get("speaker", "SPEAKER_00"))
                by_dur[spk] = by_dur.get(spk, 0.0) + dur
            if by_dur:
                # Highest duration; ties → lexicographically smaller label.
                segment["speaker"] = min(
                    by_dur, key=lambda s: (-by_dur[s], s)
                )
        else:
            # No word-level timing: attribute the whole segment by overlap.
            seg_start = float(segment.get("start", 0.0) or 0.0)
            seg_end = float(segment.get("end", seg_start) or seg_start)
            pseudo = [{"start": seg_start, "end": seg_end}]
            mapped = map_words_to_speakers(pseudo, turns)
            segment["speaker"] = mapped[0]["speaker"]

    return aligned
