"""Shared, stdlib-only helpers for the text-transcript parsers.

Every parser in this package reuses these (DRY) so timestamp parsing, span
synthesis, and text cleanup behave identically across sources. Nothing here
touches the network, filesystem, or clock — pure functions over strings.

Helpers
-------
- :func:`parse_hms` — ``"H:MM:SS"`` / ``"MM:SS"`` / ``"[00:01:23]"`` → seconds.
- :func:`synthesize_spans` — ``(speaker, text)`` turns with NO timestamps →
  monotonic, non-overlapping :class:`~dlogos.schema.TranscriptSegment`s by
  cumulative word count.
- :func:`finalize_segments` — rows that *may* carry an explicit start time →
  fill missing spans monotonically; guarantee non-decreasing ordering.
- :func:`clean_text` — collapse whitespace, drop bracketed stage directions
  like ``[laughs]`` / ``[music]``.

The shared timing constant :data:`SEC_PER_WORD` (~0.4 s/word ≈ 150 wpm) sets the
synthesized-span pace; ordering is what matters here, not absolute accuracy.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from dlogos.schema import TranscriptSegment

__all__ = [
    "SEC_PER_WORD",
    "Row",
    "parse_hms",
    "synthesize_spans",
    "finalize_segments",
    "clean_text",
]

#: Default synthesized speaking pace (~150 words/minute). Ordering, not absolute
#: timing, is the goal; parsers may override via the ``sec_per_word`` arg.
SEC_PER_WORD: float = 0.4

#: A finalize input row: ``(speaker, text, t_start | None)``. ``t_start`` is an
#: explicit start in seconds when the source carried one, else ``None`` (fill it).
Row = tuple[str, str, "float | None"]


# --------------------------------------------------------------------------- #
# Timestamp parsing
# --------------------------------------------------------------------------- #
# One unsigned timestamp: optional ``[``/``(`` wrapper, optional leading
# ``H:``/``HH:``, then ``MM:SS``. Fractional seconds tolerated and truncated to
# whole-second granularity downstream (we keep them — float). Matches anywhere
# in ``s`` (e.g. a line prefix); the first hit wins.
_HMS_RE = re.compile(
    r"""
    (?P<open>[\[(])?               # optional bracket/paren wrapper
    (?:(?P<h>\d{1,2}):)?           # optional hours
    (?P<m>\d{1,2}):                # minutes
    (?P<s>\d{1,2}(?:\.\d+)?)       # seconds (optional fraction)
    (?(open)[\])])                 # matching close only if it opened
    """,
    re.VERBOSE,
)


def parse_hms(s: str) -> float | None:
    """Parse the first ``H:MM:SS`` / ``MM:SS`` timestamp in ``s`` to seconds.

    Accepts an optional ``[...]`` / ``(...)`` wrapper (``"[00:01:23]"``),
    optional hours (``"3:04"`` is 3 m 4 s, ``"1:03:04"`` is 1 h 3 m 4 s), and a
    fractional-seconds tail. Returns the offset in seconds as a float, or
    ``None`` when no timestamp is present.

    Pure and total: never raises on arbitrary input.
    """

    m = _HMS_RE.search(s)
    if m is None:
        return None
    hours = int(m.group("h")) if m.group("h") else 0
    minutes = int(m.group("m"))
    seconds = float(m.group("s"))
    return hours * 3600.0 + minutes * 60.0 + seconds


# --------------------------------------------------------------------------- #
# Span synthesis / finalization
# --------------------------------------------------------------------------- #
def _word_count(text: str) -> int:
    """Whitespace-delimited word count; a non-empty turn counts as ≥1 word."""

    n = len(text.split())
    return n if n else 1


def synthesize_spans(
    turns: Sequence[tuple[str, str]],
    *,
    sec_per_word: float = SEC_PER_WORD,
) -> list[TranscriptSegment]:
    """Assign monotonic spans to ``(speaker, text)`` turns with no timestamps.

    Each turn's duration is ``word_count * sec_per_word``; spans are laid end to
    end so the result is strictly ordered and non-overlapping. The cleaned text
    (via :func:`clean_text`) is what gets counted and stored. Empty-after-clean
    turns are dropped (no zero-content segments downstream).

    This is the fallback path for sources with no native timing (most
    name-prefixed web transcripts); ordering is faithful, absolute timing is not.
    """

    rows: list[Row] = [(spk, txt, None) for spk, txt in turns]
    return finalize_segments(rows, sec_per_word=sec_per_word)


def finalize_segments(
    rows: Iterable[Row],
    *,
    sec_per_word: float = SEC_PER_WORD,
) -> list[TranscriptSegment]:
    """Turn ``(speaker, text, t_start|None)`` rows into ordered segments.

    The single entry point parsers funnel through:

    - Text is cleaned (:func:`clean_text`); rows empty after cleaning are dropped.
    - An explicit ``t_start`` (e.g. a parsed ``HH:MM:SS``) is honored. A missing
      ``t_start`` is filled monotonically from the running cursor by word count.
    - A turn ends at the **immediately following** row's explicit start when one
      exists (so a timestamped source's turn fills exactly up to the next
      timestamp — no gaps, no overlap). Otherwise it ends at ``t_start +
      word_count * sec_per_word``. Interior synthesized turns therefore pace by
      word count; only the turn right before a timestamp stretches to meet it.
    - The cursor is clamped non-decreasing, so a stray out-of-order or backwards
      timestamp can never produce a backwards or negative span — the invariant
      that every segment has ``0 <= t_start <= t_end`` and ``t_start`` is
      non-decreasing across the list always holds.
    """

    # Materialize + clean first, so the next-row lookahead is simple and
    # dropped-empty rows don't perturb timing.
    cleaned: list[tuple[str, str, float | None]] = []
    for speaker, text, t_start in rows:
        ctext = clean_text(text)
        if not ctext:
            continue
        cleaned.append((speaker, ctext, t_start))

    out: list[TranscriptSegment] = []
    cursor = 0.0
    for i, (speaker, text, t_start) in enumerate(cleaned):
        start = cursor if t_start is None else max(float(t_start), cursor)

        # Only the *immediately* following row's explicit start bounds this turn,
        # so a turn right before a timestamp fills the gap up to it while interior
        # turns still pace by word count.
        nxt = cleaned[i + 1][2] if i + 1 < len(cleaned) else None
        if nxt is not None and nxt > start:
            end = nxt
        else:
            end = start + _word_count(text) * sec_per_word
        if end <= start:
            # Degenerate guard (zero-word turn, equal timestamps): keep ordering
            # strictly forward so spans never collapse or go backwards.
            end = start + sec_per_word

        out.append(
            TranscriptSegment(
                speaker=speaker, text=text, t_start=start, t_end=end
            )
        )
        cursor = end

    return out


# --------------------------------------------------------------------------- #
# Text cleanup
# --------------------------------------------------------------------------- #
# Bracketed stage directions / non-speech cues: ``[laughs]``, ``[music]``,
# ``[crosstalk]``, ``[inaudible 00:12]``, ``(laughs)``. We strip short
# parenthetical cues too, but conservatively — only single-token-ish cues and
# the common non-speech words — so we don't eat legitimate parenthetical asides.
_BRACKET_CUE_RE = re.compile(r"\[[^\]]*\]")
_PAREN_CUE_RE = re.compile(
    r"""\(
        \s*
        (?:laughs?|laughing|laughter|music|applause|crosstalk|inaudible
           |unintelligible|sighs?|pause|silence|noise|chuckles?|coughs?)
        [^)]*
    \)""",
    re.IGNORECASE | re.VERBOSE,
)
_WS_RE = re.compile(r"\s+")


def clean_text(s: str) -> str:
    """Collapse whitespace and strip leftover non-speech stage directions.

    Removes ``[...]`` bracket cues entirely (``[laughs]``, ``[music]``,
    ``[inaudible 00:12]``) and a conservative set of parenthetical non-speech
    cues (``(laughs)``, ``(applause)``), then squeezes all runs of whitespace —
    including newlines from wrapped source lines — down to single spaces and
    strips the ends. Legitimate parenthetical asides (``(more on that later)``)
    are left intact.

    Returns ``""`` for input that is empty or only cues/whitespace, so callers
    can drop content-free turns.
    """

    s = _BRACKET_CUE_RE.sub(" ", s)
    s = _PAREN_CUE_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()
