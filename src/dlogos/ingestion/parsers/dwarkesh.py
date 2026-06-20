"""Parser for Dwarkesh Podcast web transcripts (``www.dwarkesh.com``).

The Dwarkesh transcript page (extracted to readable text by the separate
fetch layer) lays each speaker turn out as a **three-part block**::

    Dwarkesh Patel <-- speaker name on its own line (often a trailing space)
    00:00:00      <-- a bare ``HH:MM:SS`` timestamp on the very next line
    Today I'm speaking with ... <-- the utterance, spanning the lines until the
    ...               next ``name`` + ``timestamp`` block (or end of transcript)

Because every turn carries a real ``HH:MM:SS`` offset we parse it (via
:func:`dlogos.ingestion.parsers._util.parse_hms`) rather than synthesizing —
ordering *and* approximate absolute timing are faithful here.

Boilerplate the page emits around the transcript is dropped by construction:
the nav/title/byline, the "Timestamps" table of contents (whose rows look like
``(00:00:00) – label`` — a *parenthesized* timestamp) and the per-section
headers (``00:00:00 – label`` — a timestamp *followed by text on the same
line*) never match the turn shape, which requires a name line **immediately
followed by a line that is nothing but a timestamp**. Everything before the
first real turn is discarded.

Pure + stdlib-only: regex over the already-extracted text, no network/IO.
"""

from __future__ import annotations

import re

from dlogos.ingestion.parsers._util import Row, finalize_segments, parse_hms
from dlogos.schema import TranscriptSegment

__all__ = ["parse"]

# A line that is *only* an ``HH:MM:SS`` (or ``H:MM:SS``) timestamp — no
# surrounding text. This is the second line of every speaker block. We anchor
# both ends so a section header like ``00:00:00 – AGI is still a decade away``
# (timestamp + text) does NOT match: that line carries trailing prose.
_TIMESTAMP_ONLY_RE = re.compile(r"^\d{1,2}:\d{2}:\d{2}$")

# A plausible speaker-name line: 1–5 capitalized/cased words, letters plus a
# few name punctuation marks (``.'-``), no digits, no sentence punctuation.
# Matches "Dwarkesh Patel", "Andrej Karpathy"; rejects prose lines (which carry
# lowercase-initial words, digits, or terminal punctuation) and the parenthesized
# TOC rows. Kept deliberately tight so a mid-sentence wrapped line cannot pass.
_NAME_RE = re.compile(
    r"^[A-Z][\w.'-]*(?: [A-Z][\w.'-]*){0,4}$",
    re.UNICODE,
)


def _is_name(line: str) -> bool:
    """True when ``line`` looks like a standalone speaker-name label."""

    return bool(_NAME_RE.match(line.strip()))


def parse(text: str) -> list[TranscriptSegment]:
    """Parse a Dwarkesh web transcript into ordered :class:`TranscriptSegment`s.

    Detects the recurring ``name`` / ``HH:MM:SS`` / ``utterance...`` blocks,
    attaches each block's text up to the next block, parses the real timestamp
    as the turn start, and funnels everything through
    :func:`~dlogos.ingestion.parsers._util.finalize_segments` (which fills each
    turn's end up to the following turn's start, cleans text, drops empties, and
    guarantees monotonic non-overlapping spans). Boilerplate before the first
    turn is dropped; ``speaker`` is the human name exactly as written.
    """

    lines = text.splitlines()
    rows: list[Row] = []

    # Accumulator for the turn currently being read.
    cur_speaker: str | None = None
    cur_start: float | None = None
    cur_text: list[str] = []

    def _flush() -> None:
        if cur_speaker is not None:
            rows.append((cur_speaker, "\n".join(cur_text), cur_start))

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        # A new turn begins when this line is a name AND the next non-skipped
        # line is a bare timestamp. We peek the immediate next line only — the
        # source always places the timestamp directly under the name.
        nxt = lines[i + 1].strip() if i + 1 < n else ""
        if stripped and _is_name(stripped) and _TIMESTAMP_ONLY_RE.match(nxt):
            _flush()
            cur_speaker = stripped
            cur_start = parse_hms(nxt)
            cur_text = []
            i += 2  # consume the name line and the timestamp line
            continue

        # Otherwise it's body text for the open turn (or pre-first-turn
        # boilerplate, which is ignored because no turn is open yet).
        if cur_speaker is not None:
            cur_text.append(line)
        i += 1

    _flush()
    return finalize_segments(rows)
