"""Parser for 80,000 Hours podcast transcripts (80000hours.org).

The 80,000 Hours episode pages render their transcript as readable text in a
very regular shape (this is what ``bs4 .get_text("\\n")`` yields):

- A **speaker label** is a line that is *exactly* a human name followed by a
  colon — ``"Rob Wiblin:"``, ``"Ryan Greenblatt:"``, ``"Yoshua Bengio:"``. The
  speaker's words follow on the subsequent lines until the next label.
- The transcript body is broken up by **section headers** that carry a real
  inline timestamp: ``"Cold open [00:00:00]"``,
  ``"Who's Ryan Greenblatt? [00:01:10]"``. These are not speaker turns; they
  anchor the chapter that follows to a real audio offset.

So this source gives us *real* timestamps for free — not per-utterance, but
per-chapter. We use each section header's ``[HH:MM:SS]`` to anchor the
``t_start`` of the next speaker turn, and let
:func:`~dlogos.ingestion.parsers._util.finalize_segments` fill the in-between
turns monotonically by word count. The result is ordered, non-overlapping, and
pinned to the true chapter boundaries.

Everything before the first speaker turn (page chrome, the "Highlights" teaser,
the table of contents) and the trailing site boilerplate carries no speaker
label, so it is naturally dropped: we only ever emit text that belongs to an
open turn.

Pure and stdlib-only — regex over the already-extracted readable text. The
fetch + HTML→text layer lives elsewhere and is not imported here.
"""

from __future__ import annotations

import re

from dlogos.ingestion.parsers._util import Row, finalize_segments, parse_hms
from dlogos.schema import TranscriptSegment

__all__ = ["parse"]


# A speaker label line: the whole (stripped) line is ``Name:`` and nothing more.
# The name is one or more capitalised-ish words — letters plus the punctuation
# that shows up in real names (``.'-`` and spaces, e.g. "Rob Wiblin",
# "Holden Karnofsky", "Ajeya Cotra"). We cap the word count and length so a
# stray sentence that merely ends in a colon ("So here's the thing:") cannot be
# mistaken for a speaker. The text after the colon must be empty — on this
# source the words always start on the *next* line.
_SPEAKER_RE = re.compile(
    r"""
    ^\s*
    (?P<name>
        [A-Z][\w.'’-]*           # first name token, starts uppercase
        (?:\s+[A-Z][\w.'’-]*){0,3}  # up to 3 more capitalised tokens
    )
    \s*:\s*$
    """,
    re.VERBOSE,
)

# A section/chapter header carrying a real timestamp, e.g.
# ``"Cold open [00:00:00]"`` or ``"Why AI companies ... [00:13:01]"``. The line
# *ends* with a bracketed ``[HH:MM:SS]`` and is not itself a speaker label.
_SECTION_TS_RE = re.compile(r"\[\s*\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?\s*\]\s*$")


def _speaker_of(line: str) -> str | None:
    """Return the speaker name if ``line`` is a bare ``Name:`` label, else None."""

    m = _SPEAKER_RE.match(line)
    return m.group("name").strip() if m else None


def parse(text: str) -> list[TranscriptSegment]:
    """Parse an 80,000 Hours transcript into ordered :class:`TranscriptSegment`s.

    Walks the readable text line by line:

    - A bare ``Name:`` line opens a new turn for that speaker. Any pending
      section-header timestamp becomes that turn's anchored ``t_start``.
    - A ``... [HH:MM:SS]`` section header sets the pending timestamp for the
      *next* turn (it is not emitted as speech).
    - Every other line is appended to the currently open turn.

    Lines before the first speaker label (page chrome / TOC / Highlights teaser)
    and trailing boilerplate carry no open turn and are dropped. Rows are funnelled
    through :func:`finalize_segments`, which honours the anchored chapter
    timestamps and fills the rest monotonically — guaranteeing non-decreasing,
    non-overlapping spans.
    """

    rows: list[Row] = []
    cur_speaker: str | None = None
    cur_lines: list[str] = []
    cur_start: float | None = None
    pending_ts: float | None = None

    def flush() -> None:
        if cur_speaker is not None and cur_lines:
            rows.append((cur_speaker, " ".join(cur_lines), cur_start))

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        speaker = _speaker_of(line)
        if speaker is not None:
            # Close the previous turn, open a new one anchored to any pending
            # chapter timestamp.
            flush()
            cur_speaker = speaker
            cur_lines = []
            cur_start = pending_ts
            pending_ts = None
            continue

        if _SECTION_TS_RE.search(line):
            # A chapter header: remember its real offset for the next turn; the
            # header text itself is navigation, not speech.
            pending_ts = parse_hms(line)
            continue

        # Ordinary content line: only kept if a turn is open (drops pre-transcript
        # chrome and trailing boilerplate, which never sit under a speaker label).
        if cur_speaker is not None:
            cur_lines.append(line)

    flush()

    return finalize_segments(rows)
