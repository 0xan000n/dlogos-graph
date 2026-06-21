"""Parser for lexfridman.com episode transcripts (stdlib-only, pure).

Lex Fridman publishes each episode's transcript as a column of ``ts-segment``
blocks. After the HTML→text fetch layer (``get_text("\\n")``) renders the page,
each block becomes a **multi-line** run rather than a single line: the speaker
name, the ``(HH:MM:SS)`` timestamp, and the spoken text each land on their own
line, separated by blank lines::

    Lex Fridman
    (00:00:00)
    It's hard for us humans to make any kind of clean predictions ...

    Demis Hassabis
    (00:00:12)
    Yes, exactly. I mean, fluid dynamics, Navier-Stokes equations ...

    (00:00:27)
    But again, if you look at something like Veo, our video ...

So the **anchor of a turn is the bracketed timestamp line**, not the name. The
line immediately above a timestamp is the speaker name *when a new speaker takes
the floor*; a long monologue split across several blocks emits continuation
blocks whose timestamp is preceded directly by the previous turn's text (or a
chapter heading) — i.e. **no fresh name line** — so we carry the last speaker
forward.

Two things interleave with the turns and must not be mistaken for speakers:

* **Chapter headings** ("Introduction", "Learnable patterns in nature",
  "Google and the race to AGI", ...). Usually a heading is followed by a real
  name line before the next timestamp, so it never sits adjacent to a timestamp.
  But a few headings fall *mid-monologue*, landing directly above a continuation
  timestamp with no name line — structurally indistinguishable from a speaker
  header in isolation.
* **Pre-transcript chrome** (page title, bio, "Menu", nav) — carries no
  timestamp at all and falls away.

To separate a real speaker header from a one-off heading without hard-coding any
names, we use a cheap first pass: a real speaker recurs as a header dozens of
times across the episode, whereas a chapter heading appears at most once directly
above a timestamp. So a name candidate is adopted as a speaker only if it occurs
**more than once** in that header position; a singleton candidate is treated as a
heading and the previous speaker is carried forward.

Real timestamps are parsed (via
:func:`~dlogos.ingestion.parsers._util.parse_hms`) and handed to
:func:`~dlogos.ingestion.parsers._util.finalize_segments`, which honors them
while guaranteeing monotonic, non-overlapping spans.

Pure and stdlib-only: regex over already-extracted readable text. It never
fetches or parses HTML itself.
"""

from __future__ import annotations

import re
from collections import Counter

from dlogos.schema import TranscriptSegment

from ._util import Row, finalize_segments, parse_hms

__all__ = ["parse"]

# A standalone timestamp line: just ``(HH:MM:SS)`` (or ``(MM:SS)``), nothing
# else. This is the anchor that starts a turn.
_TS_LINE_RE = re.compile(r"^\(\d{1,2}:\d{2}(?::\d{2})?\)$")

# A plausible speaker-name line: short, and not ending in sentence punctuation,
# so a wrapped paragraph (continuation text that happens to be short) can't
# masquerade as a name. The decisive discriminator is recurrence (see module
# docstring); this only screens the candidates.
_NAME_MAX_LEN = 60
_SENTENCE_TAIL = ".?!,:;…"


def _is_name_candidate(line: str) -> bool:
    """A short, sentence-free line that *could* be a speaker label."""

    return 0 < len(line) <= _NAME_MAX_LEN and line[-1] not in _SENTENCE_TAIL


def parse(text: str) -> list[TranscriptSegment]:
    """Parse a lexfridman.com readable transcript into ordered segments.

    Walks the newline-joined extracted text. Each ``(HH:MM:SS)`` line starts a
    turn; its speaker is the recurring name line directly above it (when a new
    speaker takes the floor) or, for continuation blocks of an ongoing monologue
    (and for one-off chapter headings that fall mid-monologue), the last speaker
    carried forward. The block's spoken text is the non-blank lines that follow
    the timestamp up to the next timestamp or speaker line. Real timestamps drive
    the spans (monotonic, non-overlapping); chapter headings and all
    pre-transcript chrome are dropped (no timestamp → never started).

    Pure: no network, filesystem, or clock access. Returns ``[]`` when no
    timestamped turn is present.
    """

    # The page structure is carried by line *roles*, not blank spacing, so we
    # compact down to the non-blank lines.
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    n = len(lines)

    # First pass: count how often each name candidate sits directly above a
    # timestamp. Real speakers recur many times here; a chapter heading that
    # happens to fall mid-monologue appears at most once. Speaker headers are the
    # candidates that recur (> 1).
    header_counts: Counter[str] = Counter()
    for i in range(1, n):
        if _TS_LINE_RE.match(lines[i]) and _is_name_candidate(lines[i - 1]):
            header_counts[lines[i - 1]] += 1
    speakers = {name for name, c in header_counts.items() if c > 1}

    rows: list[Row] = []
    last_speaker: str | None = None

    i = 0
    while i < n:
        if not _TS_LINE_RE.match(lines[i]):
            i += 1
            continue

        # ``lines[i]`` is a timestamp. A recurring speaker header directly above
        # it hands the floor to that speaker; anything else (continuation text or
        # a one-off heading) inherits the previous speaker.
        if i >= 1 and lines[i - 1] in speakers:
            last_speaker = lines[i - 1]
        speaker = last_speaker

        t_start = parse_hms(lines[i])

        # Collect the turn text: every following non-blank line until the next
        # timestamp, or a recurring speaker header that introduces the next turn.
        j = i + 1
        parts: list[str] = []
        while j < n:
            nxt = lines[j]
            if _TS_LINE_RE.match(nxt):
                break
            # A speaker header for the *next* turn: a recurring name directly
            # above a timestamp. Stop before it so it isn't eaten as text.
            if j + 1 < n and _TS_LINE_RE.match(lines[j + 1]) and nxt in speakers:
                break
            parts.append(nxt)
            j += 1

        if speaker is not None:
            rows.append((speaker, " ".join(parts), t_start))
        i = j

    return finalize_segments(rows)
