"""Parser for The Jim Rutt Show transcript pages (``www.jimruttshow.com``).

The show publishes one HTML page per episode under
``/the-jim-rutt-show-transcripts/transcript-of-ep-.../``. Once that page is run
through the shared HTML→text extractor (BeautifulSoup ``get_text("\\n")``), every
speaker turn lands as a **bare name line followed by a colon-prefixed text
line**, because the source markup wraps the name in its own inline element::

    Jim
    : Quick reminder to folks to check out my Substack ...
    Nate
    : Thanks for having me.

A single turn may spill across several paragraphs; the continuation paragraphs
are plain prose lines with *no* name/colon prefix and belong to the turn above::

    Jim
    : Right. But I also want to make clear ...
    And in fact, with respect to my own time on this issue ...   <- still Jim
    And I do remember at the time ...                             <- still Jim

There are no timestamps anywhere on these pages, so spans are synthesized
monotonically by word count via :func:`~dlogos.ingestion.parsers._util.\
synthesize_spans`. Speaker labels are kept verbatim ("Jim", "Nate", "Joe",
"Brendan") — the human name exactly as it appears — for the name-driven resolver
to canonicalize downstream.

This module is **pure** and **stdlib-only**: it parses an already-extracted
readable string. It never fetches URLs or parses HTML itself.
"""

from __future__ import annotations

import re

from dlogos.schema import TranscriptSegment

from ._util import synthesize_spans

__all__ = ["parse"]


# A speaker *label* line: a short bare personal name as the source renders it —
# one to four whitespace-separated Capitalized tokens, nothing else on the line.
# This matches "Jim", "Nate", "Joe", "Brendan", and (defensively) a two- or
# three-word name, while rejecting prose, the colon-text lines, and nav noise.
_LABEL_RE = re.compile(
    r"""
    ^\s*
    (?P<name>
        [A-Z][\w.'’-]*            # first capitalized token
        (?:\s+[A-Z][\w.'’-]*){0,3}  # up to 3 more capitalized tokens
    )
    \s*$
    """,
    re.VERBOSE,
)

# The colon-prefixed text line that immediately follows a label. The leading
# colon is the source's "Name: text" separator after the inline name element was
# split onto its own line; we strip it and keep the rest as the turn's opener.
_COLON_TEXT_RE = re.compile(r"^\s*:\s?(?P<text>.*\S.*)$")

# Pre-transcript disclaimer that every page carries right before the first turn.
# Used only as an extra guard so any stray boilerplate above it is dropped.
_DISCLAIMER_RE = re.compile(r"rough transcript which has not been revised", re.I)

# Nav / chrome tokens that bracket the transcript body once the readable text is
# extracted (header menu above, subscribe/archive list below). Hitting one of
# these *after* the body has started ends the transcript — it can never be a
# continuation paragraph and is never a real speaker turn.
_NAV_LINES = frozenset(
    {
        "skip to content",
        "home",
        "transcripts",
        "subscribe",
        "about",
        "search for:",
        "apple podcasts",
        "android",
        "by email",
        "rss",
        "more subscribe options",
        "new posts",
        "archives",
        "recent posts",
    }
)


def _is_nav(line: str) -> bool:
    """True for a header/footer chrome line that bounds the transcript body."""

    return line.strip().lower() in _NAV_LINES


def parse(text: str) -> list[TranscriptSegment]:
    """Parse Jim Rutt Show readable transcript text into ordered segments.

    Detects the recurring ``Name`` / ``: text`` turn pattern, attaches each
    turn's continuation paragraphs, drops the surrounding nav/disclaimer
    boilerplate, and — since the source has no timestamps — lets
    :func:`synthesize_spans` assign monotonic, non-overlapping spans by word
    count. Speaker labels are preserved verbatim.

    Pure: no network, filesystem, or clock; same input → same output.
    """

    lines = text.splitlines()
    turns: list[tuple[str, list[str]]] = []  # (speaker, [text chunks])
    started = False  # have we entered the transcript body yet?

    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i]
        line = raw.strip()

        if not line:
            i += 1
            continue

        # A turn begins when a bare-name label line is immediately followed
        # (skipping blanks) by a colon-prefixed text line.
        label = _LABEL_RE.match(line)
        if label:
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            if j < n:
                colon = _COLON_TEXT_RE.match(lines[j].strip())
                if colon:
                    turns.append((label.group("name"), [colon.group("text")]))
                    started = True
                    i = j + 1
                    continue

        # Not a new turn. Before the body has started this is leading
        # boilerplate (nav, title, disclaimer) — skip it.
        if not started:
            i += 1
            continue

        # Inside the body: a nav line means we've reached the trailing chrome;
        # the transcript is over.
        if _is_nav(line) or _DISCLAIMER_RE.search(line):
            break

        # Otherwise this is a continuation paragraph of the current turn.
        if turns:
            turns[-1][1].append(line)
        i += 1

    pairs = [(speaker, " ".join(chunks)) for speaker, chunks in turns]
    return synthesize_spans(pairs)
