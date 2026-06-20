"""Parser for singjupost.com podcast transcripts (``parse(text) -> segments``).

The Singju Post publishes hand-made podcast transcripts as HTML articles. Once
the page is rendered to readable text (the separate, lazy ``transcripts`` fetch
layer does that with BeautifulSoup — this module never touches the network), a
transcript looks like::

    TRANSCRIPT:
    Introduction
    SAM HARRIS:
     I'm here with Tristan Harris. Tristan, it's great to see you again.
    TRISTAN HARRIS:
     Sam, it's great to be back with you.
    From Social Media to AI
    TRISTAN HARRIS:
     Well, first, it's just good to be back with you, Sam...
     But to get to your question, so how did we get into AI?...

Structure this parser keys on
------------------------------
* **Speaker label** — an ALL-CAPS human name ending in a colon, e.g.
  ``SAM HARRIS:`` or ``JOHN VERVAEKE:``. The label may sit on its **own line**
  (the common case, text follows on the next line) *or* **inline** before the
  turn's first sentence (``JOHN VERVAEKE: Well, I mean...`` — seen on the
  multi-speaker Unbelievable episode). Both forms are handled.
* **Continuation lines** — a turn's later paragraphs arrive as separate,
  label-less lines; they are appended to the turn in progress.
* **Section headings** — short editorial sub-heads (``Introduction``,
  ``The AI Dilemma: Two Choices``) are interleaved between turns. They are
  *not* speech and are dropped.
* **Boilerplate** — site chrome before the first speaker turn (the leading
  ``TRANSCRIPT:`` marker, share buttons, editor's notes) and after the last
  (``Related Posts``, category lists, footer) is discarded by only emitting the
  span between the first real speaker label and the trailing nav.

Singju Post transcripts carry no timestamps, so spans are synthesized
monotonically by word count via :func:`._util.synthesize_spans`; ordering is
faithful, absolute timing is not (the explicit goal of this run).

Pure and stdlib-only: regex over the already-extracted string, no I/O.
"""

from __future__ import annotations

import re

from dlogos.schema import TranscriptSegment

from ._util import synthesize_spans

__all__ = ["parse"]


# A speaker label: one or more ALL-CAPS name tokens (letters, optional internal
# ``.`` / ``'`` / ``’`` / ``-`` as in ``O'BRIEN`` or ``J.D.``), then a colon.
# ``(?P<rest>...)`` captures any inline turn text that follows on the same line.
# Anchored at the (stripped) line start; the colon must come before any
# lowercase letter so a normal sentence with a mid-line colon is never a label.
_SPEAKER_RE = re.compile(
    r"^(?P<name>[A-Z][A-Z.'’\-]*(?:\s+[A-Z][A-Z.'’\-]*){0,4}):(?P<rest>.*)$"
)

# ALL-CAPS labels that are site chrome, not people. A name-shaped line whose
# label is one of these is dropped even though it matches the speaker pattern.
_BOILERPLATE_LABELS = frozenset(
    {
        "TRANSCRIPT",
        "RECOMMENDED",
        "ALSO READ",
        "LATEST POSTS",
        "CATEGORIES",
        "MISSION STATEMENT",
        "TERMS OF USE",
        "PRIVACY POLICY",
        "EDITOR’S NOTES",
        "EDITOR'S NOTES",
        "ADVERTISEMENT",
        "SHARE",
        "TWEET",
    }
)

# Lines (after the transcript starts) that mark the tail nav / related-content
# block. Everything from the first such line on is footer chrome — stop there.
_END_MARKERS = frozenset(
    {
        "Related Posts",
        "Continue Reading",
        "LATEST POSTS:",
        "RECOMMENDED:",
    }
)

# A speech line almost always ends in terminal punctuation (``. ? ! ” " …``) —
# possibly followed by a closing quote/paren. Section sub-heads (``Introduction``,
# ``The AI Dilemma: Two Choices``) do not. We use this, plus a word-count floor,
# to tell a turn's continuation paragraph from an interleaved heading.
_SENTENCE_END_RE = re.compile(r"[.?!…”\"'’)]\s*$")

#: A label-less line with at least this many words is treated as speech even if
#: it lacks terminal punctuation (a turn paragraph cut mid-sentence by the
#: source), so we never mistake a long utterance for a heading.
_MIN_SPEECH_WORDS = 12


def _label(line: str) -> tuple[str, str] | None:
    """Return ``(speaker_name, inline_text)`` if ``line`` is a speaker label.

    ``inline_text`` is whatever followed the colon on the same line (often empty
    for own-line labels). Returns ``None`` for non-label lines and for ALL-CAPS
    site-chrome labels in :data:`_BOILERPLATE_LABELS`.
    """

    m = _SPEAKER_RE.match(line.strip())
    if m is None:
        return None
    name = m.group("name").strip()
    if name.upper() in _BOILERPLATE_LABELS:
        return None
    return name, m.group("rest").strip()


def _is_heading(line: str) -> bool:
    """True for a short, label-less editorial sub-head (no terminal punctuation).

    Continuation speech paragraphs end in sentence punctuation or run long; a
    heading like ``The AI Dilemma: Two Choices`` does neither, so it is dropped
    rather than glued onto the current turn.
    """

    s = line.strip()
    if not s:
        return False
    if _SENTENCE_END_RE.search(s):
        return False
    return len(s.split()) < _MIN_SPEECH_WORDS


def parse(text: str) -> list[TranscriptSegment]:
    """Parse a Singju Post transcript string into ordered segments.

    Walks the readable text line by line: site chrome before the first speaker
    label is skipped, then each ``NAME:`` turn (own-line or inline) opens a turn
    whose following label-less paragraphs are appended to it, interleaved section
    headings are dropped, and the trailing related-posts / footer nav ends
    parsing. Spans are synthesized monotonically by word count.

    Returns ``[]`` if no speaker turn is found (e.g. a non-transcript page).
    """

    turns: list[tuple[str, str]] = []
    current_speaker: str | None = None
    current_parts: list[str] = []

    def _flush() -> None:
        if current_speaker is not None and current_parts:
            turns.append((current_speaker, " ".join(current_parts)))

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        # Once the body starts, the related-posts / footer block ends it.
        if current_speaker is not None and line in _END_MARKERS:
            break

        labelled = _label(line)
        if labelled is not None:
            speaker, inline = labelled
            _flush()
            current_speaker = speaker
            current_parts = [inline] if inline else []
            continue

        # Before the first real speaker label everything is page chrome.
        if current_speaker is None:
            continue

        # Label-less line inside a turn: speech paragraph vs. section heading.
        if _is_heading(line):
            continue
        current_parts.append(line)

    _flush()

    return synthesize_spans(turns)
