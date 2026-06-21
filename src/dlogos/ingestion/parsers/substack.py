"""Parser for Substack-hosted transcripts (one parser, two layouts).

Source hosts: ``centerforhumanetechnology.substack.com`` (the *Your Undivided
Attention* / YUA episodes) and ``lironshapira.substack.com`` (the *Doom Debates*
episodes). Both are Substack post pages, but Substack renders transcript turns in
two distinct shapes depending on how the author pasted them, and a single
``parse`` handles both:

**Inline shape** (YUA) — speaker, colon, and text on one line::

    Tristan Harris: Tim and Janet, welcome to Your Undivided Attention.
    Tim Fist: Thanks for having me.

**Block shape** (Doom Debates) — the speaker name on its own line, then a blank
line, then an ``HH:MM:SS`` timestamp on its own line, then the turn's text on the
following line(s)::

    Liron
    <blank>
    00:15:17
    I wanted to actually ask you more on that topic ...

In the block shape, section headers (``"Cold Open"``, ``"Introducing Dr. Ben
Goertzel"``) sit on their own lines too; they are told apart from speaker labels
because a *speaker* label is followed (within a line or two) by a timestamp,
whereas a section header is followed directly by the next speaker name.

The parser is **pure** and **stdlib-only**: it works over the already-extracted
readable text (the fetch + HTML→text layer lives elsewhere). It never touches the
network, filesystem, or clock. Timestamps in the block shape are parsed as real
offsets via :func:`._util.parse_hms`; the inline shape has no per-turn timing, so
spans are synthesized monotonically by word count. Speaker labels are the human
names exactly as they appear, which the name-driven resolver canonicalizes
(``"Ben"`` / ``"Ben Goertzel"`` collapse there, not here).
"""

from __future__ import annotations

import re

from dlogos.schema import TranscriptSegment

from ._util import Row, finalize_segments, parse_hms, synthesize_spans

__all__ = ["parse"]


# --------------------------------------------------------------------------- #
# Patterns
# --------------------------------------------------------------------------- #
# A bare ``HH:MM:SS`` (or ``MM:SS``) line — the per-turn timestamp in the block
# shape. Anchored: the WHOLE line is just the timestamp (optionally bracketed),
# which is how Substack renders the Doom Debates layout.
_TS_LINE_RE = re.compile(r"^\s*[\[(]?\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?[\])]?\s*$")

# An inline "Name: text" turn. The label is 1–4 capitalized, name-like tokens
# (letters plus ``.'-``), then a colon, then at least one non-space char of
# speech on the same line. Requiring a leading capital + a name-shaped token and
# trailing text keeps prose lines that merely contain a colon ("two reasons:
# first ...") and bare footer fragments (": Tim referred to ...") from matching.
_NAME_TOKEN = r"[A-Z][A-Za-z.'’‘‐-]*"
_INLINE_TURN_RE = re.compile(
    rf"^(?P<speaker>{_NAME_TOKEN}(?:[ .]{_NAME_TOKEN}){{0,3}}):\s+(?P<text>\S.*)$"
)

# A standalone speaker-label line in the block shape: just a short name, nothing
# else. Same name shape as inline, no trailing colon/text. The real discriminator
# is the timestamp lookahead in :func:`_parse_block`; this just keeps the
# candidate set tight.
_LABEL_LINE_RE = re.compile(
    rf"^(?P<speaker>{_NAME_TOKEN}(?:[ .]{_NAME_TOKEN}){{0,3}})\s*$"
)

# Footer/boilerplate lines that can appear after the last real turn. We stop the
# block-shape scan when we hit one of these as a standalone line so trailing nav
# never becomes a phantom turn. (The inline shape is naturally self-limiting —
# footer lines don't match ``Name: text`` — but we drop these too for safety.)
_FOOTER_MARKERS = frozenset(
    {
        "subscribe",
        "leave a comment",
        "recommended media",
        "recommended yua episodes",
        "ready for more?",
        "discussion about this post",
        "share",
        "share post",
        "get the app",
        "substack",
        "previous",
        "comments",
        "top",
        "latest",
        "discussions",
    }
)


def _is_footer(line: str) -> bool:
    """A standalone nav/footer marker line (case-insensitive, exact-ish)."""

    s = line.strip().casefold()
    return s in _FOOTER_MARKERS or s.startswith("© ")


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def parse(text: str) -> list[TranscriptSegment]:
    """Parse a readable Substack transcript into ordered ``TranscriptSegment``s.

    Detects which of the two Substack layouts the text uses and dispatches to the
    matching scanner. Returns ``[]`` for text with no recognizable speaker turns.

    Pure: same input, same output; no I/O.
    """

    lines = text.splitlines()

    # The block shape repeats a standalone speaker-label line whose next non-blank
    # line is a bare timestamp, once per turn — so a real Doom-Debates page shows
    # this pattern many times. We require ≥2 occurrences to classify as block
    # shape, since the YUA player chrome produces exactly one accidental
    # ``Share`` / ``0:00`` label/timestamp pair that must not hijack detection.
    if _count_block_turns(lines) >= 2:
        return _parse_block(lines)
    return _parse_inline(lines)


# --------------------------------------------------------------------------- #
# Layout detection
# --------------------------------------------------------------------------- #
def _count_block_turns(lines: list[str]) -> int:
    """How many ``label-line → (blank?) → timestamp-line`` pairs are present.

    One per turn in the block shape. A lone accidental pair (e.g. a ``Share`` nav
    line above the audio player's ``0:00``) stays under the ≥2 threshold the
    caller uses, so it never misclassifies an inline page as block shape.
    """

    return sum(
        1
        for i, line in enumerate(lines)
        if _LABEL_LINE_RE.match(line) and _next_is_timestamp(lines, i)
    )


def _next_is_timestamp(lines: list[str], i: int) -> bool:
    """Is the next non-blank line after index ``i`` a bare timestamp line?

    The block shape inserts a blank line between the speaker name and its
    timestamp, so we skip blanks (a bounded peek) before checking.
    """

    j = i + 1
    while j < len(lines) and not lines[j].strip():
        j += 1
    return j < len(lines) and bool(_TS_LINE_RE.match(lines[j]))


# --------------------------------------------------------------------------- #
# Block shape (Doom Debates / Liron Shapira) — real timestamps
# --------------------------------------------------------------------------- #
def _parse_block(lines: list[str]) -> list[TranscriptSegment]:
    """Parse the ``Name / <blank> / HH:MM:SS / text...`` block layout.

    A turn starts at a speaker-label line whose following non-blank line is a
    timestamp; the timestamp is parsed as a real start offset, and every line
    after it up to the next speaker label (or a footer marker / EOF) is the turn's
    text. Standalone section headers — label-shaped lines NOT followed by a
    timestamp — are skipped. Boilerplate before the first turn is implicitly
    dropped (we only emit once the first timestamped label is found).
    """

    rows: list[Row] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if _is_footer(line):
            # Footer reached after at least one turn → stop; before any turn it is
            # just header nav, so keep scanning.
            if rows:
                break
            i += 1
            continue

        if _LABEL_LINE_RE.match(line) and _next_is_timestamp(lines, i):
            speaker = line.strip()
            # Advance to the timestamp line (skipping the blank separator).
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            t_start = parse_hms(lines[j])
            # Collect text until the next speaker block / footer / EOF.
            body: list[str] = []
            k = j + 1
            while k < n:
                nxt = lines[k]
                if _is_footer(nxt):
                    break
                if _LABEL_LINE_RE.match(nxt) and _next_is_timestamp(lines, k):
                    break
                body.append(nxt)
                k += 1
            rows.append((speaker, "\n".join(body), t_start))
            i = k
            continue

        i += 1

    return finalize_segments(rows)


# --------------------------------------------------------------------------- #
# Inline shape (YUA / Center for Humane Technology) — synthesized spans
# --------------------------------------------------------------------------- #
def _parse_inline(lines: list[str]) -> list[TranscriptSegment]:
    """Parse the one-line ``Name: text`` layout (no per-turn timestamps).

    Each ``Name: text`` line opens a turn; subsequent non-label lines are
    continuation paragraphs appended to the current turn (unlabeled interstitial
    narration before the first labeled turn — e.g. the YUA host's cold-open
    monologue — is dropped as pre-roll, matching "drop boilerplate before the
    first speaker turn"). Footer markers after the last turn end the scan. Spans
    are synthesized monotonically by word count.
    """

    turns: list[tuple[str, str]] = []
    started = False
    for line in lines:
        if _is_footer(line):
            if started:
                break
            continue

        m = _INLINE_TURN_RE.match(line)
        if m:
            turns.append((m.group("speaker").strip(), m.group("text")))
            started = True
        elif started and line.strip():
            # Continuation paragraph of the current turn.
            spk, txt = turns[-1]
            turns[-1] = (spk, f"{txt}\n{line}")
        # else: pre-first-turn narration / blank → drop.

    return synthesize_spans(turns)
