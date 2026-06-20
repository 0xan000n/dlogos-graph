"""Parser for lexfridman.com episode transcripts (stdlib-only, pure).

Lex Fridman publishes each episode's transcript as a page of ``ts-segment``
blocks; the HTML→text fetch layer renders each block as one readable line of the
form::

    Lex Fridman (00:00:00) It's hard for us humans to make any kind of ...
    Demis Hassabis (00:00:12) Yes, exactly. I mean, fluid dynamics ...
    (00:00:27) But again, if you look at something like Veo ...

So a *turn header* is a human speaker name followed by a ``(HH:MM:SS)``
timestamp; the rest of the line is that turn's text. A long monologue is split
across several blocks, and every continuation block after the first has an
**empty speaker name** — the line begins directly with ``(HH:MM:SS) ...``. We
carry the last seen speaker forward across those continuation lines so the human
name is attached to every segment.

The real timestamps are parsed (via :func:`~dlogos.ingestion.parsers._util.parse_hms`)
and handed to :func:`~dlogos.ingestion.parsers._util.finalize_segments`, which
honors them while guaranteeing monotonic, non-overlapping spans. Lines before
the first timestamped turn (page title, "Episode links", sponsor blurb) and any
trailing boilerplate carry no ``(HH:MM:SS)`` header and so are never started —
they fall away naturally.

Pure and stdlib-only: regex over already-extracted readable text. It never
fetches or parses HTML itself.
"""

from __future__ import annotations

import re

from dlogos.schema import TranscriptSegment

from ._util import Row, finalize_segments, parse_hms

__all__ = ["parse"]

# A turn header at the start of a line: an optional speaker name, then a
# ``(HH:MM:SS)`` (or ``(MM:SS)``) timestamp, then the turn text. The name is
# everything before the timestamp on the line; it is empty for continuation
# blocks of an ongoing monologue (line begins ``(00:00:27) ...``).
#
#   group "name" — speaker label as printed (may be empty → carry forward)
#   group "ts"   — the bracketed timestamp, fed to parse_hms
#   group "text" — the rest of the line (this block's spoken text)
_TURN_RE = re.compile(
    r"""
    ^
    (?P<name>.*?)                       # speaker name (non-greedy; may be empty)
    \s*
    (?P<ts>\(\d{1,2}:\d{2}(?::\d{2})?\))  # (HH:MM:SS) or (MM:SS) timestamp
    \s*
    (?P<text>.*)                        # the turn's spoken text
    $
    """,
    re.VERBOSE,
)


def parse(text: str) -> list[TranscriptSegment]:
    """Parse a lexfridman.com readable transcript into ordered segments.

    Recognizes the recurring ``Name (HH:MM:SS) text`` turn-header line (with
    empty-name continuation lines inheriting the previous speaker), parses the
    real timestamps, and returns monotonic, non-overlapping
    :class:`~dlogos.schema.TranscriptSegment`s. Boilerplate before the first
    timestamped turn and after the last is dropped (no header → never started).

    Pure: no network, filesystem, or clock access. Returns ``[]`` when no turn
    header is present.
    """

    rows: list[Row] = []
    last_speaker: str | None = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        m = _TURN_RE.match(line)
        if m is None:
            # No timestamp header on this line. Before the first turn this is
            # nav/boilerplate; after a turn it is trailing boilerplate. Either
            # way there is no in-turn continuation text to attach (lexfridman
            # keeps each block on its own line), so we simply skip it.
            continue

        name = m.group("name").strip()
        speaker = name or last_speaker
        if speaker is None:
            # A bare ``(HH:MM:SS) ...`` line before any named turn — no speaker
            # to attribute it to yet; skip until we've seen the first name.
            continue

        last_speaker = speaker
        t_start = parse_hms(m.group("ts"))
        rows.append((speaker, m.group("text"), t_start))

    return finalize_segments(rows)
