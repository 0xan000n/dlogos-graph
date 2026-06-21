"""Parser for *The Great Simplification* (Nate Hagens) PDF transcripts.

Source host: ``www.thegreatsimplification.com`` — each episode is published as a
PDF whose extracted text is a flat run of speaker turns prefixed with a wall
clock timestamp and the speaker's real name::

    [00:00:00] Tristan Harris: There's a lot of different risks from ai. ...
    [00:00:47] Nate Hagens: Today I'm pleased to be joined by ...

Two real-world wrinkles this parser handles, both visible in the committed
fixture (``tests/fixtures/transcripts/tgs.txt``, a trimmed real extraction of
TGS-214):

* **Continuation timestamps.** A long turn carries *bare* ``[HH:MM:SS]`` markers
  with no following ``Name:`` — they timestamp a later sentence *within the same
  speaker's* turn. Those do NOT start a new turn; their inline marker is stripped
  and the text folds into the current turn.
* **Page-footer boilerplate.** The PDF's running footer (``"1 The Great
  Simplification"``, the auto-generated ``PLEASE NOTE: ...`` header) gets spliced
  into the extracted text. We drop everything before the first real turn and
  scrub the recurring ``N The Great Simplification`` footer fragment mid-turn.

This module is **pure** and **stdlib-only** (regex over already-extracted
readable text) per the parser contract in :mod:`dlogos.ingestion.parsers`. The
PDF→text step lives in the separate, lazily-imported fetch layer; this function
only ever sees text. TGS carries native timestamps, so real ``t_start`` values
are parsed and handed to :func:`~dlogos.ingestion.parsers._util.finalize_segments`
(no span synthesis needed for the timestamped turn-starts).
"""

from __future__ import annotations

import re

from dlogos.ingestion.parsers._util import Row, finalize_segments, parse_hms
from dlogos.schema import TranscriptSegment

__all__ = ["parse"]

# A bare timestamp marker: ``[H:MM:SS]`` (hours 1-2 digits). Used both to find
# turn boundaries (when followed by ``Name:``) and to strip continuation markers.
_TS = r"\[\d{1,2}:\d{2}:\d{2}\]"

# A speaker label: a timestamp, then 1-4 Capitalized name tokens, then a colon.
# Name tokens allow internal apostrophes/hyphens/periods (``O'Brien``, ``Jr.``).
# The colon after the name is what separates this from a *continuation* marker
# (a bare ``[ts]`` with no ``Name:``), so continuations never split a turn.
_TURN_RE = re.compile(
    r"(?P<ts>" + _TS + r")\s+"
    r"(?P<name>[A-Z][A-Za-z.'\-]*(?:\s+[A-Z][A-Za-z.'\-]*){0,3})\s*:"
)

# Bare continuation timestamp to strip out of a turn's body text.
_CONT_TS_RE = re.compile(_TS)

# The recurring PDF page-footer fragment, spliced mid-text by extraction, e.g.
# ``"... operates for 1 The Great Simplification the common good ..."``. The
# optional page number and the ``fi`` ligature variant are both tolerated.
_FOOTER_RE = re.compile(
    r"\s*\d*\s*The\s+Great\s+Simpli(?:fi|ﬁ)cation\s*",
    re.IGNORECASE,
)


def parse(text: str) -> list[TranscriptSegment]:
    """Parse a TGS transcript into ordered, timestamped segments.

    Splits ``text`` on ``[HH:MM:SS] Name:`` turn boundaries; bare ``[HH:MM:SS]``
    continuation markers fold into the current speaker's turn (their inline
    timestamp stripped). The leading auto-generated header / any nav text before
    the first real turn is dropped, and the recurring page-footer fragment is
    scrubbed from turn bodies. Each turn's wall-clock start is parsed via
    :func:`~dlogos.ingestion.parsers._util.parse_hms`; rows are finalized into
    non-overlapping, monotonic segments by
    :func:`~dlogos.ingestion.parsers._util.finalize_segments` (which also cleans
    the text and drops content-free turns).

    Pure and total: no I/O, never raises on arbitrary input; returns ``[]`` when
    no speaker turn is present.
    """

    matches = list(_TURN_RE.finditer(text))
    if not matches:
        return []

    rows: list[Row] = []
    for i, m in enumerate(matches):
        # Body runs from just after this label to the start of the next label
        # (or end of text for the final turn). Continuation timestamps inside the
        # body are stripped; the page-footer fragment is scrubbed.
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end]
        body = _CONT_TS_RE.sub(" ", body)
        body = _FOOTER_RE.sub(" ", body)

        speaker = re.sub(r"\s+", " ", m.group("name")).strip()
        t_start = parse_hms(m.group("ts"))
        rows.append((speaker, body, t_start))

    return finalize_segments(rows)
