"""Pure text-transcript parsers for the dLogos ingestion front-end.

These parsers turn an *already-extracted, readable* transcript string into the
shared :class:`~dlogos.schema.TranscriptSegment` list the pipeline consumes —
the same shape the ASR/diarization path emits, so everything downstream
(speaker resolution, extraction, grounding) is unchanged. The public transcripts
in ``docs/corpus`` already carry real speaker *names* (``"Jim Rutt:"``,
``"Tristan Harris:"``), which the name-driven resolver
(:mod:`dlogos.speakers.speaker_store`) canonicalizes directly — strictly better
than diarization labels.

The parser contract
-------------------
Every parser is a **pure** function::

    parse(text: str) -> list[TranscriptSegment]

- *Pure*: no network, no filesystem, no clock — same input, same output.
- *Input* is readable text only. Parsers NEVER fetch URLs or parse HTML/PDF
  themselves; the fetch + HTML→text + PDF→text layer is separate and lazy-imports
  its (optional ``transcripts``-extra) deps.
- ``speaker`` is the human name exactly as it appears in the source.
- Real timestamps are parsed where the source has them (e.g. Dwarkesh/Lex
  ``HH:MM:SS``). Where absent, spans are *synthesized* monotonically by
  cumulative word count (~0.4s/word) so ordering is correct — citation precision
  is explicitly not this run's goal, entity/speaker structuring is.

Parsers are **stdlib-only** (regex over the readable text) and reuse the shared
helpers in :mod:`dlogos.ingestion.parsers._util`.
"""

from __future__ import annotations

__all__: list[str] = []
