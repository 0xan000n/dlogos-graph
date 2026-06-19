"""Open-weight claim extraction (§7.4).

Three stages, all built on core deps only so unit tests need no heavy
extras and no network:

- :mod:`dlogos.extraction.chunking` — split a :class:`~dlogos.schema.Transcript`
  into overlapping, speaker-labelled windows that never split a segment.
- :mod:`dlogos.extraction.predicates` — map free model output onto the closed
  :class:`~dlogos.schema.Predicate` enum (controlled vocabulary), rejecting
  anything unmappable.
- :mod:`dlogos.extraction.extractor` — an async, OpenAI-compatible extractor
  that turns chunks into validated :class:`~dlogos.schema.ExtractedClaim`
  records, enforcing source spans and the controlled predicate vocabulary.
"""

from __future__ import annotations

from dlogos.extraction.chunking import Chunk, ChunkSegment, chunk_transcript
from dlogos.extraction.extractor import (
    ClaimExtractor,
    ExtractionError,
    PredicateVocabularyError,
)
from dlogos.extraction.predicates import (
    PREDICATE_SYNONYMS,
    PredicateMappingError,
    map_predicate,
    try_map_predicate,
)

__all__ = [
    "Chunk",
    "ChunkSegment",
    "chunk_transcript",
    "ClaimExtractor",
    "ExtractionError",
    "PredicateVocabularyError",
    "PREDICATE_SYNONYMS",
    "PredicateMappingError",
    "map_predicate",
    "try_map_predicate",
]
