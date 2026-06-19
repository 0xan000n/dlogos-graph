"""The Graphiti × open-weight extraction spike (spec §7.6).

Resolves the #1 risk *before* scaling: which integration shape wins, and does
it clear the throughput / $-per-episode gate (not only a quality bar)?

- :mod:`dlogos.spike.run_comparison` orchestrates **Approach A** (Graphiti
  native extraction via an open-weight endpoint) vs **Approach B** (our
  extractor -> resolution -> bulk load) over N transcript fixtures, capturing
  per-approach artifacts. Every collaborator is injected so it runs fully
  offline against mocks.
- :mod:`dlogos.spike.score` computes the spike metrics — throughput
  (episodes/min), estimated $/episode, claims/episode, %valid-source-span,
  %valid-speaker-attribution, and (Approach B) JSON-parse-success-rate — and
  emits ``report.md`` + ``report.json``.

Nothing heavy is imported at module load; the real Graphiti/extractor clients
are only constructed by the caller and handed in.
"""

from __future__ import annotations

from dlogos.spike.run_comparison import (
    ApproachArtifacts,
    ComparisonResult,
    SpikeRunner,
    run_spike,
)
from dlogos.spike.score import (
    ApproachScore,
    SpikeReport,
    emit_report,
    render_report_md,
    score_approach,
    score_comparison,
)

__all__ = [
    "ApproachArtifacts",
    "ApproachScore",
    "ComparisonResult",
    "SpikeReport",
    "SpikeRunner",
    "emit_report",
    "render_report_md",
    "run_spike",
    "score_approach",
    "score_comparison",
]
