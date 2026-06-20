"""Entity-fragmentation report — the resolution-quality metric.

The whole point of the incremental, Wikidata-anchored resolver is that the same
real-world entity becomes **one** canonical node across episodes. This module
turns "does it shatter?" into a number: for a small set of hand-picked
*probes* (e.g. "the Fed", "OpenAI", a recurring guest), count how many
**distinct** ``canonical_id``s that entity actually occupies in the loaded
graph. One canonical node per probe is perfect; anything more is fragmentation.

A node counts toward a probe when either:

- its display ``name`` or any of its ``aliases`` (surface-normalized) intersects
  the probe's normalized alias set, **or**
- its ``wikidata_qid`` equals the probe's ``qid`` (the anchor wins even when the
  surface form embeds/reads far away, e.g. "the iPhone maker" → ``Q312``).

The function is **pure** and operates directly on a list of
:class:`~dlogos.graph.store.EntityNode` (pull them from a store's ``.entities``
values, e.g. ``list(store.entities.values())``). It reuses
:func:`~dlogos.resolution.subjects._normalize_surface` so probe/alias matching
collapses case and whitespace exactly the way resolution does. Stdlib + the
existing schema types only — no heavy imports.
"""

from __future__ import annotations

import string
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from dlogos.graph.store import EntityNode
from dlogos.resolution.subjects import _normalize_surface


@dataclass(frozen=True)
class Probe:
    """A probed real-world entity whose fragmentation we want to measure.

    ``name`` is the human-readable label used in the report. ``aliases`` are the
    surface forms (any casing/spacing) we expect the entity to appear under;
    they are normalized at match time. ``qid`` is an optional Wikidata anchor —
    when set, any node carrying that QID counts, regardless of surface form.
    """

    name: str
    aliases: list[str] = field(default_factory=list)
    qid: str | None = None


@dataclass(frozen=True)
class ProbeResult:
    """Per-probe outcome: how many distinct canonical nodes the entity occupies.

    ``fragments`` is the count of distinct ``canonical_id``s that matched the
    probe; ``canonical_ids`` is the matched set itself (sorted, for legibility
    and stable output).
    """

    probe: Probe
    fragments: int
    canonical_ids: list[str]


@dataclass(frozen=True)
class FragReport:
    """The full fragmentation report over all probes.

    ``per_probe`` preserves the input probe order. ``mean_fragments`` is the
    average fragment count across probes (0.0 when there are no probes).
    ``worst`` is the most-fragmented probe result, or ``None`` when there are no
    probes.
    """

    per_probe: list[ProbeResult]
    mean_fragments: float
    worst: ProbeResult | None


def _norm_set(values: Iterable[str]) -> set[str]:
    """Surface-normalize an iterable of strings into a set (empties dropped).

    Reuses :func:`_normalize_surface` (casefold + whitespace collapse) and then
    trims surrounding punctuation so "Apple Inc." matches the probe alias
    "apple inc" — probe matching is intentionally more forgiving than the
    exact-name resolution key.
    """
    out: set[str] = set()
    for v in values:
        norm = _normalize_surface(v).strip(string.punctuation + " ")
        if norm:
            out.add(norm)
    return out


def _node_matches(node: EntityNode, probe_aliases: set[str], probe_qid: str | None) -> bool:
    """Whether one entity node should count toward a probe.

    Anchor (QID) match wins outright; otherwise the node's normalized
    name/aliases must intersect the probe's normalized alias set.
    """
    if probe_qid and node.wikidata_qid == probe_qid:
        return True
    if not probe_aliases:
        return False
    node_surfaces = _norm_set([node.name, *node.aliases])
    return bool(node_surfaces & probe_aliases)


def fragmentation_report(
    entity_nodes: Sequence[EntityNode],
    probes: Sequence[Probe],
) -> FragReport:
    """Count distinct canonical nodes per probe and summarize the spread.

    Parameters
    ----------
    entity_nodes:
        The canonical :class:`~dlogos.graph.store.EntityNode` records from the
        loaded graph (e.g. ``list(store.entities.values())``).
    probes:
        The entities to measure. Each contributes one :class:`ProbeResult`.

    Returns
    -------
    FragReport
        Per-probe fragment counts (input order preserved), the mean fragment
        count across probes, and the worst (most-fragmented) probe result. With
        no probes, ``mean_fragments`` is ``0.0`` and ``worst`` is ``None``.
    """
    per_probe: list[ProbeResult] = []
    for probe in probes:
        probe_aliases = _norm_set(probe.aliases)
        matched: set[str] = set()
        for node in entity_nodes:
            if _node_matches(node, probe_aliases, probe.qid):
                matched.add(node.canonical_id)
        per_probe.append(
            ProbeResult(
                probe=probe,
                fragments=len(matched),
                canonical_ids=sorted(matched),
            )
        )

    if per_probe:
        mean_fragments = sum(r.fragments for r in per_probe) / len(per_probe)
        worst = max(per_probe, key=lambda r: r.fragments)
    else:
        mean_fragments = 0.0
        worst = None

    return FragReport(per_probe=per_probe, mean_fragments=mean_fragments, worst=worst)
