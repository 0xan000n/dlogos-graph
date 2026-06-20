"""Tests for the entity-fragmentation report (resolution-quality metric).

The report turns "does the same real-world entity shatter into many canonical
nodes across episodes?" into a number: per probe, how many *distinct*
``canonical_id``s does that entity occupy. One is perfect; more is fragmented.
"""

from __future__ import annotations

from dlogos.eval.fragmentation import Probe, fragmentation_report
from dlogos.graph.store import EntityNode
from dlogos.schema import EntityType


def _ent(canonical_id: str, name: str, *, aliases=None, qid=None) -> EntityNode:
    return EntityNode(
        canonical_id=canonical_id,
        name=name,
        type=EntityType.organization,
        aliases=list(aliases or []),
        wikidata_qid=qid,
    )


def test_counts_distinct_canonical_ids_per_probe() -> None:
    # Apple shattered into 3 separate canonical nodes; OpenAI is whole (1).
    nodes = [
        _ent("ent-a1", "Apple", aliases=["apple inc"]),
        _ent("ent-a2", "the iPhone maker"),
        _ent("ent-a3", "Apple Inc."),
        _ent("ent-o1", "OpenAI", aliases=["openai"]),
    ]
    probes = [
        Probe("Apple", aliases=["apple", "apple inc", "the iphone maker"]),
        Probe("OpenAI", aliases=["openai"]),
    ]

    report = fragmentation_report(nodes, probes)

    counts = {p.probe.name: p.fragments for p in report.per_probe}
    assert counts["Apple"] == 3
    assert counts["OpenAI"] == 1


def test_overall_summary_mean_and_worst() -> None:
    nodes = [
        _ent("ent-a1", "Apple"),
        _ent("ent-a2", "Apple Inc."),
        _ent("ent-a3", "the iPhone maker"),
        _ent("ent-o1", "OpenAI"),
    ]
    probes = [
        Probe("Apple", aliases=["apple", "apple inc", "the iphone maker"]),
        Probe("OpenAI", aliases=["openai"]),
    ]

    report = fragmentation_report(nodes, probes)

    # mean over probes: (3 + 1) / 2 == 2.0; worst is the 3-way Apple split.
    assert report.mean_fragments == 2.0
    assert report.worst.probe.name == "Apple"
    assert report.worst.fragments == 3


def test_normalization_collapses_case_and_whitespace() -> None:
    # Surface forms differ only by case/spacing; the probe alias is lowercased.
    nodes = [_ent("ent-a1", "Apple  Inc")]
    probes = [Probe("Apple", aliases=["apple inc"])]

    report = fragmentation_report(nodes, probes)

    assert report.per_probe[0].fragments == 1


def test_qid_match_counts_even_when_surface_differs() -> None:
    # "the iPhone maker" never matches by surface, but carries the probe's QID.
    nodes = [
        _ent("ent-a1", "Apple", qid="Q312"),
        _ent("ent-a2", "the iPhone maker", qid="Q312"),
    ]
    probes = [Probe("Apple", aliases=["apple"], qid="Q312")]

    report = fragmentation_report(nodes, probes)

    assert report.per_probe[0].fragments == 2


def test_aliases_on_node_intersect_probe() -> None:
    # The match lives on the node's *aliases*, not its display name.
    nodes = [_ent("ent-a1", "AAPL", aliases=["Apple Inc"])]
    probes = [Probe("Apple", aliases=["apple inc"])]

    report = fragmentation_report(nodes, probes)

    assert report.per_probe[0].fragments == 1


def test_no_match_is_zero_fragments() -> None:
    nodes = [_ent("ent-o1", "OpenAI", aliases=["openai"])]
    probes = [Probe("Tesla", aliases=["tesla"])]

    report = fragmentation_report(nodes, probes)

    assert report.per_probe[0].fragments == 0


def test_empty_probes_has_zero_mean_and_no_worst() -> None:
    nodes = [_ent("ent-o1", "OpenAI")]

    report = fragmentation_report(nodes, [])

    assert report.per_probe == []
    assert report.mean_fragments == 0.0
    assert report.worst is None
