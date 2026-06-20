"""Export a :class:`~dlogos.graph.store.GraphStore` to vis-network JSON.

A pure, deterministic projection of *all* nodes and edges in a store onto the
shape the browser-side graph viewer consumes:

.. code-block:: json

    {
      "nodes": [{"id", "label", "group", "title"}, ...],
      "edges": [{"from", "to", "label", "title"}, ...]
    }

``group`` is one of ``"speaker"`` / ``"entity"`` / ``"claim"`` so the UI can
color nodes by type. Claim nodes carry a *short snippet* as their ``label`` and
the *full claim text + span + speaker* as their hover ``title`` — that title is
what makes the grounded span and attribution legible in the viewer. Edge labels
are the :class:`~dlogos.graph.store.EdgeType` value
(``asserts``/``about``/``mentions``/``disputes``/``supersedes``/``appears_in``…)
so the dialogue ontology reads directly off the picture.

Only the standard-library :mod:`json` is used. The functions are pure and the
output is sorted by id, so the same store always serializes to byte-identical
JSON — handy for snapshot tests and stable diffs. The store is read through the
:class:`~dlogos.graph.fake_store.FakeGraphStore` accessors (``.speakers`` /
``.entities`` / ``.claims`` / ``.edges`` dicts), which the real backend mirrors.
"""

from __future__ import annotations

import json
from typing import Any

# Length budget for the short claim label shown *on* the node (the full text
# lives in the hover title, so this can be aggressive without losing detail).
_SNIPPET_MAX = 60


def _truncate(text: str, limit: int = _SNIPPET_MAX) -> str:
    """One-line, length-bounded version of ``text`` for a node label."""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: max(0, limit - 1)].rstrip() + "…"  # ellipsis


def _claim_text(claim: Any) -> str:
    """Human-readable rendering of a claim: speaker → predicate/stance → object.

    ``ClaimNode`` has no free-text quote field; the legible content is the
    controlled ``predicate``, the ``stance``, and the free-text ``object``. We
    render them as one sentence-ish line so a reader can see what was claimed.
    """
    predicate = getattr(claim.predicate, "value", claim.predicate)
    stance = getattr(claim.stance, "value", claim.stance)
    obj = (claim.object or "").strip()
    return f"{predicate} ({stance}): {obj}".rstrip(": ").strip()


def _span_str(claim: Any) -> str:
    """Render a claim's grounded source span as ``ep @ [t_start-t_end s]``."""
    span = claim.source_span
    return f"{span.episode_id} @ [{span.t_start:.1f}-{span.t_end:.1f}s]"


def _speaker_label(speaker: Any) -> str:
    """Display label for a speaker node: name if known, else the id."""
    name = getattr(speaker, "name", None)
    return name if name else speaker.speaker_id


def _node_records(store: Any) -> list[dict[str, Any]]:
    """Project every speaker/entity/claim node onto vis-network node dicts."""
    nodes: list[dict[str, Any]] = []

    for speaker in store.speakers.values():
        label = _speaker_label(speaker)
        title_bits = [f"Speaker: {label}", f"id: {speaker.speaker_id}"]
        if getattr(speaker, "is_host", False):
            title_bits.append("role: host")
        qid = getattr(speaker, "wikidata_qid", None)
        if qid:
            title_bits.append(f"wikidata: {qid}")
        nodes.append(
            {
                "id": speaker.speaker_id,
                "label": label,
                "group": "speaker",
                "title": " | ".join(title_bits),
            }
        )

    for entity in store.entities.values():
        etype = getattr(entity.type, "value", entity.type)
        title_bits = [f"Entity: {entity.name}", f"type: {etype}", f"id: {entity.canonical_id}"]
        aliases = getattr(entity, "aliases", None) or []
        if aliases:
            title_bits.append(f"aliases: {', '.join(aliases)}")
        nodes.append(
            {
                "id": entity.canonical_id,
                "label": entity.name,
                "group": "entity",
                "title": " | ".join(title_bits),
            }
        )

    for claim in store.claims.values():
        text = _claim_text(claim)
        title = (
            f"Claim: {text} | span: {_span_str(claim)} | "
            f"speaker: {claim.speaker_id}"
        )
        nodes.append(
            {
                "id": claim.claim_id,
                "label": _truncate(text),
                "group": "claim",
                "title": title,
            }
        )

    nodes.sort(key=lambda n: n["id"])
    return nodes


def _edge_records(store: Any) -> list[dict[str, Any]]:
    """Project every graph edge onto a vis-network edge dict."""
    edges: list[dict[str, Any]] = []
    for edge in store.edges.values():
        etype = getattr(edge.type, "value", edge.type)
        title_bits = [f"{etype}: {edge.src_id} → {edge.dst_id}", f"id: {edge.edge_id}"]
        if getattr(edge, "invalidated", False):
            title_bits.append("invalidated")
        edges.append(
            {
                "from": edge.src_id,
                "to": edge.dst_id,
                "label": etype,
                "title": " | ".join(title_bits),
                # Carried for stable sorting; harmless extra field for the UI.
                "id": edge.edge_id,
            }
        )
    edges.sort(key=lambda e: e["id"])
    return edges


def export_graph(store: Any) -> dict[str, list[dict[str, Any]]]:
    """Read all nodes + edges from ``store`` and return vis-network JSON dict.

    Pure and deterministic: nodes and edges are sorted by id, so the same store
    always produces the same dict. ``store`` is any
    :class:`~dlogos.graph.store.GraphStore` exposing the in-memory accessors
    (``.speakers`` / ``.entities`` / ``.claims`` / ``.edges``), e.g.
    :class:`~dlogos.graph.fake_store.FakeGraphStore`.

    Returns
    -------
    dict
        ``{"nodes": [...], "edges": [...]}`` ready to hand to vis-network.
    """
    return {"nodes": _node_records(store), "edges": _edge_records(store)}


def write_graph_json(store: Any, path: Any) -> None:
    """Serialize ``export_graph(store)`` to ``path`` as pretty, stable JSON.

    Uses only the standard-library :mod:`json`. Output is deterministic
    (sorted nodes/edges, fixed indentation) so it round-trips and diffs
    cleanly. ``path`` may be a ``str`` or :class:`os.PathLike`.
    """
    graph = export_graph(store)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(graph, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
