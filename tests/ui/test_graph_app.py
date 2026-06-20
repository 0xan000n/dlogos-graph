"""Tests for the stdlib localhost graph viewer (:mod:`dlogos.ui.graph_app`).

No socket is ever bound and no network is touched: the server's whole decision
surface is the pure :func:`route` / :func:`load_graph` / :func:`read_html`
functions, and these are exercised directly. The HTTP shell is a thin delegate
over :func:`route`, so covering the pure core covers the server's behaviour.

Key behaviours asserted:
- ``route('/')`` returns ``200`` and the actual served HTML (which references
  vis-network + ``/graph.json``).
- ``route('/graph.json')`` returns the file's JSON when present, and the
  explicit empty-graph document (still ``200``) when the file is absent.
- a query string on the path is tolerated; an unknown path is ``404``.
- ``load_graph`` reads a real temp file and returns an empty graph for a missing
  path.
- the CLI parser carries the documented ``--graph`` / ``--port`` / ``--host``
  defaults.
"""

from __future__ import annotations

import json
from http import HTTPStatus
from pathlib import Path

from dlogos.ui.graph_app import (
    EMPTY_GRAPH,
    build_arg_parser,
    load_graph,
    read_html,
    route,
)


# --------------------------------------------------------------------------- #
# load_graph — pure file read
# --------------------------------------------------------------------------- #
def _sample_graph() -> dict:
    return {
        "nodes": [
            {"id": "spk-1", "label": "Host", "group": "speaker", "title": "Host"},
            {"id": "ent-1", "label": "Apple", "group": "entity", "title": "Apple"},
            {"id": "clm-1", "label": "rates_positive", "group": "claim",
             "title": "Host rates_positive Apple"},
        ],
        "edges": [
            {"from": "spk-1", "to": "clm-1", "label": "asserts"},
            {"from": "clm-1", "to": "ent-1", "label": "about"},
        ],
    }


def test_load_graph_reads_a_tmp_file(tmp_path: Path) -> None:
    graph = _sample_graph()
    p = tmp_path / "graph.json"
    p.write_text(json.dumps(graph), encoding="utf-8")

    loaded = load_graph(p)
    assert loaded == graph
    assert [n["group"] for n in loaded["nodes"]] == ["speaker", "entity", "claim"]


def test_load_graph_accepts_a_str_path(tmp_path: Path) -> None:
    p = tmp_path / "graph.json"
    p.write_text(json.dumps(_sample_graph()), encoding="utf-8")
    loaded = load_graph(str(p))
    assert len(loaded["nodes"]) == 3


def test_load_graph_missing_file_returns_empty_graph(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.json"
    loaded = load_graph(missing)
    assert loaded == {"nodes": [], "edges": []}
    # A fresh dict, not an alias of the module-level constant.
    loaded["nodes"].append({"id": "x"})
    assert EMPTY_GRAPH == {"nodes": [], "edges": []}


def test_load_graph_invalid_json_raises(tmp_path: Path) -> None:
    p = tmp_path / "broken.json"
    p.write_text("{not json", encoding="utf-8")
    try:
        load_graph(p)
    except json.JSONDecodeError:
        pass
    else:  # pragma: no cover - failure path
        raise AssertionError("expected a JSONDecodeError on corrupt graph file")


# --------------------------------------------------------------------------- #
# read_html — the served page
# --------------------------------------------------------------------------- #
def test_read_html_returns_the_viewer_page() -> None:
    html = read_html()
    assert "<!DOCTYPE html>" in html
    # It must load vis-network from a CDN and fetch the graph JSON route.
    assert "vis-network" in html
    assert "/graph.json" in html


# --------------------------------------------------------------------------- #
# route('/') — the HTML page
# --------------------------------------------------------------------------- #
def test_route_root_returns_html_200() -> None:
    status, content_type, body = route("/")
    assert status == HTTPStatus.OK
    assert content_type.startswith("text/html")
    assert b"<!DOCTYPE html>" in body
    # The served bytes are exactly the page on disk.
    assert body.decode("utf-8") == read_html()


def test_route_index_html_alias() -> None:
    status, _, body = route("/index.html")
    assert status == HTTPStatus.OK
    assert b"vis-network" in body


# --------------------------------------------------------------------------- #
# route('/graph.json') — present and absent
# --------------------------------------------------------------------------- #
def test_route_graph_json_returns_file_when_present(tmp_path: Path) -> None:
    graph = _sample_graph()
    p = tmp_path / "graph.json"
    p.write_text(json.dumps(graph), encoding="utf-8")

    status, content_type, body = route("/graph.json", graph_path=p)
    assert status == HTTPStatus.OK
    assert content_type.startswith("application/json")
    assert json.loads(body) == graph


def test_route_graph_json_returns_empty_graph_when_absent(tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"
    status, content_type, body = route("/graph.json", graph_path=missing)
    # Absent export is a normal not-yet-loaded state: 200 + empty graph, never 404.
    assert status == HTTPStatus.OK
    assert content_type.startswith("application/json")
    assert json.loads(body) == {"nodes": [], "edges": []}


def test_route_graph_json_tolerates_query_string(tmp_path: Path) -> None:
    p = tmp_path / "graph.json"
    p.write_text(json.dumps(_sample_graph()), encoding="utf-8")
    status, _, body = route("/graph.json?cachebust=123", graph_path=p)
    assert status == HTTPStatus.OK
    assert len(json.loads(body)["nodes"]) == 3


# --------------------------------------------------------------------------- #
# route(<unknown>) — 404
# --------------------------------------------------------------------------- #
def test_route_unknown_path_is_404() -> None:
    status, content_type, body = route("/favicon.ico")
    assert status == HTTPStatus.NOT_FOUND
    assert content_type.startswith("text/plain")
    assert b"404" in body
    assert b"favicon.ico" in body


# --------------------------------------------------------------------------- #
# CLI parser — documented defaults
# --------------------------------------------------------------------------- #
def test_arg_parser_defaults() -> None:
    args = build_arg_parser().parse_args([])
    assert args.graph == "out/graph.json"
    assert args.port == 8765
    assert args.host == "127.0.0.1"


def test_arg_parser_overrides() -> None:
    args = build_arg_parser().parse_args(
        ["--graph", "/tmp/g.json", "--port", "9001", "--host", "0.0.0.0"]
    )
    assert args.graph == "/tmp/g.json"
    assert args.port == 9001
    assert args.host == "0.0.0.0"
