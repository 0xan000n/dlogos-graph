"""A localhost graph viewer over a loaded dLogos graph — STDLIB ONLY.

Serves a tiny two-route site so the speakers / entities / claims / edges that a
run loaded into the graph store can be *seen* in a browser:

- ``GET /``            -> the self-contained :mod:`graph_view.html` page, which
                          (browser-side) loads ``vis-network`` from a CDN and
                          fetches ``/graph.json``.
- ``GET /graph.json``  -> the graph JSON file named by ``--graph`` (default
                          ``out/graph.json``); an explicit empty-graph document
                          when that file is absent, so the page renders cleanly
                          before anything has been exported.

HARD CONSTRAINT: this module uses ONLY the Python standard library
(``http.server``) — no new dependency. The only third-party code is
``vis-network``, fetched by the *user's browser* from a CDN, never installed
here. The request-routing and JSON-loading are factored into PURE functions
(:func:`load_graph`, :func:`route`) that take a path/string and return values,
so they are unit-tested without ever binding a socket or touching the network.

Run it::

    python -m dlogos.ui.graph_app --graph out/graph.json --port 8765

then open http://127.0.0.1:8765/ in a browser.
"""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Content types we serve.
_HTML = "text/html; charset=utf-8"
_JSON = "application/json; charset=utf-8"
_TEXT = "text/plain; charset=utf-8"

# The HTML page lives next to this module and is read at request time (not at
# import) so it can be edited without restarting an import-cached server in dev.
_HTML_PATH = Path(__file__).with_name("graph_view.html")

# An explicit empty graph — what /graph.json returns when no export exists yet.
EMPTY_GRAPH: dict[str, list] = {"nodes": [], "edges": []}


# --------------------------------------------------------------------------- #
# Pure, socket-free core (unit-tested directly)
# --------------------------------------------------------------------------- #
def load_graph(path: str | Path) -> dict:
    """Read and parse the graph JSON at ``path``.

    Returns the parsed document when the file exists, or a copy of
    :data:`EMPTY_GRAPH` when it does not — a missing export is a normal,
    not-yet-loaded state, not an error. Always returns a fresh ``dict`` so a
    caller can mutate it without aliasing :data:`EMPTY_GRAPH`. A file that
    exists but contains invalid JSON raises :class:`json.JSONDecodeError` (a
    real corruption is worth surfacing loudly).
    """

    p = Path(path)
    if not p.exists():
        return {"nodes": [], "edges": []}
    return json.loads(p.read_text(encoding="utf-8"))


def read_html() -> str:
    """Return the served HTML page source.

    Read from :data:`_HTML_PATH` at call time. Kept a function (not a constant)
    so the page and the server can be edited independently and so tests can
    assert the file is the thing actually served.
    """

    return _HTML_PATH.read_text(encoding="utf-8")


def route(path: str, *, graph_path: str | Path = "out/graph.json") -> tuple[int, str, bytes]:
    """Map a request path to ``(status, content_type, body)`` — no sockets.

    The whole routing table as one pure function so it is unit-testable without
    binding a port or issuing a real HTTP request:

    - ``/`` (and ``/index.html``) -> ``200`` with the HTML page.
    - ``/graph.json``             -> ``200`` with the graph JSON (the empty-graph
                                     document when the export file is absent).
    - anything else               -> ``404`` with a short text body.

    ``path`` may include a query string (``/graph.json?x=1``); only the path
    component is matched. ``graph_path`` is injected so tests point it at a temp
    file.
    """

    clean = path.split("?", 1)[0].split("#", 1)[0]

    if clean in ("/", "/index.html"):
        return HTTPStatus.OK, _HTML, read_html().encode("utf-8")

    if clean == "/graph.json":
        graph = load_graph(graph_path)
        body = json.dumps(graph).encode("utf-8")
        return HTTPStatus.OK, _JSON, body

    return (
        HTTPStatus.NOT_FOUND,
        _TEXT,
        f"404 Not Found: {clean}".encode("utf-8"),
    )


# --------------------------------------------------------------------------- #
# The thin stdlib HTTP shell (delegates every decision to :func:`route`)
# --------------------------------------------------------------------------- #
def make_handler(graph_path: str | Path) -> type[BaseHTTPRequestHandler]:
    """Build a :class:`BaseHTTPRequestHandler` subclass bound to ``graph_path``.

    A factory (rather than a module-global) so the served graph file is
    captured per-server, keeping the handler itself free of shared mutable
    state. ``do_GET`` is the only verb; everything else falls through to a 405.
    """

    class _GraphHandler(BaseHTTPRequestHandler):
        # Quiet the default per-request stderr logging; the launcher prints the
        # one line a human needs.
        def log_message(self, *_args) -> None:  # noqa: D401 - silence default log
            return

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            # The page fetches /graph.json from its own origin; no caching so a
            # re-export is picked up on refresh.
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler naming
            status, content_type, body = route(self.path, graph_path=graph_path)
            self._send(status, content_type, body)

        def do_HEAD(self) -> None:  # noqa: N802 - stdlib handler naming
            status, content_type, body = route(self.path, graph_path=graph_path)
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()

    return _GraphHandler


def build_server(
    *, host: str, port: int, graph_path: str | Path
) -> ThreadingHTTPServer:
    """Construct (but do not start) a :class:`ThreadingHTTPServer`.

    Returned un-started so a caller can decide whether to ``serve_forever`` or
    handle a single request; binds the socket on construction. Tests do NOT call
    this (they exercise :func:`route` / :func:`load_graph` directly) so no port
    is ever bound under test.
    """

    return ThreadingHTTPServer((host, port), make_handler(graph_path))


def serve(*, host: str, port: int, graph_path: str | Path) -> None:  # pragma: no cover - binds a socket
    """Bind the server and serve until interrupted (the runnable entry point)."""

    server = build_server(host=host, port=port, graph_path=graph_path)
    bound_host, bound_port = server.server_address[:2]
    print(
        f"dLogos graph viewer serving {graph_path} at "
        f"http://{bound_host}:{bound_port}/  (Ctrl-C to stop)"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down…")
    finally:
        server.server_close()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    """The ``--graph`` / ``--port`` / ``--host`` argument parser."""

    p = argparse.ArgumentParser(
        prog="dlogos.ui.graph_app",
        description="Serve a localhost vis-network viewer over a loaded dLogos "
        "graph JSON file (stdlib only; no new dependency).",
    )
    p.add_argument(
        "--graph",
        default="out/graph.json",
        help="Path to the graph JSON file to serve (default: out/graph.json).",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to bind (default: 8765).",
    )
    p.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host/interface to bind (default: 127.0.0.1, localhost only).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry: parse args and serve. Returns a process exit code."""

    args = build_arg_parser().parse_args(argv)
    serve(host=args.host, port=args.port, graph_path=args.graph)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
