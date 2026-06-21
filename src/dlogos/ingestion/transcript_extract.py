"""HTML→text and PDF→text extraction for public *text* transcripts.

This is the separate, **lazy-importing** fetch-layer companion to the pure
stdlib parsers in :mod:`dlogos.ingestion.parsers`. The parsers are pure
``parse(text) -> list[TranscriptSegment]`` functions over *already-extracted*
readable text; this module is the one place that turns a raw HTML string or raw
PDF bytes into that readable text — and it is the only piece that needs the
optional ``transcripts`` extra (``beautifulsoup4`` + ``pypdf``).

Both functions are **pure given their input** (same bytes/str in → same text
out): no network, no filesystem, no clock. The network GET lives one layer up in
:mod:`dlogos.ingestion.transcript_source`. The heavy deps are imported *inside*
the functions so importing this module (and anything that imports it for the
registry) never pulls in ``bs4`` / ``pypdf`` — the core test suite stays on the
core dependency group.

Why ``get_text("\\n")``: the stdlib parsers were each written against the
*newline-joined* readable text BeautifulSoup yields with a ``"\\n"`` separator
(speaker labels land on their own lines, continuation paragraphs on the next),
so this module produces exactly that shape. ``pypdf`` page text is likewise
joined with newlines between pages.
"""

from __future__ import annotations

__all__ = ["html_to_text", "pdf_to_text"]


def html_to_text(html: str) -> str:
    """Render an HTML document string to newline-joined readable text.

    Lazy-imports BeautifulSoup (the ``transcripts`` extra). Non-content nodes
    (``script`` / ``style`` / ``noscript`` / ``template``) are dropped before
    extraction so inline JS/CSS never leaks into the transcript text the parsers
    scan. Text is joined with ``"\\n"`` so that each block-level element lands on
    its own line — the exact shape the stdlib parsers key on (a bare speaker-name
    line, then its colon/timestamp/text line).

    Pure given ``html``: no network or filesystem access.
    """

    from bs4 import BeautifulSoup  # lazy: only the fetch layer needs bs4

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    return soup.get_text("\n")


def pdf_to_text(data: bytes) -> str:
    """Extract newline-joined readable text from PDF ``data`` bytes.

    Lazy-imports ``pypdf`` (the ``transcripts`` extra). Each page's extracted
    text is concatenated with a newline between pages, mirroring the readable
    shape the HTML path yields, so the same finalize/clean helpers downstream
    behave identically. A page that yields no extractable text contributes an
    empty string (it is simply skipped in the join).

    Pure given ``data``: reads the bytes via an in-memory buffer, no filesystem.
    """

    import io

    from pypdf import PdfReader  # lazy: only the fetch layer needs pypdf

    reader = PdfReader(io.BytesIO(data))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(pages)
