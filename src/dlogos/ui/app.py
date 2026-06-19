"""Side-by-side four-arm query UI + its pure render functions (spec §8, §9).

This module has two layers, deliberately separated so the rendering is testable
with the core dependency group only:

1. **Pure functions** — :func:`run_four_arms` (run a query through all four
   injected arms and collect their answers), :func:`format_citation` /
   :func:`format_citations_block` / :func:`format_arm_panel` /
   :func:`render_side_by_side` (turn the answers into Markdown panels). These
   import nothing heavy and are unit-tested directly.

2. **The Gradio shell** — :func:`build_ui` lazily imports ``gradio`` *inside the
   function* (HARD CONSTRAINT) and wires the four panels around the pure
   renderers. Importing this module never requires ``gradio``.

The arms are injected (each is an async callable ``GoldenQuery -> Answer`` as
built in :mod:`dlogos.eval.arms`), so the UI never constructs a frontier client
or a retriever itself — that keeps it deterministic under test and lets the
caller wire real or fake collaborators. The dLogos panel surfaces its
speaker-verified citations (episode + timestamp + attributed speaker) so the
provenance is inspectable, which is the whole point of the head-to-head view.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Protocol

from dlogos.eval.arms import (
    ALL_ARMS,
    ARM_DLOGOS,
    ARM_MODEL_ALONE,
    ARM_NAIVE_RAG,
    ARM_WEB_SEARCH,
    Answer,
    Citation,
)
from dlogos.eval.golden import (
    AnswerShape,
    Archetype,
    Domain,
    GoldenQuery,
)

# Human-readable column titles for each arm, in the canonical left-to-right
# order (floor -> Perplexity bar -> dumb-RAG -> dLogos).
ARM_TITLES: dict[str, str] = {
    ARM_MODEL_ALONE: "Model alone",
    ARM_WEB_SEARCH: "Model + web search",
    ARM_NAIVE_RAG: "Model + naive vector-RAG",
    ARM_DLOGOS: "Model + dLogos",
}


class Arm(Protocol):
    """An eval arm: an async callable mapping a query to an :class:`Answer`.

    Matches the arms in :mod:`dlogos.eval.arms` (each exposes ``name`` and is
    awaitable). Kept structural so fakes satisfy it in tests.
    """

    name: str

    def __call__(self, query: GoldenQuery) -> Awaitable[Answer]: ...


@dataclass
class FourArmView:
    """The collected outputs of one query across the four arms.

    ``answers`` is keyed by arm name; ``errors`` records any arm that raised so
    a single failing arm never blanks the whole side-by-side view.
    """

    query_text: str
    answers: dict[str, Answer] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Query construction
# --------------------------------------------------------------------------- #
def make_query(query_text: str) -> GoldenQuery:
    """Wrap free-text from the UI box into a minimal :class:`GoldenQuery`.

    The UI is not pre-registering answer shapes (that is the eval harness's job),
    so it supplies a permissive placeholder shape. The arms only read
    ``query_text``, so this is enough to drive them.
    """

    return GoldenQuery(
        id="ui-adhoc",
        archetype=Archetype.temporal_consensus,
        domain=Domain.technology,
        query_text=query_text,
        pre_registered_answer_shape=AnswerShape(requires_citations=False),
    )


# --------------------------------------------------------------------------- #
# Running the four arms (pure: collaborators injected, no I/O of its own)
# --------------------------------------------------------------------------- #
async def run_four_arms(query_text: str, arms: list[Arm]) -> FourArmView:
    """Run ``query_text`` through every arm and collect the answers.

    Arms run concurrently. An arm that raises is recorded in ``errors`` (with
    the exception text) instead of aborting the whole view — the head-to-head is
    still useful with three of four panels. Deterministic given deterministic
    arms; no network or randomness here.
    """

    query = make_query(query_text)
    view = FourArmView(query_text=query_text)

    async def _run(arm: Arm) -> tuple[str, Answer | Exception]:
        try:
            return arm.name, await arm(query)
        except Exception as exc:  # noqa: BLE001 - surfaced per-arm, not swallowed
            return arm.name, exc

    results = await asyncio.gather(*(_run(arm) for arm in arms))
    for name, outcome in results:
        if isinstance(outcome, Exception):
            view.errors[name] = f"{type(outcome).__name__}: {outcome}"
        else:
            view.answers[name] = outcome
    return view


def run_four_arms_sync(query_text: str, arms: list[Arm]) -> FourArmView:
    """Synchronous wrapper around :func:`run_four_arms` for the Gradio callback.

    Gradio event handlers are plain sync callables; this drives the async arms
    to completion. Kept tiny and side-effect-free beyond the event loop.
    """

    return asyncio.run(run_four_arms(query_text, arms))


# --------------------------------------------------------------------------- #
# Pure render functions (Markdown) — the unit-tested formatting surface
# --------------------------------------------------------------------------- #
def format_citation(citation: Citation) -> str:
    """One citation as a compact, inspectable Markdown line.

    Surfaces the dimensions the speaker-verified check reads: the attributed
    speaker AND the (episode, t_start, t_end) span. The optional snippet is
    quoted when present.
    """

    base = (
        f"`{citation.episode_id}` @ {citation.t_start:.1f}-{citation.t_end:.1f}s "
        f"— speaker `{citation.speaker_id}`"
    )
    if citation.snippet:
        return f"{base}: “{citation.snippet}”"
    return base


def format_citations_block(citations: list[Citation]) -> str:
    """A bulleted Markdown block of citations (or a clear 'none' marker).

    An arm with no citations (the floor / web-search arms) renders an explicit
    "no speaker-verified citations" line rather than a blank, so the contrast
    with the dLogos panel's sourced spans is visible at a glance.
    """

    if not citations:
        return "_No speaker-verified citations._"
    return "\n".join(f"- {format_citation(c)}" for c in citations)


def format_arm_panel(
    arm_name: str, answer: Answer | None, error: str | None = None
) -> str:
    """Render one arm's column: title, answer text, and its citations block.

    ``answer`` is ``None`` (with ``error`` set) when the arm failed; the panel
    then shows the error so the view degrades gracefully. The title comes from
    :data:`ARM_TITLES`, falling back to the raw arm name for unknown arms.
    """

    title = ARM_TITLES.get(arm_name, arm_name)
    lines = [f"### {title}"]
    if answer is None:
        lines.append(f"_Arm failed: {error or 'unknown error'}_")
        return "\n\n".join(lines)

    lines.append(answer.text.strip() or "_(empty answer)_")
    lines.append("**Citations**")
    lines.append(format_citations_block(answer.citations))
    return "\n\n".join(lines)


def render_side_by_side(view: FourArmView) -> dict[str, str]:
    """Render every arm's panel, keyed by arm name in canonical order.

    Returns a dict ``arm_name -> markdown`` covering all four arms in
    :data:`ALL_ARMS` even when some are missing from ``view`` (a missing arm
    renders as failed/absent), so the four columns are always populated.
    """

    panels: dict[str, str] = {}
    for arm_name in ALL_ARMS:
        answer = view.answers.get(arm_name)
        error = view.errors.get(arm_name)
        if answer is None and error is None:
            error = "no output"
        panels[arm_name] = format_arm_panel(arm_name, answer, error)
    return panels


def render_panels_in_order(view: FourArmView) -> list[str]:
    """The four panels as a list in canonical left-to-right column order.

    Convenience for the Gradio layout, which binds positional columns; tests use
    :func:`render_side_by_side` for keyed assertions.
    """

    panels = render_side_by_side(view)
    return [panels[arm_name] for arm_name in ALL_ARMS]


# --------------------------------------------------------------------------- #
# The Gradio shell — lazy ``gradio`` import, wired around the pure renderers
# --------------------------------------------------------------------------- #
def build_ui(arms: list[Arm], *, title: str = "dLogos — four-arm head-to-head"):
    """Build a Gradio Blocks app: a query box over four side-by-side panels.

    ``gradio`` is imported **inside** this function (HARD CONSTRAINT: heavy deps
    never at module top level), so importing :mod:`dlogos.ui.app` for the render
    unit tests never requires ``gradio`` to be installed.

    The submit handler runs the query through all four injected arms via the
    pure :func:`run_four_arms_sync` and feeds each arm's Markdown panel into its
    column. Launch the returned app with ``app.launch()``.
    """

    try:
        import gradio as gr  # type: ignore
    except Exception as exc:  # pragma: no cover - only when gradio is absent
        raise RuntimeError(
            "The 'gradio' optional dependency is required to build the UI. "
            "Install the 'ui' extra. (Render functions are usable without it.)"
        ) from exc

    arm_order = list(ALL_ARMS)

    with gr.Blocks(title=title) as app:
        gr.Markdown(f"# {title}")
        gr.Markdown(
            "Ask a temporal/consensus question; compare the four arms. "
            "Only the dLogos panel carries speaker-verified citations."
        )
        query_box = gr.Textbox(
            label="Query",
            placeholder="How has the consensus on X moved over the last year?",
            lines=2,
        )
        run_button = gr.Button("Run all four arms", variant="primary")

        with gr.Row():
            panel_boxes = [
                gr.Markdown(value="", label=ARM_TITLES[name]) for name in arm_order
            ]

        def _on_run(query_text: str):
            view = run_four_arms_sync(query_text, arms)
            return render_panels_in_order(view)

        run_button.click(_on_run, inputs=query_box, outputs=panel_boxes)

    return app
