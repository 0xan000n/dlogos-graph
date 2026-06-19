"""Tests for the four-arm side-by-side UI render functions (spec §8, §9).

The Gradio shell is never imported here — only the pure render/format functions
and the injected-arm runner. Arms are fakes (deterministic, no network), so the
whole suite runs with the core dependency group. The key behaviours: all four
columns always render; the dLogos panel surfaces speaker-verified citations; a
failing arm degrades to an error panel instead of blanking the view.
"""

from __future__ import annotations

import sys

from dlogos.eval.arms import (
    ALL_ARMS,
    ARM_DLOGOS,
    ARM_MODEL_ALONE,
    ARM_NAIVE_RAG,
    ARM_WEB_SEARCH,
    Answer,
    Citation,
)
from dlogos.eval.golden import GoldenQuery
from dlogos.ui.app import (
    ARM_TITLES,
    FourArmView,
    format_arm_panel,
    format_citation,
    format_citations_block,
    make_query,
    render_panels_in_order,
    render_side_by_side,
    run_four_arms,
)


class FakeArm:
    """An arm that returns a fixed answer (or raises) for any query."""

    def __init__(self, name: str, answer: Answer | None, *, raises: bool = False) -> None:
        self.name = name
        self._answer = answer
        self._raises = raises

    async def __call__(self, query: GoldenQuery) -> Answer:
        if self._raises:
            raise RuntimeError("arm exploded")
        assert self._answer is not None
        return self._answer


def _citation() -> Citation:
    return Citation(
        episode_id="ep-0003",
        t_start=600.0,
        t_end=612.0,
        speaker_id="spk-analyst",
        snippet="Apple will rebound next cycle",
    )


def _four_fake_arms():
    return [
        FakeArm(ARM_MODEL_ALONE, Answer(arm=ARM_MODEL_ALONE, text="floor answer")),
        FakeArm(ARM_WEB_SEARCH, Answer(arm=ARM_WEB_SEARCH, text="web answer")),
        FakeArm(ARM_NAIVE_RAG, Answer(arm=ARM_NAIVE_RAG, text="rag answer")),
        FakeArm(
            ARM_DLOGOS,
            Answer(
                arm=ARM_DLOGOS,
                text="consensus moved from negative to positive",
                citations=[_citation()],
            ),
        ),
    ]


# --------------------------------------------------------------------------- #
# Importing the UI module must NOT pull gradio
# --------------------------------------------------------------------------- #
def test_importing_ui_does_not_import_gradio() -> None:
    assert "gradio" not in sys.modules


def test_make_query_wraps_text() -> None:
    q = make_query("How has X moved?")
    assert isinstance(q, GoldenQuery)
    assert q.query_text == "How has X moved?"
    # The UI does not pre-register a strict answer shape.
    assert q.pre_registered_answer_shape.requires_citations is False


async def test_run_four_arms_collects_all_four() -> None:
    view = await run_four_arms("How has the consensus on Apple moved?", _four_fake_arms())
    assert isinstance(view, FourArmView)
    assert set(view.answers) == set(ALL_ARMS)
    assert not view.errors
    assert view.answers[ARM_DLOGOS].citations  # dLogos carries citations


async def test_run_four_arms_records_a_failing_arm_without_aborting() -> None:
    arms = _four_fake_arms()
    arms[1] = FakeArm(ARM_WEB_SEARCH, None, raises=True)  # web-search arm fails
    view = await run_four_arms("query", arms)
    # The three healthy arms still produced answers.
    assert set(view.answers) == {ARM_MODEL_ALONE, ARM_NAIVE_RAG, ARM_DLOGOS}
    assert ARM_WEB_SEARCH in view.errors
    assert "RuntimeError" in view.errors[ARM_WEB_SEARCH]


def test_format_citation_surfaces_speaker_and_span() -> None:
    line = format_citation(_citation())
    # The speaker-verified dimensions must be visible: speaker AND ep+timestamp.
    assert "spk-analyst" in line
    assert "ep-0003" in line
    assert "600.0" in line
    assert "612.0" in line
    assert "rebound" in line  # snippet quoted


def test_format_citations_block_marks_absence() -> None:
    assert "No speaker-verified citations" in format_citations_block([])
    block = format_citations_block([_citation()])
    assert block.startswith("- ")
    assert "spk-analyst" in block


def test_format_arm_panel_renders_title_text_and_citations() -> None:
    answer = Answer(
        arm=ARM_DLOGOS,
        text="the position moved up",
        citations=[_citation()],
    )
    panel = format_arm_panel(ARM_DLOGOS, answer)
    assert ARM_TITLES[ARM_DLOGOS] in panel
    assert "the position moved up" in panel
    assert "Citations" in panel
    assert "spk-analyst" in panel


def test_format_arm_panel_for_failed_arm() -> None:
    panel = format_arm_panel(ARM_WEB_SEARCH, None, error="boom")
    assert ARM_TITLES[ARM_WEB_SEARCH] in panel
    assert "boom" in panel
    assert "failed" in panel.lower()


async def test_render_side_by_side_covers_all_four_arms() -> None:
    view = await run_four_arms("q", _four_fake_arms())
    panels = render_side_by_side(view)
    assert set(panels) == set(ALL_ARMS)
    # Only the dLogos panel shows real citations; the floor arm shows the
    # explicit "no citations" marker.
    assert "spk-analyst" in panels[ARM_DLOGOS]
    assert "No speaker-verified citations" in panels[ARM_MODEL_ALONE]


def test_render_side_by_side_fills_missing_arms() -> None:
    # A view with only one arm answered still renders all four columns.
    view = FourArmView(query_text="q")
    view.answers[ARM_DLOGOS] = Answer(arm=ARM_DLOGOS, text="only dlogos ran")
    panels = render_side_by_side(view)
    assert set(panels) == set(ALL_ARMS)
    assert "only dlogos ran" in panels[ARM_DLOGOS]
    # The absent arms render as failed/absent rather than missing keys.
    assert "failed" in panels[ARM_MODEL_ALONE].lower()


async def test_render_panels_in_order_matches_canonical_columns() -> None:
    view = await run_four_arms("q", _four_fake_arms())
    ordered = render_panels_in_order(view)
    assert len(ordered) == len(ALL_ARMS)
    # Left-to-right column order == ALL_ARMS order; each panel carries its title.
    for panel, arm_name in zip(ordered, ALL_ARMS):
        assert ARM_TITLES[arm_name] in panel
