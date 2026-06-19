"""Thin side-by-side query UI (spec §8, §9).

A minimal web view that runs one query across the **four eval arms**
(model-alone / +web-search / +naive-vector-RAG / +dLogos) and shows the answers
next to each other, with dLogos's speaker-verified citations surfaced for
inspection. This is the greenlight artifact made interactive — *not* the
productized public API/CLI (those stay deferred, §13).

Design split (so the rendering is unit-testable without ``gradio``):

- The format/render logic is a set of **pure functions** (:mod:`dlogos.ui.app`,
  ``format_*`` / ``render_*``) over the shared ``Answer``/``Citation`` types.
  Tests exercise these directly with the core dependency group only.
- :func:`dlogos.ui.app.build_ui` lazily imports ``gradio`` *inside* the
  function and wires the panels; importing this module never requires
  ``gradio``.
"""

from __future__ import annotations

from dlogos.ui.app import (
    ARM_TITLES,
    FourArmView,
    build_ui,
    format_arm_panel,
    format_citation,
    format_citations_block,
    render_side_by_side,
    run_four_arms,
)

__all__ = [
    "ARM_TITLES",
    "FourArmView",
    "build_ui",
    "format_arm_panel",
    "format_citation",
    "format_citations_block",
    "render_side_by_side",
    "run_four_arms",
]
