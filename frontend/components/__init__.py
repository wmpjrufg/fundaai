"""Reusable Streamlit / Plotly widgets shared by pages.

Each component takes typed inputs from ``core`` (``Sapata``,
``ExperimentRun``, ``OptimisationResult``) and returns a renderable
artefact (typically a ``plotly.graph_objects.Figure``). No engineering
logic here — only presentation.

Available components
--------------------

* :func:`render_footings_3d` (``footings_3d``) — interactive 3D viewer
  of the optimised footings + pillars in elevation, with ground plane,
  hover tooltips and a legend that toggles individual elements.

Planned components
------------------

* ``ego_best_so_far_chart`` — convergence curve of the best objective
                              per EGO iteration, fed by an
                              ``ExperimentRun`` history dataframe.
* ``gpr_diagnostics``       — paired plots (residuals, std band,
                              kernel hyperparameter trace) for the
                              surrogate model.
"""

from .footings_3d import (
    DEFAULT_PILLAR_HEIGHT_M,
    footing_box,
    pillar_box,
    render_footings_3d,
)

__all__ = [
    "DEFAULT_PILLAR_HEIGHT_M",
    "footing_box",
    "pillar_box",
    "render_footings_3d",
]
