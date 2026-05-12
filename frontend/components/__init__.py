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
    CAMERA_PRESETS,
    DEFAULT_PILLAR_HEIGHT_M,
    footing_box,
    pillar_box,
    render_footings_3d,
)
from .convergence_chart import best_so_far_by_eval, render_convergence_chart
from .ego_chart import best_so_far_curves, render_ego_history
from .result_export import (
    build_export_artifacts,
    figure_to_html_bytes,
    figure_to_png_bytes,
    result_to_json_bytes,
)

__all__ = [
    "CAMERA_PRESETS",
    "DEFAULT_PILLAR_HEIGHT_M",
    "best_so_far_by_eval",
    "best_so_far_curves",
    "build_export_artifacts",
    "figure_to_html_bytes",
    "figure_to_png_bytes",
    "footing_box",
    "pillar_box",
    "render_convergence_chart",
    "render_ego_history",
    "render_footings_3d",
    "result_to_json_bytes",
]
