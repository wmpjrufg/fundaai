"""Reusable Streamlit widgets shared by pages.

Planned components (placeholder until first one lands):

* ``footings_3d_viewer``   — three-dimensional rendering of the
                              optimised footings + pillars in plan and
                              elevation.
* ``ego_best_so_far_chart`` — convergence curve of the best objective
                              per EGO iteration, fed by an
                              ``ExperimentRun`` history dataframe.
* ``gpr_diagnostics``      — paired plots (residuals, std band, kernel
                              hyperparameter trace) for the surrogate
                              model.

Each component takes typed inputs from ``core`` (``Sapata``,
``ExperimentRun``, ``OptimisationResult``) and returns a Streamlit
node — no engineering logic here, only presentation.
"""
