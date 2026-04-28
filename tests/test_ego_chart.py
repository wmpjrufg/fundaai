"""Tests for ``frontend.components.ego_chart``.

The chart is presentation code, but its data shape is testable:

    1. ``best_so_far_curves`` returns a non-increasing curve of the
       expected length and starts at the LHS minimum.
    2. ``render_ego_history`` produces a 2-row Plotly figure with the
       expected number of traces and a min/max band.
    3. The metrics annotation is added when a metrics dict is
       supplied.
    4. The log-y toggle flips the OF axis to log.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from frontend.components.ego_chart import best_so_far_curves, render_ego_history


def _hist(of_values, iters):
    return pd.DataFrame({
        "ID": list(range(len(of_values))),
        "ITER": iters,
        "X_0": np.linspace(0, 1, len(of_values)),
        "OF": of_values,
        "FIT": [1.0 / (1.0 + abs(v)) for v in of_values],
        "TIME CONSUMPTION (s)": [0.01] * len(of_values),
    })


@pytest.mark.optimization
class TestBestSoFarCurves:
    """This class verifies the per-rep best-so-far computation."""

    def test_starts_at_lhs_min_and_is_non_increasing(self):
        # Three LHS rows with min 7.0, then iters yielding 8, 5, 3.
        h = _hist([10.0, 7.0, 9.0, 8.0, 5.0, 3.0],
                  [0, 0, 0, 1, 2, 3])
        x, y = best_so_far_curves(h)
        # x = [0, 1, 2, 3], y = [7, 7, 5, 3]
        assert x.tolist() == [0, 1, 2, 3]
        assert y[0] == pytest.approx(7.0)
        assert all(y[i + 1] <= y[i] for i in range(len(y) - 1))
        assert y[-1] == pytest.approx(3.0)

    def test_empty_history_yields_empty_arrays(self):
        x, y = best_so_far_curves(pd.DataFrame(columns=["ITER", "OF"]))
        assert x.size == 0 and y.size == 0


@pytest.mark.optimization
class TestRenderEgoHistory:
    """This class verifies the Plotly figure shape."""

    def _two_runs(self):
        return {
            0: _hist([10.0, 7.0, 9.0, 8.0, 5.0, 3.0], [0, 0, 0, 1, 2, 3]),
            1: _hist([12.0, 9.0, 8.0, 7.5, 6.0, 4.0], [0, 0, 0, 1, 2, 3]),
        }

    def test_figure_has_band_and_per_rep_curves(self):
        fig = render_ego_history(self._two_runs(), show_evaluations=False)
        # 1 band + 2 per-rep lines + 1 median line + 2 time bars = 6 traces
        assert len(fig.data) == 6

    def test_with_evaluations_adds_marker_traces(self):
        fig = render_ego_history(self._two_runs(), show_evaluations=True)
        # +2 marker traces (one per rep)
        assert len(fig.data) == 6 + 2

    def test_metrics_annotation_is_attached(self):
        fig = render_ego_history(
            self._two_runs(),
            metrics={"best_of": 3.0, "mean_convergence_iter": 2.5,
                     "mean_auc_best_so_far": 0.42},
        )
        assert any(
            "best OF" in (a.text or "") for a in fig.layout.annotations
        )

    def test_log_y_flips_axis(self):
        fig = render_ego_history(self._two_runs(), log_y=True)
        assert fig.layout.yaxis.type == "log"

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            render_ego_history({})
