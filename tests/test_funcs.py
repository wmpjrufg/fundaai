"""Tests for ``core.optimization.funcs`` (LHS, fitness, dataframe helpers).

Most of the helpers in this module are exercised end-to-end by the
EGO regression suite, but a couple of them have subtle index
contracts that warrant direct unit coverage:

    - ``best_avg_worst`` uses ``idxmin`` / ``idxmax`` to find the
      best and worst rows. Pre-Sprint-4.8 the result was indexed
      positionally with ``.values[best_idx]``, which is unsafe when
      the input DataFrame has been filtered (non-default index).
      The current implementation uses ``df.loc[best_idx, col]`` so
      it works with any RangeIndex.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.optimization import funcs


@pytest.mark.optimization
class TestBestAvgWorstIndexSafety:
    """This class verifies best_avg_worst with non-default indices."""

    def _df(self, indices, of_values, x_values):
        return pd.DataFrame({
            "ID": list(range(len(indices))),
            "ITER": [0] * len(indices),
            "OF": of_values,
            "FIT": [1.0 / (1.0 + abs(v)) for v in of_values],
            "OF EVALUATIONS": [1] * len(indices),
            "X_0": x_values,
            "X_1": [v + 0.1 for v in x_values],
        }, index=indices)

    def test_default_index(self):
        """Default RangeIndex 0..n-1 — historical contract."""
        df = self._df([0, 1, 2], [5.0, 3.0, 4.0], [10.0, 20.0, 30.0])
        out = funcs.best_avg_worst(df, d=2)
        assert out.loc[0, "OF BEST"] == pytest.approx(3.0)
        assert out.loc[0, "OF WORST"] == pytest.approx(5.0)
        assert out.loc[0, "X_BEST_0"] == pytest.approx(20.0)
        assert out.loc[0, "X_WORST_0"] == pytest.approx(10.0)

    def test_non_default_index_does_not_raise(self):
        """Filtered DataFrame (index [5, 6, 7]) — would fail with .values[best_idx]."""
        df = self._df([5, 6, 7], [5.0, 3.0, 4.0], [10.0, 20.0, 30.0])
        out = funcs.best_avg_worst(df, d=2)
        # Best row had index 6 (OF = 3.0, X_0 = 20.0)
        assert out.loc[0, "OF BEST"] == pytest.approx(3.0)
        assert out.loc[0, "X_BEST_0"] == pytest.approx(20.0)
        # Worst row had index 5 (OF = 5.0, X_0 = 10.0)
        assert out.loc[0, "OF WORST"] == pytest.approx(5.0)
        assert out.loc[0, "X_WORST_0"] == pytest.approx(10.0)

    def test_negative_indices_in_dataframe_label(self):
        """Pathological: negative integer labels still work via .loc."""
        df = self._df([-3, -2, -1], [5.0, 3.0, 4.0], [10.0, 20.0, 30.0])
        out = funcs.best_avg_worst(df, d=2)
        assert out.loc[0, "X_BEST_0"] == pytest.approx(20.0)
        assert out.loc[0, "X_WORST_0"] == pytest.approx(10.0)
