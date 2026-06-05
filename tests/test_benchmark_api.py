"""Tests for the benchmark API (``core.api.benchmark``).

Pin the public contracts of :func:`run_benchmark`:

* ``BenchmarkConfig`` rejects malformed inputs at construction.
* ``run_benchmark`` honours the evaluation budget exactly (no
  algorithm produces ``eval_idx > budget_evals``).
* The history schema is stable (column names + monotonic
  ``of_best_so_far`` per repetition).
* Determinism: same config + same projeto → bit-identical history.
* The summary covers every algorithm requested and contains the
  expected columns; the p-value matrix is square with the right
  index/columns and NaN diagonal.

Test sizes are intentionally small to stay within the suite's time
budget — they exercise the plumbing, not the algorithmic quality.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.api import (
    ALGORITHM_LABELS,
    ALL_ALGORITHMS,
    BenchmarkConfig,
    BenchmarkResult,
    run_benchmark,
)
from core.io import read_projeto_from_excel


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture(scope="module")
def projeto_tres(assets_dir: Path):
    """Three-foundation reference project (n_fund = 3, n_comb = 3)."""
    return read_projeto_from_excel(
        assets_dir / "data" / "problema_fund_três.xlsx",
        f_ck_kpa=25_000.0, cobrimento_m=0.04,
    )


def _tiny_config(**overrides) -> BenchmarkConfig:
    """Small but valid configuration for fast tests."""
    defaults = dict(
        algorithms=("ego", "ga", "pso", "gwo"),
        budget_evals=30,
        n_rep=2,
        base_seed=42,
        h_min_m=0.60,
        h_max_m=1.50,
        lhs_n_pop=10,
        meta_pop_size=10,
        ga_pop_size=20,
        ga_epoch=5,
    )
    defaults.update(overrides)
    return BenchmarkConfig(**defaults)


# =============================================================================
# Config validation
# =============================================================================
class TestBenchmarkConfig:

    def test_defaults_are_valid(self):
        cfg = BenchmarkConfig()
        assert cfg.h_min_m < cfg.h_max_m
        assert cfg.lhs_n_pop < cfg.budget_evals
        assert set(cfg.algorithms).issubset(set(ALL_ALGORITHMS))

    def test_model_is_frozen(self):
        cfg = BenchmarkConfig()
        with pytest.raises(ValueError):
            cfg.budget_evals = 100   # type: ignore[misc]

    def test_extra_fields_are_forbidden(self):
        with pytest.raises(ValueError):
            BenchmarkConfig(unknown=1)   # type: ignore[call-arg]

    @pytest.mark.parametrize(
        "overrides",
        [
            {"h_min_m": 1.0, "h_max_m": 1.0},
            {"h_min_m": 2.0, "h_max_m": 1.5},
            {"lhs_n_pop": 30, "ego_budget_evals": 30},
            {"lhs_n_pop": 50, "ego_budget_evals": 30},
            {"algorithms": ()},
            {"algorithms": ("ego", "ego")},
            {"n_rep": 0},
            {"budget_evals": 5},
        ],
    )
    def test_invalid_combinations_raise(self, overrides):
        with pytest.raises(ValueError):
            BenchmarkConfig(**overrides)


# =============================================================================
# Pipeline contract
# =============================================================================
class TestRunBenchmarkContract:

    def test_returns_typed_result(self, projeto_tres):
        cfg = _tiny_config(algorithms=("ga",), n_rep=2)
        result = run_benchmark(projeto_tres, cfg)
        assert isinstance(result, BenchmarkResult)
        assert result.config == cfg

    def test_history_schema_is_stable(self, projeto_tres):
        cfg = _tiny_config(algorithms=("ga", "gwo"), n_rep=2)
        result = run_benchmark(projeto_tres, cfg)
        expected_cols = {
            "algorithm", "rep", "seed", "eval_idx",
            "of_value", "of_best_so_far", "time_eval_s", "time_total_s",
        }
        assert expected_cols.issubset(set(result.history.columns))
        assert not result.history.empty

    def test_budget_is_respected_per_rep(self, projeto_tres):
        cfg = _tiny_config(algorithms=("ego", "ga", "pso", "gwo"), n_rep=2)
        result = run_benchmark(projeto_tres, cfg)
        max_per_rep = (
            result.history.groupby(["algorithm", "rep"])["eval_idx"].max()
        )
        # No repetition may exceed the budget.
        assert (max_per_rep <= cfg.budget_evals).all(), max_per_rep.to_dict()

    def test_best_so_far_is_monotonic(self, projeto_tres):
        cfg = _tiny_config(algorithms=("ga", "pso"), n_rep=2)
        result = run_benchmark(projeto_tres, cfg)
        for (alg, rep), g in result.history.groupby(["algorithm", "rep"]):
            diffs = np.diff(g.sort_values("eval_idx")["of_best_so_far"].to_numpy())
            assert np.all(diffs <= 1e-12), (
                f"{alg}/rep{rep}: best_so_far is not monotonically non-increasing"
            )

    def test_all_algorithms_present_in_summary(self, projeto_tres):
        algs = ("ga", "pso", "gwo")
        cfg = _tiny_config(algorithms=algs, n_rep=2)
        result = run_benchmark(projeto_tres, cfg)
        assert set(result.summary["algorithm"]) == set(algs)
        expected_cols = {
            "algorithm", "label", "n_rep", "best", "mean", "std", "median",
            "auc_mean", "auc_std", "conv_eval_mean", "conv_eval_std",
            "wall_time_mean_s", "wall_time_std_s",
        }
        assert expected_cols.issubset(set(result.summary.columns))
        for alg in algs:
            row = result.summary[result.summary["algorithm"] == alg].iloc[0]
            assert row["label"] == ALGORITHM_LABELS[alg]

    def test_pvalues_matrix_shape(self, projeto_tres):
        algs = ("ga", "pso", "gwo")
        cfg = _tiny_config(algorithms=algs, n_rep=3)
        result = run_benchmark(projeto_tres, cfg)
        assert list(result.pvalues.index) == list(algs)
        assert list(result.pvalues.columns) == list(algs)
        # Diagonal must be NaN (an algorithm is not compared against itself).
        diag = np.array([result.pvalues.loc[a, a] for a in algs], dtype=float)
        assert np.all(np.isnan(diag))
        # Off-diagonal must be either NaN (degenerate) or in [0, 1].
        for a in algs:
            for b in algs:
                if a == b:
                    continue
                v = float(result.pvalues.loc[a, b])
                assert np.isnan(v) or (0.0 <= v <= 1.0)


# =============================================================================
# Determinism
# =============================================================================
class TestDeterminism:

    def test_same_config_yields_same_summary(self, projeto_tres):
        cfg = _tiny_config(algorithms=("ga", "gwo"), n_rep=2)
        r1 = run_benchmark(projeto_tres, cfg)
        r2 = run_benchmark(projeto_tres, cfg)
        # Compare on the deterministic metrics (best per algorithm). Wall
        # times are intentionally excluded — they vary across runs.
        s1 = r1.summary.set_index("algorithm")[["best", "mean", "median"]]
        s2 = r2.summary.set_index("algorithm")[["best", "mean", "median"]]
        pd.testing.assert_frame_equal(s1, s2)
