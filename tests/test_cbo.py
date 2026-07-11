"""Tests for the Constrained Bayesian Optimization architecture (Frente C).

Pins three contracts:

    1. The acquisition building blocks (Expected Improvement and the
       probability of feasibility) against hand-computed values, plus
       the degenerate constant-constraint handling.
    2. ``cbo_01_architecture``: history shape, budget of real
       evaluations (LHS + n_gen) and bit-reproducibility under the
       same seed.
    3. The benchmark integration: the ``cbo`` algorithm honours the
       EGO budget, produces the standard trace/feasibility columns and
       is deterministic for a fixed configuration.

Test sizes are intentionally tiny — they exercise plumbing and
mathematical correctness, not optimisation quality (that is the job of
the frozen experimental protocol).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mealpy import GA
from scipy.stats import norm

from core.api import BenchmarkConfig, run_benchmark
from core.api.objective import avaliar_projeto_componentes
from core.io import read_projeto_from_excel
from core.optimization import cbo_01_architecture, initial_population_01
from core.optimization.cbo import (
    _ConstantConstraint,
    _expected_improvement,
    _prob_feasibility,
)
from fundacao import constroi_kernel


# =============================================================================
# Acquisition building blocks
# =============================================================================
@pytest.mark.optimization
class TestAcquisitionParts:
    """This class verifies EI and PoF against closed-form hand values."""

    def test_expected_improvement_hand_value(self):
        """EI(mu=1, sigma=1, fmin=2) = 1*Phi(1) + 1*phi(1)."""
        esperado = 1.0 * norm.cdf(1.0) + 1.0 * norm.pdf(1.0)
        assert _expected_improvement(1.0, 1.0, 2.0) == pytest.approx(esperado, rel=1e-12)

    def test_expected_improvement_no_improvement_region(self):
        """Com mu >> fmin e sigma pequeno, EI tende a zero."""
        assert _expected_improvement(10.0, 1e-9, 1.0) == pytest.approx(0.0, abs=1e-12)

    def test_prob_feasibility_hand_values(self):
        """PoF = Phi(-mu/sigma): centrado em 0 vale 0,5; mu<0 aproxima 1."""
        assert _prob_feasibility(0.0, 1.0) == pytest.approx(0.5, rel=1e-12)
        assert _prob_feasibility(-3.0, 1.0) == pytest.approx(norm.cdf(3.0), rel=1e-12)
        assert _prob_feasibility(3.0, 1.0) == pytest.approx(norm.cdf(-3.0), rel=1e-12)

    def test_sigma_floor_avoids_division_blowup(self):
        """Sigma zero cai no piso numérico em vez de dividir por zero."""
        assert 0.0 <= _prob_feasibility(1.0, 0.0) <= 1.0
        assert _expected_improvement(0.5, 0.0, 1.0) >= 0.0

    def test_constant_constraint_is_deterministic(self):
        """Alvo constante vira probabilidade degenerada 0/1."""
        assert _ConstantConstraint(0.0).prob_feasible(np.zeros((1, 3))) == 1.0
        assert _ConstantConstraint(-1.0).prob_feasible(np.zeros((1, 3))) == 1.0
        assert _ConstantConstraint(0.25).prob_feasible(np.zeros((1, 3))) == 0.0


# =============================================================================
# Architecture contract
# =============================================================================
@pytest.mark.optimization
class TestCboArchitecture:
    """This class verifies the CBO loop plumbing on the one-footing case."""

    def _setup(self, assets_dir: Path):
        df = pd.read_excel(assets_dir / "data" / "problema_fund_um.xlsx")
        args = (df, 3, 25_000.0, 0.04)
        x_lower, x_upper = [0.6] * 3, [3.0] * 3
        x_ini = initial_population_01(8, 3, x_lower, x_upper, seed=7, use_lhs=True)
        paras_opt = {"optimizer algorithm": GA.BaseGA(epoch=3, pop_size=10)}
        paras_kernel = {"kernel": constroi_kernel()[-1]}
        return args, x_ini, x_lower, x_upper, paras_opt, paras_kernel

    def test_history_shape_and_budget(self, assets_dir: Path):
        """LHS de 8 + 3 iterações = 11 avaliações reais, ITER coerente."""
        args, x_ini, lo, up, popt, pker = self._setup(assets_dir)
        best_x, best_of, df = cbo_01_architecture(
            avaliar_projeto_componentes, 3, x_ini, lo, up, popt, pker,
            args=args, seed=7,
        )
        assert len(df) == 8 + 3
        assert set(df["ITER"].unique()) == {0, 1, 2, 3}
        assert {"OF", "VOLUME", "G_SOB", "G_PUN", "G_TEN", "G_GEO"} <= set(df.columns)
        assert len(best_x) == 3
        assert best_of == pytest.approx(df["OF"].min())
        # volume nunca excede Theta do mesmo ponto (penalidade >= 0)
        assert (df["VOLUME"] <= df["OF"] + 1e-12).all()

    def test_reproducible_under_same_seed(self, assets_dir: Path):
        """Duas execuções com a mesma seed produzem o mesmo histórico."""
        args, x_ini, lo, up, _, pker = self._setup(assets_dir)
        popt1 = {"optimizer algorithm": GA.BaseGA(epoch=3, pop_size=10)}
        popt2 = {"optimizer algorithm": GA.BaseGA(epoch=3, pop_size=10)}
        _, of1, df1 = cbo_01_architecture(
            avaliar_projeto_componentes, 3, x_ini, lo, up, popt1, pker,
            args=args, seed=11,
        )
        _, of2, df2 = cbo_01_architecture(
            avaliar_projeto_componentes, 3, x_ini, lo, up, popt2, pker,
            args=args, seed=11,
        )
        assert of1 == of2
        pd.testing.assert_series_equal(df1["OF"], df2["OF"])


# =============================================================================
# Benchmark integration
# =============================================================================
@pytest.mark.optimization
class TestCboBenchmark:
    """This class verifies the ``cbo`` algorithm inside run_benchmark."""

    @pytest.fixture(scope="class")
    def projeto_um(self, assets_dir: Path):
        return read_projeto_from_excel(
            assets_dir / "data" / "problema_fund_um.xlsx",
            f_ck_kpa=25_000.0, cobrimento_m=0.04,
        )

    def _cfg(self, **overrides) -> BenchmarkConfig:
        defaults = dict(
            algorithms=("cbo",),
            budget_evals=20,
            ego_budget_evals=14,
            n_rep=2,
            base_seed=42,
            h_min_m=0.60,
            h_max_m=3.00,
            lhs_n_pop=8,
            meta_pop_size=10,
            ga_pop_size=10,
            ga_epoch=3,
            cbo_constraint_restarts=1,
        )
        defaults.update(overrides)
        return BenchmarkConfig(**defaults)

    def test_budget_and_trace_schema(self, projeto_um):
        """CBO respeita ego_budget_evals e produz o trace padrão."""
        result = run_benchmark(projeto_um, self._cfg())
        per_rep_evals = result.history.groupby("rep")["eval_idx"].max()
        assert (per_rep_evals == 14).all()
        assert set(result.history["algorithm"]) == {"cbo"}
        assert (result.history.groupby("rep")["of_best_so_far"]
                .apply(lambda s: (np.diff(s) <= 1e-12).all()).all())
        # feasibility report presente
        assert {"feasible", "volume_m3", "max_violation"} <= set(result.per_rep.columns)
        assert result.summary.iloc[0]["label"] == "CBO (ECI)"

    def test_deterministic_history(self, projeto_um):
        """Mesma configuração → histórico determinístico (colunas sem relógio)."""
        r1 = run_benchmark(projeto_um, self._cfg())
        r2 = run_benchmark(projeto_um, self._cfg())
        cols = ["algorithm", "rep", "seed", "eval_idx",
                "of_value", "of_best_so_far"]
        pd.testing.assert_frame_equal(r1.history[cols], r2.history[cols])
