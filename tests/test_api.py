"""Tests for the API layer (``core.api``).

The API layer is the public surface consumed by the Streamlit page,
the notebooks and any future CLI. The tests below pin the public
contracts:

    * ``OptimisationConfig`` and ``EvaluationResult`` validate their
      fields up front.
    * ``evaluate`` reproduces the Sprint 2 regression baseline
      ``of = 19.70604234767181`` when fed the canonical three-foundation
      project and the canonical seed-42 design vector.
    * ``optimize`` runs to completion with independent seeds across
      repetitions, returns a typed result with the right shape and is
      reproducible when called twice with the same configuration.

The optimisation tests intentionally use small ``n_pop``/``n_gen`` to
stay within the regression suite's time budget; they exercise the
plumbing, not the algorithmic quality.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.api import (
    EvaluationResult,
    OptimisationConfig,
    OptimisationResult,
    evaluate,
    optimize,
)
from core.api._adapter import (
    design_vector_to_sapatas,
    projeto_to_dataframe,
    sapatas_to_design_vector,
)
from core.domain import FundacaoProjeto, Sapata
from core.io import read_projeto_from_excel


# =============================================================================
# Adapter (round-trip)
# =============================================================================
class TestAdapter:
    """This class verifies the internal domain<->DataFrame adapter."""

    def test_projeto_to_dataframe_matches_template_columns(self, assets_dir: Path):
        """This test ensures the rebuilt DataFrame has the same columns as the template.

        :return: None (internal asserts)
        """
        proj = read_projeto_from_excel(
            assets_dir / "data" / "problema_fund_três.xlsx", f_ck_kpa=25_000.0, cobrimento_m=0.04
        )
        df = projeto_to_dataframe(proj)
        df_excel = pd.read_excel(assets_dir / "data" / "problema_fund_três.xlsx")
        assert list(df.columns) == list(df_excel.columns)
        assert df.shape == df_excel.shape

    def test_design_vector_round_trip(self, assets_dir: Path):
        """This test ensures sapatas->vector->sapatas round-trips losslessly.

        :return: None (internal asserts)
        """
        proj = read_projeto_from_excel(
            assets_dir / "data" / "problema_fund_três.xlsx", f_ck_kpa=25_000.0, cobrimento_m=0.04
        )
        sapatas = [Sapata(p, h_x=2.0, h_y=1.5, h_z=0.6) for p in proj.pilares]
        vec = sapatas_to_design_vector(sapatas)
        rebuilt = design_vector_to_sapatas(vec, proj)
        assert len(rebuilt) == len(sapatas)
        for a, b in zip(rebuilt, sapatas):
            assert a.h_x == b.h_x and a.h_y == b.h_y and a.h_z == b.h_z
            assert a.pilar.rotulo == b.pilar.rotulo

    def test_design_vector_wrong_size_raises(self, assets_dir: Path):
        """This test ensures vectors with the wrong size are rejected.

        :return: None (internal asserts)
        """
        proj = read_projeto_from_excel(
            assets_dir / "data" / "problema_fund_três.xlsx", f_ck_kpa=25_000.0, cobrimento_m=0.04
        )
        with pytest.raises(ValueError, match="design vector"):
            design_vector_to_sapatas([0.0, 0.0], proj)


# =============================================================================
# OptimisationConfig
# =============================================================================
class TestOptimisationConfig:
    """This class verifies the OptimisationConfig validation invariants."""

    def test_defaults_are_valid(self):
        """This test ensures the default configuration is internally consistent.

        :return: None (internal asserts)
        """
        cfg = OptimisationConfig()
        assert cfg.h_min_m < cfg.h_max_m
        assert cfg.n_rep >= 1

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"h_min_m": 0.0},
            {"h_max_m": -0.5},
            {"h_min_m": 1.0, "h_max_m": 1.0},
            {"n_gen": 0},
            {"n_pop": 1},
            {"n_rep": 0},
            {"ga_epoch": 0},
            {"ga_pop_size": 1},
            {"penalty": 0.0},
            {"penalty": -1.0},
        ],
    )
    def test_invalid_combinations_raise(self, kwargs):
        """This test ensures invalid configurations are rejected at construction.

        :param kwargs: Override that should make the configuration invalid

        :return: None (internal asserts)
        """
        with pytest.raises(ValueError):
            OptimisationConfig(**kwargs)

    def test_penalty_none_is_accepted(self):
        """This test ensures the default ``penalty=None`` falls through to the engine.

        :return: None (internal asserts)
        """
        cfg = OptimisationConfig(penalty=None)
        assert cfg.penalty is None

    def test_extra_fields_are_forbidden(self):
        """This test ensures Pydantic rejects unknown fields at construction.

        Catches typos in callers before they silently bypass validation
        (e.g. ``pop_size`` when the field is actually ``ga_pop_size``).

        :return: None (internal asserts)
        """
        with pytest.raises(ValueError):
            OptimisationConfig(unknown_field=1)   # type: ignore[call-arg]

    def test_model_is_frozen(self):
        """This test ensures ``OptimisationConfig`` instances are immutable.

        :return: None (internal asserts)
        """
        cfg = OptimisationConfig()
        with pytest.raises(ValueError):
            cfg.h_min_m = 1.0   # type: ignore[misc]

    def test_model_dump_round_trip(self):
        """This test ensures the model serialises and re-loads losslessly.

        Useful for persisting the configuration alongside experiment
        results (mlflow / parquet / json reports).

        :return: None (internal asserts)
        """
        original = OptimisationConfig(
            h_min_m=0.5, h_max_m=2.0, n_gen=3, n_pop=100, n_rep=4,
            base_seed=7, kernel_index=2, ga_epoch=20, ga_pop_size=40,
            penalty=12.5,
        )
        as_dict = original.model_dump()
        round_tripped = OptimisationConfig(**as_dict)
        assert round_tripped == original

    def test_json_schema_is_self_describing(self):
        """This test ensures Pydantic generates a JSON schema with all fields.

        Confirms that downstream tools (FastAPI, OpenAPI, docs) will see
        the configuration surface as a documented contract.

        :return: None (internal asserts)
        """
        schema = OptimisationConfig.model_json_schema()
        assert schema["type"] == "object"
        properties = schema["properties"]
        for field_name in (
            "h_min_m", "h_max_m", "n_gen", "n_pop", "n_rep",
            "base_seed", "kernel_index", "ga_epoch", "ga_pop_size", "penalty",
        ):
            assert field_name in properties, f"missing field in schema: {field_name}"
            # Every documented field carries a description for tooling
            assert properties[field_name].get("description"), (
                f"field {field_name!r} is missing a description in the JSON schema"
            )


# =============================================================================
# evaluate — preserves the regression baseline
# =============================================================================
class TestEvaluate:
    """This class verifies that ``evaluate`` preserves the Sprint 2 baseline."""

    def _seed42_sapatas(
        self, projeto: FundacaoProjeto
    ) -> list[Sapata]:
        """This helper builds the canonical seed-42 sapatas for the baseline.

        Reproduces the design vector used by ``test_avaliar_projeto.py``:
        ``np.random.seed(42); np.random.uniform(0.6, 3.0, size=3*N)``.

        :return: List of Sapata entities decoded from the canonical x_seed42
        """
        np.random.seed(42)
        x = np.random.uniform(0.6, 3.0, size=3 * projeto.n_fund)
        return design_vector_to_sapatas(x.tolist(), projeto)

    def test_baseline_matches_19_706(self, assets_dir: Path):
        """This test ensures evaluate reproduces of = 19.70604234767181.

        Same baseline as ``test_baseline_three_foundations_returns_19_706``
        in ``tests/test_avaliar_projeto.py``, but exercising the API
        layer end-to-end (Excel reader -> domain entities -> evaluate).

        :return: None (internal asserts)
        """
        proj = read_projeto_from_excel(
            assets_dir / "data" / "problema_fund_três.xlsx",
            f_ck_kpa=25_000.0,
            cobrimento_m=0.04,
        )
        sapatas = self._seed42_sapatas(proj)
        result = evaluate(proj, sapatas)
        assert isinstance(result, EvaluationResult)
        assert result.of_total == pytest.approx(19.70604234767181, rel=1e-12)

    def test_constraints_table_is_per_element(self, assets_dir: Path):
        """This test ensures every pillar shows up in the constraints mapping.

        :return: None (internal asserts)
        """
        proj = read_projeto_from_excel(
            assets_dir / "data" / "problema_fund_três.xlsx",
            f_ck_kpa=25_000.0,
            cobrimento_m=0.04,
        )
        sapatas = self._seed42_sapatas(proj)
        result = evaluate(proj, sapatas)
        assert set(result.constraints) == {p.rotulo for p in proj.pilares}
        for table in result.constraints.values():
            assert {"g sobreposicao", "g punção secao C", "g tensao", "g geometria"} <= set(table)

    def test_penalty_override_is_honoured(self, assets_dir: Path):
        """This test ensures the explicit penalty argument actually changes the OF.

        :return: None (internal asserts)
        """
        proj = read_projeto_from_excel(
            assets_dir / "data" / "problema_fund_três.xlsx",
            f_ck_kpa=25_000.0,
            cobrimento_m=0.04,
        )
        sapatas = self._seed42_sapatas(proj)
        a = evaluate(proj, sapatas, penalty=10.0).of_total
        b = evaluate(proj, sapatas, penalty=1e6).of_total
        assert b > a * 1_000.0   # penalty mudou drasticamente o OF

    def test_wrong_number_of_sapatas_raises(self, assets_dir: Path):
        """This test ensures evaluate rejects mismatched ``len(sapatas)``.

        :return: None (internal asserts)
        """
        proj = read_projeto_from_excel(
            assets_dir / "data" / "problema_fund_três.xlsx",
            f_ck_kpa=25_000.0,
            cobrimento_m=0.04,
        )
        sapatas = self._seed42_sapatas(proj)[:1]
        with pytest.raises(ValueError, match="expected"):
            evaluate(proj, sapatas)


# =============================================================================
# optimize — orchestration plumbing
# =============================================================================
class TestOptimize:
    """This class verifies the orchestration of ``optimize``.

    Uses very small ``n_pop`` / ``n_gen`` / ``n_rep`` to stay within the
    regression suite time budget. The point is to validate the plumbing
    (return shape, seed propagation, reproducibility) — not algorithm
    quality.
    """

    def _small_config(self) -> OptimisationConfig:
        """This helper returns a tight config that runs in a few seconds.

        ``ga_pop_size`` stays at 20 because mealpy's k-way tournament
        selection breaks with very small populations. This is a
        plumbing test, not a tuning study.

        :return: OptimisationConfig with small generations, population and reps
        """
        return OptimisationConfig(
            h_min_m=0.60,
            h_max_m=1.50,
            n_gen=1,
            n_pop=12,
            n_rep=2,
            base_seed=123,
            kernel_index=-1,
            ga_epoch=4,
            ga_pop_size=20,
            penalty=10.0,
        )

    def test_optimize_returns_typed_result(self, assets_dir: Path):
        """This test ensures optimize returns OptimisationResult with the right shape.

        :return: None (internal asserts)
        """
        proj = read_projeto_from_excel(
            assets_dir / "data" / "problema_fund_um.xlsx",
            f_ck_kpa=25_000.0,
            cobrimento_m=0.04,
        )
        cfg = self._small_config()
        result = optimize(proj, cfg)
        assert isinstance(result, OptimisationResult)
        assert len(result.sapatas) == proj.n_fund
        assert len(result.per_rep_of) == cfg.n_rep
        assert result.best_of <= min(result.per_rep_of) + 1e-9
        # best_seed is one of the actual repetition seeds
        assert result.best_seed in {cfg.base_seed + r for r in range(cfg.n_rep)}

    def test_optimize_is_reproducible(self, assets_dir: Path):
        """This test ensures two calls with the same config yield the same result.

        :return: None (internal asserts)
        """
        proj = read_projeto_from_excel(
            assets_dir / "data" / "problema_fund_um.xlsx",
            f_ck_kpa=25_000.0,
            cobrimento_m=0.04,
        )
        cfg = self._small_config()
        r1 = optimize(proj, cfg)
        r2 = optimize(proj, cfg)
        assert r1.best_of == pytest.approx(r2.best_of, rel=1e-12)
        assert r1.per_rep_of == r2.per_rep_of
        assert r1.best_seed == r2.best_seed

    def test_sapatas_respect_bounds(self, assets_dir: Path):
        """This test ensures the optimised sapatas stay within the configured bounds.

        :return: None (internal asserts)
        """
        proj = read_projeto_from_excel(
            assets_dir / "data" / "problema_fund_um.xlsx",
            f_ck_kpa=25_000.0,
            cobrimento_m=0.04,
        )
        cfg = self._small_config()
        result = optimize(proj, cfg)
        for s in result.sapatas:
            assert cfg.h_min_m <= s.h_x <= cfg.h_max_m
            assert cfg.h_min_m <= s.h_y <= cfg.h_max_m
            assert cfg.h_min_m <= s.h_z <= cfg.h_max_m
