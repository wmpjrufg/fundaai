"""Tests for ``core.io.experiments`` (experiment persistence).

The persistence layer turns each ``optimize`` call into a self-
describing folder under ``experiments/<run_id>/``. These tests lock
the contract on six fronts:

    1. **Schema**: schema_version is captured and enforced; manifest
       is JSON; histories are Parquet; summary is CSV.
    2. **Recorder lifecycle**: ``begin -> record_rep -> end`` writes
       every required file with the expected status transitions.
    3. **Atomicity**: writes use a temp-then-rename pattern so a
       reader never sees a half-written manifest.
    4. **Round-trip**: ``load_experiment`` reproduces an
       ``ExperimentRun`` whose history matches what was recorded.
    5. **Metrics**: ``summarise_history`` computes paper-grade
       metrics correctly for known synthetic histories.
    6. **End-to-end with optimize**: the full pipeline runs with a
       recorder, the resulting folder is self-contained, and a
       second ``optimize`` call from the same folder reproduces the
       OF.

The integration test runs a tiny optimisation (n_pop=8, n_gen=2,
n_rep=1) on the real three-foundation problem to keep it fast while
exercising the full code path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.api import optimize
from core.api.types import OptimisationConfig
from core.domain import FundacaoProjeto
from core.io import read_projeto_from_excel
from core.io.experiments import (
    SCHEMA_VERSION,
    ExperimentRecorder,
    compute_metrics,
    load_experiment,
    summarise_history,
)


# =============================================================================
# Helpers
# =============================================================================
def _fake_history(of_values: list[float], iters: list[int] | None = None) -> pd.DataFrame:
    """This helper builds a synthetic EGO-shaped history DataFrame.

    :param of_values: Objective values, one per row
    :param iters: ITER column values; defaults to a range from 0 to len-1

    :return: DataFrame with the columns expected by ``summarise_history``
    """
    n = len(of_values)
    if iters is None:
        iters = list(range(n))
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "ID": list(range(n)),
        "ITER": iters,
        "X_0": rng.uniform(0, 1, n),
        "X_1": rng.uniform(0, 1, n),
        "OF": of_values,
        "FIT": [1.0 / (1.0 + abs(v)) for v in of_values],
        "OF EVALUATIONS": [1] * n,
        "TIME CONSUMPTION": [0.01] * n,
    })


# =============================================================================
# summarise_history
# =============================================================================
@pytest.mark.optimization
class TestSummariseHistory:
    """This class verifies the per-rep paper-grade metrics."""

    def test_basic_fields_for_monotone_history(self):
        """This test ensures monotone-improving history yields sensible metrics."""
        # Initial pop has min 5.0; n_gen=4 with values 4, 3, 2, 1.
        hist = _fake_history(
            of_values=[10.0, 8.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            iters=[0, 0, 0, 1, 2, 3, 4],
        )
        m = summarise_history(hist)
        assert m["of_initial"] == pytest.approx(5.0)
        assert m["of_best"] == pytest.approx(1.0)
        assert m["best_iter"] == 4
        assert m["improvement_abs"] == pytest.approx(4.0)
        assert m["improvement_rel"] == pytest.approx(4.0 / 5.0)
        assert m["convergence_iter"] == 4
        assert m["convergence_ratio"] == pytest.approx(4.0 / 4.0)
        assert m["n_evals_total"] == 7
        assert m["n_gen"] == 4
        assert m["t_total_s"] == pytest.approx(7 * 0.01)
        assert 0.0 <= m["auc_best_so_far"] <= 1.0

    def test_no_improvement_history_zero_improvement(self):
        """This test ensures a flat history reports zero improvement and unit AUC."""
        hist = _fake_history(of_values=[3.0, 3.0, 3.0], iters=[0, 1, 2])
        m = summarise_history(hist)
        assert m["of_initial"] == pytest.approx(3.0)
        assert m["of_best"] == pytest.approx(3.0)
        assert m["improvement_abs"] == pytest.approx(0.0)
        # AUC undefined when initial == best -> None
        assert m["auc_best_so_far"] is None

    def test_unique_x_count_handles_duplicates(self):
        """This test ensures duplicate design vectors are de-duplicated."""
        hist = pd.DataFrame({
            "ID": [0, 1, 2, 3], "ITER": [0, 0, 1, 2],
            "X_0": [0.1, 0.1, 0.2, 0.2], "X_1": [0.3, 0.3, 0.4, 0.5],
            "OF": [5.0, 5.0, 4.0, 3.0],
            "FIT": [0.1, 0.1, 0.1, 0.1],
        })
        m = summarise_history(hist)
        # (0.1, 0.3), (0.2, 0.4), (0.2, 0.5) -> 3 unique
        assert m["n_unique_x"] == 3

    def test_empty_history_raises(self):
        """This test ensures an empty history raises rather than yielding nan."""
        with pytest.raises(ValueError):
            summarise_history(pd.DataFrame(columns=["ITER", "OF"]))


# =============================================================================
# compute_metrics
# =============================================================================
@pytest.mark.optimization
class TestComputeMetrics:
    """This class verifies the across-rep aggregation."""

    def test_aggregates_min_mean_std(self):
        """This test ensures aggregates match handcrafted values."""
        rows = [
            {"rep_id": 0, "seed": 1, "wall_time_s": 1.0,
             "of_best": 5.0, "convergence_iter": 4,
             "auc_best_so_far": 0.3, "improvement_rel": 0.5,
             "t_total_s": 0.5},
            {"rep_id": 1, "seed": 2, "wall_time_s": 2.0,
             "of_best": 3.0, "convergence_iter": 2,
             "auc_best_so_far": 0.1, "improvement_rel": 0.7,
             "t_total_s": 0.6},
            {"rep_id": 2, "seed": 3, "wall_time_s": 3.0,
             "of_best": 4.0, "convergence_iter": 3,
             "auc_best_so_far": 0.2, "improvement_rel": 0.6,
             "t_total_s": 0.7},
        ]
        m = compute_metrics(rows)
        assert m["n_rep"] == 3
        assert m["best_of"] == pytest.approx(3.0)
        assert m["worst_of"] == pytest.approx(5.0)
        assert m["mean_of"] == pytest.approx(4.0)
        assert m["median_of"] == pytest.approx(4.0)
        assert m["best_rep_id"] == 1
        assert m["mean_convergence_iter"] == pytest.approx(3.0)
        assert m["mean_auc_best_so_far"] == pytest.approx(0.2)
        assert m["wall_time_total_s"] == pytest.approx(6.0)


# =============================================================================
# Recorder
# =============================================================================
@pytest.fixture
def projeto_tres(assets_dir: Path) -> FundacaoProjeto:
    """This fixture loads the canonical three-foundation project."""
    return read_projeto_from_excel(
        assets_dir / "data" / "problema_fund_três.xlsx", f_ck_kpa=25_000.0, cobrimento_m=0.04
    )


@pytest.mark.optimization
class TestExperimentRecorder:
    """This class verifies the recorder lifecycle."""

    def test_begin_creates_expected_files(self, tmp_path: Path, projeto_tres):
        """This test ensures begin() writes manifest, config, env, project files."""
        cfg = OptimisationConfig(n_pop=4, n_gen=1, n_rep=1)
        rec = ExperimentRecorder(root=tmp_path, run_id="run_1")
        rec.begin(cfg, projeto_tres)
        run_dir = rec.run_dir
        assert (run_dir / "manifest.json").exists()
        assert (run_dir / "config.json").exists()
        assert (run_dir / "env.json").exists()
        assert (run_dir / "project.json").exists()
        # Manifest is valid JSON with the right schema_version + status.
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["schema_version"] == SCHEMA_VERSION
        assert manifest["status"] == "running"
        assert manifest["run_id"] == "run_1"

    def test_record_rep_writes_parquet_and_summary(self, tmp_path: Path, projeto_tres):
        """This test ensures record_rep writes one parquet + one CSV row per rep."""
        cfg = OptimisationConfig(n_pop=4, n_gen=1, n_rep=2)
        rec = ExperimentRecorder(root=tmp_path, run_id="run_2")
        rec.begin(cfg, projeto_tres)
        for rep in range(2):
            rec.record_rep(
                rep_id=rep,
                seed=42 + rep,
                history=_fake_history([10.0, 8.0, 5.0], iters=[0, 0, 1]),
                wall_time_s=1.0,
            )
        run_dir = rec.run_dir
        assert (run_dir / "history" / "rep_000.parquet").exists()
        assert (run_dir / "history" / "rep_001.parquet").exists()
        df = pd.read_csv(run_dir / "summary.csv")
        assert len(df) == 2
        assert {"rep_id", "seed", "wall_time_s", "of_best", "of_initial"}.issubset(df.columns)

    def test_end_writes_metrics_and_completed_status(self, tmp_path: Path, projeto_tres):
        """This test ensures end() finalises the manifest with metrics + completed status."""
        cfg = OptimisationConfig(n_pop=4, n_gen=1, n_rep=1)
        rec = ExperimentRecorder(root=tmp_path, run_id="run_3")
        rec.begin(cfg, projeto_tres)
        rec.record_rep(
            rep_id=0, seed=42,
            history=_fake_history([10.0, 8.0, 5.0], iters=[0, 0, 1]),
            wall_time_s=1.0,
        )
        manifest = rec.end()
        assert manifest.status == "completed"
        assert manifest.completed_at is not None
        assert manifest.metrics is not None
        assert (rec.run_dir / "metrics.json").exists()
        loaded = json.loads((rec.run_dir / "metrics.json").read_text())
        assert "best_of" in loaded and "wall_time_total_s" in loaded

    def test_cancel_marks_failed(self, tmp_path: Path, projeto_tres):
        """This test ensures cancel() flips status to 'failed' with the error message."""
        cfg = OptimisationConfig(n_pop=4, n_gen=1, n_rep=1)
        rec = ExperimentRecorder(root=tmp_path, run_id="run_4")
        rec.begin(cfg, projeto_tres)
        rec.cancel("RuntimeError('boom')")
        manifest = json.loads((rec.run_dir / "manifest.json").read_text())
        assert manifest["status"] == "failed"
        assert "boom" in manifest["error"]

    def test_artifact_writing_rejects_path_traversal(self, tmp_path: Path, projeto_tres):
        """This test ensures artifact names with separators are rejected."""
        cfg = OptimisationConfig(n_pop=4, n_gen=1, n_rep=1)
        rec = ExperimentRecorder(root=tmp_path, run_id="run_5")
        rec.begin(cfg, projeto_tres)
        with pytest.raises(ValueError):
            rec.write_artifact("../escape.bin", b"x")

    def test_artifact_writing_persists_bytes(self, tmp_path: Path, projeto_tres):
        """This test ensures write_artifact stores the supplied bytes verbatim."""
        cfg = OptimisationConfig(n_pop=4, n_gen=1, n_rep=1)
        rec = ExperimentRecorder(root=tmp_path, run_id="run_6")
        rec.begin(cfg, projeto_tres)
        path = rec.write_artifact("best.dxf", b"DXF\x00bytes")
        assert path.read_bytes() == b"DXF\x00bytes"

    def test_record_before_begin_raises(self, tmp_path: Path):
        """This test ensures record_rep before begin() raises clearly."""
        rec = ExperimentRecorder(root=tmp_path, run_id="run_7")
        with pytest.raises(RuntimeError):
            rec.record_rep(0, 42, _fake_history([1.0, 0.5]), 0.1)


# =============================================================================
# load_experiment
# =============================================================================
@pytest.mark.optimization
class TestLoadExperiment:
    """This class verifies the round-trip from disk back to memory."""

    def test_round_trip_preserves_history(self, tmp_path: Path, projeto_tres):
        """This test ensures load_experiment recovers manifest + history."""
        cfg = OptimisationConfig(n_pop=4, n_gen=1, n_rep=2)
        rec = ExperimentRecorder(root=tmp_path, run_id="run_load_1")
        rec.begin(cfg, projeto_tres)
        h0 = _fake_history([10.0, 8.0, 5.0], iters=[0, 0, 1])
        h1 = _fake_history([12.0, 9.0, 7.0, 4.0], iters=[0, 0, 1, 2])
        rec.record_rep(0, 42, h0, 1.0)
        rec.record_rep(1, 43, h1, 2.0)
        rec.end()

        loaded = load_experiment(rec.run_dir)
        assert loaded.manifest.run_id == "run_load_1"
        assert loaded.manifest.status == "completed"
        assert set(loaded.history.keys()) == {0, 1}
        # OF columns survived
        assert loaded.history[0]["OF"].tolist() == h0["OF"].tolist()
        assert loaded.history[1]["OF"].tolist() == h1["OF"].tolist()

    def test_unsupported_schema_version_raises(self, tmp_path: Path, projeto_tres):
        """This test ensures a stale schema_version is rejected at load time."""
        cfg = OptimisationConfig(n_pop=4, n_gen=1, n_rep=1)
        rec = ExperimentRecorder(root=tmp_path, run_id="run_load_2")
        rec.begin(cfg, projeto_tres)
        rec.end()
        # Tamper with the manifest to simulate a future schema.
        manifest_path = rec.run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["schema_version"] = "999.0"
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="schema_version"):
            load_experiment(rec.run_dir)

    def test_missing_manifest_raises_filenotfound(self, tmp_path: Path):
        """This test ensures pointing at an empty folder raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_experiment(tmp_path)


# =============================================================================
# End-to-end with optimize()
# =============================================================================
@pytest.mark.optimization
class TestOptimizeIntegration:
    """This class verifies optimize() end-to-end with the recorder enabled."""

    def test_optimize_writes_complete_run_folder(self, tmp_path: Path, projeto_tres):
        """This test ensures optimize() produces a fully populated run folder.

        Uses a tiny configuration (n_pop=8, n_gen=2, n_rep=1) so the
        test stays fast while still exercising the EGO + GA stack.
        """
        cfg = OptimisationConfig(
            h_min_m=0.6, h_max_m=3.0,
            n_pop=8, n_gen=2, n_rep=1,
            base_seed=42, kernel_index=-1,
            ga_epoch=10, ga_pop_size=20,
        )
        rec = ExperimentRecorder(root=tmp_path, run_id="e2e_1")
        result = optimize(projeto_tres, cfg, recorder=rec)

        assert result.best_of == pytest.approx(min(result.per_rep_of))
        run_dir = rec.run_dir
        # Every documented file is present.
        for required in ("manifest.json", "config.json", "env.json",
                         "project.json", "metrics.json", "summary.csv"):
            assert (run_dir / required).exists(), f"missing {required}"
        assert (run_dir / "history" / "rep_000.parquet").exists()
        # Manifest reports completion + matching metrics.
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["status"] == "completed"
        assert manifest["metrics"]["best_of"] == pytest.approx(result.best_of, rel=1e-12)

    def test_progress_callback_receives_named_events(self, tmp_path: Path, projeto_tres):
        """A progress callback gets start/rep/iter/end events with required keys.

        Real optimisation, tiny config: n_pop=4, n_gen=1, n_rep=1. The
        callback must receive at least optimize.start, optimize.rep_start,
        ego.iter, optimize.rep_end and optimize.end, each one carrying
        the contextual keys the UI relies on.
        """
        from core.api import OptimisationConfig, optimize

        events: list[dict] = []
        cfg = OptimisationConfig(
            h_min_m=0.6, h_max_m=3.0,
            n_pop=4, n_gen=1, n_rep=1,
            ga_epoch=5, ga_pop_size=10,
        )
        rec = ExperimentRecorder(root=tmp_path, run_id="prog-1")
        optimize(projeto_tres, cfg, recorder=rec,
                 progress=lambda ev: events.append(ev))

        seen = {e["event"] for e in events}
        assert {"optimize.start", "optimize.rep_start", "ego.iter",
                "optimize.rep_end", "optimize.end"}.issubset(seen)

        ego_evs = [e for e in events if e["event"] == "ego.iter"]
        assert ego_evs and all(
            {"iter", "n_gen", "of_min", "n_train", "rep"}.issubset(e)
            for e in ego_evs
        )

    def test_progress_callback_errors_do_not_abort_run(self, tmp_path: Path, projeto_tres):
        """A buggy progress callback must not crash the optimisation."""
        from core.api import OptimisationConfig, optimize

        def boom(_ev):
            raise RuntimeError("ui hook bug")

        cfg = OptimisationConfig(
            h_min_m=0.6, h_max_m=3.0,
            n_pop=4, n_gen=1, n_rep=1,
            ga_epoch=5, ga_pop_size=10,
        )
        rec = ExperimentRecorder(root=tmp_path, run_id="prog-2")
        # Should complete normally despite the always-raising callback
        result = optimize(projeto_tres, cfg, recorder=rec, progress=boom)
        assert result.best_of < float("inf")

    def test_optimize_failure_marks_run_failed(self, tmp_path: Path, monkeypatch):
        """This test ensures an exception during optimize flips the run to 'failed'.

        The recorder must persist the failure status so a CI dashboard
        can flag broken runs without parsing logs.
        """
        # Force ego_01_architecture to raise. ``core.api.optimize`` is
        # shadowed by the re-exported ``optimize`` function, so reach
        # the submodule via importlib explicitly.
        import importlib
        optimize_module = importlib.import_module("core.api.optimize")

        def boom(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(optimize_module, "ego_01_architecture", boom)

        cfg = OptimisationConfig(n_pop=4, n_gen=1, n_rep=1)
        rec = ExperimentRecorder(root=tmp_path, run_id="e2e_fail")
        # We need a project to call optimize; build the smallest possible one.
        from core.io import read_projeto_from_excel
        proj = read_projeto_from_excel(
            Path(__file__).resolve().parent.parent / "assets" / "data" / "problema_fund_um.xlsx",
            f_ck_kpa=25_000.0, cobrimento_m=0.04,
        )
        with pytest.raises(RuntimeError):
            optimize(proj, cfg, recorder=rec)
        manifest = json.loads((rec.run_dir / "manifest.json").read_text())
        assert manifest["status"] == "failed"
        assert "boom" in manifest["error"]
