"""Tests for ``frontend.components.result_export``."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import plotly.graph_objects as go
import pytest

from core.api import OptimisationConfig, OptimisationResult, optimize
from core.io import read_projeto_from_excel
from frontend.components import (
    build_export_artifacts,
    figure_to_html_bytes,
    figure_to_png_bytes,
    result_to_json_bytes,
)


@pytest.fixture
def small_result(assets_dir: Path) -> OptimisationResult:
    """Run a tiny optimize so the export panel has something to consume."""
    proj = read_projeto_from_excel(
        assets_dir / "data" / "problema_fund_um.xlsx",
        f_ck_kpa=25_000.0, cobrimento_m=0.04,
    )
    cfg = OptimisationConfig(
        h_min_m=0.6, h_max_m=3.0,
        n_pop=4, n_gen=1, n_rep=1,
        ga_epoch=5, ga_pop_size=10,
    )
    return optimize(proj, cfg)


@pytest.mark.optimization
class TestResultToJsonBytes:
    """This class verifies the JSON snapshot of the result."""

    def test_payload_carries_expected_keys(self, small_result):
        payload = json.loads(result_to_json_bytes(small_result, run_id="r1"))
        assert payload["run_id"] == "r1"
        assert "best_of" in payload and "best_seed" in payload
        assert "per_rep_of" in payload and "sapatas" in payload
        assert len(payload["sapatas"]) == len(small_result.sapatas)

    def test_metrics_are_attached_when_supplied(self, small_result):
        payload = json.loads(result_to_json_bytes(
            small_result, metrics={"best_of": 1.23}
        ))
        assert payload["metrics"]["best_of"] == 1.23


@pytest.mark.optimization
class TestFigureExports:
    """This class verifies the per-figure helpers."""

    def test_figure_to_html_bytes_includes_plotly(self):
        fig = go.Figure()
        out = figure_to_html_bytes(fig, title="x")
        # CDN script include for Plotly is the canary
        assert b"plotly" in out.lower()
        assert b"<title>x</title>" in out

    def test_figure_to_png_bytes_returns_bytes_or_none(self):
        # Kaleido is optional; the helper must not raise either way.
        fig = go.Figure()
        out = figure_to_png_bytes(fig)
        assert out is None or isinstance(out, bytes)


@pytest.mark.optimization
class TestBuildExportArtifacts:
    """This class verifies the unified bundle."""

    def test_dxf_and_json_always_present(self, small_result):
        bundle = build_export_artifacts(small_result)
        assert "dxf" in bundle and bundle["dxf"]
        assert "json" in bundle and bundle["json"]
        assert "html_3d" not in bundle
        assert "html_history" not in bundle

    def test_html_3d_present_when_fig_supplied(self, small_result):
        bundle = build_export_artifacts(small_result, fig_3d=go.Figure())
        assert "html_3d" in bundle and bundle["html_3d"]

    def test_html_history_present_when_fig_supplied(self, small_result):
        bundle = build_export_artifacts(small_result, fig_history=go.Figure())
        assert "html_history" in bundle and bundle["html_history"]
