"""Unified export panel for the FundaIA result section.

Centralises the artifacts the user can take home after an
optimisation: DXF (CAD-ready arrangement), JSON (structured run
summary), interactive HTML (the 3D viewer as a stand-alone file),
PNG (rasterised snapshot of the EGO history chart). The component
returns the ``bytes`` for each artifact so the caller decides how
to expose them — typically a row of ``st.download_button``.

Resumo em português:
    Bloco unificado de exportação. Recebe o ``OptimisationResult``
    e, opcionalmente, o ``ExperimentRun`` da última run; devolve
    bytes para DXF, JSON, HTML 3D e PNG do histórico, prontos
    para botões de download.
"""

from __future__ import annotations

import dataclasses
import io
import json
from typing import Any, Mapping, Optional, Sequence

from core.api import OptimisationResult
from core.domain import Sapata
from core.io import sapatas_to_dxf_bytes


__all__ = [
    "result_to_json_bytes",
    "figure_to_html_bytes",
    "figure_to_png_bytes",
    "build_export_artifacts",
]


def result_to_json_bytes(
    result: OptimisationResult,
    *,
    metrics: Optional[Mapping[str, Any]] = None,
    run_id: Optional[str] = None,
) -> bytes:
    """Serialise the optimisation result as a JSON document.

    The payload is a self-contained snapshot suitable for ingestion
    by a sibling tool: best objective, per-rep trajectory, sapata
    list (with pillar metadata) and any aggregated metrics passed
    in. Keys are stable across runs so the JSON can be diffed.

    :param result: The ``OptimisationResult`` returned by ``optimize``
    :param metrics: Optional metrics dict (typically from
                    ``ExperimentRun.manifest.metrics``)
    :param run_id: Optional run identifier (typically from
                   ``ExperimentRecorder.run_id``)

    :return: UTF-8 encoded JSON bytes (indent=2, sorted keys)
    """
    payload: dict[str, Any] = {
        "run_id": run_id,
        "best_of": result.best_of,
        "best_seed": result.best_seed,
        "per_rep_of": list(result.per_rep_of),
        "sapatas": [_sapata_dict(s) for s in result.sapatas],
    }
    if metrics is not None:
        payload["metrics"] = dict(metrics)
    return json.dumps(payload, indent=2, sort_keys=True, default=str).encode("utf-8")


def _sapata_dict(s: Sapata) -> dict:
    """Turn a domain :class:`Sapata` into a plain JSON-friendly dict."""
    return {
        "rotulo": s.pilar.rotulo,
        "xg_m": s.pilar.xg, "yg_m": s.pilar.yg,
        "a_p_m": s.pilar.a_p, "b_p_m": s.pilar.b_p,
        "h_x_m": s.h_x, "h_y_m": s.h_y, "h_z_m": s.h_z,
        "volume_m3": s.volume,
    }


def figure_to_html_bytes(fig, *, title: Optional[str] = None) -> bytes:
    """Render a Plotly figure as a stand-alone HTML document.

    The resulting HTML embeds the Plotly bundle so the user can
    open the file in any browser — no internet, no Streamlit, no
    Python — and keep all the interactive controls (rotate, zoom,
    legend toggles).

    :param fig: ``plotly.graph_objects.Figure``
    :param title: Optional ``<title>`` for the HTML document

    :return: UTF-8 encoded HTML bytes
    """
    html = fig.to_html(include_plotlyjs="cdn", full_html=True,
                       config={"displaylogo": False, "responsive": True})
    if title:
        html = html.replace("<head>", f"<head><title>{title}</title>", 1)
    return html.encode("utf-8")


def figure_to_png_bytes(fig, *, scale: float = 2.0) -> Optional[bytes]:
    """Try to render a Plotly figure as a PNG snapshot.

    Returns ``None`` when the optional ``kaleido`` engine is not
    installed; the caller should hide the PNG download button in
    that case rather than presenting a broken file. The function
    deliberately swallows export errors — PNGs are nice-to-have,
    DXF and JSON are the primary deliverables.

    :param fig: ``plotly.graph_objects.Figure``
    :param scale: Resolution multiplier (1.0 = 700×500 default)

    :return: PNG bytes, or ``None`` when the engine is missing
    """
    try:
        return fig.to_image(format="png", scale=scale, engine="kaleido")
    except Exception:
        return None


def build_export_artifacts(
    result: OptimisationResult,
    *,
    fig_3d=None,
    fig_history=None,
    metrics: Optional[Mapping[str, Any]] = None,
    run_id: Optional[str] = None,
) -> dict[str, bytes]:
    """Assemble every export artifact for one optimisation result.

    Returned keys:

    * ``"dxf"``           — bytes for a CAD arrangement of the sapatas;
    * ``"json"``          — bytes of the structured JSON snapshot;
    * ``"html_3d"``       — bytes of the stand-alone 3D viewer (only
                             when ``fig_3d`` is supplied);
    * ``"html_history"``  — bytes of the stand-alone EGO history
                             chart (only when ``fig_history`` is
                             supplied);
    * ``"png_history"``   — bytes of a rasterised history chart (only
                             when ``fig_history`` is supplied **and**
                             the kaleido engine is available).

    :param result: ``OptimisationResult`` produced by ``optimize``
    :param fig_3d: Optional Plotly figure of the 3D arrangement
    :param fig_history: Optional Plotly figure of the EGO history
    :param metrics: Optional aggregated metrics dict
    :param run_id: Optional run identifier

    :return: Mapping ``artifact_name -> bytes``
    """
    artifacts: dict[str, bytes] = {
        "dxf": sapatas_to_dxf_bytes(result.sapatas),
        "json": result_to_json_bytes(result, metrics=metrics, run_id=run_id),
    }
    if fig_3d is not None:
        artifacts["html_3d"] = figure_to_html_bytes(fig_3d, title="FundaIA — 3D")
    if fig_history is not None:
        artifacts["html_history"] = figure_to_html_bytes(
            fig_history, title="FundaIA — EGO history"
        )
        png = figure_to_png_bytes(fig_history)
        if png is not None:
            artifacts["png_history"] = png
    return artifacts
