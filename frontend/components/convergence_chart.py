"""Convergence chart — multi-algorithm best-so-far comparison.

Renders the ``best_so_far`` trajectory of every algorithm in a
:class:`core.api.BenchmarkResult`, on a shared x-axis that counts
**real objective evaluations** (the metric the EGO argument depends
on). For each algorithm: a thick median line, a faint min/max
envelope and a ``±1σ`` ribbon describe the variance across
repetitions. Markers at the per-rep curves are available behind a
``legendonly`` toggle so the user can spot individual trajectories
without polluting the default view.

The figure uses the FundaIA dark Plotly template (registered as the
default in :mod:`frontend.theme.palette`) and exposes a linear/log
toggle on the OF axis to inspect long tails of convergence.

Resumo em português:
    Componente que compara a curva *best-so-far* de cada algoritmo
    sobre o mesmo orçamento de avaliações reais. Para cada algoritmo
    plota a mediana entre repetições, banda ``±1σ`` e envelope min–max;
    pontos por repetição ficam disponíveis (escondidos por padrão) na
    legenda.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from frontend.theme.palette import PALETTE


__all__ = ["render_convergence_chart", "best_so_far_by_eval"]


def best_so_far_by_eval(group: pd.DataFrame, max_evals: int) -> np.ndarray:
    """Resample one repetition's ``of_best_so_far`` onto a 1..max_evals grid.

    The recorded trace already exposes ``of_best_so_far`` per evaluation,
    so the resample is a forward-fill on the integer grid. When the
    repetition stopped before ``max_evals`` (e.g. budget hit because of
    a cancellation), the last value is carried forward so cross-rep
    aggregates remain comparable.

    :param group: Per-rep history slice with ``eval_idx`` and
                  ``of_best_so_far``
    :param max_evals: Length of the output array

    :return: ndarray of length ``max_evals`` (NaN if the group is empty)
    """
    out = np.full(max_evals, np.nan, dtype=float)
    if group.empty:
        return out
    g = group.sort_values("eval_idx")
    idx = g["eval_idx"].to_numpy(dtype=int) - 1
    of = g["of_best_so_far"].to_numpy(dtype=float)
    if idx.size == 0:
        return out
    last = of[0]
    j = 0
    for i in range(max_evals):
        while j < idx.size and idx[j] <= i:
            last = of[j]
            j += 1
        out[i] = last
    return out


def _algorithm_color_map(algorithms: Iterable[str]) -> dict[str, str]:
    """Stable colour assignment per algorithm using the categorical palette."""
    palette = PALETTE["categorical"]
    return {alg: palette[i % len(palette)] for i, alg in enumerate(algorithms)}


def render_convergence_chart(
    history: pd.DataFrame,
    *,
    labels: Optional[dict[str, str]] = None,
    summary: Optional[pd.DataFrame] = None,
    title: Optional[str] = None,
    log_y: bool = False,
    show_individual_reps: bool = True,
    show_time_panel: bool = True,
) -> go.Figure:
    """Render the convergence comparison chart as an interactive figure.

    :param history: Long-format DataFrame as returned by
                    :func:`core.api.run_benchmark`. Must contain
                    ``algorithm``, ``rep``, ``eval_idx`` and
                    ``of_best_so_far`` columns
    :param labels: Optional mapping ``algorithm -> human label`` used
                   on the legend (defaults to the canonical labels
                   exposed by ``core.api.ALGORITHM_LABELS`` if you
                   pass them through; falls back to the raw algorithm
                   key)
    :param summary: Optional per-algorithm summary DataFrame used to
                    print a compact annotation block above the chart
                    (``best``, ``mean ± std``, ``mean conv. eval``)
    :param title: Optional figure title (rendered inside Plotly)
    :param log_y: When ``True``, the OF axis uses ``type='log'``
    :param show_individual_reps: When ``True``, expose per-rep traces
                                 (hidden behind ``legendonly``) so the
                                 user can opt into the noise
    :param show_time_panel: When ``True``, add a bottom panel with the
                            mean ``time_total_s`` curve per algorithm
                            (secondary metric — wall-clock view)

    :return: ``plotly.graph_objects.Figure``

    :raises ValueError: When ``history`` is empty
    """
    if history.empty:
        raise ValueError(
            "render_convergence_chart requires a non-empty history. "
            "Run run_benchmark first."
        )

    labels = dict(labels) if labels else {}
    algorithms: list[str] = list(dict.fromkeys(history["algorithm"].tolist()))
    color_map = _algorithm_color_map(algorithms)
    max_evals = int(history["eval_idx"].max())
    grid = np.arange(1, max_evals + 1, dtype=int)

    if show_time_panel:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.22,
            row_heights=[0.66, 0.34],
            subplot_titles=(
                "Convergência por nº de avaliações reais (best-so-far)",
                "Tempo acumulado por avaliação [s]",
            ),
        )
    else:
        fig = make_subplots(rows=1, cols=1)

    for alg in algorithms:
        alg_hist = history[history["algorithm"] == alg]
        color = color_map[alg]
        label = labels.get(alg, alg)

        # Per-rep resampling on the eval grid -----------------------------
        per_rep_curves: list[np.ndarray] = []
        per_rep_ids: list[int] = []
        for rep_id, group in alg_hist.groupby("rep", sort=True):
            per_rep_curves.append(best_so_far_by_eval(group, max_evals))
            per_rep_ids.append(int(rep_id))
        if not per_rep_curves:
            continue
        stack = np.vstack(per_rep_curves)
        with np.errstate(invalid="ignore"):
            band_min = np.nanmin(stack, axis=0)
            band_max = np.nanmax(stack, axis=0)
            band_med = np.nanmedian(stack, axis=0)
            band_mean = np.nanmean(stack, axis=0)
            band_std = np.nanstd(stack, axis=0, ddof=1) if stack.shape[0] > 1 \
                else np.zeros_like(band_mean)
        sig_lo = band_mean - band_std
        sig_hi = band_mean + band_std

        legend_group = f"alg-{alg}"
        # min/max envelope
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([grid, grid[::-1]]),
                y=np.concatenate([band_max, band_min[::-1]]),
                fill="toself",
                fillcolor=_rgba(color, 0.08),
                line=dict(width=0),
                name=f"{label} · envelope min–max",
                legendgroup=legend_group,
                legendgrouptitle_text=label,
                hoverinfo="skip",
                showlegend=True,
                visible="legendonly",
            ),
            row=1, col=1,
        )
        # ±1σ ribbon
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([grid, grid[::-1]]),
                y=np.concatenate([sig_hi, sig_lo[::-1]]),
                fill="toself",
                fillcolor=_rgba(color, 0.18),
                line=dict(width=0),
                name=f"{label} · ±1σ",
                legendgroup=legend_group,
                hoverinfo="skip",
                showlegend=True,
            ),
            row=1, col=1,
        )
        # Median line
        fig.add_trace(
            go.Scatter(
                x=grid, y=band_med, mode="lines",
                line=dict(color=color, width=3.0),
                name=f"{label} · mediana",
                legendgroup=legend_group,
                hovertemplate=(
                    f"<b>{label}</b>"
                    "<br>avaliação: %{x}"
                    "<br>OF mediana: %{y:.6f} m³"
                    "<extra></extra>"
                ),
            ),
            row=1, col=1,
        )
        # Individual reps (legendonly by default)
        if show_individual_reps:
            for rep_id, curve in zip(per_rep_ids, per_rep_curves):
                fig.add_trace(
                    go.Scatter(
                        x=grid, y=curve, mode="lines",
                        line=dict(color=color, width=1.0, dash="dot"),
                        opacity=0.55,
                        name=f"{label} · rep {rep_id}",
                        legendgroup=legend_group,
                        showlegend=True,
                        visible="legendonly",
                        hovertemplate=(
                            f"<b>{label} · rep {rep_id}</b>"
                            "<br>avaliação: %{x}"
                            "<br>OF best-so-far: %{y:.6f} m³"
                            "<extra></extra>"
                        ),
                    ),
                    row=1, col=1,
                )

        # Bottom panel — mean wall time per evaluation
        if show_time_panel:
            t_cols: list[np.ndarray] = []
            for _, group in alg_hist.groupby("rep", sort=True):
                g = group.sort_values("eval_idx")
                t_vec = np.full(max_evals, np.nan, dtype=float)
                idx = g["eval_idx"].to_numpy(dtype=int) - 1
                vals = g["time_total_s"].to_numpy(dtype=float)
                if idx.size:
                    last = vals[0]
                    j = 0
                    for i in range(max_evals):
                        while j < idx.size and idx[j] <= i:
                            last = vals[j]
                            j += 1
                        t_vec[i] = last
                t_cols.append(t_vec)
            if t_cols:
                t_stack = np.vstack(t_cols)
                with np.errstate(invalid="ignore"):
                    t_mean = np.nanmean(t_stack, axis=0)
                fig.add_trace(
                    go.Scatter(
                        x=grid, y=t_mean, mode="lines",
                        line=dict(color=color, width=2.0),
                        name=f"{label} · t médio",
                        legendgroup=legend_group,
                        showlegend=False,
                        hovertemplate=(
                            f"<b>{label}</b>"
                            "<br>avaliação: %{x}"
                            "<br>t médio acumulado: %{y:.3f} s"
                            "<extra></extra>"
                        ),
                    ),
                    row=2, col=1,
                )

    # ---------------------------------------------------------------- layout
    fig.update_xaxes(
        title_text="nº de avaliações reais",
        row=1, col=1,
        rangemode="nonnegative",
        range=[0.5, max_evals + 0.5],
        showspikes=True,
        spikecolor=PALETTE["accent"], spikethickness=1,
        spikemode="across", spikedash="dot", spikesnap="cursor",
    )
    fig.update_yaxes(
        title_text="OF (volume penalizado) [m³]",
        row=1, col=1,
        rangemode="tozero" if not log_y else "normal",
        showspikes=True,
        spikecolor=PALETTE["accent"], spikethickness=1, spikedash="dot",
    )
    if log_y:
        fig.update_yaxes(type="log", row=1, col=1)
    if show_time_panel:
        fig.update_xaxes(
            title_text="nº de avaliações reais",
            row=2, col=1,
            rangemode="nonnegative",
            range=[0.5, max_evals + 0.5],
        )
        fig.update_yaxes(
            title_text="tempo acumulado [s]",
            row=2, col=1, rangemode="tozero",
        )

    fig.update_layout(
        title=title,
        hovermode="closest",
        height=720 if show_time_panel else 520,
        margin=dict(l=30, r=30, t=80 if title else 60, b=50),
        legend=dict(
            orientation="v",
            yanchor="top", y=1.0,
            xanchor="left", x=1.02,
            groupclick="toggleitem",
        ),
        dragmode="zoom",
    )

    # Optional summary annotation -----------------------------------------
    if summary is not None and not summary.empty:
        parts: list[str] = []
        for _, row in summary.iterrows():
            parts.append(
                f"<b>{row['label']}</b> · best "
                f"{row['best']:.4f} · "
                f"⌀ {row['mean']:.4f} ± {row['std']:.4f}"
            )
        if parts:
            fig.add_annotation(
                xref="paper", yref="paper", x=0.0, y=1.08,
                xanchor="left", yanchor="bottom",
                text=" &nbsp;·&nbsp; ".join(parts),
                showarrow=False,
                bordercolor=PALETTE["border"], borderwidth=1,
                borderpad=8,
                bgcolor="rgba(17,26,46,0.85)",
                font=dict(color=PALETTE["text"], size=11),
            )
    return fig


# =============================================================================
# Helpers
# =============================================================================
def _rgba(hex_color: str, alpha: float) -> str:
    """Convert ``#rrggbb`` to a ``rgba(r,g,b,a)`` Plotly fill string."""
    h = hex_color.lstrip("#")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.3f})"
