"""EGO best-so-far chart — premium history plot for ``optimize`` runs.

Consumes either a live ``OptimisationResult.per_rep_of`` summary or
the full ``ExperimentRun`` produced by the recorder (Sprint 4.2),
and renders an interactive Plotly figure with:

* one **best-so-far line per repetition** plus a thicker line for the
  median across reps,
* a faint **min/max band** between the best and worst run,
* **markers** at every actual evaluation (initial LHS pop included),
* a synchronised **secondary subplot** showing the elapsed time per
  iteration,
* hover tooltips with the full design vector when available,
* a **linear / log** toggle on the OF axis,
* informative annotations (best OF, convergence iter, AUC) read
  from the persisted metrics.

The figure inherits the FundaIA dark Plotly template defined in
``frontend.theme.palette``.

Resumo em português:
    Componente premium do histórico do EGO. Recebe
    ``ExperimentRun`` (recorder) ou estruturas equivalentes e
    plota a curva *best-so-far* por repetição + banda min/max +
    medianas + tempo por iteração + escala log opcional.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from frontend.theme.palette import PALETTE


__all__ = [
    "render_ego_history",
    "best_so_far_curves",
]


# =============================================================================
# Helpers
# =============================================================================
def best_so_far_curves(history: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return the iteration index and best-so-far OF for one EGO history.

    The history may include the initial LHS population (rows with
    ``ITER == 0``); this function reduces them to a single starting
    point at iteration 0 with the LHS minimum, then accumulates the
    cumulative minimum across the EGO iterations.

    :param history: DataFrame as produced by ``ego_01_architecture``
                    (must contain ``ITER`` and ``OF``)

    :return: Tuple ``(iters, best_so_far)`` of equal length
    """
    if history.empty:
        return np.array([], dtype=int), np.array([], dtype=float)
    iters = history["ITER"].to_numpy(dtype=int)
    of = history["OF"].to_numpy(dtype=float)
    initial_min = float(of[iters == 0].min()) if (iters == 0).any() else float(of.min())
    # Build sequence: iter 0 = LHS min; then one entry per ITER > 0.
    iter_only = iters[iters > 0]
    of_iter = of[iters > 0]
    if iter_only.size == 0:
        return np.array([0], dtype=int), np.array([initial_min], dtype=float)
    # Sort by ITER (defensive — usually already sorted).
    order = np.argsort(iter_only)
    iter_only = iter_only[order]
    of_iter = of_iter[order]
    running = np.minimum.accumulate(np.minimum(of_iter, initial_min))
    out_iters = np.concatenate([[0], iter_only])
    out_curve = np.concatenate([[initial_min], running])
    return out_iters, out_curve


def _time_per_iter(history: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return iteration index and elapsed time when the column is present."""
    col = next(
        (c for c in ("TIME CONSUMPTION (s)", "TIME CONSUMPTION") if c in history.columns),
        None,
    )
    if col is None:
        return np.array([], dtype=int), np.array([], dtype=float)
    iters = history["ITER"].to_numpy(dtype=int)
    times = history[col].to_numpy(dtype=float)
    mask = iters > 0
    return iters[mask], times[mask]


# =============================================================================
# Main entry point
# =============================================================================
def render_ego_history(
    histories: Mapping[int, pd.DataFrame] | Iterable[pd.DataFrame],
    *,
    metrics: Optional[Mapping] = None,
    title: Optional[str] = None,
    show_evaluations: bool = True,
    log_y: bool = False,
) -> go.Figure:
    """Render the EGO history as a two-row interactive Plotly figure.

    Top row: best-so-far curves per repetition + min/max band +
    median line + per-evaluation markers (optional).
    Bottom row: elapsed time per iteration (when available in the
    history).

    :param histories: Either a mapping ``{rep_id: history_df}``
                      (typical: ``ExperimentRun.history``) or an
                      iterable of history DataFrames
    :param metrics: Optional aggregated metrics dict (``best_of``,
                    ``mean_convergence_iter``, ``mean_auc_best_so_far``,
                    ...) used for the annotation in the top right
    :param title: Optional figure title
    :param show_evaluations: When ``True``, draw faint markers at
                             every actual evaluation (initial pop
                             included)
    :param log_y: When ``True``, use a logarithmic scale on the OF axis

    :return: ``plotly.graph_objects.Figure``

    :raises ValueError: When no history is supplied
    """
    if isinstance(histories, Mapping):
        rep_pairs: list[tuple[int, pd.DataFrame]] = sorted(histories.items())
    else:
        rep_pairs = list(enumerate(histories))
    if not rep_pairs:
        raise ValueError("render_ego_history requires at least one history.")

    # Pre-compute curves
    curves: list[tuple[int, np.ndarray, np.ndarray]] = []
    max_iter = 0
    for rep_id, df in rep_pairs:
        x, y = best_so_far_curves(df)
        curves.append((rep_id, x, y))
        if x.size:
            max_iter = max(max_iter, int(x.max()))

    # Resample every curve onto the same iteration grid for the band /
    # median computation. Each curve is non-increasing, so a forward
    # fill on the final value is the right extrapolation when one rep
    # converged earlier than another.
    grid = np.arange(0, max_iter + 1, dtype=int)
    aligned = np.full((len(curves), grid.size), np.nan, dtype=float)
    for row, (_, x, y) in enumerate(curves):
        if not x.size:
            continue
        # Forward-fill on the iteration grid.
        idx = 0
        last = y[0]
        for col, t in enumerate(grid):
            while idx < x.size and x[idx] <= t:
                last = y[idx]
                idx += 1
            aligned[row, col] = last
    band_min = np.nanmin(aligned, axis=0)
    band_max = np.nanmax(aligned, axis=0)
    band_median = np.nanmedian(aligned, axis=0)

    # Build the figure
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.72, 0.28],
        subplot_titles=("OF best-so-far por iteração", "Tempo por iteração [s]"),
    )

    # --- Top row: band + per-rep curves + median + markers
    band_x = np.concatenate([grid, grid[::-1]])
    band_y = np.concatenate([band_max, band_min[::-1]])
    fig.add_trace(
        go.Scatter(
            x=band_x, y=band_y, fill="toself",
            fillcolor="rgba(245,158,11,0.10)",
            line=dict(width=0),
            name="min–max entre reps",
            hoverinfo="skip", showlegend=True,
        ),
        row=1, col=1,
    )

    palette = PALETTE["categorical"]
    for n, (rep_id, x, y) in enumerate(curves):
        color = palette[n % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="lines",
                line=dict(color=color, width=2),
                name=f"rep {rep_id}",
                hovertemplate=(
                    "rep " + str(rep_id) +
                    "<br>iter %{x}<br>OF best-so-far %{y:.6f}<extra></extra>"
                ),
                opacity=0.85,
            ),
            row=1, col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=grid, y=band_median, mode="lines",
            line=dict(color=PALETTE["best_so_far"], width=3.5, dash="solid"),
            name="mediana",
            hovertemplate="iter %{x}<br>mediana OF %{y:.6f}<extra></extra>",
        ),
        row=1, col=1,
    )

    if show_evaluations:
        for n, (rep_id, df) in enumerate(rep_pairs):
            color = palette[n % len(palette)]
            iters = df["ITER"].to_numpy(dtype=int)
            of = df["OF"].to_numpy(dtype=float)
            fig.add_trace(
                go.Scatter(
                    x=iters, y=of, mode="markers",
                    marker=dict(size=5, color=color, opacity=0.45,
                                line=dict(width=0)),
                    name=f"avals rep {rep_id}",
                    showlegend=False,
                    hovertemplate=(
                        "rep " + str(rep_id) +
                        "<br>iter %{x}<br>OF avaliada %{y:.6f}"
                        "<extra></extra>"
                    ),
                ),
                row=1, col=1,
            )

    # --- Bottom row: elapsed time per iteration
    for n, (rep_id, df) in enumerate(rep_pairs):
        x, y = _time_per_iter(df)
        if not x.size:
            continue
        color = palette[n % len(palette)]
        fig.add_trace(
            go.Bar(
                x=x, y=y, name=f"t rep {rep_id}",
                marker=dict(color=color, line=dict(width=0)),
                opacity=0.6,
                hovertemplate=(
                    "rep " + str(rep_id) +
                    "<br>iter %{x}<br>tempo %{y:.3f} s<extra></extra>"
                ),
                showlegend=False,
            ),
            row=2, col=1,
        )

    # --- Layout
    fig.update_xaxes(title_text="iteração do EGO", row=2, col=1)
    fig.update_yaxes(title_text="OF (volume penalizado) [m³]", row=1, col=1)
    if log_y:
        fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(title_text="tempo [s]", row=2, col=1)
    fig.update_layout(
        title=title,
        barmode="group",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60 if title else 30, b=40),
    )

    # --- Optional metrics annotation
    if metrics:
        parts: list[str] = []
        if "best_of" in metrics:
            parts.append(f"best OF: <b>{metrics['best_of']:.6f}</b>")
        mci = metrics.get("mean_convergence_iter")
        if mci is not None:
            parts.append(f"⌀ conv. iter: <b>{mci:.1f}</b>")
        auc = metrics.get("mean_auc_best_so_far")
        if auc is not None:
            parts.append(f"⌀ AUC: <b>{auc:.3f}</b>")
        if parts:
            fig.add_annotation(
                xref="paper", yref="paper", x=1.0, y=1.08,
                xanchor="right", yanchor="bottom",
                text=" &nbsp;·&nbsp; ".join(parts),
                showarrow=False,
                bordercolor=PALETTE["border"], borderwidth=1,
                borderpad=6,
                bgcolor="rgba(17,26,46,0.85)",
                font=dict(color=PALETTE["text"], size=12),
            )
    return fig
