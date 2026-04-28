"""Colour palette and Plotly template for the FundaIA front end.

A single source of truth so every chart, every component and every
CSS rule reuse the same colours. The palette mirrors
``.streamlit/config.toml`` (dark base + warm accent) and is
designed to keep contrast comfortable on Plotly 3D meshes (the 3D
footings viewer) and on time-series plots (the EGO history chart).
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import plotly.io as pio


PALETTE: dict[str, Any] = {
    # Surfaces (mirror config.toml)
    "bg":            "#0b1220",   # main background
    "surface":       "#111a2e",   # elevated surface (cards, tabs, sidebar)
    "surface_alt":   "#0f1729",   # hover / pressed
    "border":        "#1f2a44",   # subtle dividers
    # Foreground
    "text":          "#e5e7eb",   # primary text
    "text_muted":    "#9aa3b2",   # secondary labels, axis ticks
    # Accents
    "accent":        "#f59e0b",   # primary accent — warm amber
    "accent_strong": "#fbbf24",   # accent used on hover / active
    "best_so_far":   "#f59e0b",   # reserved for the EGO best-so-far line
    "ok":            "#10b981",   # feasible / success
    "warn":          "#f97316",   # warning / borderline constraint
    "fail":          "#ef4444",   # infeasible / violation
    # Categorical sequence (high-contrast on dark background)
    "categorical": [
        "#60a5fa",   # blue
        "#f59e0b",   # amber (accent)
        "#34d399",   # emerald
        "#a78bfa",   # violet
        "#f472b6",   # pink
        "#fb923c",   # orange
        "#22d3ee",   # cyan
        "#facc15",   # yellow
        "#fb7185",   # rose
        "#4ade80",   # green
    ],
    # Sequential ramp for "colour by volume" (dark-friendly Viridis)
    "viridis": [
        "#440154", "#482878", "#3e4989", "#31688e", "#26828e",
        "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725",
    ],
}


def plotly_template() -> go.layout.Template:
    """Return the Plotly template used by every chart in the front end.

    The template configures dark-friendly backgrounds, subtle
    gridlines, the categorical sequence above and the default font.
    Charts can override any property locally without leaking back
    into the global default.

    :return: Configured ``plotly.graph_objects.layout.Template``
    """
    layout = go.Layout(
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["bg"],
        font=dict(color=PALETTE["text"], size=13,
                  family="Inter, system-ui, -apple-system, Segoe UI, "
                         "Roboto, Helvetica, Arial, sans-serif"),
        colorway=PALETTE["categorical"],
        xaxis=dict(
            gridcolor=PALETTE["border"],
            zerolinecolor=PALETTE["border"],
            linecolor=PALETTE["border"],
            tickcolor=PALETTE["text_muted"],
            color=PALETTE["text_muted"],
        ),
        yaxis=dict(
            gridcolor=PALETTE["border"],
            zerolinecolor=PALETTE["border"],
            linecolor=PALETTE["border"],
            tickcolor=PALETTE["text_muted"],
            color=PALETTE["text_muted"],
        ),
        scene=dict(
            bgcolor=PALETTE["bg"],
            xaxis=dict(backgroundcolor=PALETTE["bg"],
                       gridcolor=PALETTE["border"],
                       zerolinecolor=PALETTE["border"],
                       color=PALETTE["text_muted"]),
            yaxis=dict(backgroundcolor=PALETTE["bg"],
                       gridcolor=PALETTE["border"],
                       zerolinecolor=PALETTE["border"],
                       color=PALETTE["text_muted"]),
            zaxis=dict(backgroundcolor=PALETTE["bg"],
                       gridcolor=PALETTE["border"],
                       zerolinecolor=PALETTE["border"],
                       color=PALETTE["text_muted"]),
        ),
        legend=dict(
            bgcolor="rgba(17, 26, 46, 0.85)",   # surface with alpha
            bordercolor=PALETTE["border"],
            borderwidth=1,
            font=dict(color=PALETTE["text"]),
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        hoverlabel=dict(
            bgcolor=PALETTE["surface"],
            bordercolor=PALETTE["border"],
            font=dict(color=PALETTE["text"]),
        ),
        transition=dict(duration=350, easing="cubic-in-out"),
    )
    return go.layout.Template(layout=layout)


# Register the template under a stable name and make it the default,
# so any later go.Figure() inherits the FundaIA look automatically.
pio.templates["fundaia_dark"] = plotly_template()
pio.templates.default = "fundaia_dark"
