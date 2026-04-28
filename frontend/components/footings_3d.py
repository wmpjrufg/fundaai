"""3D footings viewer (Plotly) — interactive component for Streamlit.

Renders the optimised footing layout as a fully interactive 3D scene:
each ``Sapata`` becomes a parallelepiped buried below the soil
surface (``z`` from ``-h_z`` to ``0``); each pillar above the footing
becomes a thinner box rising from ``z = 0`` to a configurable
visualisation height. The user can rotate, zoom, toggle traces and
read per-element details from the hover tooltip.

The component is **framework-aware but Streamlit-agnostic**: it
returns a ``plotly.graph_objects.Figure``. The caller (a Streamlit
page or a notebook) decides how to render — `st.plotly_chart(fig)`,
`fig.show()` or `fig.write_html(...)`.

Resumo em português:
    Visualizador 3D interativo (Plotly) das sapatas otimizadas.
    Cada sapata vira um paralelepípedo enterrado, cada pilar uma
    caixa fina acima. Hover mostra rótulo, dimensões, volume e
    coordenadas. A função devolve uma ``Figure`` Plotly, então
    serve tanto para a página Streamlit quanto para notebooks.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import plotly.graph_objects as go

from core.domain import Sapata


__all__ = [
    "render_footings_3d",
    "footing_box",
    "pillar_box",
    "DEFAULT_PILLAR_HEIGHT_M",
]


DEFAULT_PILLAR_HEIGHT_M = 1.5
"""Visualisation-only height for the pillar boxes [m].

The domain ``Pilar`` entity does not carry a structural height (only
the in-plane dimensions ``ap`` and ``bp``), so the renderer uses a
fixed value for the upward extrusion. The user may override it via
``render_footings_3d(..., pillar_height_m=...)`` to match the
expected slab elevation.
"""


# =============================================================================
# Geometry helpers — produce the 12-triangle Mesh3d data for an AABB box.
# =============================================================================
def _box_mesh(
    xc: float, yc: float, zc: float,
    dx: float, dy: float, dz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, y, z, i, j, k) arrays describing a centred AABB box.

    The box is centred on ``(xc, yc, zc)`` with extents ``dx, dy, dz``.
    ``i``, ``j``, ``k`` are integer arrays indexing into ``x/y/z`` that
    define the 12 triangles of the closed surface (Plotly convention).

    :param xc: Centre x coordinate [m]
    :param yc: Centre y coordinate [m]
    :param zc: Centre z coordinate [m]
    :param dx: Total extent on x [m]
    :param dy: Total extent on y [m]
    :param dz: Total extent on z [m]

    :return: Six 1-D numpy arrays in the order ``(x, y, z, i, j, k)``
    """
    hx, hy, hz = dx / 2.0, dy / 2.0, dz / 2.0
    # 8 vertices of the AABB
    x = np.array([xc - hx, xc + hx, xc + hx, xc - hx,
                  xc - hx, xc + hx, xc + hx, xc - hx], dtype=float)
    y = np.array([yc - hy, yc - hy, yc + hy, yc + hy,
                  yc - hy, yc - hy, yc + hy, yc + hy], dtype=float)
    z = np.array([zc - hz, zc - hz, zc - hz, zc - hz,
                  zc + hz, zc + hz, zc + hz, zc + hz], dtype=float)
    # 12 triangles (two per face)
    i = np.array([0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3], dtype=int)
    j = np.array([1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4], dtype=int)
    k = np.array([2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7], dtype=int)
    return x, y, z, i, j, k


def footing_box(sapata: Sapata, *, color: str | None = None,
                opacity: float = 0.55, name: str | None = None) -> go.Mesh3d:
    """Build a ``Mesh3d`` trace for a single footing buried below ``z = 0``.

    :param sapata: Domain entity carrying ``h_x``, ``h_y``, ``h_z`` and ``pilar``
    :param color: Plotly colour spec; defaults to a Tab10-like palette index
    :param opacity: Alpha in ``[0, 1]``
    :param name: Trace name for the legend; defaults to the pillar label

    :return: ``plotly.graph_objects.Mesh3d`` ready to be added to a Figure
    """
    xc, yc = sapata.pilar.xg, sapata.pilar.yg
    zc = -sapata.h_z / 2.0   # top at z=0, bottom at z=-h_z
    x, y, z, i, j, k = _box_mesh(xc, yc, zc, sapata.h_x, sapata.h_y, sapata.h_z)
    label = name or sapata.pilar.rotulo
    hover = (
        f"<b>{label} (footing)</b><br>"
        f"hx = {sapata.h_x:.3f} m<br>"
        f"hy = {sapata.h_y:.3f} m<br>"
        f"hz = {sapata.h_z:.3f} m<br>"
        f"V  = {sapata.volume:.3f} m³<br>"
        f"xg = {xc:.3f} m, yg = {yc:.3f} m"
    )
    return go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k,
        color=color or "#1f77b4",
        opacity=opacity,
        flatshading=True,
        name=label,
        showlegend=True,
        hovertemplate=hover + "<extra></extra>",
    )


def pillar_box(sapata: Sapata, *, height_m: float = DEFAULT_PILLAR_HEIGHT_M,
               color: str = "#7f7f7f", opacity: float = 0.85) -> go.Mesh3d:
    """Build a ``Mesh3d`` trace for the column above a footing.

    The column extrudes upward from ``z = 0`` to ``z = height_m``,
    centred on the pillar centroid with section ``ap × bp``.

    :param sapata: Domain entity (the pillar is read from ``sapata.pilar``)
    :param height_m: Visualisation-only column height [m]
    :param color: Plotly colour spec
    :param opacity: Alpha in ``[0, 1]``

    :return: ``plotly.graph_objects.Mesh3d``
    """
    pilar = sapata.pilar
    xc, yc = pilar.xg, pilar.yg
    zc = height_m / 2.0
    x, y, z, i, j, k = _box_mesh(xc, yc, zc, pilar.a_p, pilar.b_p, height_m)
    hover = (
        f"<b>{pilar.rotulo} (pillar)</b><br>"
        f"a_p = {pilar.a_p:.3f} m<br>"
        f"b_p = {pilar.b_p:.3f} m<br>"
        f"xg = {xc:.3f} m, yg = {yc:.3f} m"
    )
    return go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k,
        color=color, opacity=opacity, flatshading=True,
        name=f"{pilar.rotulo} (pillar)",
        showlegend=False,
        hovertemplate=hover + "<extra></extra>",
    )


# =============================================================================
# Main entry point
# =============================================================================
def _palette(n: int) -> list[str]:
    """Return a list of ``n`` distinct hex colours (Tab10 cycled)."""
    base = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    return [base[i % len(base)] for i in range(n)]


def _ground_plane(sapatas: Sequence[Sapata], *, margin: float = 1.0) -> go.Mesh3d:
    """Return a faint horizontal ground plane sized to enclose every footing.

    :param sapatas: Sequence of footings (used to compute extents)
    :param margin: Additional margin around the bounding box [m]

    :return: ``plotly.graph_objects.Mesh3d`` for the rectangle at ``z = 0``
    """
    xs, ys = [], []
    for s in sapatas:
        x_c, y_c = s.pilar.xg, s.pilar.yg
        xs += [x_c - s.h_x / 2 - margin, x_c + s.h_x / 2 + margin]
        ys += [y_c - s.h_y / 2 - margin, y_c + s.h_y / 2 + margin]
    x_min, x_max = float(min(xs)), float(max(xs))
    y_min, y_max = float(min(ys)), float(max(ys))
    x = np.array([x_min, x_max, x_max, x_min], dtype=float)
    y = np.array([y_min, y_min, y_max, y_max], dtype=float)
    z = np.zeros(4, dtype=float)
    i = np.array([0, 0], dtype=int)
    j = np.array([1, 2], dtype=int)
    k = np.array([2, 3], dtype=int)
    return go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k,
        color="#e9e9e9", opacity=0.35, flatshading=True,
        name="ground (z=0)", hoverinfo="skip", showlegend=True,
    )


def render_footings_3d(
    sapatas: Iterable[Sapata],
    *,
    show_pillars: bool = True,
    show_ground: bool = True,
    pillar_height_m: float = DEFAULT_PILLAR_HEIGHT_M,
    title: Optional[str] = None,
    colour_by: str = "label",
) -> go.Figure:
    """Build a 3D Plotly figure showing the optimised footings + pillars.

    Each footing is a translucent box buried below ``z = 0``; the
    column above is a thinner box rising to ``pillar_height_m``. A
    faint ground plane at ``z = 0`` indicates the soil-footing
    interface. The figure has equal aspect on the three axes and the
    camera starts from a moderately-elevated isometric view.

    Resumo em português:
        Constrói uma `plotly.graph_objects.Figure` 3D com cada sapata
        como caixa enterrada, cada pilar como caixa acima, plano de
        solo (z=0) opcional e proporção de eixos equal-aspect. Hover
        traz dimensões, volume e coordenadas; a legenda permite
        ligar/desligar cada elemento.

    :param sapatas: Sequence (or any iterable) of optimised
                    ``core.domain.Sapata`` instances
    :param show_pillars: When ``True`` (default), draw the columns
                         above the footings
    :param show_ground: When ``True`` (default), draw a faint ground
                        plane at ``z = 0``
    :param pillar_height_m: Visualisation-only column height [m]
    :param title: Optional figure title
    :param colour_by: ``"label"`` colours each footing differently;
                      ``"volume"`` interpolates a Viridis ramp from
                      smallest to largest concrete volume

    :return: ``plotly.graph_objects.Figure`` ready for
             ``st.plotly_chart(fig, use_container_width=True)``

    :raises ValueError: When ``sapatas`` is empty or ``colour_by`` is
                        unknown
    """
    sapatas = list(sapatas)
    if not sapatas:
        raise ValueError("render_footings_3d requires at least one Sapata.")
    if colour_by not in {"label", "volume"}:
        raise ValueError(
            f"unknown colour_by={colour_by!r}; expected 'label' or 'volume'."
        )

    # Colour assignment
    if colour_by == "label":
        colours = _palette(len(sapatas))
    else:
        volumes = np.array([s.volume for s in sapatas], dtype=float)
        v_min, v_max = float(volumes.min()), float(volumes.max())
        # Map to a 10-step Viridis palette
        viridis = ["#440154", "#482878", "#3e4989", "#31688e", "#26828e",
                   "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725"]
        if v_max - v_min < 1e-12:
            colours = [viridis[0]] * len(sapatas)
        else:
            indices = ((volumes - v_min) / (v_max - v_min) * (len(viridis) - 1)).round().astype(int)
            colours = [viridis[i] for i in indices]

    fig = go.Figure()
    if show_ground:
        fig.add_trace(_ground_plane(sapatas))
    for s, c in zip(sapatas, colours):
        fig.add_trace(footing_box(s, color=c))
        if show_pillars:
            fig.add_trace(pillar_box(s, height_m=pillar_height_m))

    # Equal-aspect 3D layout with informative axis titles
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="x [m]"),
            yaxis=dict(title="y [m]"),
            zaxis=dict(title="z [m]"),
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.1)),
        ),
        margin=dict(l=0, r=0, t=40 if title else 10, b=0),
        legend=dict(itemsizing="constant"),
    )
    return fig
