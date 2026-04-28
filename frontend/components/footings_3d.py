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
from frontend.theme.palette import PALETTE


__all__ = [
    "render_footings_3d",
    "footing_box",
    "pillar_box",
    "CAMERA_PRESETS",
    "DEFAULT_PILLAR_HEIGHT_M",
]


CAMERA_PRESETS: dict[str, dict] = {
    "isométrica": dict(eye=dict(x=1.6, y=-1.6, z=1.1),
                       up=dict(x=0, y=0, z=1)),
    "topo":       dict(eye=dict(x=0.0, y=0.0, z=2.5),
                       up=dict(x=0, y=1, z=0)),
    "lateral X":  dict(eye=dict(x=2.5, y=0.0, z=0.4),
                       up=dict(x=0, y=0, z=1)),
    "lateral Y":  dict(eye=dict(x=0.0, y=-2.5, z=0.4),
                       up=dict(x=0, y=0, z=1)),
    "perspectiva":dict(eye=dict(x=2.0, y=-2.0, z=0.7),
                       up=dict(x=0, y=0, z=1)),
}
"""Named camera presets exposed by the 3D viewer.

Each entry feeds directly into ``go.layout.scene.camera`` and pairs
with one of the buttons rendered in the Streamlit page.
"""


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


_LIGHTING = dict(
    ambient=0.55, diffuse=0.85, specular=0.25, roughness=0.55, fresnel=0.10,
)
_LIGHT_POSITION = dict(x=120, y=-120, z=160)


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
        lighting=_LIGHTING, lightposition=_LIGHT_POSITION,
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
        lighting=_LIGHTING, lightposition=_LIGHT_POSITION,
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


def _terrain_bounds(sapatas: Sequence[Sapata], margin: float) -> tuple[float, float, float, float]:
    """Return the (xmin, xmax, ymin, ymax) AABB of every footing plus margin."""
    xs: list[float] = []
    ys: list[float] = []
    for s in sapatas:
        x_c, y_c = s.pilar.xg, s.pilar.yg
        xs += [x_c - s.h_x / 2 - margin, x_c + s.h_x / 2 + margin]
        ys += [y_c - s.h_y / 2 - margin, y_c + s.h_y / 2 + margin]
    return float(min(xs)), float(max(xs)), float(min(ys)), float(max(ys))


def _ground_plane(
    sapatas: Sequence[Sapata], *, margin: float = 1.0,
) -> list[go.BaseTraceType]:
    """Return the traces composing the terrain ground plane.

    Composed of:

    * a translucent ``Mesh3d`` rectangle at ``z = 0`` matching the
      AABB of every footing plus ``margin``;
    * a denser grid of thin ``Scatter3d`` lines so the user sees
      proportions and distances at a glance;
    * a thicker contour outlining the lot bounds.

    :param sapatas: Sequence of footings (used to compute extents)
    :param margin: Additional margin around the bounding box [m]

    :return: List of Plotly traces ready to add to a Figure
    """
    x_min, x_max, y_min, y_max = _terrain_bounds(sapatas, margin)
    rect_x = np.array([x_min, x_max, x_max, x_min], dtype=float)
    rect_y = np.array([y_min, y_min, y_max, y_max], dtype=float)
    rect_z = np.zeros(4, dtype=float)
    rect = go.Mesh3d(
        x=rect_x, y=rect_y, z=rect_z,
        i=np.array([0, 0]), j=np.array([1, 2]), k=np.array([2, 3]),
        color=PALETTE["surface"], opacity=0.55, flatshading=True,
        lighting=dict(ambient=0.9), name="terreno", showlegend=True,
        hoverinfo="skip",
    )

    # Grid spacing: ~10 lines on the longer side
    longer = max(x_max - x_min, y_max - y_min)
    step = max(round(longer / 10.0, 1), 0.5)
    grid_x = np.arange(np.floor(x_min / step) * step, x_max + step, step)
    grid_y = np.arange(np.floor(y_min / step) * step, y_max + step, step)

    grid_xs: list[float] = []
    grid_ys: list[float] = []
    grid_zs: list[float] = []
    for gx in grid_x:
        grid_xs += [float(gx), float(gx), None]
        grid_ys += [y_min, y_max, None]
        grid_zs += [0.0, 0.0, None]
    for gy in grid_y:
        grid_xs += [x_min, x_max, None]
        grid_ys += [float(gy), float(gy), None]
        grid_zs += [0.0, 0.0, None]
    grid = go.Scatter3d(
        x=grid_xs, y=grid_ys, z=grid_zs,
        mode="lines",
        line=dict(color=PALETTE["border"], width=1),
        opacity=0.7,
        hoverinfo="skip", showlegend=False, name="grid",
    )

    # Lot contour
    contour = go.Scatter3d(
        x=[x_min, x_max, x_max, x_min, x_min],
        y=[y_min, y_min, y_max, y_max, y_min],
        z=[0, 0, 0, 0, 0],
        mode="lines",
        line=dict(color=PALETTE["accent"], width=4),
        name="contorno do terreno",
        hoverinfo="skip", showlegend=False,
    )
    return [rect, grid, contour]


def render_footings_3d(
    sapatas: Iterable[Sapata],
    *,
    show_pillars: bool = True,
    show_ground: bool = True,
    pillar_height_m: float = DEFAULT_PILLAR_HEIGHT_M,
    title: Optional[str] = None,
    colour_by: str = "label",
    camera: str | dict | None = None,
    terrain_margin_m: float = 1.0,
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
    :param camera: Either the string name of a preset declared in
                   :data:`CAMERA_PRESETS` (``"isométrica"``,
                   ``"topo"``, ``"lateral X"``, ``"lateral Y"``,
                   ``"perspectiva"``) or a raw camera dict; defaults
                   to the isometric preset when ``None``
    :param terrain_margin_m: Margin around the footings AABB used to
                             draw the terrain rectangle and grid [m]

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
        for trace in _ground_plane(sapatas, margin=terrain_margin_m):
            fig.add_trace(trace)
    for s, c in zip(sapatas, colours):
        fig.add_trace(footing_box(s, color=c))
        if show_pillars:
            fig.add_trace(pillar_box(s, height_m=pillar_height_m))

    # Resolve camera preset
    if camera is None:
        camera_dict = CAMERA_PRESETS["isométrica"]
    elif isinstance(camera, str):
        if camera not in CAMERA_PRESETS:
            raise ValueError(
                f"unknown camera preset {camera!r}; "
                f"expected one of {sorted(CAMERA_PRESETS)}."
            )
        camera_dict = CAMERA_PRESETS[camera]
    else:
        camera_dict = dict(camera)

    # Equal-aspect 3D layout with informative axis titles
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="x [m]"),
            yaxis=dict(title="y [m]"),
            zaxis=dict(title="z [m]"),
            aspectmode="data",
            camera=camera_dict,
        ),
        margin=dict(l=0, r=0, t=40 if title else 10, b=0),
        legend=dict(itemsizing="constant"),
    )
    return fig
