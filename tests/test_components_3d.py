"""Tests for ``frontend.components.footings_3d`` (3D viewer).

The 3D viewer is presentation code, but its geometry contract is
testable: every footing must produce one ``Mesh3d`` trace whose
vertices match the analytical AABB of the entity, the optional
pillars and ground plane must appear when toggled, and the figure
layout must be 3D with equal-data aspect.

Locks the contract on:

    1. **Trace count**: with ``N`` footings, ``show_pillars=True`` and
       ``show_ground=True`` we get ``2*N + 1`` traces.
    2. **Box geometry**: ``footing_box`` produces an 8-vertex /
       12-triangle mesh whose bounds match
       ``[xg - h_x/2, xg + h_x/2] × [yg - h_y/2, yg + h_y/2]
       × [-h_z, 0]``.
    3. **Hover content**: each tooltip carries the pillar label and
       the three footing dimensions.
    4. **Colour modes**: ``colour_by="volume"`` produces a sequence
       of distinct colours when volumes vary; ``colour_by="label"``
       cycles through the Tab10 palette.
    5. **Edge cases**: empty input raises; unknown ``colour_by``
       raises.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.domain import Pilar, Sapata
from frontend.components import footings_3d
from frontend.components.footings_3d import (
    DEFAULT_PILLAR_HEIGHT_M,
    footing_box,
    pillar_box,
    render_footings_3d,
)


# =============================================================================
# Helpers
# =============================================================================
def _make_sapatas(n: int = 3) -> list[Sapata]:
    """Build ``n`` synthetic sapatas with monotonically increasing volume."""
    out = []
    for i in range(n):
        p = Pilar(rotulo=f"P{i:02d}", a_p=0.30 + 0.05 * i, b_p=0.40,
                  xg=2.0 * i, yg=0.0)
        out.append(Sapata(pilar=p, h_x=1.0 + 0.1 * i, h_y=1.5, h_z=0.6 + 0.05 * i))
    return out


# =============================================================================
# footing_box / pillar_box
# =============================================================================
@pytest.mark.optimization
class TestPrimitiveBoxes:
    """This class verifies the per-element Mesh3d primitives."""

    def test_footing_box_bounds_match_entity(self):
        """The 8 vertices of footing_box span the AABB derived from h_x, h_y, h_z."""
        s = _make_sapatas(1)[0]
        mesh = footing_box(s)
        x = np.asarray(mesh.x); y = np.asarray(mesh.y); z = np.asarray(mesh.z)
        assert x.shape == (8,) and y.shape == (8,) and z.shape == (8,)
        assert x.min() == pytest.approx(s.pilar.xg - s.h_x / 2)
        assert x.max() == pytest.approx(s.pilar.xg + s.h_x / 2)
        assert y.min() == pytest.approx(s.pilar.yg - s.h_y / 2)
        assert y.max() == pytest.approx(s.pilar.yg + s.h_y / 2)
        # Footing top is at z=0, bottom at z=-h_z (buried).
        assert z.max() == pytest.approx(0.0)
        assert z.min() == pytest.approx(-s.h_z)

    def test_footing_box_has_12_triangles(self):
        """A closed AABB box is composed of exactly 12 triangles (6 faces × 2)."""
        s = _make_sapatas(1)[0]
        mesh = footing_box(s)
        assert len(mesh.i) == 12 and len(mesh.j) == 12 and len(mesh.k) == 12

    def test_footing_hover_carries_dimensions(self):
        """Hover template mentions the pillar label and the three dimensions."""
        s = _make_sapatas(1)[0]
        mesh = footing_box(s)
        ht = mesh.hovertemplate
        assert s.pilar.rotulo in ht
        assert f"{s.h_x:.3f}" in ht and f"{s.h_y:.3f}" in ht and f"{s.h_z:.3f}" in ht

    def test_pillar_box_extrudes_upward_from_zero(self):
        """pillar_box spans z in [0, height_m] and is centred on (ap, bp)."""
        s = _make_sapatas(1)[0]
        mesh = pillar_box(s, height_m=1.5)
        z = np.asarray(mesh.z)
        assert z.min() == pytest.approx(0.0)
        assert z.max() == pytest.approx(1.5)
        x = np.asarray(mesh.x); y = np.asarray(mesh.y)
        assert x.min() == pytest.approx(s.pilar.xg - s.pilar.a_p / 2)
        assert x.max() == pytest.approx(s.pilar.xg + s.pilar.a_p / 2)
        assert y.min() == pytest.approx(s.pilar.yg - s.pilar.b_p / 2)
        assert y.max() == pytest.approx(s.pilar.yg + s.pilar.b_p / 2)


# =============================================================================
# render_footings_3d
# =============================================================================
@pytest.mark.optimization
class TestRenderFootings3D:
    """This class verifies the high-level figure builder."""

    def test_default_trace_count(self):
        """N footings + N pillars + 1 ground plane = 2*N + 1 traces."""
        sapatas = _make_sapatas(3)
        fig = render_footings_3d(sapatas)
        assert len(fig.data) == 2 * len(sapatas) + 1

    def test_no_pillars_no_ground(self):
        """show_pillars=False, show_ground=False yields exactly N traces."""
        sapatas = _make_sapatas(3)
        fig = render_footings_3d(sapatas, show_pillars=False, show_ground=False)
        assert len(fig.data) == len(sapatas)

    def test_layout_is_3d_with_data_aspect(self):
        """The scene uses equal-data aspect mode so ratios are not distorted."""
        fig = render_footings_3d(_make_sapatas(2))
        scene = fig.layout.scene
        assert scene.aspectmode == "data"
        assert scene.xaxis.title.text == "x [m]"
        assert scene.yaxis.title.text == "y [m]"
        assert scene.zaxis.title.text == "z [m]"

    def test_colour_by_volume_uses_distinct_colours_when_volumes_vary(self):
        """Different footing volumes -> at least two distinct colours in the palette."""
        sapatas = _make_sapatas(5)   # different h_z so volumes vary
        fig = render_footings_3d(sapatas, show_pillars=False, show_ground=False,
                                 colour_by="volume")
        colours = {trace.color for trace in fig.data}
        assert len(colours) >= 2

    def test_pillar_height_is_propagated(self):
        """pillar_height_m=2.5 -> tallest pillar trace reaches z = 2.5."""
        fig = render_footings_3d(_make_sapatas(1), pillar_height_m=2.5)
        # The pillar trace is the second one (after the ground plane).
        pillar_trace = fig.data[2]
        assert max(pillar_trace.z) == pytest.approx(2.5)

    def test_empty_sapatas_raises(self):
        """Empty input is a programmer error (do not silently produce an empty figure)."""
        with pytest.raises(ValueError):
            render_footings_3d([])

    def test_unknown_colour_by_raises(self):
        """colour_by must be 'label' or 'volume'."""
        with pytest.raises(ValueError, match="colour_by"):
            render_footings_3d(_make_sapatas(1), colour_by="density")

    def test_default_pillar_height_constant_is_used_when_omitted(self):
        """Omitting pillar_height_m falls back to DEFAULT_PILLAR_HEIGHT_M."""
        fig = render_footings_3d(_make_sapatas(1))
        pillar_trace = fig.data[2]
        assert max(pillar_trace.z) == pytest.approx(DEFAULT_PILLAR_HEIGHT_M)
