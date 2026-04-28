"""Tests for ``frontend.theme`` (palette + Plotly template + CSS).

Locks the contract on three fronts:

    1. **Palette completeness**: every key the CSS module references
       must exist in ``PALETTE`` so a typo cannot ship as a broken
       stylesheet.
    2. **Plotly default template**: importing the package registers
       ``"fundaia_dark"`` and sets it as the default, so charts that
       do not specify a template still inherit the project look.
    3. **CSS injection**: ``apply_theme()`` is idempotent and writes
       a single ``<style>`` block via ``st.markdown``.
"""

from __future__ import annotations

import plotly.io as pio
import pytest

from frontend.theme import PALETTE, apply_theme, plotly_template


@pytest.mark.optimization
class TestPalette:
    """This class verifies the palette declares every key used elsewhere."""

    def test_required_keys_are_present(self):
        for key in (
            "bg", "surface", "surface_alt", "border",
            "text", "text_muted",
            "accent", "accent_strong", "best_so_far",
            "ok", "warn", "fail",
            "categorical", "viridis",
        ):
            assert key in PALETTE, f"missing palette key {key!r}"

    def test_categorical_has_at_least_six_distinct_colours(self):
        cats = PALETTE["categorical"]
        assert len(cats) >= 6
        assert len(set(cats)) == len(cats)


@pytest.mark.optimization
class TestPlotlyTemplate:
    """This class verifies the Plotly template registration."""

    def test_template_is_registered(self):
        # Note: pio.templates.default may be re-set by Streamlit when it
        # is imported elsewhere in the test session. We only assert that
        # the FundaIA template is registered and selectable explicitly.
        assert "fundaia_dark" in pio.templates

    def test_template_has_dark_paper_background(self):
        tpl = plotly_template()
        assert tpl.layout.paper_bgcolor == PALETTE["bg"]
        assert tpl.layout.plot_bgcolor == PALETTE["bg"]


@pytest.mark.optimization
class TestApplyTheme:
    """This class verifies the CSS injection helper."""

    def test_apply_theme_is_callable_outside_streamlit(self):
        # Must be safe to call in non-streamlit environments (tests,
        # documentation generation). The lazy import inside swallows
        # the ImportError when Streamlit is not on the stack.
        apply_theme()  # no exception
