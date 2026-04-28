"""CSS overrides injected on every Streamlit page.

Streamlit's native dark theme (declared in ``.streamlit/config.toml``)
covers most of what the FundaIA UI needs. The CSS below polishes the
last 10%: rounded corners on cards/tabs, softer dividers, accent on
focus rings, monospaced run-id chips and a denser sidebar header.

The function ``apply_theme()`` is idempotent — calling it multiple
times in the same session is harmless. Pages should call it as the
first Streamlit operation after ``st.set_page_config`` (or right at
import, before any widget renders).
"""

from __future__ import annotations

from .palette import PALETTE


_CSS = f"""
<style>
/* --- Layout ---------------------------------------------------------- */
.stApp {{
  background:
    radial-gradient(1200px 600px at 100% -10%,
                    rgba(245,158,11,0.08), transparent 70%),
    radial-gradient(900px 450px at -10% 110%,
                    rgba(96,165,250,0.10), transparent 70%),
    {PALETTE["bg"]};
}}

[data-testid="stSidebar"] {{
  background: {PALETTE["surface"]};
  border-right: 1px solid {PALETTE["border"]};
}}

/* Tighter top padding so the page does not feel empty */
.block-container {{
  padding-top: 2.0rem;
  padding-bottom: 2.5rem;
}}

/* --- Headings -------------------------------------------------------- */
h1, h2, h3, h4 {{
  letter-spacing: -0.01em;
}}
h1 {{
  font-weight: 700;
}}

/* --- Cards / containers ---------------------------------------------- */
[data-testid="stMetric"], [data-testid="stExpander"] {{
  background: {PALETTE["surface"]};
  border: 1px solid {PALETTE["border"]};
  border-radius: 12px;
}}
[data-testid="stExpander"] details summary {{
  font-weight: 600;
}}

/* --- Tabs ------------------------------------------------------------- */
.stTabs [data-baseweb="tab-list"] {{
  gap: 4px;
  border-bottom: 1px solid {PALETTE["border"]};
}}
.stTabs [data-baseweb="tab"] {{
  background: transparent;
  border-radius: 10px 10px 0 0;
  padding: 8px 16px;
  color: {PALETTE["text_muted"]};
}}
.stTabs [aria-selected="true"] {{
  background: {PALETTE["surface"]};
  color: {PALETTE["text"]};
  border-bottom: 2px solid {PALETTE["accent"]};
}}

/* --- Buttons --------------------------------------------------------- */
.stButton > button, .stDownloadButton > button {{
  border-radius: 10px;
  border: 1px solid {PALETTE["border"]};
  background: {PALETTE["surface"]};
  color: {PALETTE["text"]};
  font-weight: 600;
  transition: all 160ms ease-in-out;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
  border-color: {PALETTE["accent"]};
  color: {PALETTE["accent_strong"]};
  transform: translateY(-1px);
}}
.stButton > button[kind="primary"] {{
  background: {PALETTE["accent"]};
  color: #1a1206;
  border-color: {PALETTE["accent"]};
}}
.stButton > button[kind="primary"]:hover {{
  background: {PALETTE["accent_strong"]};
}}

/* --- Inputs / focus -------------------------------------------------- */
input:focus, textarea:focus, [data-baseweb="select"] > div:focus-within {{
  outline: 2px solid {PALETTE["accent"]} !important;
  outline-offset: 1px;
}}

/* --- Sliders --------------------------------------------------------- */
[data-baseweb="slider"] [role="slider"] {{
  border: 2px solid {PALETTE["accent"]} !important;
}}

/* --- DataFrames ------------------------------------------------------ */
[data-testid="stDataFrame"] {{
  border: 1px solid {PALETTE["border"]};
  border-radius: 10px;
  overflow: hidden;
}}

/* --- Code / chips ---------------------------------------------------- */
.fundaia-chip {{
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  background: {PALETTE["surface"]};
  border: 1px solid {PALETTE["border"]};
  color: {PALETTE["text_muted"]};
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
  letter-spacing: 0.02em;
}}
.fundaia-chip--ok    {{ color: {PALETTE["ok"]};   border-color: {PALETTE["ok"]}; }}
.fundaia-chip--warn  {{ color: {PALETTE["warn"]}; border-color: {PALETTE["warn"]}; }}
.fundaia-chip--fail  {{ color: {PALETTE["fail"]}; border-color: {PALETTE["fail"]}; }}
.fundaia-chip--accent{{ color: {PALETTE["accent_strong"]}; border-color: {PALETTE["accent"]}; }}

/* --- Plotly modebar (subtle) ----------------------------------------- */
.modebar {{
  background: rgba(17,26,46,0.6) !important;
  border: 1px solid {PALETTE["border"]} !important;
  border-radius: 6px !important;
}}
.modebar-btn:hover svg {{
  fill: {PALETTE["accent_strong"]} !important;
}}
</style>
"""


def apply_theme() -> None:
    """Inject the FundaIA CSS overrides into the current Streamlit page.

    Imports Streamlit lazily so this module remains importable in
    pure-Python contexts (tests, notebooks, static analysis). Safe to
    call repeatedly: Streamlit deduplicates identical ``st.markdown``
    calls within the same render cycle.

    :return: Nothing (side effect: ``st.markdown`` injection)
    """
    try:
        import streamlit as st
    except Exception:   # pragma: no cover  (only when streamlit absent)
        return
    st.markdown(_CSS, unsafe_allow_html=True)
