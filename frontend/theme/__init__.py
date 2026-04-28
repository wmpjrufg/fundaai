"""Visual theme — paleta, CSS e template Plotly compartilhados.

This package centralises the visual identity of the FundaIA front
end: a dark-mode colour palette (mirrored from
``.streamlit/config.toml``), a Plotly template that aligns chart
backgrounds and gridlines with the Streamlit canvas, and an
``apply_theme()`` helper that injects a small CSS layer for the
final touches that ``config.toml`` cannot reach (rounded cards,
softer dividers, accent on focus rings, monospaced run-id chips).

Resumo em português:
    Tema visual unificado. Espelha a paleta do dark theme
    declarado em ``.streamlit/config.toml``, oferece um template
    Plotly coerente (fundo, grade, fonte) e um ``apply_theme()``
    que injeta um CSS pequeno para acabamentos que o tema
    nativo do Streamlit não cobre.
"""

from .palette import PALETTE, plotly_template
from .css import apply_theme

__all__ = [
    "PALETTE",
    "apply_theme",
    "plotly_template",
]
