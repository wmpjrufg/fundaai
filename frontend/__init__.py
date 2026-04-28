"""Frontend layer — Streamlit pages, reusable components and i18n.

This package is the only place that touches Streamlit, plotting widgets
and locale dictionaries. Domain, engineering, optimisation, IO and API
layers stay framework-free under ``core/``. The ``app.py`` entry point
loads the pages declared in ``frontend.pages``; reusable widgets
(3D viewer, EGO best-so-far curve, GPR diagnostics) belong in
``frontend.components``; localisation strings belong in ``frontend.i18n``.

Resumo em português:
    Camada de apresentação. Único lugar que importa Streamlit ou
    plotting libraries. ``frontend.pages`` agrupa as páginas
    (``home``, ``sapatas``, futuras), ``frontend.components`` os
    widgets reutilizáveis e ``frontend.i18n`` os dicionários PT/EN.
"""
