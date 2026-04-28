"""Core package of the FundaIA project — pure domain code.

This package hosts the architectural layers that are independent of any
presentation framework (Streamlit, CLI, notebooks). It is the long-term
home of the engineering, optimisation, I/O and API layers, and is being
populated incrementally during Sprint 3 of the refactor roadmap.

Camadas (referência rápida em português):
    * ``core.domain``       — business entities (Solo, Pilar, Sapata, ...).
    * ``core.engineering``  — pure analytical checks (NBR 6118 / 6122).
    * ``core.optimization`` — EGO/GA/GWO algorithms (will absorb metapy_toolbox).
    * ``core.io``           — Excel readers/writers and DXF export.
    * ``core.api``          — high-level functions used by the UI/CLI.

Migration policy (current sprint — 3.1):
    The package starts empty on purpose. No production code from
    ``fundacao.py`` or ``metapy_toolbox/`` has been moved yet. The
    intent is to ship the skeleton first, validate that the test suite
    stays green, and then migrate logic file by file in subsequent
    sprints — always preserving the regression baseline
    ``of = 19.70604234767181`` (see ``tests/test_avaliar_projeto.py``).
"""
