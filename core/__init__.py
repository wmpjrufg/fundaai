"""Core package of the FundaIA project — framework-free project code.

Hosts the architectural layers that are independent of any
presentation framework (Streamlit, CLI, notebooks). The full layout
materialised across Sprints 3.1 → 4.7 of the refactor roadmap.

Camadas:
    * ``core.domain``         — business entities (Solo, Pilar,
                                 Combinacao, Sapata, FundacaoProjeto).
    * ``core.engineering``    — pure analytical checks (NBR 6118 / 6122).
    * ``core.optimization``   — EGO / GA / GWO algorithms + benchmark
                                 functions + ``SurrogateCache`` (the
                                 retired ``metapy_toolbox`` was folded
                                 here in Sprint 3.6 and removed in 4.3).
    * ``core.io``             — Excel reader, DXF writer and the
                                 ``ExperimentRecorder`` /
                                 ``load_experiment`` persistence layer.
    * ``core.api``            — high-level functions consumed by the
                                 UI / CLI / notebooks (``optimize``,
                                 ``evaluate``, ``OptimisationConfig``).
    * ``core.observability``  — structured JSON logging primitives
                                 (``configure_logging``, ``run_context``,
                                 ``get_logger``) shared by every layer.

Dependency direction (enforced by convention, see ARCHITECTURE.md):
    domain  ←  engineering / optimization / io  ←  api  ←  frontend.

Acceptance criterion preserved across every commit:
    ``tests/test_avaliar_projeto.py::test_baseline_three_foundations_returns_19_706``
    must keep locking ``of = 19.70604234767181`` (rel=1e-12).
"""
