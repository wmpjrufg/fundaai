"""API layer — high-level entry points consumed by UI, CLI and notebooks.

This subpackage will expose stable, framework-free functions that
orchestrate domain, engineering, optimisation and I/O layers. The
flagship function ``optimize(project, config) -> OptimisationResult``
will replace the inline orchestration currently embedded in
``pages/sapatas.py``.

Resumo em português:
    Camada de API. Funções de alto nível (sem dependência de
    Streamlit/CLI) que orquestram domínio, engenharia, otimização e
    I/O. ``optimize(project, config)`` será a porta de entrada
    principal, substituindo a lógica embutida em ``pages/sapatas.py``.
"""
