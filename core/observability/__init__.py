"""Observability layer — structured logging primitives.

Cross-cutting concern that does not belong inside any of the existing
core sub-packages: every layer (engineering, optimization, io, api)
needs to emit progress and diagnostic events without coupling to a
concrete sink. The ``logging`` module here builds on the Python stdlib
``logging`` package and provides:

* ``configure_logging(...)``     — opt-in single setup. The default is
                                    "do nothing": loggers are silent
                                    until configured, so library users
                                    are never surprised by stderr noise.
* ``JsonFormatter``               — emits one JSON object per record,
                                    suitable for live ingestion (jq,
                                    journald, structured log shippers).
* ``get_logger(name)``            — namespaced logger; conventional
                                    namespaces are ``fundaia.optimize``,
                                    ``fundaia.ego``, ``fundaia.cache``,
                                    ``fundaia.experiments``.
* ``run_context(run_id)``         — context manager that tags every
                                    log record emitted inside the
                                    block with ``run_id``. Used by the
                                    ``ExperimentRecorder`` so a
                                    persisted run is searchable in the
                                    log stream by id.

Resumo em português:
    Camada de observabilidade. Configura o logging stdlib para emitir
    JSON por linha, com contexto por run. Cada camada (api,
    optimization, io) emite eventos com nomes estáveis (e.g.
    ``ego.iter``, ``cache.hit``, ``experiment.record_rep``) que podem
    ser filtrados em tempo real ou pós-processados em parquet.
"""

from .logging import (
    DEFAULT_NAMESPACE,
    JsonFormatter,
    configure_logging,
    get_logger,
    run_context,
)

__all__ = [
    "DEFAULT_NAMESPACE",
    "JsonFormatter",
    "configure_logging",
    "get_logger",
    "run_context",
]
