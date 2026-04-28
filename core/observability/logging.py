"""Structured logging built on the Python stdlib ``logging`` module.

Design goals:

1. **Silent by default**: importing the FundaIA codebase does not
   reconfigure the global ``logging`` setup. Callers opt in via
   ``configure_logging(...)`` once at the start of a script or test.
2. **One event per line**: every log record becomes a JSON object on
   a single line. Easy to grep, easy to ship, easy to load back as
   a parquet/JSONL frame for analysis.
3. **Stable event names**: producers attach a ``event`` key (e.g.
   ``"ego.iter"``, ``"cache.hit"``) so downstream consumers can
   filter without parsing free-form messages.
4. **Run-scoped context**: ``run_context(run_id)`` injects the
   current run identifier into every record emitted from the
   block, mirroring the ``ExperimentRecorder.run_id``.

Resumo em português:
    Logging estruturado em JSON por linha, opcional, com contexto de
    run via ``contextvars``. Não substitui o ``ExperimentRecorder`` —
    complementa: o recorder grava arquivos consolidados ao final, o
    logger emite eventos em tempo real durante a execução.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional


__all__ = [
    "JsonFormatter",
    "configure_logging",
    "get_logger",
    "run_context",
    "DEFAULT_NAMESPACE",
]


DEFAULT_NAMESPACE = "fundaia"
"""Top-level logger namespace.

All FundaIA loggers descend from ``fundaia`` (e.g. ``fundaia.optimize``,
``fundaia.ego``). Configuring a single handler on this namespace
captures every event the codebase produces.
"""


# Run-scoped identifier: ``run_context`` sets it for the duration of a
# ``with`` block; the formatter reads it for every record so the
# producer does not need to thread the id through every call site.
_RUN_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "fundaia_run_id", default=None
)


# Standard LogRecord attributes that are not "extra" payload and must
# not be serialised twice.
_RECORD_RESERVED = {
    "args", "asctime", "created", "exc_info", "exc_text", "filename",
    "funcName", "levelname", "levelno", "lineno", "message", "module",
    "msecs", "msg", "name", "pathname", "process", "processName",
    "relativeCreated", "stack_info", "thread", "threadName", "taskName",
}


class JsonFormatter(logging.Formatter):
    """Format LogRecord objects as one-line JSON documents.

    Output keys (always present):

    * ``ts``     — ISO 8601 UTC timestamp of the record.
    * ``level``  — logger level name (``INFO``, ``WARNING``, ...).
    * ``logger`` — logger name (``fundaia.optimize``, ...).
    * ``msg``    — formatted message string.
    * ``run_id`` — current ``run_context`` id, or ``None``.

    Any non-reserved attribute attached via ``extra={...}`` is copied
    verbatim, so producers can write::

        logger.info("ego iteration", extra={"event": "ego.iter",
                                            "rep": 0, "iter": 4,
                                            "of_min": 19.706})
    """

    def format(self, record: logging.LogRecord) -> str:
        """Serialise ``record`` as a single-line JSON string.

        :param record: stdlib LogRecord produced by a logger call

        :return: One-line JSON document terminated by no newline
        """
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "run_id": _RUN_ID.get(),
        }
        for key, value in record.__dict__.items():
            if key in _RECORD_RESERVED or key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def get_logger(name: str = DEFAULT_NAMESPACE) -> logging.Logger:
    """Return a namespaced logger under ``fundaia``.

    Producers should call this once at module load with a stable
    sub-namespace (``"fundaia.optimize"``, ``"fundaia.ego"``,
    ``"fundaia.cache"``, ``"fundaia.experiments"``).

    :param name: Full logger name. Defaults to ``"fundaia"``

    :return: stdlib ``logging.Logger`` instance
    """
    if not name.startswith(DEFAULT_NAMESPACE):
        # Make sure all FundaIA loggers descend from a single root so a
        # single ``configure_logging`` call captures everything.
        name = f"{DEFAULT_NAMESPACE}.{name}"
    return logging.getLogger(name)


def configure_logging(
    level: int | str = logging.INFO,
    *,
    stream: Any = sys.stderr,
    log_file: Optional[Path | str] = None,
    json: bool = True,
    propagate: bool = False,
) -> logging.Logger:
    """Configure the FundaIA logger root once and return it.

    Idempotent: calling it twice replaces the previous handlers (no
    duplicates, no leaks). Tests can call it freely.

    :param level: Threshold for the FundaIA root logger
    :param stream: File-like object for the stream handler. Pass
                   ``None`` to disable the stream handler entirely
    :param log_file: Optional path for an additional file handler.
                     Parent directories are created on demand
    :param json: When ``True`` (default), use :class:`JsonFormatter`;
                 when ``False``, use a plain ``%(asctime)s …`` format
                 — useful for ad-hoc humans reading the console
    :param propagate: Whether records should bubble up to the stdlib
                      root logger. Default ``False`` so configuring
                      pytest's caplog or a parent application does
                      not double-print

    :return: Configured ``fundaia`` root logger
    """
    root = logging.getLogger(DEFAULT_NAMESPACE)
    # Drop any previous FundaIA handlers so re-config is idempotent.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level)
    root.propagate = propagate

    formatter: logging.Formatter
    if json:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s :: %(message)s"
        )

    if stream is not None:
        sh = logging.StreamHandler(stream)
        sh.setFormatter(formatter)
        root.addHandler(sh)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(formatter)
        root.addHandler(fh)

    return root


@contextmanager
def run_context(run_id: Optional[str]) -> Iterator[None]:
    """Tag every log record emitted inside the block with ``run_id``.

    Implemented with :class:`contextvars.ContextVar`, so it is safe
    under ``asyncio`` and threads. Nested calls restore the outer
    value cleanly.

    :param run_id: Identifier to attach (typically the
                   ``ExperimentRecorder.run_id``). ``None`` clears
                   the context for the block

    :return: Yields nothing; the side effect is the context var
    """
    token = _RUN_ID.set(run_id)
    try:
        yield
    finally:
        _RUN_ID.reset(token)
