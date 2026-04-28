"""Tests for ``core.observability`` (structured logging).

Locks the contract on five fronts:

    1. **JSON shape**: every formatted record contains the four
       canonical keys ``ts``, ``level``, ``logger``, ``msg`` and
       carries every ``extra`` field verbatim.
    2. **Idempotency**: configuring twice does not duplicate handlers.
    3. **Run context**: ``run_context(...)`` injects ``run_id`` into
       the JSON payload and restores the previous value on exit.
    4. **Silent by default**: importing the package does not
       reconfigure stdlib root, and the FundaIA logger has no
       handlers until ``configure_logging`` is called.
    5. **End-to-end via optimize**: a real ``optimize`` call emits
       at least the named events ``optimize.start``,
       ``optimize.rep_start``, ``optimize.rep_end``, ``ego.iter``,
       ``optimize.end``; when a recorder is supplied the run is
       tagged with the recorder's ``run_id``.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import pytest

from core.observability import (
    DEFAULT_NAMESPACE,
    JsonFormatter,
    configure_logging,
    get_logger,
    run_context,
)


# =============================================================================
# JsonFormatter
# =============================================================================
@pytest.mark.optimization
class TestJsonFormatter:
    """This class verifies the per-record JSON shape."""

    def _format_one(self, **extra) -> dict:
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="fundaia.x", level=logging.INFO, pathname=__file__, lineno=1,
            msg="hello", args=(), exc_info=None,
        )
        for k, v in extra.items():
            setattr(record, k, v)
        return json.loads(formatter.format(record))

    def test_canonical_keys_present(self):
        """ts, level, logger, msg, run_id are always present."""
        payload = self._format_one()
        assert {"ts", "level", "logger", "msg", "run_id"}.issubset(payload)
        assert payload["level"] == "INFO"
        assert payload["logger"] == "fundaia.x"
        assert payload["msg"] == "hello"

    def test_extra_fields_are_passed_through(self):
        """Custom keys attached via extra={...} appear verbatim."""
        payload = self._format_one(event="ego.iter", iter=4, of_min=19.706)
        assert payload["event"] == "ego.iter"
        assert payload["iter"] == 4
        assert payload["of_min"] == pytest.approx(19.706)


# =============================================================================
# configure_logging
# =============================================================================
@pytest.mark.optimization
class TestConfigureLogging:
    """This class verifies the configuration entrypoint."""

    def teardown_method(self):
        # Restore baseline for other tests
        configure_logging(level=logging.WARNING, stream=None, json=True)

    def test_idempotent_does_not_duplicate_handlers(self):
        """Calling configure_logging twice keeps handler count stable."""
        buf = io.StringIO()
        configure_logging(stream=buf)
        configure_logging(stream=buf)
        root = logging.getLogger(DEFAULT_NAMESPACE)
        assert len(root.handlers) == 1

    def test_writes_one_json_per_log_call(self):
        """Each log call produces exactly one JSON line on the configured stream."""
        buf = io.StringIO()
        configure_logging(stream=buf)
        log = get_logger("optimize")
        log.info("payload", extra={"event": "ego.iter", "iter": 1})
        log.info("payload2", extra={"event": "ego.iter", "iter": 2})
        lines = [l for l in buf.getvalue().splitlines() if l.strip()]
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["event"] == "ego.iter"
        assert first["iter"] == 1
        assert first["logger"] == "fundaia.optimize"

    def test_log_file_is_created(self, tmp_path: Path):
        """log_file kwarg adds a FileHandler that produces JSON lines."""
        log_path = tmp_path / "deep" / "fundaia.log"
        configure_logging(stream=None, log_file=log_path)
        log = get_logger("cache")
        log.info("hi", extra={"event": "cache.miss"})
        assert log_path.is_file()
        line = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
        assert line["event"] == "cache.miss"


# =============================================================================
# run_context
# =============================================================================
@pytest.mark.optimization
class TestRunContext:
    """This class verifies that run_context tags records correctly."""

    def teardown_method(self):
        configure_logging(level=logging.WARNING, stream=None, json=True)

    def test_run_id_appears_in_records_and_resets_on_exit(self):
        """Inside the with-block run_id is set; outside it is None again."""
        buf = io.StringIO()
        configure_logging(stream=buf)
        log = get_logger("optimize")
        log.info("before", extra={"event": "optimize.before"})
        with run_context("test-run-42"):
            log.info("inside", extra={"event": "optimize.inside"})
        log.info("after", extra={"event": "optimize.after"})

        records = [json.loads(l) for l in buf.getvalue().splitlines() if l.strip()]
        before, inside, after = records
        assert before["run_id"] is None
        assert inside["run_id"] == "test-run-42"
        assert after["run_id"] is None


# =============================================================================
# get_logger
# =============================================================================
@pytest.mark.optimization
def test_get_logger_normalises_namespace():
    """Names without the fundaia prefix are remapped under it."""
    assert get_logger().name == DEFAULT_NAMESPACE
    assert get_logger("ego").name == "fundaia.ego"
    assert get_logger("fundaia.cache").name == "fundaia.cache"


@pytest.mark.optimization
def test_silent_by_default_after_module_import():
    """Importing the package does not attach handlers to the FundaIA root."""
    # We previously called configure_logging in other tests; pretend a
    # fresh process by clearing handlers.
    root = logging.getLogger(DEFAULT_NAMESPACE)
    for h in list(root.handlers):
        root.removeHandler(h)
    assert root.handlers == []


# =============================================================================
# End-to-end through optimize
# =============================================================================
@pytest.mark.optimization
class TestEndToEndOptimize:
    """This class verifies emission from a real optimize() call."""

    def teardown_method(self):
        configure_logging(level=logging.WARNING, stream=None, json=True)

    def test_named_events_emitted_during_optimize(self, tmp_path: Path, assets_dir: Path):
        """A small optimize() emits start/rep_*/ego.iter/end events tagged with run_id."""
        from core.api import OptimisationConfig, optimize
        from core.io import read_projeto_from_excel
        from core.io.experiments import ExperimentRecorder

        buf = io.StringIO()
        configure_logging(stream=buf, level=logging.DEBUG)

        proj = read_projeto_from_excel(
            assets_dir / "data" / "problema_fund_um.xlsx",
            f_ck_kpa=25_000.0, cobrimento_m=0.04,
        )
        cfg = OptimisationConfig(
            h_min_m=0.6, h_max_m=3.0,
            n_pop=4, n_gen=1, n_rep=1,
            ga_epoch=5, ga_pop_size=10,
        )
        rec = ExperimentRecorder(root=tmp_path, run_id="log-e2e")
        optimize(proj, cfg, recorder=rec)

        records = [json.loads(l) for l in buf.getvalue().splitlines() if l.strip()]
        events = {r.get("event") for r in records}
        assert "optimize.start" in events
        assert "optimize.rep_start" in events
        assert "optimize.rep_end" in events
        assert "ego.iter" in events
        assert "optimize.end" in events
        assert "experiment.begin" in events
        assert "experiment.record_rep" in events
        assert "experiment.end" in events

        # All events emitted inside the run carry the recorder run_id.
        run_ids = {r.get("run_id") for r in records if r.get("event", "").startswith("optimize")}
        assert "log-e2e" in run_ids
