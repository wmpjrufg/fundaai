"""Experiment persistence — record, reload and aggregate optimisation runs.

This module turns each ``optimize`` call into a self-describing folder
under ``experiments/<run_id>/`` containing the input configuration, a
fingerprint of the project, the runtime environment, the full per-rep
EGO history (one parquet file per repetition), a ergonomic CSV summary
and a paper-grade ``metrics.json`` aggregate. The data layout is
deliberately stable and versioned so the artifacts produced today can
be loaded by tomorrow's plotting notebooks and by the eventual paper
companion repository.

The recorder is **opt-in**: ``optimize`` only writes to disk when an
``ExperimentRecorder`` is supplied; the regression baseline path
(``cache=None`` and ``recorder=None``) is byte-for-byte unchanged.

Folder layout::

    experiments/<run_id>/
      manifest.json     # schema_version, status, timestamps, counts
      config.json       # OptimisationConfig.model_dump (round-trippable)
      env.json          # python, packages, OS, git rev/branch/dirty
      project.json      # FundacaoProjeto fingerprint + summary
      summary.csv       # one row per repetition (rep_id, seed, of_best, ...)
      metrics.json      # aggregated paper-grade metrics
      history/
        rep_000.parquet # full ego_01_architecture history per rep
        rep_001.parquet
        ...
      artifacts/        # optional binary blobs (DXF, plots, ...)

Resumo em português:
    Camada de persistência de experimentos. Cada chamada a ``optimize``
    com um ``ExperimentRecorder`` produz uma pasta autodescritiva em
    ``experiments/<run_id>/`` com configuração, ambiente, fingerprint
    do projeto, histórico completo do EGO por repetição (Parquet),
    resumo em CSV e métricas agregadas em JSON. Pensado para alimentar
    diretamente os plots e tabelas do artigo da IC.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from core.domain import FundacaoProjeto
from core.observability import get_logger

if TYPE_CHECKING:  # imported for type hints only — breaks the
                   # core.io <-> core.api circular import cycle.
    from core.api.types import OptimisationConfig

# numpy>=2.0 renamed np.trapz to np.trapezoid; keep working under both.
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")

_log = get_logger("experiments")


__all__ = [
    "SCHEMA_VERSION",
    "ExperimentManifest",
    "ExperimentRecorder",
    "ExperimentRun",
    "compute_metrics",
    "load_experiment",
    "summarise_history",
]


SCHEMA_VERSION = "1.0"
"""Schema version for experiment folders.

Increment when the on-disk layout changes in a way that breaks
backward compatibility. ``load_experiment`` reads this field from
``manifest.json`` and rejects unsupported versions explicitly.
"""


# =============================================================================
# Dataclasses
# =============================================================================
@dataclass(frozen=True, slots=True)
class ExperimentManifest:
    """This class holds the top-level metadata of one persisted run.

    Fields mirror ``manifest.json`` on disk one-to-one. Optional fields
    (``completed_at``, ``metrics``, ``summary``) are populated only
    after ``ExperimentRecorder.end`` is called.

    :param schema_version: Layout version; matches :data:`SCHEMA_VERSION`
    :param run_id: Unique identifier (timestamp + short uuid)
    :param created_at: ISO 8601 UTC timestamp set at ``begin``
    :param completed_at: ISO 8601 UTC timestamp set at ``end`` (or ``None`` while running)
    :param status: ``"running"``, ``"completed"`` or ``"failed"``
    :param config: Round-trippable Pydantic dump of ``OptimisationConfig``
    :param env: Snapshot of the runtime environment (versions, OS, git)
    :param project: Fingerprint of the input ``FundacaoProjeto``
    :param metrics: Aggregated paper-grade metrics (``None`` while running)
    :param summary: Per-rep summary rows (``None`` while running)
    :param error: Optional traceback / message when ``status == "failed"``
    """

    schema_version: str
    run_id: str
    created_at: str
    completed_at: Optional[str]
    status: str
    config: dict
    env: dict
    project: dict
    metrics: Optional[dict] = None
    summary: Optional[list[dict]] = None
    error: Optional[str] = None


@dataclass(frozen=True, slots=True)
class ExperimentRun:
    """This class is the in-memory view of a persisted run loaded from disk.

    Returned by :func:`load_experiment`. ``history`` keeps each
    repetition as its own DataFrame, in registration order, so plotting
    code can iterate ``run.history`` directly.

    :param manifest: ``ExperimentManifest`` reconstructed from ``manifest.json``
    :param history: Mapping ``rep_id -> EGO history DataFrame``
    :param run_dir: Absolute path to the original folder
    """

    manifest: ExperimentManifest
    history: dict[int, pd.DataFrame]
    run_dir: Path


# =============================================================================
# Environment / project / config capture
# =============================================================================
def _git_value(args: list[str]) -> Optional[str]:
    """This helper invokes ``git`` quietly and returns stripped stdout or None.

    :param args: Argv tail passed to ``git`` (e.g. ``["rev-parse", "HEAD"]``)

    :return: Output string, or ``None`` if git is unavailable / fails
    """
    try:
        out = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _capture_env() -> dict:
    """This helper captures the runtime environment at the moment of recording.

    Returns a JSON-serialisable dict with Python version, OS string,
    pinned versions of the libraries that materially affect numerical
    results, and a best-effort git snapshot. Fields that fail to be
    captured (e.g. git not available) become ``None``.

    :return: Dictionary describing the runtime environment
    """
    pkgs: dict[str, Optional[str]] = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "mealpy",
                 "pydantic", "joblib", "pyarrow", "ezdxf"):
        try:
            mod = __import__(name.replace("-", "_"))
            pkgs[name] = getattr(mod, "__version__", None)
        except Exception:
            pkgs[name] = None

    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": pkgs,
        "git": {
            "rev": _git_value(["rev-parse", "HEAD"]),
            "branch": _git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty": (_git_value(["status", "--porcelain"]) or "") != "",
        },
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }


def _project_fingerprint(projeto: FundacaoProjeto) -> dict:
    """This helper computes a stable fingerprint of a ``FundacaoProjeto``.

    The fingerprint combines a SHA-256 hash over the canonical JSON
    serialisation of the project's per-element data with a small set of
    human-readable summary scalars (``n_fund``, ``n_comb``, ``f_ck``,
    ``cobrimento``). The hash lets two runs assert they were optimised
    against the same input even if the file path differs; the summary
    keeps the manifest readable without loading the full project.

    :param projeto: Validated input project

    :return: Dictionary with ``hash`` (sha256 hex) and human-readable summary
    """
    payload = {
        "n_fund": projeto.n_fund,
        "n_comb": projeto.n_comb,
        "f_ck_kpa": projeto.f_ck_kpa,
        "cobrimento_m": projeto.cobrimento_m,
        "pilares": [dataclasses.asdict(p) for p in projeto.pilares],
        "solo_por_pilar": {
            rot: dataclasses.asdict(solo)
            for rot, solo in projeto.solo_por_pilar.items()
        },
        "combinacoes_por_pilar": {
            rot: [dataclasses.asdict(c) for c in combs]
            for rot, combs in projeto.combinacoes_por_pilar.items()
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "hash": digest,
        "n_fund": projeto.n_fund,
        "n_comb": projeto.n_comb,
        "f_ck_kpa": projeto.f_ck_kpa,
        "cobrimento_m": projeto.cobrimento_m,
        "pilar_labels": [p.rotulo for p in projeto.pilares],
    }


# =============================================================================
# Metrics
# =============================================================================
def summarise_history(history: pd.DataFrame) -> dict:
    """This function returns one paper-grade row from a single EGO history.

    Computes:

    * ``of_initial``         — best objective from the LHS initial population
                                (rows with ``ITER == 0``).
    * ``of_best``             — best objective achieved at any point.
    * ``best_iter``           — first ``ITER`` where ``of_best`` was hit.
    * ``improvement_abs``     — ``of_initial - of_best`` (always non-negative).
    * ``improvement_rel``     — improvement normalised by ``|of_initial|``.
    * ``convergence_iter``    — first iter at which the running best
                                reached within ``1e-6`` (relative) of the
                                final ``of_best``. Always between 0 and
                                ``n_gen``.
    * ``convergence_ratio``   — ``convergence_iter / n_gen``.
    * ``auc_best_so_far``     — area under the normalised best-so-far
                                curve, in [0, 1]; lower means faster
                                convergence.
    * ``n_evals_total``       — ``len(history)`` (initial pop + n_gen).
    * ``n_unique_x``          — number of distinct design vectors visited.
    * ``t_total_s``           — sum of ``TIME CONSUMPTION`` across rows
                                (when available).
    * ``mean_t_per_iter_s``   — mean of ``TIME CONSUMPTION`` for ``ITER > 0``.

    Resumo em português:
        Resume um histórico EGO em uma linha de métricas. Calcula
        melhor OF inicial e final, ganho absoluto e relativo, iteração
        de convergência (primeira em que o melhor-até-aqui chegou no
        valor final), AUC da curva best-so-far e tempos totais.

    :param history: DataFrame as produced by ``ego_01_architecture``
                    (must contain columns ``ITER``, ``OF`` and the
                    ``X_*`` design columns; ``TIME CONSUMPTION`` is
                    optional)

    :return: Dictionary of paper-grade metrics
    """
    if history.empty:
        raise ValueError("Cannot summarise an empty history.")

    of = history["OF"].to_numpy(dtype=float)
    iters = history["ITER"].to_numpy(dtype=int)
    of_best = float(np.min(of))
    best_idx = int(np.argmin(of))
    best_iter = int(iters[best_idx])

    initial_mask = iters == 0
    of_initial = float(np.min(of[initial_mask])) if initial_mask.any() else float("nan")
    improvement_abs = of_initial - of_best
    denom = abs(of_initial) if of_initial != 0 else 1.0
    improvement_rel = improvement_abs / denom

    # Best-so-far curve along iteration order (after initial pop).
    of_iter_only = of[~initial_mask]
    n_gen = int(iters.max()) if len(iters) else 0
    best_so_far = np.minimum.accumulate(of_iter_only) if of_iter_only.size else np.array([])

    convergence_iter: Optional[int] = None
    if best_so_far.size:
        tol = 1e-6 * (abs(of_best) if of_best != 0 else 1.0)
        for k, v in enumerate(best_so_far, start=1):
            if abs(v - of_best) <= tol:
                convergence_iter = int(k)
                break

    # AUC of normalised best-so-far curve in [0, 1] over iteration index.
    auc = None
    if best_so_far.size and of_initial != of_best:
        norm = (best_so_far - of_best) / (of_initial - of_best)
        norm = np.clip(norm, 0.0, 1.0)
        auc = float(_trapezoid(norm) / max(len(norm) - 1, 1)) if len(norm) > 1 else float(norm[0])

    # Time accounting (best-effort: column name varies across versions).
    t_total = None
    t_mean_iter = None
    time_col = next(
        (c for c in ("TIME CONSUMPTION (s)", "TIME CONSUMPTION") if c in history.columns),
        None,
    )
    if time_col is not None:
        times = history[time_col].to_numpy(dtype=float)
        t_total = float(np.nansum(times))
        iter_times = times[~initial_mask]
        if iter_times.size:
            t_mean_iter = float(np.nanmean(iter_times))

    # Unique design vectors visited.
    x_cols = [c for c in history.columns if c.startswith("X_")]
    if x_cols:
        n_unique_x = int(history[x_cols].drop_duplicates().shape[0])
    else:
        n_unique_x = int(len(history))

    return {
        "of_initial": of_initial,
        "of_best": of_best,
        "best_iter": best_iter,
        "improvement_abs": float(improvement_abs),
        "improvement_rel": float(improvement_rel),
        "convergence_iter": convergence_iter,
        "convergence_ratio": (convergence_iter / n_gen) if (convergence_iter is not None and n_gen) else None,
        "auc_best_so_far": auc,
        "n_evals_total": int(len(history)),
        "n_unique_x": n_unique_x,
        "n_gen": n_gen,
        "t_total_s": t_total,
        "mean_t_per_iter_s": t_mean_iter,
    }


def compute_metrics(per_rep_summary: Iterable[Mapping[str, Any]]) -> dict:
    """This function aggregates per-repetition rows into paper-grade metrics.

    Produces a flat dictionary with the best/mean/std/median of the
    objective across repetitions, the mean convergence iter and the
    total elapsed time. Rows missing a field contribute ``nan``.

    :param per_rep_summary: Iterable of per-rep dicts as produced by
                            :func:`summarise_history`, each augmented
                            with ``rep_id``, ``seed`` and ``wall_time_s``

    :return: Aggregated metrics dictionary
    """
    rows = list(per_rep_summary)
    if not rows:
        raise ValueError("compute_metrics requires at least one row.")

    def _col(name: str) -> np.ndarray:
        return np.array([r.get(name, np.nan) for r in rows], dtype=float)

    def _safe_nanmean(arr: np.ndarray) -> Optional[float]:
        """Return nanmean(arr) or None when every entry is NaN."""
        if arr.size == 0 or np.all(np.isnan(arr)):
            return None
        return float(np.nanmean(arr))

    of_best = _col("of_best")
    return {
        "n_rep": len(rows),
        "best_of": float(np.nanmin(of_best)),
        "worst_of": float(np.nanmax(of_best)),
        "mean_of": float(np.nanmean(of_best)),
        "std_of": float(np.nanstd(of_best, ddof=0)),
        "median_of": float(np.nanmedian(of_best)),
        "best_rep_id": int(rows[int(np.nanargmin(of_best))]["rep_id"]),
        "mean_convergence_iter": _safe_nanmean(_col("convergence_iter")),
        "mean_auc_best_so_far": _safe_nanmean(_col("auc_best_so_far")),
        "mean_improvement_rel": _safe_nanmean(_col("improvement_rel")),
        "mean_t_total_s": _safe_nanmean(_col("t_total_s")),
        "wall_time_total_s": float(np.nansum(_col("wall_time_s"))),
    }


# =============================================================================
# Recorder
# =============================================================================
def _new_run_id() -> str:
    """This helper builds a sortable run identifier (timestamp + short uuid).

    :return: String like ``"20260428T193245Z-a1b2c3d4"``
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}-{uuid.uuid4().hex[:8]}"


def _atomic_write_json(path: Path, payload: Any) -> None:
    """This helper writes JSON to a temp file and renames into place atomically.

    Prevents readers from observing a half-written manifest. Uses the
    same directory as the target so the rename is on the same
    filesystem (POSIX-atomic).

    :param path: Destination path
    :param payload: JSON-serialisable object

    :return: Nothing (writes to disk)
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


class ExperimentRecorder:
    """Record one optimisation run as a self-describing folder on disk.

    Lifecycle::

        recorder = ExperimentRecorder(root="experiments")
        recorder.begin(config, projeto)
        for rep in range(config.n_rep):
            history = ego_01_architecture(...)
            recorder.record_rep(rep_id=rep, seed=..., history=history,
                                wall_time_s=...)
        recorder.write_artifact("best_design.dxf", dxf_bytes)   # optional
        recorder.end()

    On any exception during ``begin`` -> ``end``, call ``cancel(error)``
    to mark the manifest as ``failed``. The recorder keeps every write
    atomic so a crash mid-run still leaves a readable folder.

    :ivar run_dir: Absolute path of this run's folder
    :ivar manifest_path: Convenience pointer to ``run_dir / 'manifest.json'``
    """

    def __init__(self, root: Path | str, run_id: Optional[str] = None) -> None:
        """This constructor allocates a fresh run folder under ``root``.

        :param root: Parent directory; created if it does not exist
        :param run_id: Optional explicit identifier (mostly useful for tests).
                       When ``None`` a timestamped+uuid id is generated

        :return: Nothing (instance attributes initialised)
        """
        self._root = Path(root)
        self._run_id = run_id or _new_run_id()
        self._run_dir = self._root / self._run_id
        (self._run_dir / "history").mkdir(parents=True, exist_ok=True)
        (self._run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        self._summary_rows: list[dict] = []
        self._summary_per_rep: list[dict] = []
        self._created_at: Optional[str] = None
        self._config_dump: Optional[dict] = None
        self._env: Optional[dict] = None
        self._project: Optional[dict] = None

    # ------------------------------------------------------------------ paths
    @property
    def run_dir(self) -> Path:
        """This property returns the run folder absolute path."""
        return self._run_dir

    @property
    def run_id(self) -> str:
        """This property returns the run identifier."""
        return self._run_id

    @property
    def manifest_path(self) -> Path:
        """This property returns the manifest absolute path."""
        return self._run_dir / "manifest.json"

    # ------------------------------------------------------------------ begin
    def begin(self, config: "OptimisationConfig", projeto: FundacaoProjeto) -> None:
        """This method writes the initial manifest, config, env and project files.

        Must be called exactly once before any ``record_rep`` /
        ``write_artifact`` / ``end`` call. The manifest is written with
        ``status = "running"`` so an interrupted run is observable
        from disk.

        :param config: Validated ``OptimisationConfig`` driving the run
        :param projeto: Validated ``FundacaoProjeto`` being optimised

        :return: Nothing (mutates the run folder)
        """
        self._created_at = datetime.now(timezone.utc).isoformat()
        self._config_dump = config.model_dump()
        self._env = _capture_env()
        self._project = _project_fingerprint(projeto)

        _atomic_write_json(self._run_dir / "config.json", self._config_dump)
        _atomic_write_json(self._run_dir / "env.json", self._env)
        _atomic_write_json(self._run_dir / "project.json", self._project)
        self._flush_manifest(status="running")
        _log.info("experiment begin",
                  extra={"event": "experiment.begin", "run_id": self._run_id,
                         "run_dir": str(self._run_dir)})

    # ------------------------------------------------------------ record_rep
    def record_rep(
        self,
        rep_id: int,
        seed: int,
        history: pd.DataFrame,
        wall_time_s: float,
    ) -> dict:
        """This method persists one EGO repetition to disk and updates the summary.

        Writes ``history/rep_<rep_id:03d>.parquet`` and appends a row to
        the in-memory summary. The same row is returned for the caller's
        convenience (e.g. logging).

        :param rep_id: Zero-based repetition index
        :param seed: Seed used by this repetition (``base_seed + rep``)
        :param history: Full DataFrame returned by ``ego_01_architecture``
        :param wall_time_s: Wall-clock seconds spent on this repetition

        :return: The summary row dict that was appended
        """
        if self._created_at is None:
            raise RuntimeError("ExperimentRecorder.record_rep called before begin().")

        # Persist history as Parquet (typed, compressed).
        target = self._run_dir / "history" / f"rep_{rep_id:03d}.parquet"
        history.to_parquet(target, index=False, engine="pyarrow")

        per_rep = summarise_history(history)
        row = {"rep_id": int(rep_id), "seed": int(seed), "wall_time_s": float(wall_time_s),
               **per_rep}
        self._summary_rows.append(row)
        self._summary_per_rep.append(row)
        # Keep summary.csv up to date so a crash mid-run leaves partial data.
        pd.DataFrame(self._summary_rows).to_csv(self._run_dir / "summary.csv", index=False)
        self._flush_manifest(status="running")
        _log.info("rep recorded",
                  extra={"event": "experiment.record_rep",
                         "rep_id": int(rep_id), "seed": int(seed),
                         "of_best": per_rep.get("of_best"),
                         "wall_time_s": float(wall_time_s)})
        return row

    # ----------------------------------------------------------- artifacts
    def write_artifact(self, name: str, data: bytes) -> Path:
        """This method writes a binary artifact under ``artifacts/<name>``.

        :param name: Relative file name (no directory traversal)
        :param data: Raw bytes payload

        :return: Absolute path of the written artifact

        :raises ValueError: When ``name`` contains path separators
        """
        if "/" in name or "\\" in name or ".." in name:
            raise ValueError(f"Artifact name must be a plain file name; got {name!r}.")
        target = self._run_dir / "artifacts" / name
        target.write_bytes(data)
        return target

    # -------------------------------------------------------------------- end
    def end(self, status: str = "completed") -> ExperimentManifest:
        """This method writes the final manifest with aggregated metrics.

        Marks the run as ``completed`` (or any explicit final status)
        and writes ``metrics.json`` and ``summary.csv`` one last time.

        :param status: Final status to record (typically ``"completed"``)

        :return: The fully populated ``ExperimentManifest`` written to disk
        """
        if self._created_at is None:
            raise RuntimeError("ExperimentRecorder.end called before begin().")
        metrics = compute_metrics(self._summary_per_rep) if self._summary_per_rep else None
        if metrics is not None:
            _atomic_write_json(self._run_dir / "metrics.json", metrics)
        _log.info("experiment end",
                  extra={"event": "experiment.end", "run_id": self._run_id,
                         "status": status,
                         "best_of": (metrics or {}).get("best_of"),
                         "n_rep": (metrics or {}).get("n_rep")})
        return self._flush_manifest(status=status,
                                    completed_at=datetime.now(timezone.utc).isoformat(),
                                    metrics=metrics)

    def cancel(self, error: str) -> ExperimentManifest:
        """This method marks the run as ``failed`` and stores the error message.

        :param error: Short human-readable description (e.g. exception repr)

        :return: The final manifest written to disk
        """
        _log.error("experiment cancel",
                   extra={"event": "experiment.cancel",
                          "run_id": self._run_id, "error": error})
        return self._flush_manifest(status="failed",
                                    completed_at=datetime.now(timezone.utc).isoformat(),
                                    error=error)

    # ----------------------------------------------------------- internals
    def _flush_manifest(
        self,
        *,
        status: str,
        completed_at: Optional[str] = None,
        metrics: Optional[dict] = None,
        error: Optional[str] = None,
    ) -> ExperimentManifest:
        """This helper writes the current manifest snapshot atomically.

        :return: The ``ExperimentManifest`` instance corresponding to disk
        """
        manifest = ExperimentManifest(
            schema_version=SCHEMA_VERSION,
            run_id=self._run_id,
            created_at=self._created_at or datetime.now(timezone.utc).isoformat(),
            completed_at=completed_at,
            status=status,
            config=self._config_dump or {},
            env=self._env or {},
            project=self._project or {},
            metrics=metrics,
            summary=self._summary_rows or None,
            error=error,
        )
        _atomic_write_json(self.manifest_path, dataclasses.asdict(manifest))
        return manifest


# =============================================================================
# Loading
# =============================================================================
def load_experiment(run_dir: Path | str) -> ExperimentRun:
    """This function reads a persisted run folder back into memory.

    Verifies the schema version and rebuilds an :class:`ExperimentRun`
    with the manifest and every per-rep history DataFrame loaded from
    Parquet. Useful for paper plots, diff between runs and CI checks.

    :param run_dir: Path of the folder produced by ``ExperimentRecorder``

    :return: ``ExperimentRun`` with manifest + history dict + run_dir

    :raises FileNotFoundError: When ``manifest.json`` is missing
    :raises ValueError: When the on-disk schema version is unsupported
    """
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found under {run_dir}.")

    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = raw.get("schema_version")
    if schema != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported experiment schema_version={schema!r} "
            f"(this loader expects {SCHEMA_VERSION!r})."
        )
    manifest = ExperimentManifest(**raw)

    history: dict[int, pd.DataFrame] = {}
    history_dir = run_dir / "history"
    if history_dir.exists():
        for parquet_path in sorted(history_dir.glob("rep_*.parquet")):
            rep_id = int(parquet_path.stem.split("_")[1])
            history[rep_id] = pd.read_parquet(parquet_path, engine="pyarrow")

    return ExperimentRun(manifest=manifest, history=history, run_dir=run_dir)
