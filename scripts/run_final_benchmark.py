"""Final experimental protocol for the IC manuscript (artigo 1).

Runs the frozen head-to-head comparison between EGO+GPR and the pure
baselines (GA, PSO, GWO, uniform random search) on the three canonical
case studies, under two budget scenarios:

* **S1 — equal budget**: every algorithm spends exactly the same number
  of real objective evaluations per repetition (``BUDGET_S1``). This is
  the sample-efficiency question: *who finds the best design with few
  real evaluations?*
* **S2 — extended budget**: the pure baselines run with ``BUDGET_S2``
  real evaluations (cheap vectorised FO makes this take < 1 s), while
  EGO keeps the S1 result as its reference. This is the wall-clock
  question: *when the FO is cheap, what do the baselines deliver in a
  fraction of EGO's wall time?*

Every repetition is seeded (``base_seed + rep``) and the whole run is
bit-reproducible given the same library versions. Results are persisted
under ``experiments/protocolo_final/<case>/<scenario>/`` as:

    history.parquet   one row per real evaluation
    per_rep.csv       final outcome per (algorithm, repetition)
    summary.csv       aggregated statistics per algorithm
    pvalues.csv       pairwise Wilcoxon-Holm p-values on paired per-rep best
    config.json       BenchmarkConfig round-trip
    meta.json         environment snapshot (git rev, versions, timing)

Usage (from the repository root):

    .venv/bin/python scripts/run_final_benchmark.py

Resumo em português:
    Protocolo experimental final do artigo 1. Compara EGO+GPR contra
    GA/PSO/GWO/busca aleatória nos casos de 1, 2 e 3 sapatas, com 30
    repetições semeadas e dois cenários de orçamento (igual e
    estendido). Persiste histórico, sumário, p-valores e metadados de
    ambiente por caso/cenário.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

# Allow running as `python scripts/run_final_benchmark.py` from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.api import BenchmarkConfig, run_benchmark  # noqa: E402
from core.io import read_projeto_from_excel  # noqa: E402

# =============================================================================
# Frozen protocol parameters (do not edit between partial runs)
# =============================================================================
OUT_ROOT = REPO_ROOT / "experiments" / "protocolo_final"

CASES: dict[str, str] = {
    "caso1_um":   "assets/data/problema_fund_um.xlsx",    # 1 fund, dim 3
    "caso2_dois": "assets/data/problema_fund_dois.xlsx",  # 2 fund, dim 6
    "caso3_tres": "assets/data/problema_fund_três.xlsx",  # 3 fund, dim 9
}

F_CK_KPA = 25_000.0       # C25 [kPa]
COBRIMENTO_M = 0.04       # [m]
# Bounds follow the manuscript configuration (secao 6): h in [0.60, 3.00] m.
# 3.00 m is required for feasibility of caso1 (P08 has a_p = 2.10 m, so the
# geometric constraint needs h_x >= 2.30 m — impossible under h_max = 1.50).
H_MIN_M, H_MAX_M = 0.60, 3.00
N_REP = 30                # seeds base_seed .. base_seed + 29
BASE_SEED = 42
BUDGET_S1 = 150           # real evaluations per rep — every algorithm
BUDGET_S2 = 3_000         # real evaluations per rep — pure baselines only
LHS_FACTOR = 10           # EGO initial sample = 10 * dim (Jones et al. 1998)
META_POP_SIZE = 30        # GA / PSO / GWO population
GA_POP_SIZE = 50          # inner-EI GA (surrogate only) — Sprint 4.12 default
GA_EPOCH = 30             # inner-EI GA epochs — Sprint 4.12 default
KERNEL_INDEX = -1         # production kernel: Matern(nu=2.5)


def _env_snapshot() -> dict:
    """Capture the reproducibility metadata of this run.

    :return: Mapping with python/platform/library versions and git state
    """
    def _git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True,
                text=True, check=True,
            ).stdout.strip()
        except Exception:
            return "unknown"

    libs = {}
    for pkg in ("numpy", "pandas", "scipy", "scikit-learn", "mealpy", "pydantic"):
        try:
            libs[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            libs[pkg] = "absent"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "libs": libs,
        "git_rev": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def _persist(result, out_dir: Path, wall_s: float) -> None:
    """Write every artefact of one (case, scenario) run.

    :param result: BenchmarkResult returned by run_benchmark
    :param out_dir: Destination folder (created if missing)
    :param wall_s: Total wall time of the run [s]
    :return: None
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    result.history.to_parquet(out_dir / "history.parquet", index=False)
    result.per_rep.to_csv(out_dir / "per_rep.csv", index=False)
    result.summary.to_csv(out_dir / "summary.csv", index=False)
    result.pvalues.to_csv(out_dir / "pvalues.csv")
    (out_dir / "config.json").write_text(
        result.config.model_dump_json(indent=2), encoding="utf-8"
    )
    meta = _env_snapshot()
    meta["wall_time_total_s"] = wall_s
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    """Run the two scenarios on the three frozen cases and persist everything.

    :return: None
    """
    t_protocol = time.perf_counter()
    for case_name, rel_path in CASES.items():
        projeto = read_projeto_from_excel(
            REPO_ROOT / rel_path, f_ck_kpa=F_CK_KPA, cobrimento_m=COBRIMENTO_M
        )
        dim = 3 * projeto.n_fund
        lhs = LHS_FACTOR * dim

        # ------------------------------------------------------------------
        # S1 — equal budget (sample efficiency)
        # ------------------------------------------------------------------
        cfg_s1 = BenchmarkConfig(
            algorithms=("ego", "ga", "pso", "gwo", "random"),
            budget_evals=BUDGET_S1,
            ego_budget_evals=BUDGET_S1,
            n_rep=N_REP,
            base_seed=BASE_SEED,
            h_min_m=H_MIN_M,
            h_max_m=H_MAX_M,
            lhs_n_pop=lhs,
            meta_pop_size=META_POP_SIZE,
            kernel_index=KERNEL_INDEX,
            ga_pop_size=GA_POP_SIZE,
            ga_epoch=GA_EPOCH,
        )
        print(f"[{case_name}] S1 equal-budget: {BUDGET_S1} evals × {N_REP} reps "
              f"(dim={dim}, lhs={lhs}) ...", flush=True)
        t0 = time.perf_counter()
        res_s1 = run_benchmark(projeto, cfg_s1, progress=_progress)
        wall = time.perf_counter() - t0
        _persist(res_s1, OUT_ROOT / case_name / "S1_orcamento_igual", wall)
        print(f"[{case_name}] S1 done in {wall/60:.1f} min", flush=True)

        # ------------------------------------------------------------------
        # S2 — extended budget for the cheap baselines (wall-clock regime)
        # ------------------------------------------------------------------
        cfg_s2 = BenchmarkConfig(
            algorithms=("ga", "pso", "gwo", "random"),
            budget_evals=BUDGET_S2,
            ego_budget_evals=BUDGET_S1,   # unused (no EGO here); kept for record
            n_rep=N_REP,
            base_seed=BASE_SEED,
            h_min_m=H_MIN_M,
            h_max_m=H_MAX_M,
            lhs_n_pop=lhs,
            meta_pop_size=META_POP_SIZE,
            kernel_index=KERNEL_INDEX,
            ga_pop_size=GA_POP_SIZE,
            ga_epoch=GA_EPOCH,
        )
        print(f"[{case_name}] S2 extended-budget: {BUDGET_S2} evals × {N_REP} reps ...",
              flush=True)
        t0 = time.perf_counter()
        res_s2 = run_benchmark(projeto, cfg_s2, progress=_progress)
        wall = time.perf_counter() - t0
        _persist(res_s2, OUT_ROOT / case_name / "S2_orcamento_estendido", wall)
        print(f"[{case_name}] S2 done in {wall/60:.1f} min", flush=True)

    print(f"PROTOCOL COMPLETE in {(time.perf_counter() - t_protocol)/60:.1f} min",
          flush=True)


def _progress(ev: dict) -> None:
    """Print a compact heartbeat so the background log shows liveness.

    :param ev: Progress payload emitted by run_benchmark
    :return: None
    """
    if ev.get("event") == "benchmark.rep_end":
        print(f"    {ev['algorithm']:>6} rep {ev['rep'] + 1:>2} "
              f"best={ev['best']:.4f} evals={ev['n_evals']} "
              f"t={ev['wall_time_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
