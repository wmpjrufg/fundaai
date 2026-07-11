"""Build every figure and table of the IC manuscript from the frozen runs.

Consumes the artefacts persisted by ``scripts/run_final_benchmark.py``
(``experiments/protocolo_final/``) and ``scripts/run_gpr_kernel_study.py``
(``experiments/estudo_gpr/``) and regenerates, deterministically:

Figures (PDF + PNG) → ``docs/artigo_ic_lucas/figuras/``

    fig_convergencia_s1      median best-so-far vs real evaluations (3 cases)
    fig_dist_best_s1         final-best distribution per algorithm (box + strip)
    fig_s1_vs_s2             equal vs extended budget (dumbbell) + EGO reference
    fig_gpr_obs_pred         observed vs predicted, best kernel, alpha 10 vs 1e6
    fig_gpr_kernels_r2       R-squared per kernel and penalty (dot + range)
    fig_violacoes_s1         max constraint violation of final designs (symlog)

Tables (LaTeX fragments) → ``docs/artigo_ic_lucas/tabelas/``
CSV mirrors → ``assets/tables/protocolo_final/``

    tab_casos.tex            frozen case studies (pillars, loads, soil)
    tab_protocolo.tex        frozen protocol parameters
    tab_s1.tex               S1 statistics per case and algorithm
    tab_s2.tex               S2 statistics + EGO-150 reference
    tab_pvalues_s1.tex       Mann-Whitney p-value matrices (S1)
    tab_gpr_kernels.tex      GPR metrics per kernel and penalty (top + production)

Visual identity: validated 5-slot categorical palette (see the dataviz
skill's reference palette) with one fixed hue per algorithm; hairline
grid, muted axis ink, direct labels where they fit, single y-axis per
panel, white (paper) surface.

Usage (from the repository root, after both runs complete):

    .venv/bin/python scripts/make_paper_artifacts.py

Resumo em português:
    Gera todas as figuras (PDF/PNG) e tabelas (fragmentos LaTeX + CSV)
    do artigo a partir dos resultados congelados do protocolo final e
    do estudo de kernels, com identidade visual validada e formatação
    pt-BR (vírgula decimal).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# =============================================================================
# Inputs / outputs
# =============================================================================
PROTO_DIR = REPO_ROOT / "experiments" / "protocolo_final"
GPR_DIR = REPO_ROOT / "experiments" / "estudo_gpr"
FIG_DIR = REPO_ROOT / "docs" / "artigo_ic_lucas" / "figuras"
TAB_DIR = REPO_ROOT / "docs" / "artigo_ic_lucas" / "tabelas"
CSV_DIR = REPO_ROOT / "assets" / "tables" / "protocolo_final"

CASES: dict[str, dict] = {
    "caso1_um":   {"titulo": "Caso 1 — 1 sapata (dim. 3)",  "curto": "Caso 1"},
    "caso2_dois": {"titulo": "Caso 2 — 2 sapatas (dim. 6)", "curto": "Caso 2"},
    "caso3_tres": {"titulo": "Caso 3 — 3 sapatas (dim. 9)", "curto": "Caso 3"},
}
S1, S2 = "S1_orcamento_igual", "S2_orcamento_estendido"

# Fixed identity → hue map (validated categorical palette, light mode,
# white surface; aqua/yellow carry the relief rule → direct labels + tables).
ALG_COLOR = {
    "ego":    "#2a78d6",   # slot 1 — blue
    "ga":     "#1baf7a",   # slot 2 — aqua
    "pso":    "#eda100",   # slot 3 — yellow
    "gwo":    "#008300",   # slot 4 — green
    "random": "#4a3aa7",   # slot 5 — violet
}
ALG_ORDER = ["ego", "ga", "pso", "gwo", "random"]
ALG_LABEL = {
    "ego": "EGO+GPR", "ga": "GA", "pso": "PSO",
    "gwo": "GWO", "random": "Aleatória",
}
PENALTY_COLOR = {10.0: "#2a78d6", 1e6: "#e34948"}   # blue vs red (slot 6)

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

KERNEL_SHORT = {
    "k00": "RBF", "k01": "RBF+RBF", "k02": r"RBF$\times$RBF",
    "k03": r"Matérn $\nu{=}0{,}5$", "k04": r"Matérn $\nu{=}1{,}5$",
    "k05": r"Matérn $\nu{=}2{,}5$", "k06": r"Matérn $1{,}5{+}2{,}5$",
    "k07": r"RQ $\alpha{=}1$", "k08": r"RQ $\alpha{=}0{,}1$",
    "k09": r"RQ $\alpha{=}10$", "k10": "DP+RBF", "k11": r"DP+Matérn $1{,}5$",
    "k12": r"DP$_{0{,}1}$+RBF", "k13": "DP (linear)", "k14": "ExpSine",
    "k15": r"RBF$\times$ExpSine", "k16": r"Matérn$\times$ExpSine",
    "k17": "RBF+White", "k18": r"Matérn $2{,}5$+White", "k19": "RQ+White",
    "k20": r"Matérn $\nu{=}2{,}5$ (produção)",
}


def _style() -> None:
    """Apply the manuscript-wide matplotlib style (print, white surface).

    :return: None
    """
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300,
        "font.family": "sans-serif", "font.size": 8.5,
        "axes.titlesize": 9, "axes.labelsize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "text.color": INK, "axes.labelcolor": INK_2,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def _fmt_br(x: float, nd: int = 2) -> str:
    """Format a number with Brazilian comma decimals for LaTeX tables.

    :param x: Value to format
    :param nd: Number of decimal places
    :return: Formatted string (``--`` for NaN)
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:,.{nd}f}".replace(",", "@").replace(".", ",").replace("@", ".")


def _fmt_pvalue(p: float) -> str:
    """Format a p-value for the manuscript (bold when < 0.05).

    :param p: Two-sided p-value
    :return: LaTeX-formatted string
    """
    if np.isnan(p):
        return "--"
    txt = r"$<$0,001" if p < 0.001 else _fmt_br(p, 3)
    return rf"\textbf{{{txt}}}" if p < 0.05 else txt


def _save(fig, name: str) -> None:
    """Persist a figure as PDF (vector, for LaTeX) and PNG (for review).

    :param fig: Matplotlib figure
    :param name: Base file name without extension
    :return: None
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  figura: {name}.pdf/.png")


def _load(case: str, scenario: str) -> dict:
    """Load the persisted artefacts of one (case, scenario) run.

    :param case: Case folder name
    :param scenario: Scenario folder name
    :return: Mapping with history / per_rep / summary / pvalues frames
    """
    base = PROTO_DIR / case / scenario
    return {
        "history": pd.read_parquet(base / "history.parquet"),
        "per_rep": pd.read_csv(base / "per_rep.csv"),
        "summary": pd.read_csv(base / "summary.csv"),
        "pvalues": pd.read_csv(base / "pvalues.csv", index_col=0),
    }



# =============================================================================
# Best feasible designs (deterministic reproduction for the layout figure)
# =============================================================================
def _best_feasible_designs() -> dict:
    """Reproduce the best strictly feasible EGO design of each case.

    The per-rep table stores which repetition produced the best feasible
    design (min ``volume_m3`` among ``feasible`` rows) but not the design
    vector itself. Because every repetition is fully seeded, re-running
    that single repetition reproduces the trajectory bit-for-bit; the
    decoded ``best_sapatas`` are cached in ``best_designs.json`` so the
    (~2 min) reproduction happens only once.

    :return: Mapping ``case -> list of dicts`` with pillar + footing geometry
    """
    import json
    cache = PROTO_DIR / "best_designs.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    from core.api import BenchmarkConfig, run_benchmark
    from core.io import read_projeto_from_excel

    spreadsheets = {
        "caso1_um": "problema_fund_um.xlsx",
        "caso2_dois": "problema_fund_dois.xlsx",
        "caso3_tres": "problema_fund_três.xlsx",
    }
    designs: dict[str, list[dict]] = {}
    for case, fname in spreadsheets.items():
        per_rep = pd.read_csv(PROTO_DIR / case / S1 / "per_rep.csv")
        feas = per_rep[(per_rep["algorithm"] == "ego") & per_rep["feasible"]]
        row = feas.loc[feas["volume_m3"].idxmin()]
        proj = read_projeto_from_excel(
            REPO_ROOT / "assets" / "data" / fname,
            f_ck_kpa=25_000.0, cobrimento_m=0.04,
        )
        cfg_json = json.loads((PROTO_DIR / case / S1 / "config.json")
                              .read_text(encoding="utf-8"))
        cfg = BenchmarkConfig(**{**cfg_json,
                                 "algorithms": ("ego",),
                                 "n_rep": 1,
                                 "base_seed": int(row["seed"])})
        res = run_benchmark(proj, cfg)
        if abs(res.best_of_value - float(row["best"])) > 1e-9:
            raise RuntimeError(
                f"{case}: reproduced best {res.best_of_value} != "
                f"stored {row['best']} â seeds/config drifted."
            )
        designs[case] = [
            {"rotulo": s.pilar.rotulo, "xg": s.pilar.xg, "yg": s.pilar.yg,
             "ap": s.pilar.a_p, "bp": s.pilar.b_p,
             "hx": s.h_x, "hy": s.h_y, "hz": s.h_z}
            for s in res.best_sapatas
        ]
        print(f"  design factível reproduzido: {case} "
              f"(seed {int(row['seed'])}, V={row['volume_m3']:.3f} m³)")
    cache.write_text(json.dumps(designs, indent=2, ensure_ascii=False),
                     encoding="utf-8")
    return designs


def fig_planta_casos(designs: dict) -> None:
    """Plan-view layout of the best strictly feasible EGO design per case.

    :param designs: Mapping produced by :func:`_best_feasible_designs`
    :return: None
    """
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.7))
    for ax, (case, meta) in zip(axes, CASES.items()):
        for s in designs[case]:
            ax.add_patch(Rectangle(
                (s["xg"] - s["hx"] / 2, s["yg"] - s["hy"] / 2),
                s["hx"], s["hy"],
                facecolor=ALG_COLOR["ego"] + "22",
                edgecolor=ALG_COLOR["ego"], linewidth=1.4,
            ))
            ax.add_patch(Rectangle(
                (s["xg"] - s["ap"] / 2, s["yg"] - s["bp"] / 2),
                s["ap"], s["bp"],
                facecolor=INK_2, edgecolor=INK, linewidth=0.8,
            ))
            ax.annotate(
                f"{s['rotulo']}\n"
                f"{s['hx'] * 100:.0f}×{s['hy'] * 100:.0f}×{s['hz'] * 100:.0f}",
                (s["xg"], s["yg"] - s["hy"] / 2),
                textcoords="offset points", xytext=(0, -4),
                ha="center", va="top", fontsize=7, color=INK,
            )
        xs = [s["xg"] for s in designs[case]]
        ys = [s["yg"] for s in designs[case]]
        half = max(max(s["hx"], s["hy"]) for s in designs[case]) / 2
        pad = half + 0.9
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad - 0.6, max(ys) + pad)
        ax.set_aspect("equal")
        ax.set_title(meta["titulo"], color=INK, pad=6)
        ax.set_xlabel("x [m]")
        ax.grid(True)
    axes[0].set_ylabel("y [m]")
    fig.tight_layout()
    _save(fig, "fig_planta_casos")


# =============================================================================
# Figures
# =============================================================================
def fig_convergencia_s1(data: dict) -> None:
    """Median best-so-far (with interquartile band) vs real evaluations.

    :param data: ``data[case][scenario]`` mapping of loaded artefacts
    :return: None
    """
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.4), sharex=True)
    for ax, (case, meta) in zip(axes, CASES.items()):
        hist = data[case][S1]["history"]
        budget = int(hist["eval_idx"].max())
        for alg in ALG_ORDER:
            g = hist[hist["algorithm"] == alg]
            mat = (g.pivot_table(index="eval_idx", columns="rep",
                                 values="of_best_so_far", aggfunc="last")
                    .sort_index().ffill())
            med = mat.median(axis=1)
            q25, q75 = mat.quantile(0.25, axis=1), mat.quantile(0.75, axis=1)
            ax.fill_between(mat.index, q25, q75,
                            color=ALG_COLOR[alg], alpha=0.13, linewidth=0)
            ax.plot(mat.index, med, color=ALG_COLOR[alg], linewidth=1.8,
                    solid_capstyle="round", label=ALG_LABEL[alg])
        ax.set_yscale("log")
        ax.set_xlim(0, budget)
        ax.set_title(meta["titulo"], color=INK, pad=6)
        ax.set_xlabel("Avaliações reais de $\\Theta$")
    axes[0].set_ylabel("Melhor $\\Theta$ até então [m³]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.13), columnspacing=1.4,
               handlelength=1.6)
    fig.tight_layout()
    _save(fig, "fig_convergencia_s1")


def fig_dist_best_s1(data: dict) -> None:
    """Distribution of the final best per algorithm (box + jittered strip).

    :param data: Loaded artefacts per case/scenario
    :return: None
    """
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.4))
    for ax, (case, meta) in zip(axes, CASES.items()):
        per_rep = data[case][S1]["per_rep"]
        series = [per_rep.loc[per_rep["algorithm"] == a, "best"].to_numpy()
                  for a in ALG_ORDER]
        bp = ax.boxplot(series, positions=range(len(ALG_ORDER)),
                        widths=0.55, showfliers=False, patch_artist=True,
                        medianprops=dict(color=INK, linewidth=1.4),
                        whiskerprops=dict(color=AXIS, linewidth=0.9),
                        capprops=dict(color=AXIS, linewidth=0.9),
                        boxprops=dict(linewidth=1.0))
        for patch, alg in zip(bp["boxes"], ALG_ORDER):
            patch.set_facecolor("white")
            patch.set_edgecolor(ALG_COLOR[alg])
        for i, (alg, vals) in enumerate(zip(ALG_ORDER, series)):
            x = i + rng.uniform(-0.13, 0.13, size=len(vals))
            ax.scatter(x, vals, s=6, color=ALG_COLOR[alg],
                       alpha=0.55, linewidths=0, zorder=3)
        ax.set_xticks(range(len(ALG_ORDER)))
        ax.set_xticklabels([ALG_LABEL[a] for a in ALG_ORDER],
                           rotation=35, ha="right")
        ax.set_title(meta["titulo"], color=INK, pad=6)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Melhor $\\Theta$ final [m³]")
    fig.tight_layout()
    _save(fig, "fig_dist_best_s1")


def fig_s1_vs_s2(data: dict) -> None:
    """Dumbbell plot: equal vs extended budget, with the EGO-150 reference.

    :param data: Loaded artefacts per case/scenario
    :return: None
    """
    metas = ["ga", "pso", "gwo", "random"]
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.5))
    budget_s1 = int(data["caso1_um"][S1]["history"]["eval_idx"].max())
    budget_s2 = int(data["caso1_um"][S2]["history"]["eval_idx"].max())
    for ax, (case, meta) in zip(axes, CASES.items()):
        pr_s1 = data[case][S1]["per_rep"]
        pr_s2 = data[case][S2]["per_rep"]
        ego_med = pr_s1.loc[pr_s1["algorithm"] == "ego", "best"].median()
        for i, alg in enumerate(metas):
            v1 = pr_s1.loc[pr_s1["algorithm"] == alg, "best"].median()
            v2 = pr_s2.loc[pr_s2["algorithm"] == alg, "best"].median()
            ax.plot([i, i], [v1, v2], color=ALG_COLOR[alg],
                    linewidth=1.2, alpha=0.8, zorder=2)
            ax.scatter([i], [v1], s=34, facecolors="white",
                       edgecolors=ALG_COLOR[alg], linewidths=1.4, zorder=3)
            ax.scatter([i], [v2], s=34, color=ALG_COLOR[alg], zorder=3)
        ax.axhline(ego_med, color=ALG_COLOR["ego"], linewidth=1.4,
                   linestyle=(0, (4, 2)), zorder=1)
        ax.text(len(metas) - 0.42, ego_med, "EGO (150)", color=ALG_COLOR["ego"],
                fontsize=7.5, va="bottom", ha="right")
        ax.set_xticks(range(len(metas)))
        ax.set_xticklabels([ALG_LABEL[a] for a in metas],
                           rotation=35, ha="right")
        ax.set_title(meta["titulo"], color=INK, pad=6)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Mediana do melhor $\\Theta$ [m³]")
    proxy = [
        plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="white",
                   markeredgecolor=INK_2, label=f"{budget_s1} avaliações"),
        plt.Line2D([], [], marker="o", linestyle="", color=INK_2,
                   label=f"{budget_s2} avaliações"),
    ]
    fig.legend(handles=proxy, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    _save(fig, "fig_s1_vs_s2")


def fig_violacoes_s1(data: dict) -> None:
    """Max constraint violation of each final design (strip, symlog scale).

    :param data: Loaded artefacts per case/scenario
    :return: None
    """
    rng = np.random.default_rng(11)
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.4), sharey=True)
    for ax, (case, meta) in zip(axes, CASES.items()):
        per_rep = data[case][S1]["per_rep"]
        for i, alg in enumerate(ALG_ORDER):
            vals = per_rep.loc[per_rep["algorithm"] == alg,
                               "max_violation"].to_numpy()
            x = i + rng.uniform(-0.16, 0.16, size=len(vals))
            ax.scatter(x, vals, s=7, color=ALG_COLOR[alg],
                       alpha=0.6, linewidths=0, zorder=3)
        ax.axhline(0.0, color=AXIS, linewidth=0.9, zorder=1)
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.set_xticks(range(len(ALG_ORDER)))
        ax.set_xticklabels([ALG_LABEL[a] for a in ALG_ORDER],
                           rotation=35, ha="right")
        ax.set_title(meta["titulo"], color=INK, pad=6)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Violação máxima $\\max_k g_k$ [--]")
    fig.tight_layout()
    _save(fig, "fig_violacoes_s1")


def fig_gpr(metrics: pd.DataFrame, preds: pd.DataFrame) -> None:
    """Observed-vs-predicted scatter and per-kernel R-squared dot plot.

    :param metrics: metrics.csv frame of the kernel study
    :param preds: predictions.parquet frame of the kernel study
    :return: None
    """
    # --- observed vs predicted (best kernel at alpha=10) -----------------
    mean_r2 = (metrics[metrics["penalty"] == 10.0]
               .groupby("kernel_id")["r2"].mean().sort_values(ascending=False))
    best_k = mean_r2.index[0]
    seed = int(metrics["seed"].min())
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9))
    for ax, pen in zip(axes, (10.0, 1e6)):
        sel = preds[(preds["kernel_id"] == best_k)
                    & (preds["penalty"] == pen) & (preds["seed"] == seed)]
        r2 = metrics[(metrics["kernel_id"] == best_k)
                     & (metrics["penalty"] == pen)
                     & (metrics["seed"] == seed)]["r2"].iloc[0]
        lo = min(sel["y_test"].min(), sel["y_pred"].min())
        hi = max(sel["y_test"].max(), sel["y_pred"].max())
        pad = 0.04 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                color=AXIS, linewidth=0.9, zorder=1)
        ax.scatter(sel["y_test"], sel["y_pred"], s=8,
                   color=PENALTY_COLOR[pen], alpha=0.55, linewidths=0, zorder=2)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal")
        alpha_txt = "10^{1}" if pen == 10.0 else "10^{6}"
        ax.set_title(rf"$\alpha = {alpha_txt}$ — {KERNEL_SHORT[best_k]}",
                     color=INK, pad=6)
        ax.set_xlabel(r"$\Theta$ observado [m³]")
        ax.text(0.04, 0.93, rf"$R^2 = {_fmt_br(r2, 3)}$",
                transform=ax.transAxes, fontsize=8.5, color=INK)
    axes[0].set_ylabel(r"$\Theta$ predito [m³]")
    fig.tight_layout()
    _save(fig, "fig_gpr_obs_pred")

    # --- R2 per kernel and penalty (dot + min-max range) -----------------
    order = mean_r2.index.tolist()
    fig, ax = plt.subplots(figsize=(6.3, 4.6))
    for j, pen in enumerate((10.0, 1e6)):
        sub = metrics[metrics["penalty"] == pen]
        grp = sub.groupby("kernel_id")["r2"]
        mean = grp.mean().reindex(order)
        lo = grp.min().reindex(order).clip(lower=-0.02)
        hi = grp.max().reindex(order).clip(lower=-0.02)
        y = np.arange(len(order)) + (0.18 if pen == 1e6 else -0.18)
        clipped = mean < -0.02
        m = mean.clip(lower=-0.02)
        ax.hlines(y, lo, hi, color=PENALTY_COLOR[pen],
                  linewidth=1.1, alpha=0.7)
        face = PENALTY_COLOR[pen] if pen == 10.0 else "white"
        ax.scatter(m, y, s=22, facecolors=face,
                   edgecolors=PENALTY_COLOR[pen], linewidths=1.2, zorder=3,
                   label=(r"$\alpha=10^{1}$" if pen == 10.0
                          else r"$\alpha=10^{6}$"))
        for yi, was_clipped in zip(y, clipped):
            if was_clipped:
                ax.annotate("<0", (-0.02, yi), textcoords="offset points",
                            xytext=(-14, -2.5), fontsize=6.5, color=MUTED)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{k} · {KERNEL_SHORT[k]}" for k in order], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(-0.05, 1.02)
    ax.set_xlabel(r"$R^2$ no conjunto de teste (média e amplitude, 3 réplicas)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, "fig_gpr_kernels_r2")


# =============================================================================
# Tables
# =============================================================================
def _write_tex(name: str, content: str) -> None:
    """Write a LaTeX table fragment.

    :param name: File name (with .tex)
    :param content: Full LaTeX content
    :return: None
    """
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    (TAB_DIR / name).write_text(content, encoding="utf-8")
    print(f"  tabela: {name}")


def tab_casos() -> None:
    """Frozen case-study table straight from the input spreadsheets.

    :return: None
    """
    rows = []
    for case, meta in CASES.items():
        fname = {"caso1_um": "problema_fund_um.xlsx",
                 "caso2_dois": "problema_fund_dois.xlsx",
                 "caso3_tres": "problema_fund_três.xlsx"}[case]
        df = pd.read_excel(REPO_ROOT / "assets" / "data" / fname)
        fz = [c for c in df.columns if c.startswith("Fz-")]
        mx = [c for c in df.columns if c.startswith("Mx-")]
        my = [c for c in df.columns if c.startswith("My-")]
        for _, r in df.iterrows():
            rows.append(
                f"{meta['curto']} & {r['Elemento']} & "
                f"{_fmt_br(r['ap (m)'])} & {_fmt_br(r['bp (m)'])} & "
                f"{int(r['spt'])} & {str(r['solo']).capitalize()} & "
                f"{_fmt_br(df.loc[r.name, fz].min(), 1)}--"
                f"{_fmt_br(df.loc[r.name, fz].max(), 1)} & "
                f"{_fmt_br(df.loc[r.name, mx].abs().max(), 1)} & "
                f"{_fmt_br(df.loc[r.name, my].abs().max(), 1)} \\\\"
            )
    body = "\n        ".join(rows)
    _write_tex("tab_casos.tex", rf"""% Gerada por scripts/make_paper_artifacts.py — nao editar manualmente.
\begin{{table*}}[!t]
    \centering
    \caption{{Casos de estudo congelados: geometria dos pilares, sondagem e envelope dos esforços característicos nas três combinações de carregamento.}}
    \label{{tab:casos}}
    \small
    \begin{{tabular}}{{llccclccc}}
        \toprule
        Caso & Elem. & $a_p$ [m] & $b_p$ [m] & $N_{{\mathrm{{spt}}}}$ & Solo & $F_z$ [kN] & $|M_x|_{{\max}}$ & $|M_y|_{{\max}}$ \\
        \midrule
        {body}
        \bottomrule
    \end{{tabular}}
    \caption*{{\footnotesize Fonte: planilhas oficiais do repositório (\texttt{{assets/data/}}). Momentos em kN\,m.}}
\end{{table*}}
""")


def tab_protocolo(data: dict) -> None:
    """Frozen protocol parameter table (from the persisted configs).

    :param data: Loaded artefacts per case/scenario
    :return: None
    """
    import json
    cfg = json.loads((PROTO_DIR / "caso3_tres" / S1 / "config.json")
                     .read_text(encoding="utf-8"))
    cfg2 = json.loads((PROTO_DIR / "caso3_tres" / S2 / "config.json")
                      .read_text(encoding="utf-8"))
    _write_tex("tab_protocolo.tex", rf"""% Gerada por scripts/make_paper_artifacts.py — nao editar manualmente.
\begin{{table*}}[!t]
    \centering
    \caption{{Protocolo experimental congelado: dois cenários de orçamento sob as mesmas $n_{{\mathrm{{rep}}}} = {cfg['n_rep']}$ repetições semeadas (\textit{{seeds}} ${cfg['base_seed']}$ a ${cfg['base_seed'] + cfg['n_rep'] - 1}$).}}
    \label{{tab:protocolo}}
    \small
    \begin{{tabular}}{{p{{0.42\textwidth}}p{{0.24\textwidth}}p{{0.24\textwidth}}}}
        \toprule
        Parâmetro & S1 (orçamento igual) & S2 (orçamento estendido) \\
        \midrule
        Algoritmos & EGO, GA, PSO, GWO, aleatória & GA, PSO, GWO, aleatória \\
        Avaliações reais de $\Theta$ por repetição & {cfg['budget_evals']} (todos) & {cfg2['budget_evals']} \\
        Amostra inicial do EGO (LHS) & $10d$ por caso & --- \\
        População das metaheurísticas & {cfg['meta_pop_size']} & {cfg2['meta_pop_size']} \\
        AG interno do EI (\textit{{pop\_size}} $\times$ \textit{{epoch}}) & ${cfg['ga_pop_size']} \times {cfg['ga_epoch']}$ & --- \\
        \textit{{Kernel}} do GPR & Matérn $\nu{{=}}2{{,}}5$ (produção) & --- \\
        Limites $h_x, h_y, h_z$ [m] & \multicolumn{{2}}{{c}}{{$[{_fmt_br(cfg['h_min_m'])};\ {_fmt_br(cfg['h_max_m'])}]$}} \\
        Penalidade & \multicolumn{{2}}{{c}}{{exterior linear, $\alpha = 10$, $p = 1$}} \\
        $f_{{ck}}$; cobrimento & \multicolumn{{2}}{{c}}{{\SI{{25}}{{\mega\pascal}}; \SI{{4}}{{\centi\meter}}}} \\
        \bottomrule
    \end{{tabular}}
    \caption*{{\footnotesize Fonte: \texttt{{config.json}} persistido por caso/cenário em \texttt{{experiments/protocolo\_final/}}.}}
\end{{table*}}
""")


def _stats_block(data: dict, scenario: str, algs: list[str]) -> str:
    """Build the per-case × per-algorithm statistics rows of a results table.

    :param data: Loaded artefacts per case/scenario
    :param scenario: Scenario folder name
    :param algs: Algorithms to include, in order
    :return: LaTeX rows
    """
    lines = []
    for case, meta in CASES.items():
        summ = data[case][scenario]["summary"].set_index("algorithm")
        for alg in algs:
            if alg not in summ.index:
                continue
            r = summ.loc[alg]
            lines.append(
                f"{meta['curto']} & {ALG_LABEL[alg]} & "
                f"{_fmt_br(r['best'], 3)} & "
                f"{_fmt_br(r['mean'], 3)} $\\pm$ {_fmt_br(r['std'], 3)} & "
                f"{_fmt_br(r['median'], 3)} & "
                f"{_fmt_br(100 * r['feasibility_rate'], 0)}\\% & "
                f"{_fmt_br(r['mean_max_violation'], 3)} & "
                f"{_fmt_br(r['best_feasible_volume_m3'], 3)} & "
                f"{_fmt_br(r['wall_time_mean_s'], 2)} \\\\"
            )
        lines.append(r"\midrule")
    if lines and lines[-1] == r"\midrule":
        lines.pop()
    return "\n        ".join(lines)


def tab_resultados(data: dict) -> None:
    """S1 and S2 statistics tables.

    :param data: Loaded artefacts per case/scenario
    :return: None
    """
    body1 = _stats_block(data, S1, ALG_ORDER)
    _write_tex("tab_s1.tex", rf"""% Gerada por scripts/make_paper_artifacts.py — nao editar manualmente.
\begin{{table*}}[!t]
    \centering
    \caption{{Cenário S1 (orçamento igual de 150 avaliações reais): estatísticas do melhor $\Theta$ [\si{{\meter\cubed}}] em 30 repetições, factibilidade da solução final (tolerância $g_k \le 10^{{-9}}$), violação máxima média, melhor volume factível $V^{{\mathrm{{feas}}}}_{{\min}}$ [\si{{\meter\cubed}}] e tempo de parede.}}
    \label{{tab:s1}}
    \small
    \begin{{tabular}}{{llccccccc}}
        \toprule
        Caso & Algoritmo & Melhor & Média $\pm$ DP & Mediana & Fact. & $\overline{{\max g_k}}$ & $V^{{\mathrm{{feas}}}}_{{\min}}$ & Tempo [s] \\
        \midrule
        {body1}
        \bottomrule
    \end{{tabular}}
\end{{table*}}
""")
    body2 = _stats_block(data, S2, ["ga", "pso", "gwo", "random"])
    _write_tex("tab_s2.tex", rf"""% Gerada por scripts/make_paper_artifacts.py — nao editar manualmente.
\begin{{table*}}[!t]
    \centering
    \caption{{Cenário S2 (orçamento estendido de 3\,000 avaliações reais para as buscas diretas): estatísticas em 30 repetições, mesmas \textit{{seeds}} do S1.}}
    \label{{tab:s2}}
    \small
    \begin{{tabular}}{{llccccccc}}
        \toprule
        Caso & Algoritmo & Melhor & Média $\pm$ DP & Mediana & Fact. & $\overline{{\max g_k}}$ & $V^{{\mathrm{{feas}}}}_{{\min}}$ & Tempo [s] \\
        \midrule
        {body2}
        \bottomrule
    \end{{tabular}}
\end{{table*}}
""")


def tab_pvalues(data: dict) -> None:
    """Mann-Whitney p-value matrices for S1, one block per case.

    :param data: Loaded artefacts per case/scenario
    :return: None
    """
    blocks = []
    for case, meta in CASES.items():
        pv = data[case][S1]["pvalues"]
        header = " & ".join(ALG_LABEL[a] for a in ALG_ORDER[1:])
        lines = [rf"\multicolumn{{5}}{{l}}{{\textit{{{meta['titulo']}}}}} \\", r"\midrule"]
        for a in ALG_ORDER[:-1]:
            cells = []
            for b in ALG_ORDER[1:]:
                if ALG_ORDER.index(b) <= ALG_ORDER.index(a):
                    cells.append("")
                else:
                    cells.append(_fmt_pvalue(float(pv.loc[a, b])))
            lines.append(f"{ALG_LABEL[a]} & " + " & ".join(cells) + r" \\")
        blocks.append((header, "\n        ".join(lines)))
    header = blocks[0][0]
    body = ("\n        \\addlinespace[6pt]\n        ").join(b for _, b in blocks)
    _write_tex("tab_pvalues_s1.tex", rf"""% Gerada por scripts/make_paper_artifacts.py — nao editar manualmente.
\begin{{table}}[!t]
    \centering
    \caption{{Matriz triangular de p-valores (Mann--Whitney~$U$ bilateral) sobre o melhor $\Theta$ por repetição no cenário S1; valores em negrito indicam $p < 0{{,}}05$.}}
    \label{{tab:pvalues_s1}}
    \small
    \begin{{tabular}}{{lcccc}}
        \toprule
         & {header} \\
        \midrule
        {body}
        \bottomrule
    \end{{tabular}}
\end{{table}}
""")


def _gpr_feasible_rmse(preds: pd.DataFrame) -> pd.DataFrame:
    """RMSE restricted to the feasible test points, per kernel and penalty.

    Both penalty labels coincide exactly on feasible designs (no
    violation → no penalty term), so the feasible subset is identified
    positionally as the points where the two label vectors match. This
    is the error that matters to the optimisation: global R² is
    scale-invariant and hides the fact that, under alpha = 1e6, the
    surrogate's absolute error near the feasible region is orders of
    magnitude above the volume scale.

    :param preds: predictions.parquet frame of the kernel study
    :return: One row per kernel with mean feasible-region RMSE per penalty
    """
    rows = []
    for (k, s), g in preds.groupby(["kernel_id", "seed"]):
        g10 = g[g["penalty"] == 10.0].reset_index(drop=True)
        g6 = g[g["penalty"] == 1e6].reset_index(drop=True)
        mask = np.isclose(g10["y_test"].to_numpy(), g6["y_test"].to_numpy())
        if mask.sum() < 5:
            continue
        rows.append({
            "kernel_id": k, "seed": s,
            "rmse_feas_a10": float(np.sqrt(np.mean(
                (g10["y_pred"][mask] - g10["y_test"][mask]) ** 2))),
            "rmse_feas_a1e6": float(np.sqrt(np.mean(
                (g6["y_pred"][mask] - g6["y_test"][mask]) ** 2))),
        })
    return (pd.DataFrame(rows).groupby("kernel_id")
            [["rmse_feas_a10", "rmse_feas_a1e6"]].mean())


def tab_gpr(metrics: pd.DataFrame, preds: pd.DataFrame) -> None:
    """GPR metrics table: top kernels by R2 at alpha=10 plus the production one.

    :param metrics: metrics.csv frame of the kernel study
    :param preds: predictions.parquet frame (for the feasible-region RMSE)
    :return: None
    """
    agg = (metrics.groupby(["kernel_id", "penalty"])
           .agg(r2_m=("r2", "mean"), r2_s=("r2", "std"),
                mae_m=("mae", "mean"), rmse_m=("rmse", "mean"))
           .reset_index())
    feas = _gpr_feasible_rmse(preds)
    rank = (agg[agg["penalty"] == 10.0]
            .sort_values("r2_m", ascending=False)["kernel_id"].tolist())
    keep = rank[:8]
    if "k20" not in keep:
        keep.append("k20")
    rows = []
    for k in keep:
        a10 = agg[(agg["kernel_id"] == k) & (agg["penalty"] == 10.0)].iloc[0]
        a1e6 = agg[(agg["kernel_id"] == k) & (agg["penalty"] == 1e6)].iloc[0]
        marker = r"$^\dagger$" if k == "k20" else ""
        rows.append(
            f"{k}{marker} & {KERNEL_SHORT[k]} & "
            f"{_fmt_br(a10['r2_m'], 3)} $\\pm$ {_fmt_br(a10['r2_s'], 3)} & "
            f"{_fmt_br(a10['rmse_m'], 2)} & "
            f"{_fmt_br(feas.loc[k, 'rmse_feas_a10'], 2)} & "
            f"{_fmt_br(a1e6['r2_m'], 3)} & "
            f"\\num{{{feas.loc[k, 'rmse_feas_a1e6']:.3g}}} \\\\"
        )
    body = "\n        ".join(rows)
    _write_tex("tab_gpr_kernels.tex", rf"""% Gerada por scripts/make_paper_artifacts.py — nao editar manualmente.
\begin{{table*}}[!t]
    \centering
    \caption{{Qualidade preditiva do GPR por \textit{{kernel}} (média em 3 réplicas independentes de amostragem/partição; caso de 3 sapatas; 900 amostras LHS; partição 70/30): as oito melhores configurações sob $\alpha = 10^{{1}}$ e o \textit{{kernel}} de produção. RMSE$_{{\mathrm{{feas}}}}$ é o erro restrito aos pontos de teste factíveis — a região onde a função de aquisição decide.}}
    \label{{tab:gpr_kernels}}
    \small
    \begin{{tabular}}{{llccccc}}
        \toprule
        & & \multicolumn{{3}}{{c}}{{$\alpha = 10^{{1}}$}} & \multicolumn{{2}}{{c}}{{$\alpha = 10^{{6}}$}} \\
        \cmidrule(lr){{3-5}} \cmidrule(lr){{6-7}}
        Id & \textit{{Kernel}} & $R^2$ & RMSE & RMSE$_{{\mathrm{{feas}}}}$ & $R^2$ & RMSE$_{{\mathrm{{feas}}}}$ \\
        \midrule
        {body}
        \bottomrule
    \end{{tabular}}
    \caption*{{\footnotesize $^\dagger$\,\textit{{kernel}} adotado em produção no \fundaai{{}} (\texttt{{constroi\_kernel()[-1]}}). RMSE em \si{{\meter\cubed}}.}}
\end{{table*}}
""")


def _csv_mirrors(data: dict, metrics: pd.DataFrame | None) -> None:
    """Persist CSV mirrors of every aggregated table.

    :param data: Loaded artefacts per case/scenario
    :param metrics: Kernel-study metrics (or None when absent)
    :return: None
    """
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for case in CASES:
        for scen in (S1, S2):
            s = data[case][scen]["summary"].copy()
            s.insert(0, "case", case)
            s.insert(1, "scenario", scen)
            frames.append(s)
    pd.concat(frames, ignore_index=True).to_csv(
        CSV_DIR / "summary_all.csv", index=False)
    if metrics is not None:
        metrics.to_csv(CSV_DIR / "gpr_metrics.csv", index=False)
    print(f"  csv: {CSV_DIR.relative_to(REPO_ROOT)}/")


def main() -> None:
    """Regenerate every manuscript artefact from the frozen runs.

    :return: None
    """
    _style()
    data = {case: {scen: _load(case, scen) for scen in (S1, S2)}
            for case in CASES}
    print("Figuras:")
    fig_convergencia_s1(data)
    fig_dist_best_s1(data)
    fig_s1_vs_s2(data)
    fig_violacoes_s1(data)
    fig_planta_casos(_best_feasible_designs())

    metrics = None
    if (GPR_DIR / "metrics.csv").exists():
        metrics = pd.read_csv(GPR_DIR / "metrics.csv")
        preds = pd.read_parquet(GPR_DIR / "predictions.parquet")
        fig_gpr(metrics, preds)
    else:
        print("  (estudo GPR ausente — figuras/tabela de kernels puladas)")

    print("Tabelas:")
    tab_casos()
    tab_protocolo(data)
    tab_resultados(data)
    tab_pvalues(data)
    if metrics is not None:
        tab_gpr(metrics, preds)
    _csv_mirrors(data, metrics)
    print("ARTEFATOS COMPLETOS.")


if __name__ == "__main__":
    main()
