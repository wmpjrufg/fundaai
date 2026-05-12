"""Streamlit page — head-to-head benchmark bench (EGO vs pure metaheuristics).

Sister page of ``/sapatas``. While the design page is a tool for
engineers, this one is a **scientific bench**: same objective
function, same evaluation budget, ``n_rep`` repetitions per
algorithm with seeded reproducibility, and figures/tables ready to
be dropped into the IC manuscript.

The page is a thin shell over :func:`core.api.run_benchmark`. It
keeps Streamlit-specific concerns here (uploads, progress, downloads)
and delegates orchestration and statistics to the core layer.

Resumo em português:
    Página dedicada a comparativos científicos entre EGO e
    metaheurísticas puras. Roda :func:`core.api.run_benchmark`,
    gera a curva de convergência multi-algoritmo, a tabela-resumo
    com média ± desvio e a matriz de p-valores de Mann–Whitney, e
    oferece um bundle de download (histórico Parquet, sumário CSV,
    p-values CSV, HTML/PNG do gráfico).
"""

from __future__ import annotations

import io
import json
import queue
import threading
import time
import zipfile
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np
import pandas as pd
import streamlit as st

from core.api import (
    ALGORITHM_LABELS,
    ALL_ALGORITHMS,
    BenchmarkConfig,
    BenchmarkResult,
    run_benchmark,
)
from core.io import read_projeto_from_excel
from frontend.components import (
    figure_to_html_bytes,
    figure_to_png_bytes,
    render_convergence_chart,
)
from frontend.theme import apply_theme

apply_theme()

POLL_INTERVAL_S = 0.4


# =============================================================================
# Page
# =============================================================================
st.title("🧪 Bancada de Experimentos — EGO vs Metaheurísticas")
st.caption(
    "Comparativo controlado entre EGO+GPR e metaheurísticas puras "
    "(GA / PSO / GWO) sobre o mesmo problema, com o mesmo orçamento "
    "de avaliações reais e seeds reprodutíveis. Os artefatos gerados "
    "aqui são pensados para o relatório científico."
)

if "bench_calc" not in st.session_state:
    st.session_state["bench_calc"] = False
if "bench_show_log" not in st.session_state:
    st.session_state["bench_show_log"] = False


# --- Inputs ----------------------------------------------------------------
st.subheader("⚙️ Configuração do experimento")
cfg_a, cfg_b, cfg_c = st.columns(3)
with cfg_a:
    f_ck_mpa = st.number_input(
        "fck do concreto (MPa)",
        min_value=15.0, max_value=90.0, step=5.0, value=25.0,
    )
    cob_cm = st.number_input(
        "Cobrimento (cm)", step=0.5, value=4.0, format="%.1f",
    )
with cfg_b:
    h_min_cm = st.number_input(
        "Dimensão mínima (cm)", min_value=20.0, step=0.5, value=60.0,
    )
    h_max_cm = st.number_input(
        "Dimensão máxima (cm)", min_value=20.0, step=0.5, value=150.0,
    )
with cfg_c:
    budget_evals = st.number_input(
        "Orçamento de avaliações reais (por repetição)",
        min_value=20, max_value=5000, step=10, value=150,
        help=(
            "Número máximo de chamadas à função objetivo real **por "
            "repetição**, compartilhado entre todos os algoritmos. É "
            "o eixo X do gráfico de convergência."
        ),
    )
    n_rep = st.number_input(
        "Repetições por algoritmo (n_rep)",
        min_value=2, max_value=50, step=1, value=5,
        help=(
            "Número de execuções independentes com seeds diferentes "
            "(``base_seed + rep``). Necessário para média ± desvio e "
            "para o teste de Mann–Whitney."
        ),
    )
    base_seed = st.number_input(
        "Seed base", min_value=0, step=1, value=42,
    )

st.markdown("**Algoritmos a comparar**")
alg_cols = st.columns(len(ALL_ALGORITHMS))
selected: list[str] = []
for col, alg in zip(alg_cols, ALL_ALGORITHMS):
    with col:
        default = True
        if st.checkbox(ALGORITHM_LABELS[alg], value=default, key=f"alg_{alg}"):
            selected.append(alg)

with st.expander("🔧 Parâmetros avançados", expanded=False):
    adv_a, adv_b = st.columns(2)
    with adv_a:
        lhs_n_pop = st.number_input(
            "EGO · tamanho do LHS inicial",
            min_value=4, max_value=2000, step=2, value=20,
            help=(
                "Quantos pontos LHS o EGO avalia antes de iniciar o "
                "loop do surrogate. Deve ser estritamente menor que o "
                "orçamento total. Valores mais altos dão amostragem "
                "inicial mais densa, mas comem do orçamento que iria "
                "para o loop guiado por EI."
            ),
        )
        meta_pop_size = st.number_input(
            "GA/PSO/GWO · tamanho da população",
            min_value=4, max_value=500, step=2, value=40,
            help=(
                "Tamanho da população dos metaheurísticos puros. "
                "Como o custo aqui é só de avaliar a função real (que é "
                "barata), pode-se elevar sem grande impacto de tempo. "
                "Com budget=150 e pop=40, cada algoritmo faz ~4 gerações."
            ),
        )
    with adv_b:
        ga_pop_size = st.number_input(
            "EGO · população do GA interno (EI)",
            min_value=10, max_value=1000, step=10, value=50,
            help=(
                "Otimizador interno que maximiza a função Expected "
                "Improvement. Atua sobre o **surrogate** (não chama a "
                "função objetivo real), então é o principal custo "
                "computacional do EGO em problemas onde a função real "
                "é barata. Reduzir aqui acelera muito sem perder "
                "qualidade significativa."
            ),
        )
        ga_epoch = st.number_input(
            "EGO · épocas do GA interno (EI)",
            min_value=5, max_value=500, step=5, value=30,
        )

if not selected:
    st.warning("Selecione ao menos um algoritmo para rodar o comparativo.")
    st.stop()

st.divider()

# --- Upload ---------------------------------------------------------------
st.subheader("📥 Planilha do projeto")
uploaded_file = st.file_uploader(
    "Excel do problema (mesmo formato usado em /sapatas)",
    type=["xlsx", "xls"], key="bench_upload",
)
if uploaded_file is None:
    st.info("Carregue um arquivo Excel para habilitar a execução.")
    st.stop()

try:
    projeto = read_projeto_from_excel(
        uploaded_file,
        f_ck_kpa=float(f_ck_mpa) * 1000.0,
        cobrimento_m=float(cob_cm) / 100.0,
    )
except (ValueError, FileNotFoundError) as exc:
    st.error("Erro ao ler a planilha.")
    st.exception(exc)
    st.stop()

p_a, p_b, p_c = st.columns(3)
p_a.metric("Pilares", projeto.n_fund)
p_b.metric("Combinações", projeto.n_comb)
p_c.metric("Dimensão do vetor", 3 * projeto.n_fund)


# --- Threaded execution ---------------------------------------------------
def _spawn_benchmark(projeto, config: BenchmarkConfig, run_state: dict) -> dict:
    """Launch ``run_benchmark`` on a daemon thread and wire progress."""
    events_q: queue.Queue = queue.Queue()
    cancel_event = threading.Event()
    holder: dict = {}

    def _runner() -> None:
        try:
            holder["result"] = run_benchmark(
                projeto, config,
                progress=lambda ev: events_q.put(ev),
                should_stop=cancel_event.is_set,
            )
        except Exception as exc:   # pragma: no cover
            holder["error"] = exc
        finally:
            holder["done"] = True

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    run_state.update({
        "queue": events_q,
        "cancel_event": cancel_event,
        "holder": holder,
        "thread": thread,
        "events_seen": [],
        "total_units": len(config.algorithms) * config.n_rep,
        "best_overall": float("inf"),
        "config": config,
    })
    return run_state


col_btn, col_estimate = st.columns([1, 3])
with col_btn:
    trigger = st.button(
        "🚀 Executar comparativo",
        type="primary",
        disabled="bench_run" in st.session_state,
    )
with col_estimate:
    total_evals = int(budget_evals) * int(n_rep) * len(selected)
    st.caption(
        f"Total estimado: **{total_evals:,}** avaliações reais "
        f"({len(selected)} algoritmos × {int(n_rep)} reps × "
        f"{int(budget_evals)} aval./rep)."
    )

if trigger:
    try:
        bench_cfg = BenchmarkConfig(
            algorithms=tuple(selected),   # type: ignore[arg-type]
            budget_evals=int(budget_evals),
            n_rep=int(n_rep),
            base_seed=int(base_seed),
            h_min_m=float(h_min_cm) / 100.0,
            h_max_m=float(h_max_cm) / 100.0,
            lhs_n_pop=int(lhs_n_pop),
            meta_pop_size=int(meta_pop_size),
            ga_pop_size=int(ga_pop_size),
            ga_epoch=int(ga_epoch),
        )
    except (ValueError, Exception) as exc:
        st.error("Configuração inválida.")
        st.exception(exc)
        st.stop()

    st.session_state["bench_run"] = _spawn_benchmark(projeto, bench_cfg, {})
    st.session_state["bench_calc"] = False
    st.rerun()


# --- Live progress --------------------------------------------------------
def _render_bench_progress(run_state: dict) -> None:
    events_q: queue.Queue = run_state["queue"]
    cancel_event: threading.Event = run_state["cancel_event"]
    holder: dict = run_state["holder"]

    while True:
        try:
            ev = events_q.get_nowait()
            run_state["events_seen"].append(ev)
            if ev.get("event") == "benchmark.rep_end" and "best" in ev:
                run_state["best_overall"] = min(
                    run_state["best_overall"], float(ev["best"]),
                )
        except queue.Empty:
            break

    seen = run_state["events_seen"]
    units_done = sum(
        1 for e in seen if e.get("event") == "benchmark.rep_end"
    )
    total_units = run_state["total_units"]
    pct = min(units_done / max(total_units, 1), 1.0)
    last = seen[-1] if seen else {}
    last_kind = last.get("event")

    if cancel_event.is_set():
        headline = "⏹️ Cancelando — aguarde o fim da repetição atual..."
    elif last_kind == "benchmark.rep_start":
        alg = last.get("algorithm", "?")
        rep = last.get("rep", 0) + 1
        n_rep_total = last.get("n_rep", "?")
        headline = (
            f"▶️ {ALGORITHM_LABELS.get(alg, alg)} — rep {rep}/{n_rep_total} "
            f"em execução…"
        )
    elif last_kind == "benchmark.rep_end":
        alg = last.get("algorithm", "?")
        rep = last.get("rep", 0) + 1
        best = last.get("best", float("nan"))
        headline = (
            f"✅ {ALGORITHM_LABELS.get(alg, alg)} — rep {rep} concluída · "
            f"best `{best:.6f} m³`"
        )
    elif last_kind == "benchmark.end":
        headline = "✅ Comparativo concluído"
    elif last_kind == "benchmark.cancelled":
        headline = "⏹️ Comparativo cancelado"
    else:
        headline = "⏳ Iniciando comparativo…"

    best_so_far = run_state["best_overall"]
    label = (
        f"{units_done}/{total_units} repetições · best global "
        f"`{best_so_far:.6f} m³`"
        if best_so_far != float("inf")
        else f"{units_done}/{total_units} repetições"
    )
    st.progress(pct, text=label)

    with st.status(
        "Benchmark em andamento",
        expanded=True,
        state=("running" if not holder.get("done")
               else ("error" if "error" in holder else "complete")),
    ):
        st.markdown(headline)
        cfg: BenchmarkConfig = run_state["config"]
        st.caption(
            f"Orçamento por rep: **{cfg.budget_evals}** avaliações reais · "
            f"Algoritmos: {', '.join(ALGORITHM_LABELS[a] for a in cfg.algorithms)} · "
            f"Seed base: {cfg.base_seed}"
        )

    if not holder.get("done"):
        if cancel_event.is_set():
            st.warning("Cancelamento solicitado — interrompendo no próximo ponto seguro.")
        else:
            if st.button("⏹️ Parar comparativo", key="bench_cancel"):
                cancel_event.set()
                st.rerun()


if "bench_run" in st.session_state:
    run_state = st.session_state["bench_run"]
    _render_bench_progress(run_state)
    holder = run_state["holder"]
    if not holder.get("done"):
        time.sleep(POLL_INTERVAL_S)
        st.rerun()

    if "result" in holder:
        st.session_state["bench_result"] = holder["result"]
        st.session_state["bench_calc"] = True
        st.success("Comparativo concluído.")
    elif "error" in holder:
        st.error("Falha no comparativo.")
        st.exception(holder["error"])

    del st.session_state["bench_run"]
    st.rerun()


# --- Results ---------------------------------------------------------------
if not st.session_state.get("bench_calc"):
    st.stop()

result: BenchmarkResult = st.session_state["bench_result"]
cfg = result.config
summary = result.summary
history = result.history
pvalues = result.pvalues

st.divider()
st.subheader("📊 Resultados")

# KPI strip ----------------------------------------------------------------
best_row = summary.loc[summary["best"].idxmin()]
total_evals = int(len(history))
total_time = float(history["time_total_s"].groupby(
    [history["algorithm"], history["rep"]]
).max().sum())

kpi_a, kpi_b, kpi_c, kpi_d = st.columns(4)
kpi_a.metric("Melhor algoritmo (best abs.)",
             best_row["label"], f"{best_row['best']:.4f} m³")
kpi_b.metric("Algoritmos avaliados", f"{len(summary)}")
kpi_c.metric("Avaliações reais totais", f"{total_evals:,}")
kpi_d.metric("Tempo total acumulado", f"{total_time:.1f} s")


# Convergence chart --------------------------------------------------------
st.markdown("### 📈 Curva de convergência (best-so-far por nº de avaliações)")
log_y = st.toggle(
    "Eixo OF em escala logarítmica", value=False, key="bench_log_y",
)
fig = render_convergence_chart(
    history,
    labels=ALGORITHM_LABELS,
    summary=summary,
    log_y=log_y,
    show_individual_reps=True,
    show_time_panel=True,
)
st.plotly_chart(
    fig, use_container_width=True,
    config={
        "displaylogo": False, "responsive": True,
        "scrollZoom": True, "doubleClick": "reset",
    },
)
st.caption(
    "Linha sólida = mediana entre repetições · Faixa preenchida = "
    "**±1 desvio padrão** · Envelope min–max (off por padrão, ative na "
    "legenda) · Trajetórias individuais ficam ocultas; clique no nome "
    "do algoritmo na legenda para alternar a visibilidade."
)


# Summary table ------------------------------------------------------------
st.markdown("### 📋 Estatísticas por algoritmo")
fmt_summary = summary.assign(
    best=lambda d: d["best"].map(lambda v: f"{v:.6f}"),
    mean=lambda d: d.apply(lambda r: f"{r['mean']:.6f} ± {r['std']:.6f}", axis=1),
    median=lambda d: d["median"].map(lambda v: f"{v:.6f}"),
    auc=lambda d: d.apply(lambda r: f"{r['auc_mean']:.4f} ± {r['auc_std']:.4f}", axis=1),
    conv_eval=lambda d: d.apply(
        lambda r: f"{r['conv_eval_mean']:.1f} ± {r['conv_eval_std']:.1f}", axis=1,
    ),
    wall_time=lambda d: d.apply(
        lambda r: f"{r['wall_time_mean_s']:.2f} ± {r['wall_time_std_s']:.2f}", axis=1,
    ),
)[
    ["label", "n_rep", "best", "mean", "median", "auc", "conv_eval", "wall_time"]
].rename(columns={
    "label": "Algoritmo",
    "n_rep": "n_rep",
    "best": "best OF [m³]",
    "mean": "média ± desvio [m³]",
    "median": "mediana [m³]",
    "auc": "AUC normalizada",
    "conv_eval": "aval. até best global",
    "wall_time": "tempo [s]",
})
st.dataframe(fmt_summary, use_container_width=True, hide_index=True)
st.caption(
    "**AUC normalizada**: área sob a curva ``of_best_so_far`` dividida "
    "pelo span de avaliações — quanto menor, mais rápido o algoritmo "
    "encontra valores baixos da função objetivo. **Aval. até best global**: "
    "número médio de avaliações para atingir, em cada repetição, um valor "
    "dentro de 0,1% do melhor encontrado em todo o experimento."
)


# P-values matrix ----------------------------------------------------------
st.markdown("### 🧮 Significância estatística (Mann–Whitney U, bilateral)")
display_pvalues = pvalues.copy()
display_pvalues.index = [ALGORITHM_LABELS.get(a, a) for a in pvalues.index]
display_pvalues.columns = [ALGORITHM_LABELS.get(a, a) for a in pvalues.columns]
styled = display_pvalues.style.format(
    lambda v: "—" if pd.isna(v) else f"{v:.4f}",
).map(
    lambda v: (
        "background-color: rgba(16,185,129,0.18);" if (not pd.isna(v) and v < 0.05)
        else ("background-color: rgba(239,68,68,0.10);" if not pd.isna(v) else "")
    ),
)
st.dataframe(styled, use_container_width=True)
st.caption(
    "Células em verde indicam diferença significativa entre os pares de "
    "algoritmos ao nível α = 0,05 sobre os valores ``best`` por repetição. "
    "Diagonal não definida (algoritmo não se compara a si mesmo)."
)


# Optional log viewer ------------------------------------------------------
with st.expander("🔬 Inspeção do histórico bruto", expanded=False):
    st.dataframe(history.head(500), use_container_width=True, hide_index=True)
    st.caption(
        f"Mostrando as primeiras 500 linhas de **{len(history):,}**. "
        "Use o botão de download abaixo para baixar a tabela completa em Parquet."
    )


# --- Export ---------------------------------------------------------------
st.divider()
st.subheader("📦 Exportar bundle do experimento")


def _build_bundle(result: BenchmarkResult, fig) -> bytes:
    """Build the downloadable zip with every research-ready artefact."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # history (parquet) — full per-eval trace
        h_buf = io.BytesIO()
        result.history.to_parquet(h_buf, index=False)
        zf.writestr("history.parquet", h_buf.getvalue())

        # history (csv) — for spreadsheets / reviewers without parquet
        zf.writestr("history.csv", result.history.to_csv(index=False))

        # summary
        zf.writestr("summary.csv", result.summary.to_csv(index=False))

        # p-values
        zf.writestr("pvalues.csv", result.pvalues.to_csv())

        # config + metadata
        meta = {
            "config": result.config.model_dump(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "n_history_rows": int(len(result.history)),
            "algorithms": [str(a) for a in result.config.algorithms],
        }
        zf.writestr("metadata.json", json.dumps(meta, indent=2))

        # chart (html + png)
        zf.writestr("convergence.html", figure_to_html_bytes(fig))
        try:
            zf.writestr("convergence.png", figure_to_png_bytes(fig))
        except Exception:
            # kaleido/static export pode não estar disponível em todos
            # os ambientes — não bloqueia o bundle.
            pass
    return buf.getvalue()


bundle_bytes = _build_bundle(result, fig)
ts = datetime.now().strftime("%Y%m%d-%H%M%S")
ex_a, ex_b, ex_c = st.columns(3)
with ex_a:
    st.download_button(
        "📦 Bundle completo (zip)",
        data=bundle_bytes,
        file_name=f"fundaia_benchmark_{ts}.zip",
        mime="application/zip",
    )
with ex_b:
    st.download_button(
        "📋 Sumário (CSV)",
        data=result.summary.to_csv(index=False),
        file_name=f"summary_{ts}.csv",
        mime="text/csv",
    )
with ex_c:
    st.download_button(
        "📈 Gráfico (HTML)",
        data=figure_to_html_bytes(fig),
        file_name=f"convergence_{ts}.html",
        mime="text/html",
    )
