"""Streamlit page — optimised footing design (premium UI shell).

This module is intentionally a thin presentation layer. All
engineering and optimisation logic lives in ``core/``:
``core.io.read_projeto_from_excel`` parses the upload,
``core.api.optimize`` orchestrates the EGO+GPR+GA pipeline (now with
the experiment recorder switched on by default so the EGO history is
always available for the "Ver histórico" button), and the export
panel hands the user every artifact they may need (DXF, JSON, HTML
3D, PNG history).

Resumo em português:
    Página Streamlit (camada fina sobre ``core.api``). Aplica o tema
    dark do projeto, dispara a otimização com o ``ExperimentRecorder``
    ligado por padrão e oferece resultado completo: planta 2D, vista
    3D interativa, gráfico premium de histórico do EGO e bloco
    unificado de downloads.
"""

from __future__ import annotations

import queue
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from core.api import (
    OptimisationCancelled,
    OptimisationConfig,
    OptimisationResult,
    evaluate,
    optimize,
)
from core.domain import FundacaoProjeto, Sapata
from core.io import read_projeto_from_excel
from core.io.experiments import ExperimentRecorder, load_experiment
from core.optimization.cache import SurrogateCache
from frontend.components import (
    CAMERA_PRESETS,
    build_export_artifacts,
    figure_to_html_bytes,
    render_ego_history,
    render_footings_3d,
)
from frontend.theme import apply_theme

apply_theme()

EXPERIMENTS_ROOT = Path("experiments")
POLL_INTERVAL_S = 0.4   # how often the page reruns to refresh progress


# =============================================================================
# Plot helpers (Streamlit-specific; live here to keep core layers free of UI)
# =============================================================================
def _plot_layout(sapatas: Sequence[Sapata]):
    """Render the in-plane layout of the optimised footings (matplotlib).

    Tuned to match the dark Streamlit theme — transparent canvas,
    amber edges and grid, light annotations.

    :param sapatas: Sequence of optimised Sapata entities

    :return: Matplotlib figure ready to be passed to ``st.pyplot``
    """
    fig, ax = plt.subplots(figsize=(10, 10), facecolor="none")
    ax.set_facecolor("#0b1220")
    for sapata in sapatas:
        v_sw, _, _, _ = sapata.vertices
        ax.add_patch(
            patches.Rectangle(
                v_sw, sapata.h_x, sapata.h_y,
                linewidth=1.5, edgecolor="#f59e0b", facecolor="#f59e0b22",
            )
        )
        ax.scatter(sapata.pilar.xg, sapata.pilar.yg,
                   color="#fbbf24", marker="+", s=120, linewidths=2)
        ax.annotate(
            sapata.pilar.rotulo,
            (sapata.pilar.xg, sapata.pilar.yg),
            textcoords="offset points", xytext=(0, 12),
            ha="center", color="#e5e7eb", fontsize=10, fontweight="bold",
        )
    ax.set_xlabel("X (m)", color="#9aa3b2")
    ax.set_ylabel("Y (m)", color="#9aa3b2")
    ax.set_title("Posicionamento das sapatas", color="#e5e7eb")
    ax.grid(True, color="#1f2a44", linewidth=0.6)
    ax.tick_params(colors="#9aa3b2")
    for spine in ax.spines.values():
        spine.set_color("#1f2a44")
    ax.set_aspect("equal", adjustable="box")
    return fig


def _result_to_dataframe(sapatas: Sequence[Sapata]) -> pd.DataFrame:
    """Turn the result sapatas into the historical results DataFrame.

    Mirrors the legacy ``Dimensoes_Finais`` sheet so that downstream
    consumers (notebooks, the orientador's spreadsheets) keep working.

    :param sapatas: Sequence of optimised Sapata entities

    :return: DataFrame with columns ``Elemento``, ``h_x (m)``,
             ``h_y (m)``, ``h_z (m)``, ``Volume (m^3)``
    """
    return pd.DataFrame(
        [
            {
                "Elemento": s.pilar.rotulo,
                "h_x (m)": s.h_x, "h_y (m)": s.h_y, "h_z (m)": s.h_z,
                "Volume (m^3)": s.volume,
            }
            for s in sapatas
        ]
    )


def _build_results_xlsx(
    projeto: FundacaoProjeto, result: OptimisationResult
) -> bytes:
    """Build the multi-sheet xlsx report shipped to the user.

    :param projeto: Validated FundacaoProjeto
    :param result: OptimisationResult returned by ``optimize``

    :return: Binary content of the xlsx report
    """
    dados_final = _result_to_dataframe(result.sapatas)
    eval_result = evaluate(projeto, result.sapatas)
    df_verif = pd.DataFrame(
        [
            {"Elemento": rotulo, **constraints}
            for rotulo, constraints in eval_result.constraints.items()
        ]
    )
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        dados_final.to_excel(writer, index=False, sheet_name="Dimensoes_Finais")
        df_verif.to_excel(writer, index=False, sheet_name="Verificacoes_Detalhadas")
    return buf.getvalue()


# =============================================================================
# Localisation
# =============================================================================
def obter_textos() -> Dict[str, Dict[str, str]]:
    """Return the page texts in the supported languages.

    :return: Mapping ``lang -> {key -> text}`` covering both PT and EN
    """
    return {
        "pt": {
            "titulo_pagina": "🏗️ Dimensionamento Otimizado de Sapatas",
            "upload_header": "📥 Upload da planilha de dados",
            "upload_label": "Selecione o arquivo Excel",
            "upload_sucesso": "Arquivo carregado com sucesso!",
            "upload_aviso": "Por favor, selecione um arquivo Excel para continuar.",
            "preview_header": "Primeiras linhas da planilha de dados",
            "params_header": "⚙️ Parâmetros gerais de dimensionamento",
            "fck": "fck do concreto (MPa)",
            "cob": "Cobrimento do concreto (cm)",
            "h_min": "Dimensão mínima da sapata (cm)",
            "h_max": "Dimensão máxima da sapata (cm)",
            "n_gen": "Iterações do EGO por repetição",
            "n_pop": "Tamanho da população (LHS inicial)",
            "n_rep": "Repetições independentes (n_rep)",
            "btn_dimensionar": "🚀 Dimensionar",
            "info_otim": "Otimizando o sistema...",
            "sucesso_otim": "✅ Otimização concluída com sucesso!",
            "resultado_header": "📊 Resultados detalhados",
            "erro_proc": "Erro durante o processamento.",
        },
        "en": {
            "titulo_pagina": "🏗️ Optimized Footing Design",
            "upload_header": "📥 Data spreadsheet upload",
            "upload_label": "Select Excel file",
            "upload_sucesso": "File uploaded successfully!",
            "upload_aviso": "Please select an Excel file to continue.",
            "preview_header": "Data spreadsheet preview",
            "params_header": "⚙️ General design parameters",
            "fck": "Concrete fck (MPa)",
            "cob": "Concrete cover (cm)",
            "h_min": "Minimum footing dimension (cm)",
            "h_max": "Maximum footing dimension (cm)",
            "n_gen": "EGO iterations per repetition",
            "n_pop": "Population size (LHS init)",
            "n_rep": "Independent repetitions (n_rep)",
            "btn_dimensionar": "🚀 Design",
            "info_otim": "Optimizing the system...",
            "sucesso_otim": "✅ Optimization completed successfully!",
            "resultado_header": "📊 Detailed results",
            "erro_proc": "Error during processing.",
        },
    }


# =============================================================================
# Page
# =============================================================================
lang = st.session_state.get("lang", "pt")
t = obter_textos()[lang]

st.title(t["titulo_pagina"])

# Initialise persistent state
if "calculo_realizado" not in st.session_state:
    st.session_state["calculo_realizado"] = False
if "show_history" not in st.session_state:
    st.session_state["show_history"] = False

# --- Inputs --------------------------------------------------------------
st.subheader(t["params_header"])
col1, col2 = st.columns(2)

with col1:
    # n_comb is detected automatically from the spreadsheet columns
    # by ``read_projeto_from_excel`` — exposing it as an editable
    # input historically misled users (changing the field had no
    # effect on the optimisation). Sprint 4.8 removed the input.
    f_ck_mpa = st.number_input(t["fck"], min_value=15.0, max_value=90.0, step=5.0, value=25.0)
    cob_cm = st.number_input(t["cob"], step=0.5, value=4.0, format="%.1f")

with col2:
    h_min_cm = st.number_input(t["h_min"], min_value=60.0, step=0.5, value=60.0)
    h_max_cm = st.number_input(t["h_max"], min_value=60.0, step=0.5, value=150.0)
    n_gen_ui = st.number_input(t["n_gen"], min_value=2, max_value=200, step=1, value=20,
                                help="Cada iteração re-treina o GPR e seleciona o próximo "
                                     "ponto via Expected Improvement.")
    n_pop_ui = st.number_input(t["n_pop"], min_value=10, max_value=2000, step=10, value=100,
                                help="Tamanho da amostragem Latin Hypercube inicial; "
                                     "todas avaliadas com a função objetivo real (iter 0). "
                                     "Valores mais altos dão amostragem inicial mais densa "
                                     "ao custo de tempo de execução — 100 já é uma boa base "
                                     "para problemas com até 10 pilares.")
    n_rep_ui = st.number_input(t["n_rep"], min_value=1, max_value=20, step=1, value=5,
                                help="Número de execuções independentes do EGO com seeds "
                                     "diferentes; o melhor entre todas vence.")

st.divider()

# --- Upload --------------------------------------------------------------
st.subheader(t["upload_header"])
uploaded_file = st.file_uploader(t["upload_label"], type=["xlsx", "xls"])
if uploaded_file is None:
    st.warning(t["upload_aviso"])
    st.stop()

try:
    projeto = read_projeto_from_excel(
        uploaded_file,
        f_ck_kpa=float(f_ck_mpa) * 1000.0,
        cobrimento_m=float(cob_cm) / 100.0,
    )
except (ValueError, FileNotFoundError) as exc:
    st.error(t["erro_proc"])
    st.exception(exc)
    st.stop()

st.success(t["upload_sucesso"])

# Surface auto-detected counts so the user knows what was parsed.
detected_a, detected_b = st.columns(2)
detected_a.metric("Pilares detectados", projeto.n_fund)
detected_b.metric("Combinações detectadas", projeto.n_comb)

st.subheader(t["preview_header"])
uploaded_file.seek(0)
st.dataframe(pd.read_excel(uploaded_file).head(), use_container_width=True)

# --- Optimisation -------------------------------------------------------
def _spawn_optimisation_thread(projeto, config, run_state):
    """Launch ``optimize`` in a daemon thread and wire the progress queue.

    The thread pushes every progress event into ``run_state['queue']``
    and reads ``run_state['cancel_event']`` to honour cancellation
    cooperatively. The result, the cancellation flag or the exception
    is parked into ``run_state['holder']`` so the page rerun can
    finalise the session.
    """
    events_q: queue.Queue = queue.Queue()
    cancel_event = threading.Event()
    holder: dict = {}
    recorder = ExperimentRecorder(root=EXPERIMENTS_ROOT)
    cache = SurrogateCache(maxsize=128)

    def _runner():
        try:
            holder["result"] = optimize(
                projeto, config,
                recorder=recorder, cache=cache,
                progress=lambda ev: events_q.put(ev),
                should_stop=cancel_event.is_set,
            )
        except OptimisationCancelled:
            holder["cancelled"] = True
        except Exception as exc:   # pragma: no cover
            holder["error"] = exc
        finally:
            holder["done"] = True

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    run_state["queue"] = events_q
    run_state["cancel_event"] = cancel_event
    run_state["holder"] = holder
    run_state["thread"] = thread
    run_state["recorder"] = recorder
    run_state["events_seen"] = []   # accumulated for cross-rerun rendering
    run_state["total_units"] = int(config.n_rep) * (int(config.n_gen) + 1)
    run_state["best_of"] = float("inf")
    return run_state


# Trigger button: spawns the thread once.
if st.button(t["btn_dimensionar"], type="primary",
             disabled="run" in st.session_state):
    config = OptimisationConfig(
        h_min_m=float(h_min_cm) / 100.0,
        h_max_m=float(h_max_cm) / 100.0,
        n_gen=int(n_gen_ui),
        n_pop=int(n_pop_ui),
        n_rep=int(n_rep_ui),
        base_seed=42,
        kernel_index=-1,
        ga_epoch=30,
        ga_pop_size=50,
        penalty=None,
    )
    st.session_state["run"] = _spawn_optimisation_thread(
        projeto, config, run_state={"config": config}
    )
    st.session_state["calculo_realizado"] = False
    st.rerun()


def _render_progress(run_state: dict) -> None:
    """Render the live progress + cancel UI from accumulated events.

    Reads every event delivered to the queue so far, derives the
    pipeline state (current rep, current iter, running best OF) and
    paints the widgets. The caller is responsible for triggering
    ``st.rerun()`` (or letting the user click the cancel button) so
    progress keeps refreshing while the thread runs.
    """
    events_q: queue.Queue = run_state["queue"]
    cancel_event: threading.Event = run_state["cancel_event"]
    holder: dict = run_state["holder"]

    # Drain the queue into events_seen so a partial render history
    # survives reruns even if Streamlit is faster than the thread.
    while True:
        try:
            ev = events_q.get_nowait()
            run_state["events_seen"].append(ev)
            if ev.get("event") in ("ego.iter", "lhs.done") and "of_min" in ev:
                run_state["best_of"] = min(run_state["best_of"], float(ev["of_min"]))
        except queue.Empty:
            break

    seen = run_state["events_seen"]
    total_units = run_state["total_units"]
    units_done = sum(
        1 for e in seen
        if e.get("event") in ("lhs.done", "ego.iter")
    )
    pct = min(units_done / max(total_units, 1), 1.0)
    n_rep = run_state["config"].n_rep
    n_gen = run_state["config"].n_gen

    last = seen[-1] if seen else {}
    last_kind = last.get("event")

    # Header text driven by the latest event.
    if cancel_event.is_set():
        headline = "⏹️ Cancelando — aguarde o fim do trecho atual..."
    elif last_kind == "lhs.start":
        n_pop = last.get("n_pop", "?")
        rep = last.get("rep", 0) + 1
        headline = (f"📐 Rep {rep}/{n_rep} — amostrando LHS "
                    f"({n_pop} avaliações reais)...")
    elif last_kind == "lhs.eval":
        n = last.get("n", "?"); n_pop = last.get("n_pop", "?")
        rep = last.get("rep", 0) + 1
        headline = (f"📐 Rep {rep}/{n_rep} — LHS {n}/{n_pop}")
    elif last_kind == "lhs.done":
        rep = last.get("rep", 0) + 1
        of_min = last.get("of_min", float("nan"))
        headline = (f"📐 Rep {rep}/{n_rep} — LHS pronta "
                    f"(melhor OF inicial: `{of_min:.4f} m³`)")
    elif last_kind == "ego.iter":
        rep = last.get("rep", 0) + 1
        it = last.get("iter", 0)
        of_min = last.get("of_min", float("nan"))
        headline = (
            f"🧠 Rep {rep}/{n_rep} · iter {it}/{n_gen} — "
            f"re-treinando GPR + maximizando EI + avaliando candidato · "
            f"melhor OF da rep: `{of_min:.6f} m³`"
        )
    elif last_kind == "optimize.recording":
        rep = last.get("rep", 0) + 1
        headline = f"💾 Rep {rep}/{n_rep} — gravando histórico em disco..."
    elif last_kind == "optimize.rep_end":
        headline = "✅ Rep concluída — preparando próxima..."
    elif last_kind == "optimize.end":
        headline = (f"✅ Concluído — best OF: `{last.get('best_of', float('nan')):.4f} m³`")
    elif last_kind == "optimize.cancelled":
        headline = "⏹️ Otimização cancelada"
    elif last_kind == "optimize.failed":
        headline = f"❌ Falha: `{last.get('error', '?')}`"
    else:
        headline = "⏳ Iniciando..."

    progress_label = (
        f"Repetição {min(_seen_rep_index(seen) + 1, n_rep)}/{n_rep} · "
        f"melhor OF até agora: "
        f"{(run_state['best_of'] if run_state['best_of'] != float('inf') else '—')}"
        if run_state["best_of"] != float("inf")
        else f"Repetição {min(_seen_rep_index(seen) + 1, n_rep)}/{n_rep}"
    )

    st.progress(pct, text=progress_label)

    with st.status("Otimização em andamento", expanded=True,
                   state=("running" if not holder.get("done")
                          else ("error"
                                if "error" in holder or "cancelled" in holder
                                else "complete"))):
        st.markdown(headline)
        st.caption(
            f"Pipeline: cada rep faz **{run_state['config'].n_pop}** avaliações "
            f"reais (LHS, iter 0) + **{n_gen}** iterações do EGO; "
            f"em cada iteração o **GPR é re-treinado**, o ponto que maximiza "
            f"a função de aquisição (EI) é escolhido por um GA interno e "
            f"avaliado de verdade."
        )

    # Cancel control. While the thread is alive the button is the
    # cooperative cancel; once cancellation has been requested it shows
    # a hint that the optimiser will exit at the next safe point.
    if not holder.get("done"):
        if cancel_event.is_set():
            st.warning(
                "Cancelamento solicitado — a thread irá interromper na "
                "próxima iteração ou avaliação LHS."
            )
        else:
            if st.button("⏹️ Parar dimensionamento", key="cancel_btn"):
                cancel_event.set()
                st.rerun()


def _seen_rep_index(seen) -> int:
    """Largest ``rep`` value seen in any event; -1 when nothing yet."""
    rep = -1
    for e in seen:
        r = e.get("rep")
        if isinstance(r, int):
            rep = max(rep, r)
    return rep


# Live render block. Activates whenever a run is registered. Re-renders
# every POLL_INTERVAL_S until the runner thread completes.
if "run" in st.session_state:
    run_state = st.session_state["run"]
    _render_progress(run_state)

    holder = run_state["holder"]
    if not holder.get("done"):
        time.sleep(POLL_INTERVAL_S)
        st.rerun()

    # Thread finished — finalise.
    if "result" in holder:
        result = holder["result"]
        recorder = run_state["recorder"]
        st.session_state["projeto"] = projeto
        st.session_state["result"] = result
        st.session_state["dados_final_df"] = _result_to_dataframe(result.sapatas)
        st.session_state["best_of_valor"] = result.best_of
        st.session_state["excel_buffer"] = _build_results_xlsx(projeto, result)
        st.session_state["calculo_realizado"] = True
        st.session_state["run_dir"] = str(recorder.run_dir)
        st.session_state["run_id"] = recorder.run_id
        st.session_state["show_history"] = False
        st.success(t["sucesso_otim"])
    elif "cancelled" in holder:
        st.warning("⏹️ Otimização cancelada pelo usuário.")
    elif "error" in holder:
        st.error(t["erro_proc"])
        st.exception(holder["error"])

    # One-shot cleanup so the next "Dimensionar" click starts fresh.
    del st.session_state["run"]
    st.rerun()

# --- Results -------------------------------------------------------------
if st.session_state.get("calculo_realizado"):
    st.divider()
    st.subheader(t["resultado_header"])

    result_state: OptimisationResult = st.session_state["result"]

    # KPI strip ------------------------------------------------------------
    kpi_a, kpi_b, kpi_c, kpi_d = st.columns(4)
    kpi_a.metric("Volume total", f"{st.session_state['best_of_valor']:.4f} m³")
    kpi_b.metric("Sapatas", f"{len(result_state.sapatas)}")
    kpi_c.metric("Repetições (n_rep)", f"{len(result_state.per_rep_of)}")
    if result_state.per_rep_of:
        spread = max(result_state.per_rep_of) - min(result_state.per_rep_of)
        kpi_d.metric("Spread entre reps", f"{spread:.4f} m³")
    run_id = st.session_state.get("run_id")
    if run_id:
        st.markdown(
            f"<span class='fundaia-chip fundaia-chip--accent'>run {run_id}</span>",
            unsafe_allow_html=True,
        )

    st.write("")

    # ── Section 1 — Tabela + planta 2D lado a lado (compactos, 2D não
    #                precisa de muito espaço; tabela é a referência primária).
    table_col, plan2d_col = st.columns([3, 4])
    with table_col:
        st.markdown("##### 📋 Dimensões finais")
        st.dataframe(
            st.session_state["dados_final_df"],
            use_container_width=True, hide_index=True,
        )
        st.download_button(
            label="📥 Resultados (Excel)",
            data=st.session_state["excel_buffer"],
            file_name="otimizacao_fundacao.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with plan2d_col:
        st.markdown("##### 🗺️ Planta 2D")
        fig2d = _plot_layout(result_state.sapatas)
        st.pyplot(fig2d, use_container_width=True)

    st.divider()

    # ── Section 2 — Vista 3D (full-width, espaço próprio).
    st.markdown("### 🧊 Vista 3D do arranjo")
    st.caption(
        "Arraste para rotacionar · Roda do mouse para zoom · "
        "Duplo-clique reseta a câmera"
    )

    ctrl3d, scene3d = st.columns([1, 5], gap="large")
    with ctrl3d:
        st.markdown("**Visualização**")
        show_pillars = st.checkbox("Pilares", value=True)
        show_ground = st.checkbox("Terreno", value=True)
        colour_by = st.radio(
            "Cor das sapatas",
            options=["label", "volume"], index=0,
            format_func=lambda x: "por elemento" if x == "label" else "por volume",
        )
        st.markdown("**Câmera (ponto de partida)**")
        camera_preset = st.selectbox(
            "Preset",
            options=list(CAMERA_PRESETS.keys()),
            index=list(CAMERA_PRESETS.keys()).index("isométrica"),
            label_visibility="collapsed",
            help=(
                "Posição inicial da câmera. Use o mouse direto no "
                "gráfico para rotacionar livremente em torno do eixo "
                "vertical (Z fica sempre 'pra cima')."
            ),
        )
        st.markdown("**Geometria**")
        pillar_height = st.slider(
            "Altura visual do pilar (m)",
            min_value=0.5, max_value=4.0, value=1.5, step=0.1,
        )
        terrain_margin = st.slider(
            "Margem do terreno (m)",
            min_value=0.5, max_value=10.0, value=1.5, step=0.5,
        )
    with scene3d:
        fig3d = render_footings_3d(
            result_state.sapatas,
            show_pillars=show_pillars,
            show_ground=show_ground,
            pillar_height_m=pillar_height,
            colour_by=colour_by,
            camera=camera_preset,
            terrain_margin_m=terrain_margin,
            height=760,
        )
        st.plotly_chart(
            fig3d,
            use_container_width=True,
            config={
                "displaylogo": False, "responsive": True,
                "scrollZoom": True,   # roda do mouse = zoom direto
                "modeBarButtonsToRemove": ["resetCameraLastSave3d"],
            },
        )

    st.divider()

    # ── Section 3 — Histórico do EGO (full-width, dois subgráficos
    #               com hover por trace; scrollZoom permite aproximar
    #               regiões da curva sem perder a visão geral).
    st.markdown("### 📈 Histórico do EGO")
    hist_btn_col, hist_log_col = st.columns([1, 5])
    with hist_btn_col:
        if st.button(
            "Mostrar histórico" if not st.session_state["show_history"]
            else "Ocultar histórico",
            type="secondary",
        ):
            st.session_state["show_history"] = not st.session_state["show_history"]

    fig_history = None
    if st.session_state.get("show_history") and st.session_state.get("run_dir"):
        try:
            run = load_experiment(st.session_state["run_dir"])
            with hist_log_col:
                log_y = st.toggle(
                    "Eixo OF em escala logarítmica",
                    value=False, key="hist_log_y",
                )
            st.caption(
                "Arraste no gráfico para dar zoom em uma faixa · "
                "Duplo-clique reseta · Use a barra de ferramentas no "
                "canto superior direito (pan, zoom, autoescala, "
                "exportar PNG)."
            )
            fig_history = render_ego_history(
                run.history,
                metrics=run.manifest.metrics,
                title=None,
                log_y=log_y,
            )
            st.plotly_chart(
                fig_history,
                use_container_width=True,
                config={
                    "displaylogo": False, "responsive": True,
                    "scrollZoom": True,
                    "doubleClick": "reset",
                },
            )
            with st.expander("Resumo por repetição"):
                if run.manifest.summary:
                    st.dataframe(
                        pd.DataFrame(run.manifest.summary),
                        use_container_width=True, hide_index=True,
                    )
        except Exception as exc:   # pragma: no cover
            st.warning("Não foi possível carregar o histórico desta run.")
            st.exception(exc)

    # --- Unified export panel ------------------------------------------
    st.divider()
    st.subheader("📦 Exportar")
    try:
        artifacts = build_export_artifacts(
            result_state,
            fig_3d=fig3d,
            fig_history=fig_history,
            metrics=(
                load_experiment(st.session_state["run_dir"]).manifest.metrics
                if st.session_state.get("run_dir") else None
            ),
            run_id=run_id,
        )
        ex_a, ex_b, ex_c, ex_d, ex_e = st.columns(5)
        with ex_a:
            st.download_button(
                "📐 DXF (CAD)", data=artifacts["dxf"],
                file_name="arranjo_sapatas.dxf",
                mime="application/dxf",
            )
        with ex_b:
            st.download_button(
                "🧾 JSON", data=artifacts["json"],
                file_name=f"resultado_{run_id or 'fundaia'}.json",
                mime="application/json",
            )
        with ex_c:
            if "html_3d" in artifacts:
                st.download_button(
                    "🧊 HTML 3D", data=artifacts["html_3d"],
                    file_name=f"viewer_3d_{run_id or 'fundaia'}.html",
                    mime="text/html",
                )
        with ex_d:
            if "html_history" in artifacts:
                st.download_button(
                    "📈 HTML histórico", data=artifacts["html_history"],
                    file_name=f"ego_history_{run_id or 'fundaia'}.html",
                    mime="text/html",
                )
        with ex_e:
            if "png_history" in artifacts:
                st.download_button(
                    "🖼️ PNG histórico", data=artifacts["png_history"],
                    file_name=f"ego_history_{run_id or 'fundaia'}.png",
                    mime="image/png",
                )
    except Exception as exc:   # pragma: no cover
        st.warning("Não foi possível montar o painel de exportação.")
        st.exception(exc)
