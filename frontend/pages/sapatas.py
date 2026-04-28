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

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from core.api import OptimisationConfig, OptimisationResult, evaluate, optimize
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
    n_pop_ui = st.number_input(t["n_pop"], min_value=10, max_value=2000, step=10, value=250,
                                help="Tamanho da amostragem Latin Hypercube inicial; "
                                     "todas avaliadas com a função objetivo real (iter 0).")
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
if st.button(t["btn_dimensionar"], type="primary"):
    try:
        config = OptimisationConfig(
            h_min_m=float(h_min_cm) / 100.0,
            h_max_m=float(h_max_cm) / 100.0,
            n_gen=int(n_gen_ui),
            n_pop=int(n_pop_ui),
            n_rep=int(n_rep_ui),
            base_seed=42,
            kernel_index=-1,
            ga_epoch=50,
            ga_pop_size=150,
            penalty=None,
        )

        # ── Live progress UI: a status block (collapses on completion)
        #     plus a deterministic progress bar. Every milestone of the
        #     pipeline updates these widgets through the `progress=`
        #     callback wired below. The total work unit is
        #     n_rep * n_gen ego iterations; each rep_start/end shifts
        #     the pointer. The displayed status carries the running
        #     best OF so the user has immediate signal.
        total_units = int(config.n_rep) * int(config.n_gen)
        progress_bar = st.progress(0, text="Preparando...")
        status_box = st.status("⏳ Otimização em andamento...",
                               state="running", expanded=True)
        info_line = status_box.empty()
        sub_line = status_box.empty()

        progress_state = {"unit": 0, "best": float("inf"),
                          "rep": 0, "iter": 0}

        def _on_progress(ev: dict) -> None:
            """Translate optimisation events into Streamlit widgets.

            Streamlit renders mid-script updates synchronously, so
            calling ``progress_bar.progress(...)`` and
            ``info_line.write(...)`` from here is enough to surface
            the percentage and the running best OF.
            """
            kind = ev.get("event")
            if kind == "optimize.start":
                info_line.markdown(
                    f"**Configuração:** "
                    f"`n_rep={ev['n_rep']}` · `n_gen={ev['n_gen']}` · "
                    f"`n_pop={ev['n_pop']}` · `n_fund={ev['n_fund']}`"
                )
                sub_line.markdown(
                    "Iniciando população inicial via Latin Hypercube..."
                )
                progress_bar.progress(0, text="Iniciando...")
                return

            if kind == "optimize.rep_start":
                rep = ev["rep"]; seed = ev["seed"]
                progress_state["rep"] = rep
                progress_state["iter"] = 0
                sub_line.markdown(
                    f"🔁 **Repetição {rep + 1}/{ev['n_rep']}** "
                    f"(seed `{seed}`) — amostrando LHS e treinando GPR..."
                )
                return

            if kind == "ego.iter":
                rep = ev.get("rep", progress_state["rep"])
                it = ev["iter"]
                n_gen = ev["n_gen"]
                of_min = ev["of_min"]
                progress_state["unit"] += 1
                progress_state["best"] = min(progress_state["best"], of_min)
                pct = min(progress_state["unit"] / max(total_units, 1), 1.0)
                running_best = progress_state["best"]
                progress_bar.progress(
                    pct,
                    text=(f"Repetição {rep + 1}/{ev.get('n_rep', '?')} · "
                          f"iter {it}/{n_gen} · "
                          f"melhor OF até agora: {running_best:.4f} m³"),
                )
                sub_line.markdown(
                    f"🧠 Treinando GPR · iteração `{it}/{n_gen}` · "
                    f"OF da rep: `{of_min:.6f} m³`"
                )
                return

            if kind == "optimize.rep_end":
                rep = ev["rep"]
                of_rep = ev["of_rep"]
                wall = ev["wall_time_s"]
                sub_line.markdown(
                    f"✅ Repetição {rep + 1} concluída — "
                    f"OF: `{of_rep:.6f} m³` · "
                    f"tempo: `{wall:.2f} s`"
                )
                return

            if kind == "optimize.end":
                progress_bar.progress(
                    1.0,
                    text=f"✅ Concluído · best OF: {ev['best_of']:.4f} m³",
                )
                return

            if kind == "optimize.failed":
                sub_line.markdown(f"❌ Falhou: `{ev['error']}`")
                return

        try:
            # Recorder switched on by default (writes under
            # experiments/<run_id>/) so the "Ver histórico" button has
            # the EGO history available immediately after this call.
            recorder = ExperimentRecorder(root=EXPERIMENTS_ROOT)
            cache = SurrogateCache(maxsize=128)
            result: OptimisationResult = optimize(
                projeto, config,
                recorder=recorder, cache=cache,
                progress=_on_progress,
            )
        except Exception:
            status_box.update(label="❌ Otimização interrompida",
                              state="error", expanded=True)
            raise

        status_box.update(
            label=f"✅ Otimização concluída em {len(result.per_rep_of)} repetições",
            state="complete", expanded=False,
        )

        st.session_state["projeto"] = projeto
        st.session_state["result"] = result
        st.session_state["dados_final_df"] = _result_to_dataframe(result.sapatas)
        st.session_state["best_of_valor"] = result.best_of
        st.session_state["excel_buffer"] = _build_results_xlsx(projeto, result)
        st.session_state["calculo_realizado"] = True
        st.session_state["run_dir"] = str(recorder.run_dir)
        st.session_state["run_id"] = recorder.run_id
        st.session_state["show_history"] = False   # collapsed by default

        st.success(t["sucesso_otim"])
        st.rerun()
    except Exception as exc:   # pragma: no cover
        st.error(t["erro_proc"])
        st.exception(exc)

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
        st.markdown("**Câmera**")
        camera_preset = st.selectbox(
            "Preset",
            options=list(CAMERA_PRESETS.keys()),
            index=list(CAMERA_PRESETS.keys()).index("isométrica"),
            label_visibility="collapsed",
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
