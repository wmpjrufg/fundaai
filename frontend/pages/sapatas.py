"""Streamlit page — optimised footing design (thin shell over ``core.api``).

This module is intentionally a thin presentation layer. All engineering
and optimisation logic lives in ``core/``: ``core.io.read_projeto_from_excel``
parses the upload, ``core.api.optimize`` orchestrates the EGO+GPR+GA
pipeline and ``core.io.sapatas_to_dxf_bytes`` produces the CAD export.
The page itself only handles widgets, session state and rendering.

Resumo em português:
    Página Streamlit refatorada como camada fina sobre ``core.api``.
    Toda a lógica de engenharia e otimização está em ``core/``; aqui
    cuidamos apenas de widgets, estado da sessão e renderização.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, Sequence

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from core.api import OptimisationConfig, OptimisationResult, evaluate, optimize
from core.domain import FundacaoProjeto, Sapata
from core.io import read_projeto_from_excel, sapatas_to_dxf_bytes
from frontend.components import render_footings_3d


# =============================================================================
# Plot helpers (Streamlit-specific; live here to keep core layers free of UI)
# =============================================================================
def _plot_layout(sapatas: Sequence[Sapata]):
    """This function renders the in-plane layout of the optimised footings.

    :param sapatas: Sequence of optimised Sapata entities

    :return: Matplotlib figure ready to be passed to ``st.pyplot``
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    for sapata in sapatas:
        v_sw, _, _, _ = sapata.vertices
        ax.add_patch(
            patches.Rectangle(
                v_sw,
                sapata.h_x,
                sapata.h_y,
                linewidth=1,
                edgecolor="blue",
                facecolor="none",
            )
        )
        ax.scatter(
            sapata.pilar.xg,
            sapata.pilar.yg,
            color="red",
            marker="+",
            s=100,
        )
        ax.annotate(
            sapata.pilar.rotulo,
            (sapata.pilar.xg, sapata.pilar.yg),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Posicionamento das sapatas")
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")
    return fig


def _result_to_dataframe(sapatas: Sequence[Sapata]) -> pd.DataFrame:
    """This function turns the result sapatas into the historical results DataFrame.

    Mirrors the legacy ``Dimensoes_Finais`` sheet so that downstream
    consumers (notebooks, the orientador's spreadsheets) keep working.

    :param sapatas: Sequence of optimised Sapata entities

    :return: DataFrame with columns ``h_x (m)``, ``h_y (m)``, ``h_z (m)``
    """
    return pd.DataFrame(
        [{"h_x (m)": s.h_x, "h_y (m)": s.h_y, "h_z (m)": s.h_z} for s in sapatas]
    )


def _build_results_xlsx(
    projeto: FundacaoProjeto, result: OptimisationResult
) -> bytes:
    """This function builds the multi-sheet xlsx report shipped to the user.

    Sheet ``Dimensoes_Finais`` carries the optimised footing dimensions.
    Sheet ``Verificacoes_Detalhadas`` carries the per-element constraint
    table produced by ``evaluate``.

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
    """This function returns the page texts in the supported languages.

    :return: Mapping ``lang -> {key -> text}`` covering both PT and EN
    """
    return {
        "pt": {
            "titulo_pagina": "🏗️ Dimensionamento Otimizado de Sapatas",
            "upload_header": "Upload da planilha de dados",
            "upload_label": "Selecione o arquivo Excel",
            "upload_sucesso": "Arquivo carregado com sucesso!",
            "upload_aviso": "Por favor, selecione um arquivo Excel para continuar.",
            "preview_header": "Primeiras linhas da planilha de dados",
            "params_header": "Parâmetros gerais de dimensionamento",
            "n_comb": "Número de combinações",
            "fck": "fck do concreto (MPa)",
            "cob": "Cobrimento do concreto (cm)",
            "h_min": "Dimensão mínima da sapata (cm)",
            "h_max": "Dimensão máxima da sapata (cm)",
            "n_gen": "Número de gerações da otimização",
            "n_pop": "Tamanho da população",
            "btn_dimensionar": "Dimensionar",
            "info_otim": "Otimizando o sistema...",
            "sucesso_otim": "✅ Otimização concluída com sucesso!",
            "resultado_header": "📊 Resultados Detalhados",
            "erro_proc": "Erro durante o processamento.",
        },
        "en": {
            "titulo_pagina": "🏗️ Optimized Footing Design",
            "upload_header": "Data Spreadsheet Upload",
            "upload_label": "Select Excel file",
            "upload_sucesso": "File uploaded successfully!",
            "upload_aviso": "Please select an Excel file to continue.",
            "preview_header": "Data spreadsheet preview",
            "params_header": "General design parameters",
            "n_comb": "Number of combinations",
            "fck": "Concrete fck (MPa)",
            "cob": "Concrete cover (cm)",
            "h_min": "Minimum footing dimension (cm)",
            "h_max": "Maximum footing dimension (cm)",
            "n_gen": "Number of optimization generations",
            "n_pop": "Population size",
            "btn_dimensionar": "Design",
            "info_otim": "Optimizing the system...",
            "sucesso_otim": "✅ Optimization completed successfully!",
            "resultado_header": "📊 Detailed Results",
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

# --- Inputs --------------------------------------------------------------
st.subheader(t["params_header"])
col1, col2 = st.columns(2)

with col1:
    n_comb_ui = st.number_input(t["n_comb"], step=1, value=3, key="n_comb_input")
    f_ck_mpa = st.number_input(t["fck"], min_value=15.0, max_value=90.0, step=5.0, value=25.0)
    cob_cm = st.number_input(t["cob"], step=0.5, value=4.0, format="%.1f")

with col2:
    h_min_cm = st.number_input(t["h_min"], min_value=60.0, step=0.5, value=60.0)
    h_max_cm = st.number_input(t["h_max"], min_value=60.0, step=0.5, value=150.0)
    n_gen_ui = st.number_input(t["n_gen"], min_value=2, max_value=200, step=1, value=2)
    n_pop_ui = st.number_input(t["n_pop"], min_value=200, max_value=2000, step=5, value=250)

st.divider()

# --- Upload --------------------------------------------------------------
st.subheader(t["upload_header"])
uploaded_file = st.file_uploader(t["upload_label"], type=["xlsx", "xls"])
if uploaded_file is None:
    st.warning(t["upload_aviso"])
    st.stop()

# Read the project from the spreadsheet through the IO layer.
# Strict schema validation lives there; user-facing errors bubble up
# unchanged so the orientador sees the exact problem.
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
st.subheader(t["preview_header"])
# The preview shows the raw spreadsheet contents (uploaded_file was already
# consumed; pandas re-reads it for display only).
uploaded_file.seek(0)
st.dataframe(pd.read_excel(uploaded_file).head())

# --- Optimisation -------------------------------------------------------
if st.button(t["btn_dimensionar"], type="primary"):
    try:
        with st.spinner(t["info_otim"]):
            config = OptimisationConfig(
                h_min_m=float(h_min_cm) / 100.0,
                h_max_m=float(h_max_cm) / 100.0,
                n_gen=int(n_gen_ui),
                n_pop=int(n_pop_ui),
                n_rep=5,
                base_seed=42,
                kernel_index=-1,
                ga_epoch=50,
                ga_pop_size=150,
                penalty=None,
            )
            result: OptimisationResult = optimize(projeto, config)

            st.session_state["projeto"] = projeto
            st.session_state["result"] = result
            st.session_state["dados_final_df"] = _result_to_dataframe(result.sapatas)
            st.session_state["best_of_valor"] = result.best_of
            st.session_state["excel_buffer"] = _build_results_xlsx(projeto, result)
            st.session_state["calculo_realizado"] = True

            st.success(t["sucesso_otim"])
            st.rerun()
    except Exception as exc:   # pragma: no cover
        st.error(t["erro_proc"])
        st.exception(exc)

# --- Results -------------------------------------------------------------
if st.session_state.get("calculo_realizado"):
    st.divider()
    st.subheader(t["resultado_header"])

    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.dataframe(st.session_state["dados_final_df"], use_container_width=True)

    with col_b:
        st.metric("Volume Total", f"{st.session_state['best_of_valor']:.4f} m³")

        st.download_button(
            label="📥 Baixar Resultados (Excel)",
            data=st.session_state["excel_buffer"],
            file_name="otimizacao_fundacao.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.divider()
        st.subheader("🗺️ Arranjo das Sapatas")

        try:
            result_state: OptimisationResult = st.session_state["result"]
            tab_2d, tab_3d = st.tabs(["🗺️ Planta 2D", "🧊 Vista 3D"])
            with tab_2d:
                fig = _plot_layout(result_state.sapatas)
                st.pyplot(fig, use_container_width=True)
            with tab_3d:
                viewer_cols = st.columns([3, 1])
                with viewer_cols[1]:
                    show_pillars = st.checkbox("Exibir pilares", value=True)
                    show_ground = st.checkbox("Exibir plano de solo", value=True)
                    colour_by = st.radio(
                        "Cor das sapatas",
                        options=["label", "volume"],
                        index=0,
                        format_func=lambda x: "por elemento" if x == "label" else "por volume",
                        horizontal=False,
                    )
                    pillar_height = st.slider(
                        "Altura visual do pilar (m)",
                        min_value=0.5, max_value=4.0, value=1.5, step=0.1,
                    )
                with viewer_cols[0]:
                    fig3d = render_footings_3d(
                        result_state.sapatas,
                        show_pillars=show_pillars,
                        show_ground=show_ground,
                        pillar_height_m=pillar_height,
                        colour_by=colour_by,
                    )
                    st.plotly_chart(fig3d, use_container_width=True)

            dxf_bytes = sapatas_to_dxf_bytes(result_state.sapatas)
            st.download_button(
                label="📥 Baixar Arranjo (DXF)",
                data=dxf_bytes,
                file_name="arranjo_sapatas.dxf",
                mime="application/dxf",
            )
        except Exception as exc:   # pragma: no cover
            st.warning("Não foi possível gerar a plotagem/arquivo DXF com os dados atuais.")
            st.exception(exc)
