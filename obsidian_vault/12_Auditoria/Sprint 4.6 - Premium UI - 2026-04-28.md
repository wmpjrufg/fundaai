---
tags: [refactor, sprint, log, frontend, ui, dark, plotly, streamlit, premium]
data: 2026-04-28
branch: refactor/core-architecture
escopo: Sprint 4.6 — Premium UI (dark + EGO chart + 3D polido + exports)
---

# Sprint 4.6 — Premium UI

> Acabamento profissional do front-end **sem framework JS**: tema dark
> nativo do Streamlit + CSS injetado + Plotly tematizado, gráfico
> "premium" do histórico do EGO consumindo o `ExperimentRecorder`,
> visualizador 3D polido (presets de câmera, lighting, terreno com
> grade e contorno) e bloco unificado de exportação (DXF, JSON, HTML
> 3D, PNG do histórico).

## TL;DR

> A página de dimensionamento agora é **dark**, com KPIs visuais,
> botão **"📈 Ver histórico do EGO"** que abre um gráfico
> interativo com curvas por repetição, banda min/max, mediana,
> tempo por iteração, e um painel **📦 Exportar** unificado que
> entrega DXF, JSON, HTML 3D e PNG do histórico em uma linha de
> botões.

## Para leigos

A FundaIA tinha funcionalidade de sobra; faltava acabamento.
Agora a interface está em **modo escuro de verdade** (não preto
puro — um azul-petróleo profundo, com accent âmbar nos elementos
ativos), os botões e cards têm cantos arredondados, foco visível
nos campos, e cada tela respira melhor. Quando você termina uma
otimização aparece:

- **Tira de KPIs** no topo: volume total, número de sapatas,
  número de repetições e dispersão entre reps;
- **Resultado em tabela** + **viewer 3D interativo** lado a lado,
  com presets de câmera (isométrica / topo / lateral X / lateral Y
  / perspectiva), grid em metros no plano de solo, contorno do
  terreno em âmbar e iluminação realista;
- Botão **"📈 Ver histórico do EGO"**: abre um gráfico de duas
  faixas — em cima a curva do melhor valor encontrado por
  iteração (uma linha por repetição, com banda mostrando o pior
  e o melhor, mais a mediana destacada em âmbar), embaixo o
  tempo gasto por iteração;
- Painel **📦 Exportar** com DXF (CAD), JSON (resumo estruturado),
  HTML 3D (vista 3D em arquivo único, abre em qualquer browser
  sem internet) e PNG do gráfico de histórico (para artigo).

> **Por que sem framework JS?** Streamlit + Plotly cobrem
> 100% do que a IC precisa sem custo de migração, sem build
> step e sem dependência nova além de Plotly (já adicionado na
> Sprint 4.5). Se um dia o projeto precisar de uma página tipo
> SPA, dá pra evoluir; **agora não compensaria**.

## Para o time técnico

### Tema (`.streamlit/config.toml` + `frontend/theme/`)

Paleta dark com accent quente, espelhada em três camadas:

```
.streamlit/config.toml         (tema nativo do Streamlit: cores básicas)
frontend/theme/palette.py      (PALETTE dict + Plotly Template "fundaia_dark")
frontend/theme/css.py          (apply_theme() injeta CSS para os 10% que
                                config.toml não cobre — cards, tabs,
                                buttons, focus rings, modebar do Plotly)
frontend/theme/__init__.py     (re-exports)
```

A função `apply_theme()` é chamada no topo de cada page (`home.py` e
`sapatas.py`); idempotente e tolerante a contextos sem Streamlit
(retorna sem erro em testes).

### Componente: `frontend/components/ego_chart.py`

`render_ego_history(histories, *, metrics, title, show_evaluations, log_y)`:

- Aceita `Mapping[int, DataFrame]` (o que `ExperimentRun.history`
  devolve) **ou** iterável de DataFrames.
- Constrói uma figura Plotly `make_subplots(2, 1, shared_xaxes=True)`:
  - **Linha 1**: banda min/max entre reps (preenchimento âmbar
    translúcido), uma linha por rep, **mediana** em âmbar grosso,
    e marcadores dos pontos avaliados (toggle).
  - **Linha 2**: barras de **tempo por iteração** quando o
    histórico tem `TIME CONSUMPTION (s)`.
- Anotação superior com `best_of`, `mean_convergence_iter` e
  `mean_auc_best_so_far` (via `metrics`).
- `log_y=True` → eixo OF em log (toggle na UI).
- Reamostra cada curva no grid `[0, max_iter]` com forward-fill
  para a banda/mediana ficarem coerentes mesmo quando reps
  convergem em momentos diferentes.

### Componente: `frontend/components/result_export.py`

`build_export_artifacts(result, *, fig_3d, fig_history, metrics, run_id)`
devolve um dict `{name: bytes}` com:

| Chave           | Conteúdo                                                    |
|-----------------|-------------------------------------------------------------|
| `dxf`           | `core.io.sapatas_to_dxf_bytes(...)` — CAD-ready              |
| `json`          | `result_to_json_bytes` — estrutura completa + metrics + run_id |
| `html_3d`       | `figure_to_html_bytes(fig_3d)` — viewer 3D stand-alone       |
| `html_history`  | `figure_to_html_bytes(fig_history)` — gráfico stand-alone    |
| `png_history`   | `figure_to_png_bytes(fig_history)` — quando `kaleido` instalado |

Cada artifact é exposto como `st.download_button` numa linha de 5
colunas.

### 3D viewer: refinos

- **Lighting** + `lightposition` (Plotly suporta luz direcional
  em `Mesh3d`) — sapatas e pilares com sombreamento sutil.
- **Terreno** agora composto por:
  - retângulo translúcido na cor de superfície do tema;
  - **grid** em metros (Scatter3d em modo "lines" com `None` para
    quebrar segmentos), espaçamento adaptativo (~10 linhas no
    lado maior);
  - **contorno** em âmbar (`PALETTE["accent"]`) destacando o
    bounding box.
- **`CAMERA_PRESETS`** (`isométrica`, `topo`, `lateral X`,
  `lateral Y`, `perspectiva`) escolhidos via `st.selectbox`.
- **`terrain_margin_m`** ajustável (slider).
- O retângulo do terreno é o **bounding box dos pilares + margem**
  (não mexemos na planilha; a margem é parâmetro de UI).

### Wire-up no `frontend/pages/sapatas.py`

```
[KPIs: Volume | n_sapatas | n_rep | spread]   <chip run-id>

[Tabela | Viewer 3D — abas Planta 2D / Vista 3D]
                       └ controles à esquerda do viewer

[ 📈 Ver histórico do EGO ]   ← toggle
   └ ao clicar, expande:
     [ Plot Plotly: best-so-far + tempo por iter ]
     [ Resumo por repetição (expander) ]

[ 📦 Exportar ]
   [DXF] [JSON] [HTML 3D] [HTML hist] [PNG hist]
```

O **`ExperimentRecorder` ficou ligado por padrão** quando o usuário
roda pela UI — escreve em `experiments/<run_id>/`. É isso que
viabiliza o botão "Ver histórico" funcionar imediatamente sem
estado em memória adicional. O `SurrogateCache` também é instanciado
por default (acelera n_rep).

### Notas de compatibilidade

- O 2D plot matplotlib continua na primeira aba; cores ajustadas
  para combinar com o dark theme (laranja sobre azul-petróleo).
- O bloco de download Excel original continua presente.
- O export DXF que existia antes agora vive no painel unificado
  (`📐 DXF (CAD)` na primeira coluna); URL e mime preservados.

## Validação

```text
=== suite ===
  205 passed in ~7 s
    test_theme.py             5  (novo)
    test_ego_chart.py         8  (novo)
    test_result_export.py     7  (novo)
    test_components_3d.py    15  (12 anteriores + 3 novos: presets,
                                  unknown camera, terrain margin)

=== contratos travados ===
  - PALETTE declara todas as chaves usadas pelo CSS.
  - Template Plotly "fundaia_dark" registrado no pio.templates.
  - apply_theme() seguro fora do Streamlit.
  - render_ego_history(): banda + linhas + mediana + barras de tempo;
    log_y flipa o eixo; metrics annotation atachada quando dada.
  - best_so_far_curves(): não-decrescente, começa no LHS min.
  - build_export_artifacts(): dxf+json sempre presentes, html_3d/
    html_history só quando fig é dada, png só quando kaleido.
  - 3D viewer: presets de câmera funcionam, unknown levanta,
    terrain_margin propaga.
```

## Pendências relacionadas

- `frontend/i18n/` ainda vazio. Próxima micro-sprint: mover
  `titulos_nav` (de `app.py`) e os labels de
  `frontend/pages/sapatas.py` para dicionários PT/EN
  centralizados.
- Cor das sapatas por **status de restrição** (verde
  factível / vermelho violação) consumindo
  `EvaluationResult.constraints` — fica como item de roadmap;
  exige alinhar com o orientador o threshold visual.
- `frontend/components/gpr_diagnostics.py` — paired plots
  (resíduos, banda de incerteza, hiperparâmetros) — previsto
  para Sprint 4.7.

## Vínculos

- [[12_Auditoria/Sprint 4.5 - 3D footings viewer - 2026-04-28]] — sprint anterior
- [[12_Auditoria/Sprint 4.4 - Structured logging - 2026-04-28]]
- [[12_Auditoria/Sprint 4.2 - Experiment persistence - 2026-04-28]] — fonte do `ExperimentRun`
- [[10_Melhorias/MOC - Melhorias]]
