---
tags: [refactor, sprint, log, frontend, ux, ui, plotly, progresso, hover]
data: 2026-04-28
branch: refactor/core-architecture
escopo: Sprint 4.7 — Polimento de UX (gráficos, 3D, progresso, n_rep input)
---

# Sprint 4.7 — UI polish + live progress

> Sprint reativa ao feedback do usuário durante a Sprint 4.6:
> hover ribbon que bloqueava scroll, 3D piscando, gráfico do EGO
> "tampão" para `n_gen=2`, ausência de progresso visível durante
> a otimização e falta de input para `n_rep` na UI. Todos
> resolvidos.

## TL;DR

> O front-end agora mostra **progresso ao vivo** (rep, iteração,
> melhor OF), tem **gráficos com hover por trace** (não mais
> aquela ribbon vertical que cobria a página), o **3D ganhou seção
> full-width** separada da tabela 2D, eixos do gráfico ficaram
> **clamped em `>=0`** (sem zoom em iter negativa), e o input
> `n_rep` apareceu ao lado de `n_pop`.

## Para leigos — o que mudou

1. **Saber o que está acontecendo durante a otimização**: aparece
   uma barra de progresso e uma "caixa de status" mostrando
   `Repetição 3/5 · iter 12/20 · melhor OF até agora: 18.4321 m³`.
   Atualiza ao vivo, sem precisar esperar tudo acabar.
2. **Gráfico do histórico**: o tooltip agora aparece **só no
   ponto** que você está apontando, não mais uma faixa
   vertical que cobre tudo. O gráfico ficou mais alto (720px),
   os subgráficos (OF e tempo) **separados** com folga, e a
   roda do mouse dá zoom direto na faixa que interessa.
3. **Ponto inicial explicado**: "iter 0 = pop. inicial (LHS)"
   anotado no gráfico, com **markers** em cada iteração para
   ficar visível mesmo quando o `n_gen` é pequeno.
4. **Vista 3D**: agora ocupa **uma seção própria** abaixo da
   tabela 2D, com 760px de altura — espaço suficiente para
   inspecionar o arranjo. O **flicker no hover foi eliminado**:
   o terreno (grid + contorno) agora é completamente inerte ao
   cursor, então o tooltip não pisca entre as superfícies.
5. **`n_rep` é input visível** ao lado de `n_pop`. O default
   subiu de `n_gen=2` para `n_gen=20` (mais coerente com o
   tempo computacional típico de uma sapata pequena, e com o
   que dá pra ver no gráfico).

## Para o time técnico

### Mudanças no `frontend/components/ego_chart.py`

```python
# antes
make_subplots(..., shared_xaxes=True, vertical_spacing=0.08,
              row_heights=[0.72, 0.28])
fig.update_layout(hovermode="x unified")
# linhas sem markers; tooltip ribbon cobrindo tudo.

# depois
make_subplots(..., shared_xaxes=False, vertical_spacing=0.22,
              row_heights=[0.66, 0.34])
fig.update_layout(hovermode="closest", height=720,
                  legend=dict(yanchor="top", y=1.0,
                              xanchor="left", x=1.02,
                              groupclick="toggleitem"))
# Curvas em mode="lines+markers" (size 8, halo sutil).
```

Layout do eixo X agora trava em `>=0`:

```python
fig.update_xaxes(
    rangemode="nonnegative",
    range=[-0.2, max_iter + 0.2],
    tick0=0, dtick=1 if max_iter <= 12 else None,
    constrain="domain",
)
```

Anotação extra no gráfico: `"iter 0 = pop. inicial (LHS)"` ancorada
na origem.

### Mudanças no `frontend/components/footings_3d.py`

- **Lighting recalibrada** (`ambient=0.75`, `specular=0.05`,
  `fresnel=0.0`) — Plotly recomputava highlights especulares a
  cada movimento do cursor quando `fresnel > 0`, daí o flicker.
  Sombreamento agora é steady.
- **Terreno inerte ao cursor**: `hoverinfo="skip"` +
  `hovertemplate=None` em todos os 3 traces (rect + grid + contour)
  para garantir que o cursor não "salte" entre superfícies.
- **`render_footings_3d(..., height=720)`**: agora aceita altura
  como parâmetro; a página seta **760px** para a seção full-width.
- **Eixo Z travado**: `zaxis.range = [z_min - 0.2, z_max + 0.2]`
  baseado no `min(-h_z)` das sapatas e na altura visual do pilar.
- **`scene.dragmode="orbit"` + `hovermode="closest"`** explícitos.

### Mudanças no `frontend/pages/sapatas.py`

**Layout em 3 seções, cada uma com seu espaço:**

```
[ KPIs ] [ chip run-id ]

── Section 1 ─────────────────────────────────
[ Tabela (3 cols) ] [ Planta 2D (4 cols) ]

── Section 2 ─────────────────────────────────
### 🧊 Vista 3D do arranjo
[ controles 1 col ] [ scene 5 cols, 760px ]

── Section 3 ─────────────────────────────────
### 📈 Histórico do EGO
[ botão mostrar/ocultar ]
[ caption: "Arraste para zoom · duplo-click reseta" ]
[ chart 720px com hover closest, scrollZoom ]

── Section 4 ─────────────────────────────────
### 📦 Exportar
[ DXF | JSON | HTML 3D | HTML hist | PNG hist ]
```

**Inputs novos:**

- `n_rep_ui = st.number_input("Repetições...", value=5)` ao lado
  de `n_pop`.
- Tooltips (`help=...`) em `n_gen`, `n_pop`, `n_rep` explicando
  o papel de cada parâmetro no pipeline.
- Default de `n_gen` subiu de 2 para **20** (mais informativo).

**Progresso ao vivo:**

```python
progress_bar = st.progress(0, text="Preparando...")
status_box = st.status("⏳ Otimização em andamento...",
                       state="running", expanded=True)
info_line = status_box.empty()
sub_line = status_box.empty()

def _on_progress(ev: dict) -> None:
    if ev["event"] == "ego.iter":
        progress_state["unit"] += 1
        pct = progress_state["unit"] / total_units
        progress_bar.progress(pct, text=(...))
        sub_line.markdown(f"🧠 Treinando GPR · iter {it}/{n_gen} ...")
    # ... outros eventos: rep_start, rep_end, end, failed

optimize(projeto, config, recorder=rec, cache=cache,
         progress=_on_progress)
```

### Mudanças na API `core.api.optimize` e `core.optimization.ego.ego_01_architecture`

Adicionado parâmetro **opcional** `progress: Callable[[dict], None] | None = None`.

Eventos emitidos pelo `optimize`:

| Evento                | Campos extras                                                       |
|-----------------------|---------------------------------------------------------------------|
| `optimize.start`      | `n_rep`, `n_gen`, `n_pop`, `n_fund`, `base_seed`                    |
| `optimize.rep_start`  | `rep`, `seed`, `n_rep`, `n_gen`                                     |
| `ego.iter`            | `iter`, `n_gen`, `of_min`, `n_train`, `rep`, `seed`, `n_rep`        |
| `optimize.rep_end`    | `rep`, `seed`, `of_rep`, `wall_time_s`, `n_rep`                     |
| `optimize.end`        | `best_of`, `best_seed`, `wall_time_s`                               |
| `optimize.failed`     | `error`, `wall_time_s`                                              |

Excepções levantadas pelo callback são engolidas — UI bugada **não
aborta** a otimização.

## Validação

```text
=== suite ===
  211 passed in ~8 s
    test_ego_chart.py        +2  (hover closest; height >= 600)
    test_components_3d.py    +2  (height param; scene hovermode)
    test_experiments.py      +2  (progress callback contrato + tolerância
                                  a callback que levanta)
```

Baseline `of = 19.70604234767181` permanece intocado (o callback de
progresso é opt-in; `progress=None` mantém o caminho histórico).

## Vínculos

- [[12_Auditoria/Sprint 4.6 - Premium UI - 2026-04-28]] — sprint anterior (alvo do feedback)
- [[10_Melhorias/MOC - Melhorias]]
