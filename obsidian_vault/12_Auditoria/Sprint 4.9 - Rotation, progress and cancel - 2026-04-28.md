---
tags: [refactor, sprint, log, frontend, ux, threads, cancelamento, plotly]
data: 2026-04-28
branch: refactor/core-architecture
escopo: Sprint 4.9 — Rotação 3D travada + progresso coerente + cancel cooperativo
---

# Sprint 4.9 — Rotation, progress and cancel

> Sprint reativa ao feedback pós-4.7/4.8: rotação 3D livre demais
> entortando o eixo, mensagens de progresso confusas (parecia que o
> GPR rodava "fora" das iterações do EGO), barra de progresso que
> ficava parada ou não chegava ao fim, e ausência de um botão de
> cancelamento. Tudo resolvido em uma sprint.

## TL;DR

> **3D**: rotação agora é "turntable" (azimuth + elevação apenas;
> mundo permanece vertical). **Progresso**: peso 1 por LHS-init e 1
> por iteração de EGO → barra coerente de 0 a 100%. **Cancel**:
> botão "⏹️ Parar dimensionamento" via thread + `should_stop`
> cooperativo + `OptimisationCancelled` na API.

## Para leigos

1. **3D não vira mais de cabeça pra baixo**. Antes ao arrastar muito
   o mouse a cena ia rolando até o eixo Z ficar inclinado; agora o
   "norte" está fixo e você só consegue **girar em torno do eixo
   vertical** e **inclinar a câmera para cima/baixo** — exatamente
   o que um software de CAD faz.
2. **A barra de carregamento agora bate com a realidade**: cada rep
   tem **`n_pop` avaliações iniciais (LHS)** e **`n_gen` iterações
   do EGO**, então a barra anda em `n_rep × (n_gen + 1)` etapas.
   A mensagem mostra **em qual fase você está**: "amostrando LHS",
   "iter X/Y do EGO — re-treinando GPR + maximizando EI + avaliando
   candidato", "gravando histórico em disco".
3. **Botão "⏹️ Parar dimensionamento"** aparece durante a otimização.
   Quando você clica, a otimização para na **próxima iteração ou
   próxima avaliação LHS** (sem precisar esperar todas as 5 reps
   acabarem); o histórico parcial até esse ponto é mantido em
   `experiments/<run_id>/` com status `"failed"` e
   `error="cancelled by user"`.

> **Esclarecimento sobre o pipeline.** Sua intuição estava certa:
> o EGO **deve** treinar o GPR a cada iteração — o GPR é o "modelo
> substituto" que aprende a cada nova amostra real. Não há
> contagem dupla. O total de fits do GPR num run é exatamente
> `n_rep × n_gen`. Se você define `n_rep=5, n_gen=20` o GPR é
> treinado **100 vezes** ao todo, e há mais `n_rep × n_pop = 1.250`
> avaliações reais para "alimentar" esse aprendizado. As mensagens
> agora deixam isso explícito.

## Para o time técnico

### 3D — rotação travada

Mudanças em `frontend/components/footings_3d.py`:

```python
fig.update_layout(scene=dict(
    aspectmode="data",
    camera={**camera_dict, "up": dict(x=0, y=0, z=1)},  # +z fixo
    hovermode="closest",
    dragmode="turntable",   # azimuth + elevation only; no roll
))
```

`dragmode="turntable"` é o equivalente Plotly de turntable em CAD:
arrastar horizontalmente faz azimuth ao redor do eixo Z; arrastar
verticalmente muda a elevação. **Não é possível rolar** (roll), o
mundo nunca sai da vertical. Combinado com `camera.up = +z`, a
câmera começa sempre alinhada com a "cena no chão".

Os `CAMERA_PRESETS` agora todos têm `up=(0,0,1)` (antes o "topo"
tinha `up=(0,1,0)`).

Teste novo:
```python
def test_scene_uses_turntable_dragmode_with_z_up(self):
    fig = render_footings_3d(...)
    assert fig.layout.scene.dragmode == "turntable"
    cam_up = fig.layout.scene.camera.up
    assert (cam_up.x, cam_up.y, cam_up.z) == (0.0, 0.0, 1.0)
```

### Progress — eventos novos + denominador correto

Em `core.optimization.ego.ego_01_architecture` adicionei eventos
para a fase LHS:

| Evento     | Onde                                  | Campos                          |
|------------|---------------------------------------|---------------------------------|
| `lhs.start`| Antes do laço da pop. inicial         | `n_pop`                         |
| `lhs.eval` | A cada 10 avaliações + a última       | `n`, `n_pop`                    |
| `lhs.done` | Depois do laço, antes do EGO          | `n_pop`, `of_min`               |
| `ego.iter` | A cada iteração do EGO (já existia)   | `iter`, `n_gen`, `of_min`, `n_train` |

E em `core.api.optimize`, um evento extra antes da gravação em disco:

| Evento                | Quando                                                        |
|-----------------------|---------------------------------------------------------------|
| `optimize.recording`  | Antes de `recorder.record_rep` (gravação Parquet+CSV+manifest) |

Na UI:

```python
total_units = n_rep * (n_gen + 1)   # +1 = a fase LHS
units_done = sum(1 for e in seen
                 if e.get("event") in ("lhs.done", "ego.iter"))
pct = units_done / total_units
```

Cada `lhs.done` ou `ego.iter` consumido conta como 1 unidade.
Quando o último `ego.iter` da última rep chega, `units_done ==
total_units` e a barra está exatamente em 100%. As mensagens
diferenciam claramente "amostrando LHS", "iter X/Y", "gravando
histórico" e "concluído".

### Cancel cooperativo

API nova em `core.api`:

```python
class OptimisationCancelled(Exception): ...

optimize(projeto, config, *,
         recorder=..., cache=...,
         progress=..., should_stop=lambda: cancel_event.is_set())
```

Implementação:

- `core.optimization.ego` define um `_CancelSentinel(BaseException)`
  interno (herda de `BaseException` para escapar de `except
  Exception` dentro do mealpy/SciPy).
- `should_stop` é polled em **três lugares**: antes de cada
  `lhs.eval`, antes de cada `ego.iter`, e no fim de cada rep
  dentro de `optimize`.
- Em `core.api.optimize`, um `try/except _CancelSentinel` traduz
  o sinal para `OptimisationCancelled` e marca o recorder como
  `failed` com `error="cancelled by user"`.

### UI Streamlit — runner em thread

A página de dimensionamento agora roda `optimize()` em um
**daemon thread** com:

- `queue.Queue()` para os eventos de progresso (thread-safe).
- `threading.Event()` para o flag de cancelamento.
- `dict` para o resultado / erro / cancelamento.

A página re-renderiza a cada `POLL_INTERVAL_S = 0.4 s`:

```python
if "run" in st.session_state:
    run_state = st.session_state["run"]
    _render_progress(run_state)             # drena queue + pinta widgets
    if not run_state["holder"].get("done"):
        time.sleep(POLL_INTERVAL_S)
        st.rerun()                           # auto-refresh
    else:
        # finaliza: result | cancelled | error
        ...
```

O botão **⏹️ Parar dimensionamento** aparece durante a otimização;
ao ser clicado faz `cancel_event.set()` e dispara `st.rerun()`. A
thread vê o flag no próximo polling, levanta `_CancelSentinel`,
o `optimize` traduz para `OptimisationCancelled` e a página
mostra "⏹️ Otimização cancelada pelo usuário".

> **Nota de design**: cancelamento é cooperativo, não preemptivo.
> Se o GA interno do mealpy estiver no meio de uma rodada longa
> de cruzamento, o cancel só toma efeito quando ele retorna
> (próximo `ego.iter`). Em casos típicos (`ga_epoch=50`,
> `ga_pop_size=150`) é uma fração de segundo.

## Validação

```text
=== suite ===
  224 passed in ~8 s

  novos testes:
    test_components_3d.py     +1  (test_scene_uses_turntable_dragmode_with_z_up)
    test_experiments.py       +2  (should_stop sintético + flag mid-run)
```

Baseline `of = 19,70604234767181` permanece intocado — todos os
parâmetros novos (`should_stop`, eventos LHS) são **opt-in** com
default `None`/inertes; o caminho `progress=None, should_stop=None`
do `optimize` é byte-exato ao histórico.

## Vínculos

- [[12_Auditoria/Sprint 4.8 - Audit cleanup - 2026-04-28]] — sprint anterior
- [[12_Auditoria/Sprint 4.7 - UI polish + live progress - 2026-04-28]] — origem do feedback
- [[10_Melhorias/MOC - Melhorias]]
