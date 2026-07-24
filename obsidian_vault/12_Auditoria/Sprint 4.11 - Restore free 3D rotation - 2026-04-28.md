---
tags: [refactor, sprint, log, frontend, ux, plotly, 3d, rotacao]
data: 2026-04-28
branch: refactor/core-architecture
escopo: Sprint 4.11 — Reverter para rotação livre por mouse (default)
---

# Sprint 4.11 — Restore free 3D rotation

> Sprint reativa: na 4.10 eu interpretei errado o pedido — travei
> a elevação (em vez de só travar contra rolagem) e introduzi
> sliders de azimuth/elevação. O usuário só queria **rotação livre
> com o mouse, com o eixo Z mantido como "pra cima"** (sem rolar).
> A 4.10 tornou-se o oposto do desejado. Sprint 4.11 desfaz a UI
> da 4.10 e mantém apenas o que a Sprint 4.9 já tinha de bom.

## TL;DR

> 3D volta a ser **rotação livre com mouse** por padrão. Os
> sliders de Azimuth / Elevação foram removidos, o toggle
> "Rotação livre" também. Restou apenas o **selectbox de
> ponto de partida da câmera** + os controles de geometria
> (altura do pilar, margem do terreno).

## Para leigos

Antes (4.10): tinha que mexer nos sliders pra rotacionar; arrastar
o mouse não rotacionava. Você reclamou (com razão) que perdeu a
naturalidade. Agora (4.11):

- **Arrasta o mouse no gráfico → rotaciona**, igual a antes.
- **Z continua sendo "pra cima"** — o mundo nunca rola lateralmente.
- **Roda do mouse → zoom**.
- **Selectbox "Câmera"** define só o ponto de partida (isométrica,
  topo, lateral X, lateral Y, perspectiva). Quando você muda o
  preset, a câmera salta pro ângulo desejado; daí em diante você
  rotaciona livre com o mouse.

## Para o time técnico

### `render_footings_3d` — assinatura limpa

A função volta à assinatura compacta:

```python
render_footings_3d(
    sapatas, *,
    show_pillars=True, show_ground=True,
    pillar_height_m=1.5,
    title=None,
    colour_by="label",        # "label" | "volume"
    camera=None,               # preset name | dict | None
    terrain_margin_m=1.0,
    height=720,
)
```

Os 3 parâmetros adicionados na 4.10 (`axis_lock`, `azimuth_deg`,
`elevation_deg`) foram **removidos**. O código das presets
voltou ao caminho simples (preset → `camera_dict`).

### `scene` configuration

```python
scene=dict(
    aspectmode="data",
    camera={**camera_dict, "up": dict(x=0, y=0, z=1)},   # Z = up
    hovermode="closest",
    dragmode="turntable",                                # mouse rota livre
    uirevision="fundaia_3d_camera",                      # estado preservado
)
```

`dragmode="turntable"` é o modo CAD do Plotly: arraste horizontal
faz orbit em torno de Z; arraste vertical faz tilt; **mas não há
roll** (Z fica sempre vertical). É exatamente o que o usuário pediu.

### Streamlit page

Coluna esquerda do 3D agora tem só:

```
┌─ Visualização ─┐
   ☑ Pilares
   ☑ Terreno
   Cor das sapatas: ( ) por elemento  ( ) por volume

┌─ Câmera (ponto de partida) ─┐
   ▾ isométrica

┌─ Geometria ─┐
   Altura visual do pilar (m): ──●──── 1.5
   Margem do terreno (m):      ─●──── 1.5
```

Sem toggle de "rotação livre", sem sliders de azimuth/elevação.

## Validação

```text
=== suite ===
  224 passed in ~8 s

  testes 3D ajustados:
    test_default_uses_turntable_with_z_up        (substitui 4.10)
    test_camera_preset_applied                   (volta ao default)
    test_unknown_camera_preset_raises            (volta ao default)

  testes 4.10 removidos (não fazem mais sentido):
    test_default_disables_drag_rotation_for_axis_lock
    test_axis_lock_none_uses_turntable
    test_axis_lock_elevation_camera_follows_azimuth_slider
    test_unknown_axis_lock_raises
```

Baseline `of = 19,70604234767181` permanece intocado.

## Vínculos

- [[12_Auditoria/Sprint 4.10 - 3D elevation lock - 2026-04-28]] — sprint revertida
- [[12_Auditoria/Sprint 4.9 - Rotation, progress and cancel - 2026-04-28]]
- [[10_Melhorias/MOC - Melhorias]]
