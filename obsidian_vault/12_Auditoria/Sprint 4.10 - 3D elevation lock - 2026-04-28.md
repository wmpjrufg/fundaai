---
tags: [refactor, sprint, log, frontend, ux, plotly, 3d, rotacao]
data: 2026-04-28
branch: refactor/core-architecture
escopo: Sprint 4.10 — Bloqueio de elevação no 3D (default)
---

# Sprint 4.10 — 3D elevation lock

> Sprint reativa a "esta rotacionando num eixo que entorta… 1 desses
> eixos nao era pra girar pra ficar travado". Reinterpretado: o
> usuário queria orbit horizontal (em torno do eixo Z) com **a
> elevação travada**, sem o mouse-drag vertical podendo flipar a
> cena. Sprint 4.9 tentou resolver com `dragmode="turntable"`, mas
> turntable ainda permite mudar elevação livremente (e cruzar 90°,
> que é o que dá a sensação de "entorta").

## TL;DR

> 3D agora **não rotaciona com mouse-drag por padrão**. Você gira em
> torno do eixo vertical com um **slider de Azimuth** (0–360°) e
> ajusta a inclinação com um **slider de Elevação** (10–80°),
> exatamente como a câmera de um software de CAD. Um toggle
> "🔓 Rotação livre (mouse)" reabilita o turntable do Plotly se você
> quiser arrastar livremente.

## Para leigos

Antes você arrastava o mouse e podia inclinar a câmera até passar
"de cabeça pra baixo", o que dava aquela sensação de que um eixo
estava sendo distorcido. Agora:

- **Por default**: você usa dois sliders na coluna esquerda do 3D:
  - **Azimuth (°)** — gira em torno do eixo vertical (0° a 360°).
  - **Elevação (°)** — quão alta está a câmera (10° = quase rasante;
    80° = quase de cima).
  O mouse pode dar **zoom (roda)** e **pan (arrastar)**, mas **não
  rotaciona** — então o "entorta" desaparece de vez.
- Se quiser o comportamento antigo (girar arrastando o mouse), liga
  o toggle **🔓 Rotação livre (mouse)**. Aí o mouse-drag rotaciona
  no modo turntable do Plotly (o eixo Z continua travado contra
  rolagem, mas a elevação fica solta).

## Para o time técnico

### `render_footings_3d` ganhou três parâmetros

```python
render_footings_3d(
    sapatas, *,
    axis_lock="elevation",      # "elevation" (default) | "none"
    azimuth_deg=45.0,
    elevation_deg=30.0,
    ...
)
```

- `axis_lock="elevation"` — modo default. A câmera é **reconstruída**
  a partir de `azimuth_deg` + `elevation_deg`:

  ```python
  r = 2.4
  eye = (r * cos(elev) * sin(azim),
         r * cos(elev) * cos(azim),
         r * sin(elev))
  ```

  `dragmode="pan"` no scene → **mouse não rotaciona**, só arrasta
  e dá zoom. Resultado: zero "entorta", sem nenhuma chance de o
  usuário flipar a cena.

- `axis_lock="none"` — comportamento da Sprint 4.9 (turntable do
  Plotly). Útil para quem quer rotacionar com mouse.

- Validação: `axis_lock` desconhecido levanta `ValueError`. Se
  azimuth/elevação ficarem fora dos limites razoáveis a função
  clampa internamente (`elevation` em `[5, 85]°`) para evitar
  câmeras degeneradas.

### Streamlit page

```
┌─ Visualização ─┐    Pilares  ☑   Terreno  ☑   Cor (...)
                                  Azimuth   ───●────  45°
┌─ Câmera ──────┐    🔓 Rotação livre (mouse)  ☐   ▾
                     [livre off]                          [livre on]
                     Azimuth (slider)                     Preset (selectbox)
                     Elevação (slider)
```

Quando "Rotação livre" está **off** (default): aparecem os 2
sliders, e o `axis_lock="elevation"` é passado pro componente.

Quando está **on**: aparece o selectbox de presets clássicos
(isométrica/topo/lateral X/lateral Y/perspectiva), e
`axis_lock="none"` é passado.

`uirevision="fundaia_3d_camera"` no scene preserva a câmera entre
reruns do Streamlit (o slider do azimuth não reseta a vista a
cada interação em outro widget).

### Por que não usar só `dragmode="turntable"` (4.9)?

Turntable mantém o eixo "up" travado (não rola), **mas permite
mudança livre de elevação** via drag vertical. Quando o usuário
arrasta muito pra baixo, a câmera passa de elevation=90°,
seguindo pelo "outro lado" da esfera, e a perspectiva
foreshortening cria a sensação de "entorta". Plotly não expõe
clamping de elevação direto; o jeito limpo é desabilitar drag
e controlar a câmera por sliders.

## Validação

```text
=== suite ===
  227 passed in ~8 s

  testes 3D atualizados:
    test_default_disables_drag_rotation_for_axis_lock          (novo)
    test_axis_lock_none_uses_turntable                         (novo)
    test_axis_lock_elevation_camera_follows_azimuth_slider     (novo)
    test_unknown_axis_lock_raises                              (novo)
    test_camera_preset_applied                                 (atualizado)
    test_unknown_camera_preset_raises                          (atualizado)
```

Baseline `of = 19,70604234767181` permanece intocado — sprint
puramente visual; o pipeline numérico não foi tocado.

## Vínculos

- [[12_Auditoria/Sprint 4.9 - Rotation, progress and cancel - 2026-04-28]] — sprint anterior
- [[12_Auditoria/Sprint 4.7 - UI polish + live progress - 2026-04-28]]
- [[10_Melhorias/MOC - Melhorias]]
