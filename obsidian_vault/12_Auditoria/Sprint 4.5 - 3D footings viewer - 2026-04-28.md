---
tags: [refactor, sprint, log, frontend, visualizacao, plotly, 3d]
data: 2026-04-28
branch: refactor/core-architecture
escopo: Sprint 4.5 — Visualizador 3D das sapatas
---

# Sprint 4.5 — 3D footings viewer

> Primeiro componente populado em `frontend/components/`. Visualizador
> 3D interativo (Plotly) das sapatas otimizadas: cada sapata vira um
> paralelepípedo enterrado, cada pilar uma caixa fina acima do solo,
> com plano de solo opcional, hover com dimensões/volume e cor por
> elemento ou por volume.

## TL;DR

> A página de dimensionamento agora tem uma **aba 3D** ao lado da
> planta 2D: o usuário roda, dá zoom e vê cada sapata com suas
> dimensões reais — pronto para mostrar pro orientador, colocar no
> artigo (export HTML estático ou PNG) e diagnosticar visualmente
> resultados óbvios e não óbvios da otimização.

## Para leigos

Antes só dava pra ver as sapatas em planta (vista de cima). Agora há
**duas abas**: a planta 2D (que já existia) e uma **vista 3D
interativa**. A vista 3D mostra:

- **Sapata** como um bloco enterrado embaixo do solo (z = 0 é o
  topo da sapata).
- **Pilar** como uma caixa mais fina subindo a partir do solo,
  na altura visual configurada pelo slider.
- **Plano de solo** translúcido em z = 0 mostrando a interface
  solo–fundação.
- **Tooltip** ao passar o mouse: rótulo do pilar, h_x/h_y/h_z,
  volume de concreto e coordenadas.
- **Legenda clicável**: você pode esconder pilares ou sapatas
  específicas para focar em uma região.

Você pode arrastar para rotacionar, dar zoom com a roda do mouse, e
exportar a vista atual em PNG (botão padrão do Plotly).

> **Por que importa para o artigo?** Reviewer e leitor enxergam
> imediatamente o que a otimização produziu. A visualização 3D
> também denuncia visualmente situações ruins (sapata muito alta,
> pilar excêntrico, sobreposição residual) sem ter que ler a
> tabela de restrições.

## Para o time técnico

### Onde mora

```
frontend/components/
├── __init__.py             # exporta render_footings_3d, footing_box, pillar_box
└── footings_3d.py          # implementação Plotly
```

### API

```python
from frontend.components import render_footings_3d

fig = render_footings_3d(
    result.sapatas,            # Iterable[core.domain.Sapata]
    show_pillars=True,
    show_ground=True,
    pillar_height_m=1.5,       # apenas visualização (Pilar não tem altura no domínio)
    title=None,
    colour_by="label",         # "label" | "volume"
)
# Streamlit:
st.plotly_chart(fig, use_container_width=True)
# Notebook:
fig.show()
# Export estático:
fig.write_html("artigo/figs/3d.html")
fig.write_image("artigo/figs/3d.png")
```

### Decisões

1. **Plotly em vez de matplotlib 3D**. Plotly traz interatividade
   real (rotate/zoom/toggle) no Streamlit/Jupyter sem
   configuração extra. Adicionado `plotly>=5,<7` em
   `requirements.txt`.
2. **Mesh3d com 8 vértices + 12 triângulos** por caixa.
   Geometria explícita evita dependências mais pesadas (e.g.
   `pyvista`, `vtk`).
3. **z = 0 = interface solo-fundação**. Sapatas enterradas
   (`z ∈ [-h_z, 0]`); pilares acima (`z ∈ [0, height_m]`).
   Convenção física natural; alinha com a `tensao_adm_solo`.
4. **`pillar_height_m` é parâmetro de visualização**, não de
   domínio. O `Pilar` não carrega altura estrutural; o slider
   permite ajustar para casar com a expectativa de pé direito.
5. **Função pura, framework-agnóstica**. Devolve
   `plotly.graph_objects.Figure` — quem rendera é responsabilidade
   do chamador (Streamlit, notebook, export).
6. **Equal-data aspect** (`scene.aspectmode="data"`) garante que
   as proporções X/Y/Z não fiquem distorcidas — uma sapata
   1×1×0.6 aparece com o aspecto certo.
7. **Cor por volume** usa rampa Viridis discretizada em 10 níveis,
   suficiente para diferenciar sapatas em runs típicos sem
   excessivo "ruído visual".

### Wire-up no Streamlit

`frontend/pages/sapatas.py` agora separa a visualização em **duas
abas**:

```
┌─────────────────────────────────────────────────────────┐
│  🗺️ Planta 2D    │  🧊 Vista 3D                       │
└─────────────────────────────────────────────────────────┘
                                    Coluna 3:1 — viewer | controles
                                      ✓ Exibir pilares
                                      ✓ Exibir plano de solo
                                      Cor: [por elemento] [por volume]
                                      Altura visual do pilar: |--•----|
```

A export DXF foi mantida abaixo das duas abas (download do arranjo).

## Validação

```text
=== suite ===
  183 passed in ~7 s
    test_components_3d.py    12  (novo)

=== contratos travados pelos testes ===
  - 8 vértices + 12 triângulos por caixa (closed AABB).
  - footing.z em [-h_z, 0], pillar.z em [0, height_m].
  - Bounds X/Y de footing == [xg ± h_x/2] e [yg ± h_y/2].
  - Bounds X/Y de pillar  == [xg ± a_p/2] e [yg ± b_p/2].
  - Trace count  = 2*N + 1 (sapata + pilar + ground).
  - colour_by="volume" produz ≥ 2 cores quando volumes variam.
  - Hover carrega rótulo + 3 dimensões.
  - Empty / unknown colour_by levantam ValueError.
```

## Pendências relacionadas (próximas iterações de frontend)

- `frontend/components/ego_chart.py` — curva *best-so-far* por
  iteração, consumindo `ExperimentRun.history` (Sprint 4.6 ou
  4.7).
- `frontend/components/gpr_diagnostics.py` — paired plots
  (resíduos, banda de incerteza, hiperparâmetros do kernel)
  consumindo um `Pipeline` GPR + split de teste.
- `frontend/i18n/` — mover `titulos_nav` de `app.py` e os labels
  de `frontend/pages/sapatas.py` para dicionários PT/EN
  centralizados.
- Sapatas com cores ligadas ao **status de restrição**
  (verde = factível, vermelho = violação) consumindo uma
  `EvaluationResult.constraints` paralela.

## Vínculos

- [[12_Auditoria/Sprint 4.4 - Structured logging - 2026-04-28]] — sprint anterior
- [[12_Auditoria/Sprint 4.3 - Reorg + docs - 2026-04-28]] — `frontend/components/` scaffolded aqui
- [[10_Melhorias/MOC - Melhorias]]
