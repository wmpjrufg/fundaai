---
tags: [codigo, streamlit, ui, otimizacao]
file: pages/sapatas.py
loc: 530
---

# `pages/sapatas.py`

Página principal de **dimensionamento otimizado**. Orquestra todo o pipeline:

1. Inputs de parâmetros (n_comb, f_ck, cob, h_min, h_max, n_gen, n_pop).
2. Upload do Excel ([[05_Dados/Schema das Planilhas]]).
3. Botão "Dimensionar" → chama `ego_01_architecture` 5 vezes, retém o melhor.
4. Mostra resultados (DataFrame, métrica de volume, plot 2D, DXF).

## Funções utilitárias

| Função | Propósito |
|---|---|
| `plot_data(data)` | matplotlib: retângulos das sapatas + marker '+' nos pilares |
| `save_dxf(data) -> bytes` | gera DXF R2010 via `ezdxf` |
| `build_plot_payload(df_input, dados_final)` | monta `{label, x, y, L x, L y}` para os dois acima |
| `obter_textos()` | dicionário de tradução PT/EN |

## Configuração padrão da otimização (após Sprint 1)

```python
n_rep = 5
base_seed = 42
paras_opt    = {'optimizer algorithm': GA.BaseGA(epoch=50, pop_size=150)}
paras_kernel = {'kernel': constroi_kernel()[-1]}   # Matern ν=2.5 com bounds estendidos (k20)

for rep in range(n_rep):
    rep_seed = base_seed + rep
    x_ini = initial_population_01(
        n_pop, 3 * n_fun, x_l, x_u,
        seed=rep_seed, use_lhs=True,
    )
    x_new, best_of, _ = ego_01_architecture(
        ..., seed=rep_seed,
    )
```

> [!success] Sprint 1 — n_rep agora independente (2026-04-27)
> Cada uma das 5 repetições parte de uma população LHS **independente**
> e a sequência inteira é **reprodutível** via `base_seed`. A semente é
> também propagada ao EGO. Ver
> [[07_Issues/Issue - n_rep reusa população inicial]] (resolvida).

## Estado da página

- `st.session_state['calculo_realizado']` — flag de "já rodou".
- `st.session_state['dados_final_df']`, `['best_of_valor']`, `['excel_buffer']`.

## Issues

- ✅ ~~Duplicação 326–531~~ — resolvida na Sprint 0.
- ✅ ~~5 repetições do EGO partem do mesmo LHS~~ — resolvida na Sprint 1.
- ⏳ **`save_dxf` cria `NamedTemporaryFile(delete=False)` e não remove.** Em execuções repetidas deixa arquivos órfãos em `/tmp`. Ver [[07_Issues/Issue - DXF tempfile não removido]].

## Vínculos

- [[04_Codigo/fundacao.py]] (FO + kernels)
- [[04_Codigo/metapy_toolbox - ego.py]] (motor)
- [[04_Codigo/metapy_toolbox - funcs.py]] (LHS)
- [[01_Projeto/Pipeline de Execução]]
