---
tags: [dados, graficos, gpr]
folder: assets/graphics/
---

# Assets — Gráficos GPR

40 PNGs gerados por [[06_Notebooks/testes_otm_lucas]].

## Padrão de nomes

| Padrão | O que mostra |
|---|---|
| `z_GPR_gpr_com_kernel_k{NN}_pen_1e1_vs_1e6.png` | Scatter `observado × predito` para o kernel `kNN` em duas escalas de penalidade. 20 figuras. |
| `z_GPR_test_size_900_vs_600_comparison_gpr_com_kernel_k{NN}.png` | Comparativo `treino=900` vs `treino=600`. 20 figuras. |

## Como interpretar

- Eixo X: volume observado (m³) — saída real do `obj_teste`.
- Eixo Y: volume predito pelo GPR.
- Diagonal = ajuste perfeito.
- Pontos próximos da reta ⇒ R² alto.

## Conclusão típica observada nos experimentos

- Penalidade 1e1 → landscape suave → GPR ajusta bem.
- Penalidade 1e6 → landscape com saltos → GPR sofre a aproximar regiões inviáveis.

## Tabelas associadas

- `assets/tables/tabela_metricas_gpr_toy_problem_all_penaltys.xlsx`
- `assets/tables/tabela_metricas_gpr_toy_problem_all_splits.xlsx`

## Links

- [[03_Otimizacao/Kernels GPR]]
- [[03_Otimizacao/Penalização de Restrições]]
