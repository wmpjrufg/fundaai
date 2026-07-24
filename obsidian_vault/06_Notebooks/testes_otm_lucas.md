---
tags: [notebook, gpr, otimizacao]
file: testes_otm_lucas.ipynb
size: 3.8 MB
cells: 38
origem: IC atual
---

# `testes_otm_lucas.ipynb`

Estudo combinado **GPR + escala de penalidade + split treino/teste**.

## Experimentos

### A. Sensibilidade à penalidade

```python
args_A = [df, n_comb, f_ck, cob_m, 1e1]   # penalidade leve
args_B = [df, n_comb, f_ck, cob_m, 1e6]   # penalidade pesada
```

Treina os 20 kernels para cada caso (`df_gpr_A`, `df_gpr_B`).

⚠️ Note: o quinto elemento (`1e1` / `1e6`) **não é usado** pela função `obj_teste` atual em [[04_Codigo/fundacao.py]] (que tem o fator 10 hardcoded). Pode indicar que existia/existirá uma versão paramétrica.

### B. Sensibilidade ao split

`test_size ∈ {0.10, 0.20, 0.30, 0.40, 0.50}` para o cenário A.

### C. Visualização

`plot_side_by_side(res_left, res_right, ...)` com formatação serif/CM (qualidade publicação) salva em `assets/graphics/z_GPR_*.png`.

### D. Tabelas finais

`monta_tabela_metricas(...)` exporta para `assets/tables/`.

## Headers

1. Bibliotecas
2. Carregando dados
3. Pop inicial de possíveis soluções
4. Gerando dataset completo
5. Aprendizado de máquina do dataset
6. Separação em x e y
7. Treinamento com adição de amostras
8. Gráficos
9. Gráficos

## Vínculos

- [[06_Notebooks/testes_gpr_lucas]] (preparação)
- [[03_Otimizacao/Kernels GPR]]
- [[03_Otimizacao/Penalização de Restrições]]
- [[05_Dados/Modelos GPR Treinados]]
- [[05_Dados/Assets - Gráficos GPR]]
