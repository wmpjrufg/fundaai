---
tags: [notebook, gpr]
file: testes_gpr_lucas.ipynb
size: 1.4 MB
cells: 22
origem: IC atual
---

# `testes_gpr_lucas.ipynb`

Estudo do **Gaussian Process Regressor** isolado (sem o EGO).

## Headers

1. Bibliotecas
2. Carregando dados (`toy_problem_copy.xlsx`)
3. Pop inicial de possíveis soluções
4. Gerando dataset completo (FO em todos os pontos da pop. inicial)
5. Aprendizado de máquina do dataset
6. Separação em x e y
7. Treinamento com adição de amostras
8. Gráficos
9. Gráficos (continuação)

## Objetivo

Avaliar **qualidade preditiva** dos 20 kernels ([[03_Otimizacao/Kernels GPR]]) sobre uma base sintética gerada amostrando o espaço de design e calculando o volume penalizado real.

Resultados são salvos em `models/*.pkl` ([[05_Dados/Modelos GPR Treinados]]) e visualizados em `assets/graphics/` ([[05_Dados/Assets - Gráficos GPR]]).

## Vínculos

- [[06_Notebooks/testes_otm_lucas]] (continua o estudo com EGO)
- [[03_Otimizacao/Gaussian Process Regressor]]
