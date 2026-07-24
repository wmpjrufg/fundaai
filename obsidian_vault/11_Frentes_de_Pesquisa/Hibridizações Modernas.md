---
tags: [pesquisa, hibridizacao, metaheuristica]
---

# Hibridizações Modernas

> [!note] Frente
> Coerente com o foco "metaheurísticas e/ou hibridizações" do escopo ([[01_Projeto/Escopo da IC]]).

## Direções

### 1. Memetic Algorithms (já em [[10_Melhorias/Hibridização Memética]])
GA + busca local — clássico mas sempre forte baseline.

### 2. Surrogate-Assisted Evolutionary Algorithms (SAEA)
- **CMAES + GPR** — Hansen, Loshchilov.
- **DE + RBF surrogate** (Krityakierne et al.).
- O EGO atual já é uma instância — pode generalizar para outras metaheurísticas.

### 3. Cooperative Coevolution (CC)
- Decomposição do espaço de variáveis em grupos (uma sapata por grupo, ou (h_x, h_y, h_z) por grupo).
- Otimiza cada grupo com sub-população, alterna.
- Bom para D alto (FundaIA com 30 fundações = D=90).

### 4. Algoritmos populares (estado da arte 2023–2025)
- **CMA-ES** com restart.
- **Dual-Strategy Differential Evolution** (DSDE).
- **L-SHADE** e variantes (vencedoras de competições CEC).
- **EOSMA** (Equilibrium Optimizer + Slime Mould Algorithm) — recente.

### 5. Híbridos com aprendizado
- **Q-learning para escolher operador** (operador mutação adaptativo guiado por RL).
- Ver [[11_Frentes_de_Pesquisa/Reinforcement Learning para Otimização]].

## Critério de comparação

Para fazer ciência de verdade: **rodar 30 sementes** de cada algoritmo no mesmo problema, reportar média ± std e usar **Wilcoxon signed-rank test** para diferença significativa.

## Vínculos

- [[03_Otimizacao/Algoritmo Genético]]
- [[03_Otimizacao/Grey Wolf Optimizer]]
- [[10_Melhorias/Hibridização Memética]]
- [[10_Melhorias/Validação contra problema-benchmark]]
