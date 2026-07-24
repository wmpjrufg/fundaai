---
tags: [otimizacao, gwo, metaheuristica]
aliases: [GWO]
---

# Grey Wolf Optimizer (GWO)

Metaheurística baseada na hierarquia de caça de lobos cinzentos (α, β, δ + ω). Implementada em [[04_Codigo/metapy_toolbox - grey_wolf.py]] como `grey_wolf_optimizer_01`.

## Esquema

- `a` decresce linearmente de 2 a 0 ao longo das iterações.
- Cada lobo se move em direção a `α`, `β`, `δ` (top-3 fitness) por médias ponderadas.
- Verificação de bounds via `funcs.check_interval_01`.

## Status no FundaIA

- Disponível, mas **não invocado** pela UI atual.
- ⚠️ A linha 134 contém placeholder: `df['DIVERSITY'] = 'aqui implementa função lucas'`. Ver [[07_Issues/Issue - Placeholder Diversidade GWO]].

## Possível uso futuro

Substituir a `mealpy.GA` no laço EGO ou rodar standalone para comparação com GA.

## Links

- [[04_Codigo/metapy_toolbox - grey_wolf.py]]
- [[03_Otimizacao/Algoritmo Genético]]
- [[03_Otimizacao/EGO - Efficient Global Optimization]]
