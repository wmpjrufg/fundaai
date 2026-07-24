---
tags: [melhorias, otimizacao, hibridizacao, sugestao]
---

# Hibridização Memética

> [!note] Sugestão
> Hibridização global (metaheurística) + local (gradiente) é o caminho clássico para refinar soluções. Encaixa exatamente no foco da IC ("metaheurísticas e/ou hibridizações", ver [[01_Projeto/Escopo da IC]]).

## Esquema

```
Loop GA/PSO/GWO:
    seleção/crossover/mutação
    a cada k gerações:
        para os top-N agentes:
            x ← scipy.minimize(f, x, method='SLSQP', bounds, constraints=g_k)
```

A `mealpy` já oferece a metaheurística; a `scipy.optimize.minimize(method='SLSQP')` consegue tratar restrições explícitas (`type='ineq'`).

## Variantes

- **Lamarckian** — escreve a solução refinada de volta no agente (lex sobrevive).
- **Baldwinian** — usa o fitness pós-busca, mas mantém o agente original.
- **Memetic Algorithm Adaptativo** — ajusta `k` (frequência) e `N` (intensidade) com base no progresso.

## Compatibilidade com FundaIA

- A FO penalizada já é diferenciável quase em todo lugar (penalidades são `max(0, g)` — sub-gradiente).
- Para SLSQP com restrições: passar `g_tensao`, `g_puncao`, `g_geometria`, `g_sobreposicao` como `constraints` explícitas (em vez de penalizadas).

## Possível artigo

- Krasnogor & Smith (2005) — survey clássico de memetic algorithms.
- Moscato (1989) — paper original.

## Vínculos

- [[03_Otimizacao/Algoritmo Genético]]
- [[03_Otimizacao/EGO - Efficient Global Optimization]]
- [[01_Projeto/Escopo da IC]]
