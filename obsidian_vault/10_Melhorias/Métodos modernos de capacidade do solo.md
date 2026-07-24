---
tags: [melhorias, engenharia, geotecnia, sugestao]
---

# Métodos modernos de capacidade do solo

> [!note] Sugestão
> O `tensao_adm_solo` atual usa **método dos práticos** (`SPT/30/40/50 · 1000`). Há métodos mais precisos comuns na prática brasileira.

## Métodos a estudar

| Método | Características |
|---|---|
| **Décourt-Quaresma** | Validado para argilas e areias; base teórica robusta |
| **Aoki-Velloso** | Coeficientes `K`, `α` por tipo de solo; muito usado em estacas mas adaptável |
| **Teixeira** | Combina SPT com profundidade |
| **Meyerhof / Vesić / Hansen** | Métodos teóricos clássicos baseados em parâmetros (c, φ, γ) |

## Implicação no projeto

- A escolha do método é **input do usuário** — pode virar parâmetro `Config.metodo_capacidade`.
- Cada método pode requerer dados extras (ex.: profundidade da fundação `D_f`).

## Vínculos

- [[02_Engenharia/Tensão Admissível do Solo]]
- [[02_Engenharia/SPT - Sondagem]]
- [[10_Melhorias/Validação contra problema-benchmark]]
