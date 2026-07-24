---
tags: [engenharia, geometria]
aliases: [g_geometria, balanco minimo]
---

# Restrição de Geometria

Garante um **balanço mínimo** entre o pilar e a borda da sapata. Default: 10 cm em cada lado.

## Fórmula (em `checagem_geometria`)

$$
\delta_{ap} = \frac{2 \cdot \text{balanco\_min}}{ap}, \quad
\delta_{hx} = \frac{h_x}{ap}
$$

$$
g = 1 + \delta_{ap} - \delta_{hx} \le 0 \;\Leftrightarrow\; h_x \ge ap + 2 \cdot \text{balanco\_min}
$$

A FO calcula `g_geometria_x` e `g_geometria_y` e usa `max` dos dois.

## Motivação

- Garantir cobertura mínima da armadura.
- Permitir o "encaixe" geométrico do pilar dentro da sapata.

## Links

- [[02_Engenharia/Sapatas Isoladas]]
- [[02_Engenharia/NBR 6118]]
- [[04_Codigo/fundacao.py]]
