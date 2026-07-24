---
tags: [melhorias, otimizacao, packing, sugestao]
---

# Posicionamento como Variável de Projeto

> [!note] Sugestão
> Hoje `(xg, yg)` são **fixos** (vêm da planilha — centróide do pilar). O packing existe apenas como restrição de "não sobrepor". A IC ganharia uma frente nova ao **otimizar também a posição**.

## Casos onde isso faz sentido

1. **Pilares estruturalmente livres**: alguns projetos permitem reposicionar pilares dentro de margens.
2. **Sapatas excêntricas**: a sapata não precisa estar centrada no pilar — pode ter excentricidade controlada (gera momento adicional, mas alivia geometria).
3. **Sapatas associadas / corridas**: quando dois pilares próximos compartilham uma sapata combinada — viola o pressuposto atual de sapatas isoladas, mas é caso prático comum.

## Reformulação

Variáveis: `(h_x_i, h_y_i, h_z_i, dx_i, dy_i)` — `dx_i, dy_i` excentricidade limitada.

Restrições adicionais:
- `|dx_i|, |dy_i| ≤ excentricidade_max` (norma).
- Acréscimo de `dx_i · F_z_i` ao momento `M_y_i` (ver [[02_Engenharia/Flexão Composta - Sigma Max e Min]]).

## Conexão com packing puro

Esta é a **forma forte** do [[03_Otimizacao/Problema de Empacotamento]]. Literatura relevante:

- Strip packing 2D.
- No-Fit Polygon (NFP) — para sapatas rotacionadas.
- Bin packing com restrições estruturais (rara).

## Vínculos

- [[03_Otimizacao/Problema de Empacotamento]]
- [[11_Frentes_de_Pesquisa/Posicionamento Conjunto - Layout + Sizing]]
