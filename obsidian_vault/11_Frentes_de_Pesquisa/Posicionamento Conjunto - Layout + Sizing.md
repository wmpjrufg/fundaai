---
tags: [pesquisa, packing, layout]
---

# Posicionamento Conjunto — Layout + Sizing

> [!note] Frente
> Versão **forte** do problema atual: otimizar simultaneamente dimensão **e posição** das sapatas — ver [[10_Melhorias/Posicionamento como Variável de Projeto]].

## Conexão com literatura

### Layout optimization (mecânica)
- **Sosnovik & Oseledets** (2017) "Neural Networks for Topology Optimization".
- **SIMP** (Solid Isotropic Material with Penalization) — clássico em topology opt.
- **Density-based topology** — provavelmente exagero para sapatas.

### Packing puro
- **Strip Packing 2D** — Lodi, Martello, Vigo.
- **Irregular Packing** com NFP (No-Fit Polygon) — para o caso de sapatas rotacionadas.
- **Container Loading** — Bortfeldt & Wäscher (2013).

### Híbrido (mais raro, mais original)
- "Joint sizing and layout optimization" — busca pouco explorada na intersecção entre packing e structural sizing. **Possível contribuição original da IC**.

## Formulação proposta

Variáveis: `(h_x_i, h_y_i, h_z_i, dx_i, dy_i)` para cada `i`.

Restrições:
- Mecânica de estruturas (já no FundaIA).
- Packing **estrito** (`g_sob = 0`, não penalização).
- `|dx|, |dy| ≤ excentricidade_max`.
- Margens do terreno.

## Piloto executado

Atualização 2026-07-12: a primeira versão experimental está em `scripts/run_packing_phase_b_pilot.py`.

Caso mínimo:

- Origem: `assets/data/problema_fund_dois.xlsx`.
- Pilares reposicionados sinteticamente em linha, com espaçamento de `2,00 m`.
- Variáveis do modo packing: `(h_x, h_y, h_z, dx, dy)` por sapata.
- Restrições adicionais ao avaliador atual: contenção do pilar dentro da sapata deslocada e fronteira retangular do lote.
- Momentos efetivos usados no piloto: `Mx_eff = Mx_input - Fz * dx`, `My_eff = My_input - Fz * dy`, seguindo a convenção atual do FundaIA (`Mx = Fz * e_x`, `My = Fz * e_y`).

Resultados:

| Modo | Volume | `g_sob` | Factível | Leitura |
|---|---:|---:|---|---|
| Ótimos individuais centralizados | 4,750747 m³ | 0,2307 | Não | A decomposição por sapata falha ao montar o layout. |
| Centros fixos redimensionados | 4,929703 m³ | 0,0000 | Sim | Resolve packing por distorção dimensional. |
| Offsets de packing | 4,525122 m³ | 0,0000 | Sim | Posicionamento reduz volume factível no caso acoplado mínimo. |

Conclusão operacional: a Fase B deve evoluir para benchmark pareado CBO/EGO/metaheurísticas sobre casos acoplados congelados; este piloto ainda não é evidência estatística, mas valida a formulação 5N e a necessidade de tratar posicionamento como variável.

## Algoritmo sugerido

- Inicialização: **Bottom-Left-Fill** (BLF) heurística para packing.
- Otimização: GA com codificação `(h, posição_relativa)` + reparo de viabilidade após cada movimento.
- Pode incorporar [[11_Frentes_de_Pesquisa/Bayesian Optimization Constrained]].

## Vínculos

- [[10_Melhorias/Posicionamento como Variável de Projeto]]
- [[03_Otimizacao/Problema de Empacotamento]]
- [[02_Engenharia/Sapatas Isoladas]]
