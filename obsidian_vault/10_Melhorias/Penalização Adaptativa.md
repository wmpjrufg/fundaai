---
tags: [melhorias, otimizacao, restricoes, sugestao]
---

# Penalização Adaptativa

> [!note] Sugestão
> Hoje o fator é **fixo em 10** (ver [[03_Otimizacao/Penalização de Restrições]]). Penalidade fixa tem trade-off conhecido: alta demais ⇒ landscape rugoso; baixa demais ⇒ ótimos infactíveis.

## Esquemas clássicos

### Penalização exterior progressiva
$$
\rho(t) = \rho_0 \cdot \beta^t, \quad \beta > 1
$$
Começa permissivo, "aperta" ao longo das iterações. Hipótese: explore no início, exploit factível no fim.

### Penalidade dinâmica de Joines & Houck (1994)
$$
P(x, t) = (C \cdot t)^{\alpha} \sum_k \max(g_k, 0)^{\beta}
$$

### Penalidade adaptativa de Coello (2000)
Ajusta `ρ` baseado na fração de soluções factíveis na geração atual.

## Por que vale testar

- Contraste direto contra a penalização fixa atual.
- Os experimentos de [[06_Notebooks/testes_otm_lucas]] já comparam `1e1 × 1e6` — basta variar continuamente.

## Possível artigo seminal

- Coello Coello, C. A. (2002). "Theoretical and numerical constraint-handling techniques used with evolutionary algorithms: a survey of the state of the art". CMAME 191(11–12).

## Vínculos

- [[03_Otimizacao/Penalização de Restrições]]
- [[10_Melhorias/Tratamento de Restrições - Deb e Augmented Lagrangian]]
