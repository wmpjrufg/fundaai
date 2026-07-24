---
tags: [melhorias, otimizacao, restricoes, sugestao]
---

# Tratamento de Restrições — Deb e Augmented Lagrangian

> [!note] Sugestão
> Alternativas à penalização exterior simples.

## 1. Regras de Deb (1995, "An efficient constraint handling method for GA")

Comparação 3-regras na seleção:
1. Solução factível > solução infactível.
2. Entre duas factíveis, vence a de menor `f`.
3. Entre duas infactíveis, vence a de menor `Σ max(g_k, 0)`.

Não exige fator de penalidade. Funciona muito bem em GA. Suporta naturalmente o caso de **nenhuma solução factível** na população inicial.

## 2. Augmented Lagrangian

$$
\mathcal{L}_A(x, \lambda, \rho) = f(x) + \sum_k \lambda_k g_k(x) + \frac{\rho}{2} \sum_k \max(0, g_k(x) + \lambda_k/\rho)^2
$$

Multiplicadores `λ_k` são atualizados a cada outer iteration. Vantagem: sob convexidade local, converge ao KKT sem `ρ → ∞`.

## Comparativo proposto (experimento)

Manter o mesmo problema-teste de [[06_Notebooks/testes_otm_lucas]] mas variar:

| Estratégia | Esforço de implementação |
|---|---|
| Penalização fixa (atual) | baseline |
| Penalização adaptativa Coello | médio |
| Deb's rules | médio (mexer na seleção do GA) |
| Augmented Lagrangian | alto |
| ε-constraint method | médio |

## Quando isso ajuda na pesquisa

- Pode virar **uma seção de comparação** no relatório/artigo da IC: "qual técnica de restrições é melhor para o problema acoplado de fundações?".

## Vínculos

- [[10_Melhorias/Penalização Adaptativa]]
- [[03_Otimizacao/Penalização de Restrições]]
- [[03_Otimizacao/Algoritmo Genético]]
