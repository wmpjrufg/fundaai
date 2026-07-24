---
tags: [melhorias, refactor, performance, concluido]
status: concluido
sprint_concluido: 3.9
---

# Refactor — Vetorização da FO ✅

> [!success] Concluído no Sprint 3.9
> Todos os `df.apply` e `iterrows()` foram substituídos por numpy broadcasting em `core/api/objective.py::avaliar_projeto_fast`. Speedup medido: **62–86×** (de ~6–13 ms para ~0,1 ms/chamada).

## O que foi feito

### Sobreposição AABB
- Sprint 3.8: `sobreposicao_matrix` — laço `O(N²)` virou operação N×N numpy
- Sprint 3.9: integrada em `avaliar_projeto_fast` via `core.engineering`

### Tensão e punção
Todos os `df.apply(..., axis=1)` substituídos por operações vetoriais `(N_fund, N_comb)`:

```python
# antes (pandas):
df["sigma_max"] = df.apply(lambda row: calcular_sigma_max_min(...), axis=1)

# depois (numpy broadcasting):
sigma_max = F_z / (h_x * h_y) + np.abs(M_x) / (h_y * h_x**2 / 6) + ...  # shape (N_fund, N_comb)
```

## Resultado

| Cenário | Legacy (`_avaliar_projeto`) | Fast (`avaliar_projeto_fast`) | Speedup |
|---|---|---|---|
| 3 fund / 2 comb | 6,36 ms | 0,090 ms | ~70× |
| 10 fund / 4 comb | 10,29 ms | 0,132 ms | ~78× |
| 30 fund / 4 comb | 12,79 ms | 0,148 ms | ~86× |

Validação: `diff = 0.00e+00` — resultados numericamente idênticos (46 testes passando).

## Localização do código

- **`core/api/objective.py`** — implementação vetorizada: `avaliar_projeto_fast`, `avaliar_projeto_legacy`
- **`fundacao.py`** — `obj_felipe_lucas` virou shim com import lazy; `_avaliar_projeto` preservado para `obj_teste`

## Vínculos

- [[04_Codigo/core-api-objective.py]]
- [[04_Codigo/fundacao.py]]
- [[10_Melhorias/Questao Aberta - Custo da FO e Justificativa do EGO]]
- [[10_Melhorias/Refactor - Plano Geral]]
