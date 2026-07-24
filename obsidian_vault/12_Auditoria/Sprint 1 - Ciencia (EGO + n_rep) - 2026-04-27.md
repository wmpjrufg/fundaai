---
tags: [refactor, sprint, log, ciencia, ego]
data: 2026-04-27
branch: fix/code-sanitization-and-tests
escopo: Fase 0 — itens 6 e 7 (cientificos)
---

# Sprint 1 — Ciência (EGO + n_rep)

> Log da segunda sprint da nova rodada de refatoração. Foca em **dois
> bugs científicos** identificados pela auditoria de 2026-04-27 que
> comprometiam diretamente a interpretação dos experimentos de
> convergência e a independência das repetições.

## Escopo executado

| # | Item | Issue | Status |
|---|---|---|---|
| 1 | Histórico do EGO com `ITER` e `ID` corretos | [[07_Issues/Issue - Histórico do EGO com ITER e ID incorretos]] | ✅ |
| 2 | n_rep com populações independentes em `pages/sapatas.py` | [[07_Issues/Issue - n_rep reusa população inicial]] | ✅ |
| Bônus | `seed` como parâmetro opcional em `ego_01_architecture` | — | ✅ |

## Detalhes técnicos

### 1. Histórico do EGO (`metapy_toolbox/ego.py`)

**Antes**:
```python
aux_df = funcs.evaluation(obj, n, x_new, 0, args=args)   # ITER=0, ID=n (ultimo do loop inicial)
df = pd.concat([df, aux_df], ignore_index=True)
```

**Depois**:
```python
new_id = int(df['ID'].max()) + 1
aux_df = funcs.evaluation(obj, new_id, x_new, t, args=args)   # ITER=t, ID novo
df = pd.concat([df, aux_df], ignore_index=True)
```

Consequência: `df['ITER']` agora cobre `0..n_gen` corretamente; `df['ID']`
contém valores únicos; trajetória `best_of(t)` passa a ser plotável; análise
de avaliações por iteração e comparação entre repetições deixam de ser
inviabilizadas pelo bug.

### 2. `n_rep` com populações independentes (`pages/sapatas.py`)

**Antes**:
```python
x_ini = initial_population_01(n_pop, 3*n_fun, x_l, x_u, use_lhs=True)   # uma vez
for rep in range(n_rep):
    x_new, best_of, _ = ego_01_architecture(..., args=...)
```

**Depois**:
```python
n_rep = 5
base_seed = 42
for rep in range(n_rep):
    rep_seed = base_seed + rep
    x_ini = initial_population_01(n_pop, 3*n_fun, x_l, x_u, seed=rep_seed, use_lhs=True)
    x_new, best_of, _ = ego_01_architecture(..., args=..., seed=rep_seed)
```

Consequência: cada repetição parte de uma população LHS **independente**;
a sequência inteira é **reprodutível** via `base_seed` (alterar `base_seed`
muda toda a corrida; mantê-lo reproduz exatamente); pode-se reportar
`média ± std` honesto em relatórios.

### Bônus — parâmetro `seed` no EGO

A assinatura passou a ser:

```python
def ego_01_architecture(obj, n_gen, initial_population, x_lower, x_upper,
                        params_opt, params_kernel=None, args=None,
                        seed: Optional[int] = None) -> tuple[list, float, pd.DataFrame]:
```

`seed` é propagado para:

- `random_state` do `GaussianProcessRegressor` (substitui o `42` hardcoded quando informado);
- gerador NumPy local (`np.random.default_rng(seed)`) usado pelo `x0` dos minimizers SciPy;
- `mealpy.solve(problem_dict, seed=int(seed) + t)` quando a versão suporta (com fallback automático).

`seed=None` preserva o comportamento histórico exatamente.

## Smoke test

Executado com `metapy_toolbox.sphere`, `n_pop=8`, `n_gen=4`,
`mealpy.GA.BaseGA(epoch=10, pop_size=20)`:

```text
[1] ITER unicos = [0, 1, 2, 3, 4]
    Total linhas = 12 (esperado n_pop + n_gen = 12)
    IDs unicos? True (set tem 12 ids, total 12)
    ✓ historico correto (ITER + ID)

[2] LHS(seed=999) reproducivel? True
    of_a = 0.060777 | of_b = 0.060777 | iguais? True

[3] LHS(seed=1) != LHS(seed=2)? True
    ✓ seeds distintas geram populacoes iniciais distintas
```

## Implicações imediatas para resultados anteriores

- Quaisquer figuras ou tabelas geradas antes desta sprint que reportem
  trajetória `best_of(t)`, avaliações por iteração ou estatísticas de
  `n_rep` devem ser **regeradas**. As anteriores foram produzidas a
  partir de um histórico corrompido.
- Combinado com a Sprint 0, isso fecha o circuito de resultados
  reprodutíveis: penalty parametrizável + histórico correto + seeds
  controladas.

## Próxima sprint sugerida (Sprint 2)

1. Criar `tests/` com casos para `tensao_adm_solo`,
   `calcular_sigma_max_min`, `verificacao_puncao_sapata`,
   `sobreposicao_sapatas`, `_avaliar_projeto` e `ego_01_architecture`
   (estrutura do histórico).
2. Atualizar notebooks ([[07_Issues/Issue - Notebooks com paths quebrados]]).
3. Decidir convenção 20 vs 21 kernels e refletir em `Kernels GPR.md`.
4. Sanear benchmarks suspeitos antes de qualquer validação científica.

## Vínculos

- [[10_Melhorias/Roadmap Sugerido]]
- [[10_Melhorias/MOC - Melhorias]]
- [[07_Issues/Lista Mestre de Issues]]
- [[12_Auditoria/Sprint 0 - Saneamento - 2026-04-27]]
- [[12_Auditoria/Auditoria 2026-04-27 - Vault vs Projeto]]
