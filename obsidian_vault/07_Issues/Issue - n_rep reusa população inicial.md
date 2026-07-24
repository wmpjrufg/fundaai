---
tags: [issue, alto, ciencia, resolvido]
file: pages/sapatas.py
severity: alto
status: resolvido
resolvido_em: 2026-04-27
resolvido_em_branch: fix/code-sanitization-and-tests
---

# Issue — `n_rep = 5` reusa a mesma população inicial

> [!success] Resolvido em 2026-04-27 (Sprint 1, branch `fix/code-sanitization-and-tests`)
> A geração da população inicial (`initial_population_01`) foi movida para
> dentro do `for rep in range(n_rep)`, com seed propagada por repetição
> (`base_seed + rep`). Cada uma das 5 repetições agora parte de uma
> população **independente** e a sequência inteira é **reprodutível**.

## Sintoma original

Em [[04_Codigo/pages - sapatas.py]]:

```python
x_ini = initial_population_01(n_pop, 3*n_fun, x_l, x_u, use_lhs=True)   # gerado UMA vez

for rep in range(n_rep):                                                # 5 repetições
    x_new, best_of, _ = ego_01_architecture(
        obj_felipe_lucas, n_gen, x_ini, ...
    )
    if best_of < best_of_aux: ...
```

Todas as 5 chamadas partiam do **mesmo LHS**. A aleatoriedade vinha somente do GA interno do `mealpy` (e, possivelmente, do GPR — mas com `random_state=42` fixo, nem isso).

## Por que era problema

- Marketing dizia "fazemos 5 repetições para robustez", mas as 5 não eram independentes.
- Não dava para reportar `média ± std` de forma honesta.
- Subestimava variabilidade do método.
- Em relatório/artigo, revisor pegaria na primeira leitura.

## Correção aplicada

```python
n_rep = 5
base_seed = 42  # semente base; cada repetição usa base_seed + rep
...
for rep in range(n_rep):
    rep_seed = base_seed + rep
    x_ini = initial_population_01(
        n_pop, 3 * n_fun, x_l, x_u,
        seed=rep_seed, use_lhs=True,
    )
    x_new, best_of, _ = ego_01_architecture(
        obj_felipe_lucas, n_gen, x_ini, x_l, x_u,
        paras_opt, paras_kernel,
        args=(df, n_comb, f_ck_kpa, cob_m),
        seed=rep_seed,
    )
    ...
```

Smoke test confirma:
- `LHS(seed=999)` reproduzível: `np.allclose(x_a, x_b)` → True.
- `LHS(seed=1) ≠ LHS(seed=2)` (populações distintas).

## Vínculo

- [[04_Codigo/pages - sapatas.py]]
- [[03_Otimizacao/EGO - Efficient Global Optimization]]
- [[03_Otimizacao/Latin Hypercube Sampling]]
- [[10_Melhorias/Reprodutibilidade - Seeds e Versão]]
- [[07_Issues/Issue - Histórico do EGO com ITER e ID incorretos]] — resolvida na mesma sprint
- [[07_Issues/Lista Mestre de Issues]]
