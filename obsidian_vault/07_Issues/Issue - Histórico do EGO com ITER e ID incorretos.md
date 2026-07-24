---
tags: [issue, alto, ciencia, ego, resolvido]
file: metapy_toolbox/ego.py
severity: alto
status: resolvido
resolvido_em: 2026-04-27
resolvido_em_branch: fix/code-sanitization-and-tests
---

# Issue — Histórico do EGO com ITER/ID incorretos

> [!success] Resolvido em 2026-04-27 (Sprint 1, branch `fix/code-sanitization-and-tests`)
> O laço de iterações de `ego_01_architecture` agora registra cada novo
> ponto com `ITER = t` (índice da iteração) e `ID = max(ID) + 1`,
> garantindo identificadores únicos e cobertura completa de
> `0..n_gen` em `df['ITER']`.
>
> Smoke test (`sphere`, `n_pop=8`, `n_gen=4`):
>
> ```text
> ITER unicos = [0, 1, 2, 3, 4]
> Total linhas = 12 (esperado n_pop + n_gen = 12)
> IDs unicos? True (set tem 12 ids, total 12)
> ```

## Sintoma original

Em `ego_01_architecture` ([[04_Codigo/metapy_toolbox - ego.py]]), o ponto novo encontrado a cada iteração era registrado por:

```python
aux_df = funcs.evaluation(obj, n, x_new, 0, args=args)
df = pd.concat([df, aux_df], ignore_index=True)
```

ou seja:

- **`ITER = 0`** (deveria ser `t`) — todos os pontos novos pareciam pertencer à iteração inicial.
- **`ID = n`** — `n` era a variável que sobrou do laço de avaliação da população inicial, então **todos os pontos novos do EGO carregavam o mesmo `ID`**.

## Por que era problema

A escolha do **best_x** ainda funcionava (`df["OF"].idxmin()` ignora `ITER`/`ID`).

Mas tudo que dependia do histórico ficava corrompido:

- Trajetória `best_of(t)` — não dava pra plotar convergência.
- Contagem de avaliações por iteração.
- Gráficos comparativos entre repetições.
- Análise estatística de exploração vs exploit.
- Tabelas de "número de avaliações até atingir threshold X".

## Correção aplicada

```python
# Add new training point with correct ITER=t and a fresh ID.
new_id = int(df['ID'].max()) + 1
aux_df = funcs.evaluation(obj, new_id, x_new, t, args=args) if args is not None \
         else funcs.evaluation(obj, new_id, x_new, t)
df = pd.concat([df, aux_df], ignore_index=True)
```

Bônus da mesma sprint: foi adicionado o parâmetro `seed: Optional[int] = None`
na assinatura de `ego_01_architecture`, propagado ao `random_state` do GPR,
ao gerador NumPy do `x0` dos minimizers SciPy e ao `seed` do `mealpy.solve(...)`
quando suportado pela versão. `seed=None` mantém o comportamento histórico.

## Vínculo

- [[04_Codigo/metapy_toolbox - ego.py]]
- [[03_Otimizacao/EGO - Efficient Global Optimization]]
- [[10_Melhorias/Reprodutibilidade - Seeds e Versão]]
- [[07_Issues/Issue - n_rep reusa população inicial]] — resolvida na mesma sprint
- [[07_Issues/Lista Mestre de Issues]]
