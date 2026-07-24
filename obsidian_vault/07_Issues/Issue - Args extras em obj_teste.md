---
tags: [issue, alto, ciencia, resolvido]
severity: alto
status: resolvido
resolvido_em: 2026-04-27
resolvido_em_branch: fix/code-sanitization-and-tests
---

# Issue — Args extras passados a `obj_teste` (penalidade silenciosamente ignorada)

> [!success] Resolvido em 2026-04-27 (branch `fix/code-sanitization-and-tests`)
> A função `_unpack_args` em `fundacao.py` agora aceita explicitamente
> `args[4]` como fator de penalidade, com `_PENALTY_DEFAULT = 1e1`
> preservando o comportamento histórico quando o quinto elemento está
> ausente. Smoke test:
>
> | Chamada | OF |
> |---|---|
> | `args=(df, 3, fck, cob)` (4 args) | 19,7060 |
> | `args=(df, 3, fck, cob, 10)` | 19,7060 |
> | `args=(df, 3, fck, cob, 1e6)` | 354 645,5455 |
>
> Comprova que penalty é parametrizável e que cenários `1e1` × `1e6`
> agora produzem resultados distintos, conforme intenção original dos
> notebooks.

> [!warning] Implicação para gráficos e tabelas históricas
> As figuras e tabelas em `assets/graphics/z_GPR_*_pen_1e1_vs_1e6.png` e
> `assets/tables/tabela_metricas_gpr_toy_problem_all_penaltys.xlsx`
> precisam ser **regeradas** com o código corrigido. As versões anteriores,
> geradas antes desta correção, comparavam a mesma penalidade efetiva (10)
> rotulada com nomes diferentes.

## Sintoma original

Em [[06_Notebooks/testes_otm_lucas]], células passavam:

```python
args = [df, n_comb, f_ck, cob_m, 1e1]   # penalidade leve
of, _ = obj_teste(x, args)
```

Mas a assinatura antiga de `obj_teste` em [[04_Codigo/fundacao.py]] usava apenas `args[0..3]`. O quinto elemento era silenciosamente ignorado.

## Por que era problema

- O notebook **acreditava** estar variando a penalidade (1e1 vs 1e6) mas o código sempre usava o **fator 10 hardcoded**.
- Os experimentos de [[06_Notebooks/testes_gpr_lucas]] e [[06_Notebooks/testes_otm_lucas]] estavam comparando configurações que **não diferiam na prática**.

## Correção aplicada

Refator em `fundacao.py` (parte da sprint que também resolveu
[[07_Issues/Issue - obj_felipe_lucas vs obj_teste]]):

```python
_PENALTY_DEFAULT = 1e1

def _unpack_args(args):
    df, n_comb, f_ck, cob_m = args[0], args[1], args[2], args[3]
    penalty = args[4] if len(args) >= 5 else _PENALTY_DEFAULT
    return df, n_comb, f_ck, cob_m, penalty
```

## Vínculo

- [[03_Otimizacao/Penalização de Restrições]]
- [[04_Codigo/fundacao.py]]
- [[06_Notebooks/testes_otm_lucas]]
- [[07_Issues/Issue - obj_felipe_lucas vs obj_teste]] — resolvida na mesma sprint
- [[07_Issues/Lista Mestre de Issues]]
