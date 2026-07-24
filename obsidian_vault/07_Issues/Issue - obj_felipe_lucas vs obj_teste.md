---
tags: [issue, medio, refactor, resolvido]
file: fundacao.py
severity: medio
status: resolvido
resolvido_em: 2026-04-27
resolvido_em_branch: fix/code-sanitization-and-tests
---

# Issue — `obj_felipe_lucas` ≡ `obj_teste`

> [!success] Resolvido em 2026-04-27 (branch `fix/code-sanitization-and-tests`)
> Extração de uma única função interna `_avaliar_projeto(x, args, *, penalty=None)`
> que retorna `(of, df_anotado)`. As duas funções públicas viraram wrappers
> finos:
>
> - `obj_felipe_lucas` devolve apenas `of` (uso na otimização).
> - `obj_teste` devolve a tupla `(of, df)` (uso em notebooks/UI).
>
> Smoke test confirma equivalência numérica exata para o caso de 3 fundações
> (`of = 19.706042` em ambas as chamadas com configuração padrão).

## Sintoma original

As duas funções em [[04_Codigo/fundacao.py]] (linhas 215 e 300) eram **quase idênticas** (~80 linhas espelhadas). A única diferença era o retorno:

- `obj_felipe_lucas(x, args) -> float`
- `obj_teste(x, args) -> (float, DataFrame)`

## Por que era problema

- Bug em uma exigia correção manual na outra.
- Aumentava o risco de divergência futura (já existiam comentários comentados em ambas).

## Correção aplicada

Estrutura final em `fundacao.py`:

```python
_PENALTY_DEFAULT = 1e1   # preserva valor histórico hardcoded

def _unpack_args(args):
    """Aceita 4 ou 5 elementos; quinto é penalty (default 10)."""

def _avaliar_projeto(x, args, *, penalty=None) -> tuple[float, pd.DataFrame]:
    """Núcleo computacional compartilhado."""

def obj_felipe_lucas(x, args) -> float:
    of, _ = _avaliar_projeto(x, args)
    return of

def obj_teste(x, args) -> tuple[float, pd.DataFrame]:
    return _avaliar_projeto(x, args)
```

Bônus: a refatoração resolve simultaneamente [[07_Issues/Issue - Args extras em obj_teste]],
pois `_unpack_args` agora respeita o quinto elemento como penalidade.

## Vínculo

- [[04_Codigo/fundacao.py]]
- [[03_Otimizacao/Formulação do Problema]]
- [[07_Issues/Issue - Args extras em obj_teste]] — também resolvida na mesma sprint
- [[07_Issues/Lista Mestre de Issues]]
