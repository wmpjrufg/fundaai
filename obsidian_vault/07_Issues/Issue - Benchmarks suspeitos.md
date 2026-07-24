---
tags: [issue, medio, ciencia, resolvido]
file: metapy_toolbox/benchmark.py
severity: medio
status: resolvido
resolvido_em: 2026-04-27
resolvido_em_branch: fix/code-sanitization-and-tests
---

# Issue — Funções de benchmark suspeitas

> [!success] Resolvido em 2026-04-27 (Sprint 2, branch `fix/code-sanitization-and-tests`)
> `griewank` e `powell` corrigidos contra a referência canônica
> Surjanovic & Bingham (sfu.ca/~ssurjano). Testes regressivos
> adicionados em `tests/test_benchmark.py` (15 casos, 100% passando).
>
> ```python
> griewank([0, 0, 0]) == 0     # antes: dependia só do último x_i
> powell([0, 0, 0, 0]) == 0    # antes: estourava IndexError
> powell([1, 2, 3, 4]) == 1512 # valor de referência travado
> ```

## Sintoma original

Em [[04_Codigo/metapy_toolbox - benchmark.py]]:

- **`griewank(x)`** — o produto estava **fora do loop** (indentação errada), de modo que usava só o último `x_i` e o último índice `i`. Numericamente errado para qualquer `d > 1`.
- **`powell(x)`** — usava indexação `x[4*i]`, `x[4*i+1]`, etc., com `i` começando em 1 (estilo Fortran/MATLAB). Em Python 0-based, isso estourava o último índice quando o vetor tinha tamanho exatamente múltiplo de 4 (caso canônico do Powell, `d ∈ {4, 8, 12, ...}`).

## Por que era problema

[[10_Melhorias/Validação contra problema-benchmark]] propõe **usar essas funções como ground truth** para validar EGO/GA. Se o ground truth está errado, a validação não vale.

## Correção aplicada

### `griewank`
- Produto movido para dentro do loop, partindo de `prod = 1.0`.
- Cada iteração multiplica por `cos(x_i / sqrt(i + 1))`.
- Variáveis `sum`/`prod` renomeadas para `soma`/`produto` (evita shadowing dos built-ins Python).

### `powell`
- Indexação 1-based substituída pelo equivalente 0-based: para `i = 0..n_blocks - 1`, lê `x[4i], x[4i+1], x[4i+2], x[4i+3]`.
- `ValueError` explícito quando `len(x) % 4 != 0` — prevenção defensiva contra falha silenciosa.
- Variáveis `term1..term4` mantidas com nomes legíveis (`a, b, c, d`).

### Testes adicionados em `tests/test_benchmark.py`
- 8 testes de mínimos conhecidos (sphere, rosenbrock, rastrigin, ackley, zakharov, easom, dixon_price, goldstein_price).
- 3 testes específicos para `griewank` (mínimo em zero, produto efetivo, simetria).
- 4 testes específicos para `powell` (mínimo em zero d=4 e d=8, raise para d não-múltiplo de 4, valor pinado em (1,2,3,4) = 1512).

## Vínculo

- [[04_Codigo/metapy_toolbox - benchmark.py]]
- [[10_Melhorias/Validação contra problema-benchmark]]
- [[10_Melhorias/Testes Automatizados]]
- [[07_Issues/Lista Mestre de Issues]]
