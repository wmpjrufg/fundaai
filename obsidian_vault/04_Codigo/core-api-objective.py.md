---
tags: [codigo, api, performance]
file: core/api/objective.py
loc: ~220
sprint_criado: 3.9
---

# `core/api/objective.py`

Criado no Sprint 3.9. Lar canônico das implementações da função-objetivo. Importado por `core.api.benchmark`, `core.api.optimize`, e (via import lazy) por `fundacao.obj_felipe_lucas`.

## Funções públicas

| Função | Descrição |
|---|---|
| `avaliar_projeto_fast(x, args, *, penalty=None)` | **FO vetorizada** — numpy broadcasting puro, sem `df.apply`. Retorna escalar `of`. ~0,1 ms/chamada. **Use em loops de otimização.** |
| `avaliar_projeto_legacy(x, args)` | Wrapper fino sobre `fundacao._avaliar_projeto` (import lazy). Retorna escalar `of`. ~6–13 ms/chamada. Use apenas para comparação/validação. |

### Internals

| Função | Descrição |
|---|---|
| `_unpack(args)` | Extrai `(df, n_comb, f_ck, cob_m, pen_default)` da tupla de args |

## Benchmark (medido em 2026-06-05)

| Cenário | `avaliar_projeto_legacy` | `avaliar_projeto_fast` | Speedup |
|---|---|---|---|
| 3 fund / 2 comb | 6,36 ms | 0,090 ms | ~70× |
| 10 fund / 4 comb | 10,29 ms | 0,132 ms | ~78× |
| 30 fund / 4 comb | 12,79 ms | 0,148 ms | ~86× |

Validação numérica: `diff = 0.00e+00` para soluções factíveis e infactíveis (46 testes passando).

## Por que aqui e não em `fundacao.py`?

`fundacao.py` é um shim de compatibilidade com regra explícita: "safe to import but should not grow new functions" (ARCHITECTURE.md). `core.api` é a camada correta para orquestrar engenharia + otimização. `core/api/objective.py` segue as regras de dependência: importa de `core.engineering` (sobreposicao_matrix) e não importa Streamlit.

## Import circular (como foi resolvido)

O ciclo `fundacao → core.api.objective → core.api.benchmark → fundacao` foi quebrado com import deferido em `fundacao.obj_felipe_lucas`:

```python
def obj_felipe_lucas(x, args):
    from core.api.objective import avaliar_projeto_fast as _fast  # noqa: PLC0415
    return _fast(x, args)
```

## Links

- [[04_Codigo/fundacao.py]]
- [[10_Melhorias/Questao Aberta - Custo da FO e Justificativa do EGO]]
- [[10_Melhorias/Refactor - Vetorização da FO]]
