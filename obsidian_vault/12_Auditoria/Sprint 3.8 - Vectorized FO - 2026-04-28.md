---
tags: [refactor, sprint, log, arquitetura, performance, vetorizacao]
data: 2026-04-28
branch: refactor/core-architecture
escopo: Sprint 3.8 — Vetorizacao da funcao objetivo (overlap N×N)
---

# Sprint 3.8 — Vectorized FO

> Log da oitava (e ultima) sub-sprint da Sprint 3 (refactor estrutural).
> Substitui o laco duplo `df.iterrows()` da verificacao de sobreposicao
> em `_avaliar_projeto` por uma matriz N×N inteiramente em numpy. O
> nucleo computacional fica entre 100 e 160 vezes mais rapido para
> casos de tamanho realista, sem alterar nenhum bit do resultado.

## Escopo executado

| # | Item | Status |
|---|---|---|
| 1 | Adicionar `sobreposicao_matrix(xmin, xmax, ymin, ymax)` em `core/engineering/packing.py` | ✅ |
| 2 | Reexportar a nova funcao em `core/engineering/__init__.py` e em `fundacao.py` | ✅ |
| 3 | Reescrever o trecho de overlap em `_avaliar_projeto` para chamar `sobreposicao_matrix` | ✅ |
| 4 | Adicionar `TestSobreposicaoMatrix` em `tests/test_engenharia.py` (5 casos) | ✅ |
| 5 | Validar suite completa (122 testes) e regressao `of = 19,70604234767181` | ✅ |

## Decisao de design

### Por que uma funcao nova em vez de mudar `sobreposicao_sapatas`?

- **`sobreposicao_sapatas`** continua sendo a interface escalar
  (8 vertices por sapata) usada por testes, notebooks legados e por
  qualquer consumidor externo. Manter sua assinatura intocada evita
  quebras silenciosas.
- **`sobreposicao_matrix`** e uma funcao nova, vetorial, com assinatura
  pensada para o caminho quente (4 arrays de bounds AABB). A unica
  responsabilidade de `_avaliar_projeto` e reduzir os vertices a esses
  bounds (o que ja tinha — colunas `x1`, `x2`, `y1`, `y3` sao os bounds
  por construcao) e somar a matriz por linha.

### Algoritmo (broadcast N×N)

```python
overlap_x = np.maximum(0.0,
    np.minimum(xmax[:, None], xmax[None, :])
    - np.maximum(xmin[:, None], xmin[None, :]))
overlap_y = np.maximum(0.0,
    np.minimum(ymax[:, None], ymax[None, :])
    - np.maximum(ymin[:, None], ymin[None, :]))
overlap = overlap_x * overlap_y
np.fill_diagonal(overlap, 0.0)   # equivalente ao `if jdx == idx: continue`
g_sob = overlap.sum(axis=1) / (h_x * h_y)
```

Cada celula da matriz reproduz exatamente as operacoes do laco
escalar (mesmas subtracoes, mesmos `min`/`max`, mesma multiplicacao).
A diagonal e zerada para casar com o `j != i` original.

## Validacao

### Regressao numerica

```text
=== test_avaliar_projeto.py::test_baseline_three_foundations_returns_19_706 ===
PASSED — of == 19.70604234767181 (rel=1e-12)
```

### Suite completa

```text
=== suite ===
  122 passed in ~5 s
    test_api.py              26
    test_avaliar_projeto.py   6
    test_benchmark.py        15
    test_domain.py           15
    test_ego_historico.py     8
    test_engenharia.py       31  (26 + 5 novos)
    test_io.py               21
```

### Comparacao bit-exata vs versao escalar

```text
N=   5   loop=  0.05 ms   vec=  0.04 ms   speedup=   1.4x
N=  50   loop=  5.18 ms   vec=  0.03 ms   speedup= 162.0x
N= 200   loop=115.40 ms   vec=  1.15 ms   speedup= 100.0x
np.allclose(matrix, ref_loop, atol=0, rtol=0) -> True
```

A igualdade absoluta (`atol=0, rtol=0`) confirma que o calculo
matricial reproduz cada celula exatamente como o laco escalar antigo,
o que justifica a robustez da regressao `of = 19,70604234767181`.

## Testes adicionados

| Teste | O que valida |
|---|---|
| `test_diagonal_zerada` | Diagonal da matriz e exatamente 0 (substitui o `j != i`) |
| `test_matriz_simetrica` | `M[i,j] == M[j,i]` para qualquer par |
| `test_concorda_com_versao_escalar` | Em 4 sapatas (12 pares), cada celula coincide com `sobreposicao_sapatas(*verts_i, *verts_j)` |
| `test_caso_sem_sobreposicao_devolve_zeros` | Tres sapatas distantes -> matriz zero |
| `test_caso_unitario_n_igual_1` | Apenas 1 sapata -> matriz 1×1 com diagonal zero |

## Implicacao pratica

A funcao objetivo e chamada `n_pop * n_gen` vezes por execucao do
EGO/GA. Para `n_pop=2000` (objetivo do Roadmap Fase 2) com 50 sapatas,
o tempo gasto so na verificacao de sobreposicao caia de varios
segundos por iteracao para alguns milisegundos. O custo de uma
matriz `O(N²)` em memoria e desprezivel (200×200 floats = 320 KB).

## Pendencias para sprints futuras

A Sprint 3.7 antecipava que, apos a vetorizacao, o adapter
`projeto_to_dataframe` em `core/api/_adapter.py` poderia ser
eliminado e `_avaliar_projeto` consumiria `FundacaoProjeto`
diretamente. Esse passo e maior do que a vetorizacao em si e
esta deferido para uma Sprint 4.x: envolve reescrever
`_avaliar_projeto` para receber arrays/objetos do dominio em vez do
DataFrame anotado, redirecionar todos os consumidores existentes
(notebooks, paginas Streamlit) para a nova assinatura e migrar a
serie de colunas anotadas (e.g. `tau_sd2 - c1`) para um formato de
saida estruturado. Esta Sprint 3.8 mantem o DataFrame intacto: o
contrato externo e os testes nao mudam.

Os outros lacos `df.apply(..., axis=1)` (sigma max/min, geometria,
puncao) tambem sao vetorizaveis — `verificacao_puncao_sapata` exige
mais cuidado por ter ramificacao interna — mas tem custo unitario
varias ordens de grandeza menor que o overlap antigo, entao ficam de
fora desta sprint.

## Vinculos

- [[10_Melhorias/Roadmap Sugerido]]
- [[10_Melhorias/MOC - Melhorias]]
- [[10_Melhorias/Refactor - Vetorização da FO]] — esta sprint executa
- [[12_Auditoria/Sprint 3 - Refactor estrutural - kickoff - 2026-04-27]]
- [[12_Auditoria/Sprint 3.7 - Pydantic config - 2026-04-28]]
- [[01_Projeto/Convenções do Projeto]]
