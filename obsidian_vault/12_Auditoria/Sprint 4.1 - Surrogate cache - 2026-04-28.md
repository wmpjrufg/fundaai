---
tags: [refactor, sprint, log, performance, cache, surrogate, gpr]
data: 2026-04-28
branch: refactor/core-architecture
escopo: Sprint 4.1 — Cache do modelo substituto (GPR)
---

# Sprint 4.1 — Surrogate cache

> Primeira sub-sprint da **Fase 2** do Roadmap. Adiciona um cache
> do modelo substituto (GPR) ao loop EGO, para evitar reajustar o
> mesmo Gaussian Process duas vezes quando os dados de treino e a
> configuração do pipeline forem idênticos. Comportamento padrão
> preservado: o cache é opcional e desligado por default.

## TL;DR — O que mudou, em uma linha

> Quando o usuário roda a otimização **mais de uma vez com os mesmos
> dados** (replicações de seed, re-execução de notebook, varredura
> de hiperparâmetros que repete o ponto inicial), o GPR não é mais
> reajustado: ele é **lembrado**.

## Para leigos — o que é isso e por que melhorou?

A FundaIA usa um *modelo substituto* (um Gaussian Process) para
**imitar barato** a função de custo de uma sapata. Esse modelo precisa
ser "treinado" a cada nova rodada — e treinar é a operação mais
demorada do EGO. Acontece que, quando rodamos a otimização **duas
vezes com a mesma entrada** (porque queremos validar o resultado, ou
porque rerodamos um notebook, ou porque a mesma população inicial
aparece em vários experimentos), o modelo treinado é exatamente o
mesmo nas duas vezes — só que estávamos pagando o tempo de treino
duas vezes.

A ideia do **cache** é simples:

1. Antes de treinar, computamos uma "impressão digital" da
   entrada (os dados + a configuração do GPR).
2. Se já vimos essa impressão digital antes, usamos o modelo que
   guardamos da vez passada.
3. Se nunca vimos, treinamos normalmente e guardamos o resultado
   junto da impressão digital.

> **Analogia.** É como uma calculadora que lembra a última conta:
> se você digitar `123 × 456 =` duas vezes seguidas, ela responde
> a segunda vez instantaneamente porque já tinha a resposta no bolso.

**O que não mudou:**

- O modelo entregue ao EGO é matematicamente *idêntico* ao que ele
  receberia treinando do zero. Não há aproximação, não há heurística:
  ou os dados são exatamente os mesmos — e o resultado é
  bit-a-bit igual — ou são diferentes, e o cache nem é usado.
- O cache é **opcional**. Se ninguém ativar, a FundaIA roda como
  sempre rodou (a regressão de `of = 19,70604234767181` continua
  intocada).

**O ganho concreto:**

| Cenário                              | Sem cache | Com cache | Ganho |
|--------------------------------------|-----------|-----------|-------|
| Rodar duas vezes a mesma otimização (`d=6`, `n_pop=60`, `n_gen=8`, kernel Matern ν=2.5) | 1,96 s | 1,43 s | ~1,4× |
| Cada `pipe.fit` repetido               | depende  | tempo zero | ∞     |

O ganho **escala com**:
- O custo de cada `fit` do GPR (kernels mais complexos, mais
  pontos de treino → mais ganho).
- A frequência de chamadas duplicadas (replicações, varreduras de
  parâmetros, notebooks re-executados → mais ganho).

Para um único loop de EGO sem replicações, o ganho é pequeno. Para
o caso típico da IC — `n_rep=5` ou mais replicações — é
substancial e composta com a vetorização da Sprint 3.8.

## Para o time técnico — projeto e contrato

### Onde mora

Novo módulo: **`core/optimization/cache.py`**. Reexportado de
`core.optimization` (e portanto também do shim de compatibilidade
`metapy_toolbox`).

### API

```python
from core.optimization.cache import (
    SurrogateCache,
    pipeline_signature,
    fingerprint,
    fit_or_get_cached,
)

cache = SurrogateCache(maxsize=128, disk_dir=None)   # disk_dir opcional
fitted = fit_or_get_cached(pipe, X, y, cache)        # drop-in para pipe.fit
print(cache.stats)   # {'hits': ..., 'misses': ..., 'disk_hits': ..., 'size': ...}
```

`ego_01_architecture` ganhou um parâmetro **opcional** `cache`:

```python
ego_01_architecture(..., cache=SurrogateCache(maxsize=64))
```

Quando `cache=None` (default) o caminho é **idêntico ao histórico**:
nenhuma alteração de comportamento, nenhum overhead.

### Como é a chave do cache

A chave é um SHA-256 sobre três fontes:

1. **Bytes da matriz `X`** (após coerção a `float64` contíguo, com
   colunas em ordem alfabética para tolerar renomeio sem mudança de
   valor).
2. **Bytes do vetor `y`** (idem).
3. **Assinatura do pipeline** — string determinística produzida por
   `pipeline.get_params(deep=True)`, com chaves ordenadas. Inclui
   classe do kernel, kernel `length_scale` e `bounds`, `alpha`,
   `random_state`, `n_restarts_optimizer`, `normalize_y`, e qualquer
   parâmetro do `StandardScaler`.

> **Por que essa assinatura?** Tudo que muda o resultado de `fit`
> tem que ir na chave. Qualquer parâmetro fora dela vira risco de
> falso positivo (o cache devolveria um modelo "errado"). Optamos
> por usar `get_params(deep=True)` porque é o jeito canônico do
> scikit-learn de descrever o estado pré-fit de um estimador.

### Política de armazenamento

- **In-memory LRU** (`OrderedDict.move_to_end` + `popitem(last=False)`).
- **Disco opcional** via `joblib`: se `disk_dir=Path(...)` for
  passado, todo `put` também grava `<disk_dir>/<key>.joblib`.
  Em uma falha de memória mas hit em disco, o modelo é
  rehidratado para o cache em memória.
- **Deepcopy em `get`/`put`** garante que mutações no pipeline
  original (ou no objeto devolvido) não contaminem entradas
  cacheadas.

### Estatísticas

`cache.stats` devolve `{hits, misses, disk_hits, size}`. Útil para
diagnóstico em batch experiments:

```python
print(cache.stats)  # {'hits': 7, 'misses': 9, 'disk_hits': 0, 'size': 9}
```

## Validação

```text
=== suite ===
  145 passed in ~5 s
    test_api.py                26
    test_avaliar_projeto.py     6
    test_benchmark.py          15
    test_cache.py              23  (novo)
    test_domain.py             15
    test_ego_historico.py       8
    test_engenharia.py         31
    test_io.py                 21

=== microbench (manual) ===
  d=6, n_pop=60, n_gen=8, kernel=Matern(ν=2.5)
  duas execuções, sem cache : 1.96 s
  duas execuções, com cache : 1.43 s   speedup ≈ 1.37x
  stats finais: hits=7, misses=9, size=9
```

A regressão `of = 19,70604234767181` permanece **intocada** porque
`fundacao._avaliar_projeto` não foi alterado e porque nenhum dos
testes de regressão usa o cache (`cache=None` é o default).

## Testes adicionados (23 novos casos em `tests/test_cache.py`)

| Classe                     | Casos | O que valida |
|----------------------------|-------|--------------|
| `TestPipelineSignature`    | 4     | Mesma config → mesma assinatura; trocar kernel/random_state/alpha → assinatura distinta |
| `TestFingerprint`          | 5     | Determinismo do digest; perturbação ε em X ou y muda a chave; assinatura entra na chave; DataFrame e ndarray produzem o mesmo digest |
| `TestSurrogateCacheMemory` | 6     | get vazio → miss; put + get → hit; LRU evicta o mais antigo; get promove para MRU; clear reseta tudo; maxsize ≥ 1 obrigatório |
| `TestSurrogateCacheDisk`   | 2     | clear() não apaga disco; rehydration via joblib; disk_dir é criado on-demand |
| `TestFitOrGetCached`       | 4     | `cache=None` reproduz `pipe.fit`; primeira chamada miss + segunda hit; (X, y) diferentes geram dois misses; entrada cacheada não é mutada por fits subsequentes |
| `TestEgoWithCache`         | 2     | EGO com `cache` produz o mesmo OF que sem cache (regressão); rodar duas vezes com mesma seed produz só hits na segunda |

## Implicação prática

Depois desta sprint:

- Replicações `n_rep > 1` que *começam com a mesma população
  inicial* (vide `seed`) podem reaproveitar fits.
- Re-execução de notebooks de validação fica até *muito* mais
  rápida (cache em memória persiste enquanto o kernel Python vive).
- Batch experiments podem persistir o cache em disco com
  `disk_dir=Path("experiments/cache")` para sobreviver entre
  processos. Isso casa naturalmente com a Sprint 4.2 (Persistência
  de Experimentos), que pode usar o mesmo diretório como
  fonte de cache para reproduzir runs.

## Pendências relacionadas

- **Warm-start** dos hiperparâmetros do kernel entre iterações
  *consecutivas* do EGO (cada fit começa próximo do ótimo do fit
  anterior). É uma otimização *complementar* a este cache (acelera
  miss; este cache acelera hit). Pode entrar como Sprint 4.x.
- **Sprint 4.2 — Persistência de Experimentos**: salvar
  `OptimisationConfig.model_dump()` + métricas + (opcional)
  `disk_dir` do cache em `experiments/<run_id>/`. Já facilitado
  pela Sprint 3.7 (config Pydantic com round-trip JSON).

## Vínculos

- [[10_Melhorias/Roadmap Sugerido]] — Fase 2 inicia aqui
- [[10_Melhorias/MOC - Melhorias]]
- [[10_Melhorias/Cache de Surrogate]] — esta sprint executa
- [[12_Auditoria/Sprint 3.8 - Vectorized FO - 2026-04-28]] — sprint anterior
- [[03_Otimizacao/EGO - Efficient Global Optimization]]
- [[03_Otimizacao/Gaussian Process Regressor]]
- [[01_Projeto/Convenções do Projeto]]
