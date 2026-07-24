---
tags: [refactor, sprint, log, observabilidade, logging]
data: 2026-04-28
branch: refactor/core-architecture
escopo: Sprint 4.4 — Structured logging
---

# Sprint 4.4 — Structured logging

> Sprint enxuta de observabilidade. Adiciona um logger JSON-por-linha
> opcional em `core.observability`, com contexto de run via
> `contextvars`, e plumbing de eventos nomeados em todas as camadas
> que produzem progresso (cache, EGO, optimize, recorder).

## TL;DR

> Cada operação relevante da FundaIA agora pode emitir uma linha JSON
> com nome de evento estável (`ego.iter`, `cache.hit`,
> `optimize.start`, `experiment.record_rep`...), filtrável em tempo
> real e processável pós-execução.

## Para leigos

Logs antes eram `print` ad-hoc espalhados, perdiam contexto e
sumiam quando o terminal fechava. Agora qualquer execução pode ser
ligada a um modo "estruturado": uma linha JSON por evento, com
timestamp UTC, identificador da run e dados extras (qual iteração,
qual rep, qual `of_min`, etc.). É o tipo de log que você consegue
abrir no `jq`, jogar num `pandas.read_json`, plotar a curva de
convergência em tempo real e reproduzir tudo depois.

> **Analogia.** É a "caixa preta" do voo. O `ExperimentRecorder`
> guarda o resumo final no hangar (Sprint 4.2). O logger
> estruturado é a *cockpit voice recorder*: cada decisão durante o
> voo, com o tempo certinho e o contexto da run.

## Para o time técnico

### Onde mora

Novo pacote `core/observability/`:

```
core/observability/
├── __init__.py        # re-exports
└── logging.py         # JsonFormatter, configure_logging, get_logger, run_context
```

### API

```python
from core.observability import configure_logging, get_logger, run_context

# Setup uma vez (em scripts, notebooks, app.py se quiser)
configure_logging(level="DEBUG", log_file="experiments/run.jsonl")

# Em qualquer módulo
log = get_logger("optimize")
log.info("rep start", extra={"event": "optimize.rep_start",
                              "rep": 0, "seed": 42})

# Tag de contexto: todo evento dentro do bloco recebe run_id
with run_context("20260428T193245Z-a1b2c3d4"):
    optimize(projeto, config, recorder=rec)
```

### Decisões

1. **Stdlib `logging` puro**, sem `structlog`/`loguru`. Zero
   dependência nova; integra com pytest `caplog` e qualquer
   handler externo do usuário.
2. **Silent by default**. Importar `core` não toca em
   `logging.basicConfig`. Bibliotecas usuárias do `core` não
   recebem logs sem opt-in.
3. **JSON por linha** via `JsonFormatter`. Saída direta para
   `jq`, ingestão em parquet, busca por chave-valor.
4. **`run_context` via `contextvars`**. Funciona com
   `asyncio` e threads. Nested calls restauram o estado anterior.
5. **Idempotência**. `configure_logging` chamado N vezes não
   acumula handlers — o teste
   `test_idempotent_does_not_duplicate_handlers` garante.

### Eventos emitidos

| Origem                          | Evento                       | Extras-chave                                    |
|---------------------------------|------------------------------|-------------------------------------------------|
| `core.api.optimize`             | `optimize.start`             | `n_rep`, `n_pop`, `n_gen`, `n_fund`, `base_seed`|
| `core.api.optimize`             | `optimize.rep_start`         | `rep`, `seed`                                   |
| `core.api.optimize`             | `optimize.rep_end`           | `rep`, `seed`, `of_rep`, `wall_time_s`           |
| `core.api.optimize`             | `optimize.failed`            | `error`, `wall_time_s`                          |
| `core.api.optimize`             | `optimize.end`               | `best_of`, `best_seed`, `wall_time_s`           |
| `core.optimization.ego`         | `ego.iter`                   | `iter`, `of_min`, `n_train`                     |
| `core.optimization.cache`       | `cache.hit`/`miss`/`disk_hit`| `key` (16 chars), `size`/`disk_dir`             |
| `core.io.experiments`           | `experiment.begin`           | `run_id`, `run_dir`                             |
| `core.io.experiments`           | `experiment.record_rep`      | `rep_id`, `seed`, `of_best`, `wall_time_s`      |
| `core.io.experiments`           | `experiment.end`/`cancel`    | `run_id`, `status`, `best_of`, `n_rep`, `error` |

### Composição com sprints anteriores

- **Sprint 4.1 (cache)** — agora você consegue auditar `hit/miss`
  ao vivo (`grep '"event":"cache.hit"' run.jsonl`) e diagnosticar
  por que um run não está aproveitando o cache esperado.
- **Sprint 4.2 (recorder)** — o `run_id` do recorder é injetado
  automaticamente via `run_context`, então uma única filtragem
  por `run_id` recupera todos os eventos de um experimento
  específico (logs + manifest + parquet).

## Validação

```text
=== suite ===
  171 passed in ~6 s
    test_observability.py     9  (novo)

=== smoke ===
  Tail dos eventos durante optimize() pequena (n_pop=4, n_gen=1, n_rep=1):
    {"event":"optimize.start", "n_rep":1, ...}
    {"event":"experiment.begin", "run_dir":".../log-e2e", ...}
    {"event":"optimize.rep_start", "rep":0, "seed":42, ...}
    {"event":"ego.iter", "iter":1, "of_min":..., ...}
    {"event":"optimize.rep_end", "rep":0, "of_rep":..., ...}
    {"event":"experiment.record_rep", "rep_id":0, ...}
    {"event":"experiment.end", "status":"completed", ...}
    {"event":"optimize.end", "best_of":..., ...}
```

## Vínculos

- [[10_Melhorias/Logging Estruturado]] — esta sprint executa
- [[10_Melhorias/MOC - Melhorias]]
- [[12_Auditoria/Sprint 4.2 - Experiment persistence - 2026-04-28]] — recorder, fonte do `run_id`
- [[12_Auditoria/Sprint 4.1 - Surrogate cache - 2026-04-28]] — cache emite hit/miss
