---
tags: [refactor, sprint, log, persistencia, experimentos, mlops, artigo]
data: 2026-04-28
branch: refactor/core-architecture
escopo: Sprint 4.2 — Persistência de experimentos
---

# Sprint 4.2 — Experiment persistence

> Última sub-sprint da **Fase 2** do Roadmap. Cada chamada a
> `optimize` passa a poder ser persistida como uma **pasta
> autodescritiva** em `experiments/<run_id>/`, contendo configuração,
> ambiente, fingerprint do projeto, histórico EGO completo por
> repetição (Parquet), CSV ergonômico de resumo, métricas agregadas
> e artefatos (DXF, plots, etc.). Pensado para alimentar
> diretamente os plots e tabelas do artigo da IC.

## TL;DR — O que mudou, em uma linha

> A FundaIA agora pode **gravar todos os detalhes de cada otimização**
> em uma pasta dedicada, com tudo que é necessário para **reproduzir,
> validar e comparar** runs depois — inclusive escrever a seção de
> resultados do artigo a partir dela.

## Para leigos — o que é isso e por que melhorou?

Antes da sprint, rodar a otimização no FundaIA era como **fazer uma
prova sem caderno de rascunho**: ao final você ficava com o resultado
final, mas perdia tudo que aconteceu no caminho — quanto a função
custo melhorou em cada iteração, quanto tempo cada repetição levou,
qual era exatamente a configuração do dia, qual versão do código
gerou aquele número. Para validar resultados ou escrever o artigo,
você teria que rodar tudo de novo (e rezar para dar igual).

Agora, sempre que você passar um **gravador** (`ExperimentRecorder`)
para a otimização, a FundaIA cria uma pasta com nome único e
salva ali, em formato aberto, **tudo que aconteceu**:

```
experiments/2026-04-28T19h32m45s-a1b2c3d4/
├── manifest.json     ← cartão de visita do experimento
├── config.json       ← os parâmetros usados (n_pop, n_gen, etc.)
├── env.json          ← Python, numpy, sklearn, OS, commit do git…
├── project.json      ← qual entrada (problema) foi otimizada
├── history/
│   ├── rep_000.parquet  ← histórico inteiro da repetição 0
│   └── rep_001.parquet  ← histórico inteiro da repetição 1
├── summary.csv       ← uma linha por repetição (abre no Excel)
├── metrics.json      ← métricas agregadas para o artigo
└── artifacts/        ← DXF, plots, qualquer arquivo extra
```

> **Analogia.** É como um laboratório de química: a partir de hoje,
> cada experimento sai com a etiqueta da bancada, a data, a
> temperatura, o lote dos reagentes e a foto do resultado. Mesmo que
> daqui a três meses o aluno esqueça o que rodou, dá para abrir a
> caixa e descobrir tudo.

**O que não mudou:**

- A persistência é **opcional**. Se você não passar um gravador, a
  FundaIA roda exatamente como antes — nenhum byte vai pro disco.
- O resultado numérico é idêntico: a regressão
  `of = 19,70604234767181` continua passando.
- Não há dependência nova obrigatória: usamos `pyarrow` (que já
  vinha instalado por causa do pandas) e os formatos abertos
  JSON, CSV e Parquet — qualquer linguagem ou ferramenta lê.

**O que melhorou de fato:**

| Antes                                         | Agora                                                                 |
|-----------------------------------------------|-----------------------------------------------------------------------|
| Resultado mostrado na tela e perdido          | Pasta com tudo, indexada por timestamp                                |
| "Esse OF foi com qual config? esqueci."       | `config.json` te diz, com round-trip via Pydantic                     |
| "Quem rodou? Qual commit?"                    | `env.json` carimba versão de Python, libs e commit do git             |
| "Vou ter que rodar de novo pra plotar"        | Histórico EGO completo em Parquet, lê em < 1 ms                        |
| "Como a OF caiu ao longo das iterações?"      | `metrics.json` já traz `auc_best_so_far`, `convergence_iter` etc.     |
| "Vou ter que escrever uma tabela no artigo"   | `summary.csv` abre no LaTeX/Excel direto                               |

## Para o time técnico — projeto e contrato

### Onde mora

Novo módulo: **`core/io/experiments.py`**. Reexportado a partir de
`core.io`. Adicionado `pyarrow>=14.0,<24.0` ao `requirements.txt`
(motor Parquet usado pelo histórico).

### API pública

```python
from core.io.experiments import (
    SCHEMA_VERSION,           # "1.0"
    ExperimentRecorder,
    ExperimentManifest,
    ExperimentRun,
    summarise_history,
    compute_metrics,
    load_experiment,
)
```

#### Recorder — ciclo de vida

```python
rec = ExperimentRecorder(root="experiments")  # ou disk_dir custom + run_id custom
rec.begin(config, projeto)                    # write manifest+config+env+project
for rep in range(n_rep):
    history = ego_01_architecture(...)
    rec.record_rep(rep_id=rep, seed=..., history=history,
                   wall_time_s=...)            # write parquet + update CSV
rec.write_artifact("best_design.dxf", dxf_bytes)   # opcional
rec.end()                                      # write metrics + final manifest
# em caso de exceção:
# rec.cancel(repr(exc))                         # marca status="failed"
```

Todas as escritas usam **temp-then-rename** (POSIX-atomic) — um
`SIGKILL` no meio de uma escrita deixa, na pior hipótese, o último
manifest válido em disco, nunca um JSON pela metade.

#### `optimize()` — wire-up opt-in

```python
from core.api import optimize, OptimisationConfig
from core.io.experiments import ExperimentRecorder
from core.optimization.cache import SurrogateCache

rec   = ExperimentRecorder("experiments")
cache = SurrogateCache(maxsize=128, disk_dir="experiments/_cache")

result = optimize(projeto, config,
                  recorder=rec,    # opt-in: persiste a run
                  cache=cache)     # opt-in: cache do GPR (Sprint 4.1)
```

Default mantém comportamento histórico (regressão preservada,
`recorder=None` e `cache=None`).

### Schema da pasta — `schema_version="1.0"`

| Arquivo                         | Conteúdo                                                                                          |
|---------------------------------|---------------------------------------------------------------------------------------------------|
| `manifest.json`                 | `schema_version`, `run_id`, `created_at`/`completed_at`, `status` (running/completed/failed), `config`, `env`, `project`, `metrics`, `summary`, `error` |
| `config.json`                   | `OptimisationConfig.model_dump()` — round-trippable Pydantic                                       |
| `env.json`                      | Python version, OS, versões pinadas (`numpy`, `pandas`, `scikit-learn`, `mealpy`, `pydantic`, `joblib`, `pyarrow`, `ezdxf`), git rev/branch/dirty |
| `project.json`                  | Hash SHA-256 da serialização canônica do `FundacaoProjeto` + sumário (`n_fund`, `n_comb`, `f_ck`, `cobrimento`, `pilar_labels`) |
| `history/rep_NNN.parquet`       | DataFrame inteiro retornado por `ego_01_architecture` para a rep `NNN`                            |
| `summary.csv`                   | Uma linha por repetição (`rep_id`, `seed`, `wall_time_s`, `of_best`, `of_initial`, …)              |
| `metrics.json`                  | Agregação inter-rep (`best_of`, `mean_of`, `std_of`, `median_of`, `best_rep_id`, `mean_convergence_iter`, `mean_auc_best_so_far`, …) |
| `artifacts/`                    | Pasta livre para DXF, plots, blobs adicionais                                                       |

### Métricas paper-grade (`summarise_history`)

Para cada repetição, calculadas direto do histórico EGO:

- **`of_initial`** — melhor OF na população inicial (LHS, ITER=0).
- **`of_best`** — melhor OF do run inteiro.
- **`best_iter`** — primeira iteração em que `of_best` foi atingido.
- **`improvement_abs` / `improvement_rel`** — ganho do EGO sobre o LHS.
- **`convergence_iter`** — primeira iteração em que o melhor-até-aqui
  ficou a $\leq 10^{-6}$ (relativo) do `of_best`.
- **`convergence_ratio`** — `convergence_iter / n_gen`.
- **`auc_best_so_far`** — área sob a curva *normalizada*
  `[best_so_far(t) - of_best] / [of_initial - of_best]`. Em `[0, 1]`;
  **menor é melhor** (convergência mais rápida).
- **`n_unique_x`** — diversidade real do run (de-duplica por vetor de projeto).
- **`t_total_s` / `mean_t_per_iter_s`** — contabilidade de tempo.

E, agregado entre reps (`compute_metrics`):

- `best_of`, `worst_of`, `mean_of`, `std_of`, `median_of`,
  `best_rep_id`, `mean_convergence_iter`, `mean_auc_best_so_far`,
  `mean_improvement_rel`, `mean_t_total_s`, `wall_time_total_s`.

Todas essas saem direto para a tabela do artigo (`mean ± std`,
`best`, `convergence iter`, `wall-time`).

### Round-trip

```python
from core.io.experiments import load_experiment
run = load_experiment("experiments/2026-04-28T19h32m45s-a1b2c3d4")
print(run.manifest.status)             # "completed"
print(run.manifest.metrics["best_of"]) # número
df_rep0 = run.history[0]               # DataFrame Parquet → pandas
```

`load_experiment` rejeita explicitamente um `schema_version`
desconhecido — quando o schema evoluir (provável no futuro), há
ponto de entrada óbvio para o migrador.

## Validação

### Suite

```text
=== suite ===
  162 passed in ~6 s
    test_api.py                26
    test_avaliar_projeto.py     6
    test_benchmark.py          15
    test_cache.py              23
    test_domain.py             15
    test_ego_historico.py       8
    test_engenharia.py         31
    test_experiments.py        17  (novo)
    test_io.py                 21
```

### Smoke test (com EGO real, problema de 3 fundações)

```text
== folder layout ==
  config.json                (179 B)
  env.json                   (544 B)
  history/rep_000.parquet  (9779 B)
  history/rep_001.parquet  (9778 B)
  manifest.json             (2746 B)
  metrics.json               (394 B)
  project.json               (214 B)
  summary.csv                (454 B)

== manifest highlights ==
  schema_version : 1.0
  status         : completed
  best_of        : 10.312241
  mean_of        : 12.910844
  std_of         :  2.598603
  best_rep_id    : 1
  wall_time_total_s : 0.714
  project hash   : b36f28f9adff580f...
  python         : 3.12.12
  git rev        : 925eb9131d  dirty=True

== round-trip ==
  reps loaded   : [0, 1]
  rep 0 shape   : (10, 15)
  rep 0 cols    : ID, ITER, X_0..X_8, OF, FIT, OF EVALUATIONS,
                  TIME CONSUMPTION (s)
```

A pasta inteira para esse smoke run pesa **~25 KB**, sem artifacts.

## Testes adicionados (17 novos casos em `tests/test_experiments.py`)

| Classe                       | Casos | O que valida |
|------------------------------|-------|--------------|
| `TestSummariseHistory`       | 4     | Métricas para histórico monótono, sem-melhoria, com duplicatas; histórico vazio levanta |
| `TestComputeMetrics`         | 1     | Agregados `best/mean/std/median/best_rep_id/mean_*` corretos em mock controlado |
| `TestExperimentRecorder`     | 7     | `begin` cria 4 arquivos esperados; `record_rep` grava parquet + CSV; `end` finaliza com metrics; `cancel` marca `failed`; artifact rejeita path traversal; artifact persiste bytes; `record_rep` antes de `begin` levanta |
| `TestLoadExperiment`         | 3     | Round-trip preserva histórico; schema desconhecido é rejeitado; manifest ausente → `FileNotFoundError` |
| `TestOptimizeIntegration`    | 2     | `optimize(projeto, cfg, recorder=rec)` produz pasta completa; exceção no EGO marca run como `failed` |

## Implicações práticas

### Para o artigo da IC

- A tabela "EGO+GPR vs GA puro vs LHS" sai direto do
  `metrics.json` agregado de cada experimento.
- O gráfico "best-so-far por iteração" plota direto do
  `history/rep_*.parquet` (uma linha por rep).
- A reprodutibilidade fica **garantida em disco**: alguém clonando
  o repositório consegue carregar uma run, ler o `env.json`,
  fazer `git checkout <rev>` no commit certo, e re-rodar.

### Composição com sprints anteriores

- **Sprint 3.7 (Pydantic)** — `OptimisationConfig.model_dump()` é o
  motor do `config.json`. Adicionar campo novo na config aparece
  automaticamente no manifest, sem mudança aqui.
- **Sprint 3.8 (vetorização)** — cada repetição roda mais rápido,
  então `wall_time_total_s` no manifest reflete o ganho.
- **Sprint 4.1 (cache)** — `optimize` aceita `cache=` opcional;
  o cache pode usar `experiments/<run_id>/_cache/` (ou compartilhado)
  e o `metrics.json` da run capturado vai ter o tempo já beneficiado.

### Próximos passos sugeridos

1. **Sprint 4.3 — Logging estruturado** (já listado pendente
   em `MOC - Melhorias`): emitir eventos JSON em paralelo às escritas
   do recorder, para acompanhamento ao vivo.
2. **CLI**: um `python -m fundaia.cli list-experiments
   --where best_of < 20` lendo só os manifests é literal-mente um
   `glob + json.load`.
3. **Migração de schema**: quando o schema evoluir para `2.0`, o
   `load_experiment` já reclama; basta adicionar um `_migrate_1_0_to_2_0`
   no caminho do erro.

## Vínculos

- [[10_Melhorias/Roadmap Sugerido]] — Fase 2 fechada
- [[10_Melhorias/MOC - Melhorias]]
- [[10_Melhorias/Persistência de Experimentos]] — esta sprint executa
- [[12_Auditoria/Sprint 4.1 - Surrogate cache - 2026-04-28]] — sprint anterior
- [[12_Auditoria/Sprint 3.7 - Pydantic config - 2026-04-28]] — base do `config.json`
- [[01_Projeto/Convenções do Projeto]]
