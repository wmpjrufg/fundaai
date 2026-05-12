# FundaIA

> Plataforma computacional para o **dimensionamento otimizado de
> sapatas isoladas** em concreto armado, integrando critérios
> estruturais, geotécnicos e geométricos da NBR 6118 e da NBR 6122
> em um ambiente único de projeto. A solução é obtida por uma
> arquitetura híbrida do tipo **Efficient Global Optimization
> (EGO)**, combinando um modelo substituto baseado em **Regressão
> por Processos Gaussianos (GPR)** com um **Algoritmo Genético
> (AG)** como otimizador interno da função de aquisição *Expected
> Improvement* (EI).

[Aplicação publicada · `fundaai.streamlit.app`](https://fundaai.streamlit.app/)

---

## Sumário

- [Visão geral](#visão-geral)
- [Pipeline de execução](#pipeline-de-execução)
- [Arquitetura em camadas](#arquitetura-em-camadas)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Setup do ambiente](#setup-do-ambiente)
- [Como rodar a aplicação](#como-rodar-a-aplicação)
- [Como rodar uma otimização programaticamente](#como-rodar-uma-otimização-programaticamente)
- [Bancada de experimentos (EGO vs metaheurísticas)](#bancada-de-experimentos-ego-vs-metaheurísticas)
- [Progresso ao vivo (callback)](#progresso-ao-vivo-callback)
- [Persistência de experimentos](#persistência-de-experimentos)
- [Cache do surrogate](#cache-do-surrogate)
- [Suite de testes](#suite-de-testes)
- [Stack técnica](#stack-técnica)
- [Status atual](#status-atual)
- [Próximos passos](#próximos-passos)

---

## Visão geral

O dimensionamento de sapatas isoladas é tradicionalmente conduzido
por procedimentos iterativos de tentativa e verificação, dependentes
da experiência do projetista e frequentemente conservadores quanto
ao consumo de concreto. O **FundaIA** propõe uma alternativa
computacional que formaliza o problema como uma **otimização
mono-objetivo penalizada**, cuja função objetivo é o volume total de
concreto, sujeita a restrições de:

- **Tensão admissível do solo** (correlação empírica com SPT, NBR 6122).
- **Flexão composta na base da sapata** (σ_max e σ_min com excentricidades nos dois eixos).
- **Punção na seção crítica C** (NBR 6118 item 19.5).
- **Compatibilidade geométrica pilar-sapata** (balanço mínimo configurável).
- **Não sobreposição** entre fundações vizinhas (modelagem AABB).

A solução é obtida por uma arquitetura híbrida em três níveis: o
**EGO** ([Jones, Schonlau & Welch, 1998](https://doi.org/10.1023/A:1008306431147))
orquestra a busca, um **GPR**
([Williams & Rasmussen, 2006](https://gaussianprocess.org/gpml/))
atua como surrogate da função objetivo cara, e um **AG**
([Mealpy](https://mealpy.readthedocs.io/)) maximiza a função de
aquisição *Expected Improvement* a cada iteração.

A interface web é construída em **Streamlit** com tema dark
profissional e organiza-se em três páginas:

- **Início** — apresentação e download do template Excel.
- **Projeto de Sapatas** — ferramenta de projeto: o usuário fornece os
  dados via planilha, acompanha o progresso da otimização **ao vivo**
  (repetição corrente, iteração, melhor OF até agora), explora o
  arranjo otimizado em **planta 2D** e em **vista 3D interativa** com
  rotação livre em torno do eixo vertical, estuda a **convergência do
  EGO** em um gráfico zoomable e exporta resultado em Excel, DXF (CAD),
  JSON estruturado, HTML 3D stand-alone e PNG do gráfico de histórico.
- **Experimentos** — bancada científica de comparativos: roda EGO+GPR
  contra metaheurísticas puras (GA, PSO, GWO) sobre o mesmo problema,
  com o mesmo orçamento de avaliações reais e seeds reprodutíveis;
  entrega curva de convergência multi-algoritmo, tabela-resumo
  (best, mean ± std, AUC, avaliações até o ótimo, tempo) e matriz de
  p-valores Mann–Whitney prontas para o relatório científico.

A pesquisa associada é desenvolvida no contexto de uma Iniciação
Científica em andamento.

---

## Pipeline de execução

```
┌─────────────────┐   ┌─────────────────────┐   ┌────────────────────────┐
│ Excel de        │──▶│ core.io             │──▶│ core.api               │
│ entrada         │   │ read_projeto_       │   │ OptimisationConfig     │
│ (Nspt, ap, bp,  │   │ from_excel          │   │ (Pydantic v2,          │
│  F, M, xg, yg)  │   │ → FundacaoProjeto   │   │  validação rica)       │
└─────────────────┘   └─────────────────────┘   └────────────┬───────────┘
                                                              │
                                                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ core.api.optimize(projeto, config, *, recorder=None, cache=None)        │
│                                                                         │
│   for rep in range(config.n_rep):                                       │
│     1. População inicial via Latin Hypercube (seed = base_seed + rep)   │
│     2. Avaliação real da pseudo-objetivo penalizada Θ(x)                │
│     3. core.optimization.ego_01_architecture                            │
│        a. Treino do GPR (StandardScaler + GaussianProcessRegressor)     │
│           — opcionalmente via SurrogateCache (Sprint 4.1)               │
│        b. Maximização do Expected Improvement por GA interno (Mealpy)   │
│        c. Avaliação real do candidato e atualização da base             │
│        d. Repetir por n_gen iterações                                   │
│     4. Recorder.record_rep(rep_id, history, wall_time_s)                │
│        (manifest + parquet history + summary CSV — Sprint 4.2)          │
│   Recorder.end() → metrics.json paper-grade                             │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   ▼
            ┌───────────────────────────────────────────────────┐
            │ OptimisationResult(sapatas, best_of, best_seed,   │
            │                    per_rep_of)                     │
            │   + experiments/<run_id>/  (config, env, project,  │
            │     history, metrics, summary, artifacts)          │
            │   + DXF / Excel exportados via core.io             │
            └───────────────────────────────────────────────────┘
```

A função objetivo `_avaliar_projeto` é compartilhada por
`obj_felipe_lucas` (escalar, consumida pela otimização) e por
`obj_teste` (devolve também o DataFrame anotado com restrições e
tensões — usado por notebooks). O laço de sobreposição é vetorizado
em numpy desde a Sprint 3.8 (`core.engineering.sobreposicao_matrix`).

---

## Arquitetura em camadas

```
                      app.py (Streamlit page graph)
                                │
                          frontend/                 ← Streamlit only
                          ├── pages/
                          ├── components/   (3D viewer, EGO history
                          │                  chart, exports;
                          │                  GPR diagnostics planned)
                          ├── theme/        (palette + Plotly template
                          │                  + CSS overrides)
                          └── i18n/         (centralised PT/EN —
                                             planned, scaffolded only)
                                │
                          core.api                  ← optimize / evaluate
                          (OptimisationConfig, OptimisationResult,
                           EvaluationResult — types públicos)
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
      core.optimization   core.io          core.engineering
      (EGO+GPR+GA,        (Excel reader,   (NBR 6118/6122
       cache, benchmark)   DXF writer,      pure checks)
                           experiments)
                                │
                          core.domain                ← entidades imutáveis
                          (Solo, Pilar, Combinacao,
                           Sapata, FundacaoProjeto)
```

**Regras** (detalhadas em `ARCHITECTURE.md`):

- `core.domain` não importa nada do projeto — é Python + std-lib puro.
- `core.engineering`, `core.optimization` e `core.io` dependem
  apenas de `core.domain`. Não importam um ao outro.
- `core.api` é a **única** camada que costura todas as outras.
- `frontend/` depende apenas de `core.api`. Lógica de engenharia
  não aparece dentro do frontend.

---

## Estrutura do repositório

```
fundaIA/
├── app.py                       # Streamlit page graph entry point
├── ARCHITECTURE.md              # arquitetura-alvo (camadas + dependências)
├── README.md                    # este arquivo
├── requirements.txt             # dependências de runtime
├── pytest.ini                   # configuração da suite
│
├── core/                        # framework-free core (lógica do projeto)
│   ├── domain/                  # entidades imutáveis (Sprint 3.3)
│   ├── engineering/             # checagens NBR 6118/6122 puras (Sprint 3.2)
│   ├── optimization/            # EGO+GPR+GA, GWO, benchmark, cache (Sprints 3.6 / 4.1)
│   │   ├── ego.py
│   │   ├── genetic_algorithm.py
│   │   ├── grey_wolf.py
│   │   ├── benchmark.py
│   │   ├── funcs.py
│   │   └── cache.py
│   ├── io/                      # Excel, DXF, persistência de runs (Sprints 3.4 / 4.2)
│   │   ├── excel.py
│   │   ├── cad_dxf.py
│   │   └── experiments.py
│   └── api/                     # fachada pública (Sprint 3.5 + Pydantic 3.7)
│       ├── evaluate.py
│       ├── optimize.py
│       ├── benchmark.py         # run_benchmark + BenchmarkConfig/Result (Sprint 4.12)
│       ├── types.py
│       └── _adapter.py
│
├── frontend/                    # camada Streamlit (Sprint 4.3+)
│   ├── pages/
│   │   ├── home.py              # página inicial PT/EN + download do template
│   │   ├── sapatas.py           # página de dimensionamento (shell fino)
│   │   └── experimentos.py      # bancada de comparativos EGO vs GA/PSO/GWO (Sprint 4.12)
│   ├── components/              # widgets reutilizáveis (Sprints 4.5–4.12)
│   │   ├── footings_3d.py       # visualizador 3D (Plotly + presets/lighting)
│   │   ├── ego_chart.py         # gráfico premium do histórico do EGO
│   │   ├── convergence_chart.py # gráfico comparativo multi-algoritmo (Sprint 4.12)
│   │   └── result_export.py     # bundle DXF/JSON/HTML/PNG
│   ├── theme/                   # paleta + Plotly template + CSS (Sprint 4.6)
│   └── i18n/                    # dicionários PT/EN centralizados (planned)
│
├── fundacao.py                  # compat shim (núcleo de FO + helpers GPR)
│                                # — track de deprecação descrito em ARCHITECTURE.md
│
├── tests/                       # suite pytest (242 testes, ~25 s)
│   ├── conftest.py
│   ├── test_engenharia.py
│   ├── test_avaliar_projeto.py
│   ├── test_domain.py
│   ├── test_api.py
│   ├── test_benchmark_api.py
│   ├── test_io.py
│   ├── test_cache.py
│   ├── test_experiments.py
│   ├── test_ego_historico.py
│   └── test_benchmark.py
│
├── notebooks/                   # exploração e validação (Sprint 4.3)
│   ├── README.md
│   ├── testes_fo_filipe.ipynb
│   ├── testes_otm.ipynb
│   ├── testes_otm_lucas.ipynb
│   └── testes_gpr_lucas.ipynb
│
├── scripts/                     # helpers operacionais (Sprint 4.3)
│   ├── README.md
│   ├── env_setup.py             # bootstrap multi-OS do venv
│   ├── wake_up.py               # robô Playwright (acorda app no Cloud)
│   └── requirements.txt
│
├── archive/                     # codebase pré-Sprint-0 (não importar)
│   └── README.md
│
└── assets/
    ├── data/                    # planilhas oficiais de entrada
    │   ├── problema_fund_um.xlsx
    │   ├── problema_fund_dois.xlsx
    │   ├── problema_fund_três.xlsx
    │   └── toy_problem*.xlsx
    ├── tables/                  # tabelas exportadas pelos notebooks
    ├── graphics/                # figuras geradas pelos experimentos GPR
    └── legacy/                  # planilhas legadas (referência histórica)
```

Pastas **não versionadas** (gitignored):

- `obsidian_vault/` — vault Obsidian com mapa de orientação do projeto.
- `docs/` — biblioteca de artigos e manuscritos em construção.
- `models/` — modelos GPR persistidos (`.pkl`, ~553 MB).
- `experiments/` — pastas de runs criadas pelo `ExperimentRecorder`.
- `.venv/`, `__pycache__/`, `.claude/`.

---

## Setup do ambiente

### Requisitos
- Python 3.10 ou superior.
- Acesso a `pip` e a um shell padrão (bash, zsh, PowerShell).

### Instalação manual

```bash
# 1. Clone do repositório
git clone https://github.com/wmpjrufg/fundaIA.git
cd fundaIA

# 2. Criação do ambiente virtual
python3 -m venv .venv

# 3. Ativação (Linux / macOS)
source .venv/bin/activate
# ou (Windows PowerShell)
.venv\Scripts\Activate.ps1

# 4. Instalação das dependências
pip install -r requirements.txt
```

### Bootstrap automático

```bash
python scripts/env_setup.py
```

Cria o virtualenv em `.venv/` e instala as dependências do
`requirements.txt`.

---

## Como rodar a aplicação

```bash
streamlit run app.py
```

A interface abre no navegador (em geral
`http://localhost:8501`) com três páginas na sidebar:
**Início**, **Projeto de Sapatas** e **Experimentos**.

Fluxo principal (página **Projeto de Sapatas**):

1. Definir parâmetros gerais (`fck`, cobrimento, dimensões mínima e
   máxima da sapata, número de gerações, tamanho da população).
2. Fazer upload da planilha Excel de entrada (template disponível
   para download na página inicial).
3. Clicar em **Dimensionar** — o método EGO+GPR+AG é executado em
   `n_rep = 5` repetições com seeds independentes (`base_seed + rep`).
4. Visualizar o arranjo otimizado em planta e em 3D, e exportar os
   resultados em Excel, DXF, JSON, HTML 3D ou PNG do histórico.

Para o comparativo científico EGO vs metaheurísticas puras, ver
[Bancada de experimentos](#bancada-de-experimentos-ego-vs-metaheurísticas).

### Schema da planilha de entrada

| Coluna | Unidade | Descrição |
|---|---|---|
| `Elemento` | — | rótulo do pilar (ex.: `P04`) |
| `ap (m)`, `bp (m)` | m | dimensões do pilar |
| `spt` | — | índice de sondagem SPT |
| `solo` | — | `pedregulho`, `areia`, `silte` ou `argila` |
| `xg (m)`, `yg (m)` | m | coordenadas do centróide do pilar |
| `Fz-c{i}` | kN | carga axial característica na combinação `i` |
| `Mx-c{i}`, `My-c{i}` | kN·m | momentos característicos na combinação `i` |

---

### Estrutura da otimização (o que está rodando "por baixo")

Cada chamada a **Dimensionar** dispara o pipeline abaixo:

```
n_rep    × ( n_pop avaliações reais (LHS, iter 0)
              + n_gen iterações do EGO )

  por iteração do EGO:
     1. Treina um GPR no histórico atual (n_pop + iters anteriores)
     2. Maximiza Expected Improvement com um GA interno (mealpy)
     3. Avalia o candidato com a função objetivo real
     4. Atualiza o histórico
```

Default da UI (rebalanceado na Sprint 4.12 para ~5× mais rápido sem
regredir a baseline): `n_rep = 5`, `n_pop = 100`, `n_gen = 20`,
`ga_pop_size = 50`, `ga_epoch = 30`. Com isso o modelo substituto é
treinado **5 × 20 = 100 vezes**, e a OF real é avaliada
**5 × (100 + 20) = 600 vezes** no total. O run completo fica gravado
em `experiments/<run_id>/` (ver
[Persistência de experimentos](#persistência-de-experimentos)).

A barra de progresso na página mostra a iteração corrente
`{rep}/{n_rep} · iter {t}/{n_gen}` e o melhor OF encontrado até o
momento — atualizada **ao vivo** via callback (ver
[Progresso ao vivo](#progresso-ao-vivo-callback)).

## Como rodar uma otimização programaticamente

Sem precisar do Streamlit, a partir de um notebook ou script:

```python
from core.api import OptimisationConfig, optimize
from core.io import read_projeto_from_excel

projeto = read_projeto_from_excel(
    "assets/data/problema_fund_três.xlsx",
    f_ck_kpa=25_000.0,
    cobrimento_m=0.04,
)
config = OptimisationConfig(
    h_min_m=0.6, h_max_m=3.0,
    n_pop=250, n_gen=30, n_rep=5,
    base_seed=42,
)
result = optimize(projeto, config)

print(result.best_of, result.per_rep_of)
for s in result.sapatas:
    print(f"{s.pilar.rotulo}: h_x={s.h_x:.3f} h_y={s.h_y:.3f} h_z={s.h_z:.3f}")
```

---

## Bancada de experimentos (EGO vs metaheurísticas)

A página **Experimentos** (e a função `core.api.run_benchmark`) rodam
o mesmo projeto contra vários algoritmos — EGO+GPR, GA puro, PSO puro
e GWO puro — sob o **mesmo orçamento de avaliações reais** e seeds
controladas, produzindo material direto para o relatório científico.

```python
from core.api import BenchmarkConfig, run_benchmark
from core.io import read_projeto_from_excel

projeto = read_projeto_from_excel(
    "assets/data/problema_fund_três.xlsx",
    f_ck_kpa=25_000.0, cobrimento_m=0.04,
)
config = BenchmarkConfig(
    algorithms=("ego", "ga", "pso", "gwo"),
    budget_evals=150,    # avaliações reais por repetição (compartilhado)
    n_rep=5,             # repetições independentes por algoritmo
    base_seed=42,
    lhs_n_pop=20,        # EGO: LHS inicial
    meta_pop_size=40,    # GA/PSO/GWO: tamanho da população
    ga_pop_size=50,      # EGO: GA interno que maximiza EI (surrogate)
    ga_epoch=30,
)
result = run_benchmark(projeto, config)

print(result.summary)        # mean ± std, AUC, conv_eval, tempo por algoritmo
print(result.pvalues)        # matriz Mann–Whitney bilateral
print(result.history.head()) # uma linha por avaliação real
```

Cada algoritmo enxerga o **mesmo** objetivo via `TracedObjective`, um
wrapper que conta evaluações, registra o trace `(eval_idx, of_value,
of_best_so_far, time_eval_s, time_total_s)` e dispara um sentinela
`_BudgetExhausted` (derivado de `BaseException`, para atravessar
`except Exception` internos do `mealpy`/`scipy`) na avaliação
exatamente igual a `budget_evals`. Isso garante que a comparação seja
**estritamente justa em nº de avaliações reais**.

A página gera ainda o **bundle de download** (zip) com:
`history.parquet`, `history.csv`, `summary.csv`, `pvalues.csv`,
`metadata.json` (com round-trip do `BenchmarkConfig`),
`convergence.html` e `convergence.png` — prontos para inclusão direta
no manuscrito.

> **Por que comparar EGO com metaheurísticas se a função objetivo
> atual é barata?** O argumento metodológico do EGO é convergir em
> **poucas avaliações reais**, não em pouco tempo de parede. Quando a
> função objetivo for cara (recalque/ISE via FEM, próximas frentes do
> projeto), o ganho passa a ser também em tempo. A bancada valida o
> método no regime barato antes da migração para o regime caro.

---

## Progresso ao vivo (callback)

Tanto `core.api.optimize` quanto `core.optimization.ego_01_architecture`
aceitam um callback `progress=...` opcional que é chamado em cada
milestone do pipeline. A página Streamlit usa esse hook para
atualizar o `st.progress` e o `st.status` em tempo real; em scripts
ou notebooks você pode plugar qualquer função compatível:

```python
from core.api import OptimisationConfig, optimize
from core.io import read_projeto_from_excel

def log_progress(ev):
    if ev["event"] == "ego.iter":
        print(f"rep {ev['rep']+1}/{ev['n_rep']}  "
              f"iter {ev['iter']}/{ev['n_gen']}  "
              f"of_min={ev['of_min']:.6f}")

projeto = read_projeto_from_excel("assets/data/problema_fund_três.xlsx",
                                  f_ck_kpa=25_000.0, cobrimento_m=0.04)
optimize(projeto, OptimisationConfig(n_rep=2, n_gen=5),
         progress=log_progress)
```

Eventos emitidos: `optimize.start`, `optimize.rep_start`, `ego.iter`,
`optimize.rep_end`, `optimize.end`, `optimize.failed`. Excepções
levantadas pelo callback são silenciadamente ignoradas — um hook
de UI bugado nunca aborta a otimização.

## Persistência de experimentos

Cada chamada a `optimize` pode gravar uma pasta autodescritiva
em `experiments/<run_id>/`:

```python
from core.io.experiments import ExperimentRecorder, load_experiment

rec = ExperimentRecorder(root="experiments")
result = optimize(projeto, config, recorder=rec)

# layout produzido:
# experiments/20260428T193245Z-a1b2c3d4/
#   manifest.json   (status, schema_version, timestamps)
#   config.json     (Pydantic round-trip da OptimisationConfig)
#   env.json        (Python, libs, OS, git rev/branch/dirty)
#   project.json    (SHA-256 + sumário do FundacaoProjeto)
#   history/rep_NNN.parquet  (DataFrame inteiro do EGO por repetição)
#   summary.csv     (uma linha por rep)
#   metrics.json    (best/mean/std, AUC, convergence_iter, etc.)
#   artifacts/      (DXF, plots, blobs adicionais)

run = load_experiment(rec.run_dir)
print(run.manifest.metrics["best_of"])
df_rep0 = run.history[0]   # DataFrame Parquet → pandas
```

As métricas paper-grade calculadas automaticamente
(`summarise_history` + `compute_metrics`) incluem:

- por repetição: `of_initial`, `of_best`, `best_iter`,
  `improvement_abs`, `improvement_rel`, `convergence_iter`,
  `auc_best_so_far`, `n_unique_x`, `t_total_s`, `mean_t_per_iter_s`.
- agregado entre reps: `best/mean/std/median_of`, `best_rep_id`,
  `mean_convergence_iter`, `mean_auc_best_so_far`, `wall_time_total_s`.

---

## Cache do surrogate

Para evitar reajustar o GPR quando o conjunto `(X, y, configuração
do pipeline)` é idêntico (replicações, re-execução de notebooks,
varreduras de hiperparâmetros), passe um `SurrogateCache`:

```python
from core.optimization.cache import SurrogateCache

cache = SurrogateCache(maxsize=128, disk_dir="experiments/_cache")
optimize(projeto, config, recorder=rec, cache=cache)

print(cache.stats)
# {'hits': 7, 'misses': 9, 'disk_hits': 0, 'size': 9}
```

A chave é um SHA-256 sobre os bytes contíguos de `X`/`y` mais a
assinatura textual do pipeline (kernel, alpha, random_state,
n_restarts_optimizer, normalize_y). Cache hit devolve o modelo
exatamente igual ao que seria treinado do zero — o `optimize` com
`cache` reproduz `optimize` sem `cache` bit-a-bit.

---

## Suite de testes

A suite pytest funciona como **rede de segurança regressiva** para
o comportamento numérico atual e para o contrato de cada camada,
viabilizando refatorações sem regressão silenciosa.

```bash
# Toda a suite (242 testes em ~25 segundos)
pytest

# Por marker
pytest -m engineering   # NBR 6118/6122
pytest -m regression    # trava OF de referência (19.70604234767181)
pytest -m optimization  # contrato EGO/GPR + cache + recorder
pytest -m benchmark     # funções clássicas (sphere, rosenbrock, ...)
```

Distribuição (pós-Sprint 4.12):

| Arquivo                          | Testes |
|----------------------------------|-------:|
| `tests/test_engenharia.py`       | 37     |
| `tests/test_domain.py`           | 16     |
| `tests/test_api.py`              | 26     |
| `tests/test_benchmark_api.py`    | 18     |
| `tests/test_io.py`               | 21     |
| `tests/test_cache.py`            | 23     |
| `tests/test_experiments.py`      | 21     |
| `tests/test_ego_historico.py`    |  8     |
| `tests/test_benchmark.py`        | 15     |
| `tests/test_avaliar_projeto.py`  |  6     |
| `tests/test_observability.py`    |  9     |
| `tests/test_components_3d.py`    | 18     |
| `tests/test_ego_chart.py`        |  9     |
| `tests/test_result_export.py`    |  7     |
| `tests/test_theme.py`            |  5     |
| `tests/test_funcs.py`            |  3     |
| **Total**                        | **242**|

A regressão `of = 19,70604234767181` para o caso
`assets/data/problema_fund_três.xlsx` com seed canônica é travada
com `rel=1e-12`.

---

## Stack técnica

| Camada | Tecnologia |
|---|---|
| Linguagem | Python 3.10+ |
| Validação de configuração | [Pydantic](https://docs.pydantic.dev/) v2 |
| Interface web | [Streamlit](https://streamlit.io/) |
| Surrogate (GPR) | [scikit-learn](https://scikit-learn.org/) — `GaussianProcessRegressor` + kernels (`RBF`, `Matern`, `RationalQuadratic`, `DotProduct`, `ExpSineSquared`, `WhiteKernel`) |
| Metaheurísticas | [Mealpy](https://mealpy.readthedocs.io/) — `GA.BaseGA`, PSO, GWO |
| Otimização local | `scipy.optimize` — L-BFGS-B, SLSQP, TNC, trust-constr |
| Amostragem | `scipy.stats.qmc.LatinHypercube` |
| Persistência de modelos | [joblib](https://joblib.readthedocs.io/) |
| Persistência de experimentos | [pyarrow](https://arrow.apache.org/docs/python/) (Parquet) + JSON + CSV |
| Planilhas | [openpyxl](https://openpyxl.readthedocs.io/) + [xlsxwriter](https://xlsxwriter.readthedocs.io/) |
| Exportação CAD | [ezdxf](https://ezdxf.readthedocs.io/) |
| Testes | [pytest](https://docs.pytest.org/) |
| Operação (wake-up) | [Playwright](https://playwright.dev/) |

---

## Status atual

Marcos das sprints concluídas (detalhe completo em
`ARCHITECTURE.md` → "Sprint history"):

- ✅ **Sprints 0–2** — saneamento, correções científicas (EGO history,
  `n_rep`, `seed`), suite pytest.
- ✅ **Sprints 3.1–3.8** — refator estrutural completo: skeleton
  `core/`, migração da camada de engenharia, entidades de domínio,
  IO layer, API layer, fold do `metapy_toolbox` em
  `core.optimization`, `OptimisationConfig` em Pydantic, vetorização
  da função objetivo (~100× speedup).
- ✅ **Sprints 4.1–4.2** — Fase 2 do roadmap: cache do GPR
  (`SurrogateCache`) e persistência completa por run
  (`ExperimentRecorder` + `load_experiment` com manifest, history
  Parquet, summary CSV e metrics paper-grade).
- ✅ **Sprint 4.3** — reorganização do repositório
  (`frontend/`, `scripts/`, `notebooks/`, `archive/`,
  `assets/data/`), remoção do shim `metapy_toolbox`, atualização
  de `README.md` e `ARCHITECTURE.md`.
- ✅ **Sprint 4.4** — logging estruturado JSON-line
  (`core/observability/`) com `run_context` e eventos nomeados
  em `optimize` / `ego` / `cache` / `experiments`.
- ✅ **Sprint 4.5** — visualizador 3D Plotly em
  `frontend/components/footings_3d.py` (sapatas enterradas,
  pilares acima, hover, presets de câmera).
- ✅ **Sprint 4.6** — UI premium: tema dark
  (`.streamlit/config.toml` + `frontend/theme/`), gráfico
  premium do histórico do EGO, painel unificado de
  exportação (DXF, JSON, HTML 3D, PNG), recorder + cache
  ligados por padrão na UI.
- ✅ **Sprint 4.7** — polish de UX: progresso ao vivo via
  callback (`progress=`), gráficos com hover por trace
  (não mais ribbon que bloqueia scroll), 3D em seção
  full-width separada, eixos travados em `>=0`, input
  `n_rep` exposto na UI, default `n_gen` subido para 20,
  flicker do 3D corrigido (lighting reduzido + hover
  desabilitado em grid/contorno do terreno).
- ✅ **Sprint 4.8** — limpeza de auditoria: `Solo` deixa de
  importar `core.engineering` (pureza arquitetural), índice
  seguro em `best_avg_worst`, testes de borda para os
  helpers de engenharia, input morto `n_comb` removido,
  pendências de documentação purgadas, `env_setup.py`
  alinhado com `.venv`.
- ✅ **Sprint 4.9** — barra de progresso coerente (LHS + EGO em
  uma só métrica), botão de cancelamento cooperativo
  (`should_stop` propagado até o EGO via `_CancelSentinel`),
  travamento opcional da rotação 3D no eixo Z.
- ✅ **Sprints 4.10 → 4.11** — iteração de UX no viewer 3D:
  4.10 travou a elevação por padrão; 4.11 reverteu para
  restaurar a rotação livre do mouse, mantendo o slider de
  câmera como controle complementar.
- ✅ **Sprint 4.12** — bancada de experimentos `Experimentos`:
  `core.api.run_benchmark` compara EGO+GPR contra GA / PSO /
  GWO puros sob mesmo orçamento de avaliações, com
  `TracedObjective` cooperativo (`_BudgetExhausted`),
  componente `convergence_chart` (mediana + ±1σ + envelope),
  tabela-resumo, matriz Mann–Whitney e bundle zip
  (parquet + csv + html + png). Defaults do EGO interno
  rebalanceados (`ga_pop_size 150→50`, `ga_epoch 50→30`,
  `n_pop 250→100`) — pipeline ~5× mais rápido sem regredir a
  baseline numérica `of = 19.70604234767181`.

---

## Próximos passos

1. **Sprint 5.x — retirar `fundacao.py`**: migrar `_avaliar_projeto`,
   `obj_*`, `constroi_kernel`, `gpr_pipelines`,
   `aprendizado_maquina_paralelo` e `treino_teste_para_processo_paralelo`
   para os módulos definitivos em `core/`, eliminando o último vestígio
   da arquitetura monolítica pré-Sprint 3.
2. **Diagnóstico do GPR no frontend** — componente
   `gpr_diagnostics` em `frontend/components/` (paired plots:
   resíduos, banda de desvio, traços dos hiperparâmetros do kernel).
3. **Função objetivo cara (recalque por elemento / ISE)** — quando o
   custo de avaliação subir do regime barato atual (microssegundos)
   para o regime de centenas de ms a segundos por configuração,
   o EGO+GPR passa a vencer também em tempo de parede. A bancada de
   experimentos (Sprint 4.12) já está dimensionada para registrar
   essa transição. Frente conectada ao item 4.
4. **Frente de pesquisa principal — empacotamento + layout** —
   incorporação do problema de **packing 2D** ao processo de
   otimização, tratando as posições `(xg, yg)` como variáveis de
   projeto adicionais sob restrições rígidas de não sobreposição e
   fronteira do lote (sizing + layout acoplados). Aqui a função
   objetivo se torna naturalmente cara e o EGO passa a ser a
   escolha metodológica natural.

Para questões técnicas ou colaborações, abrir uma *issue* neste
repositório.
