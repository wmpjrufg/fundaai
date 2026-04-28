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

A interface web é construída em **Streamlit** e permite que o usuário
forneça os dados de projeto via planilha Excel, parametrize o método,
execute a otimização e exporte o resultado tanto em Excel quanto em
DXF para integração direta com o fluxo tradicional de CAD.

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
                          ├── components/   (planned: 3D viewer,
                          └── i18n/          EGO chart, GPR plots)
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
│       ├── types.py
│       └── _adapter.py
│
├── frontend/                    # camada Streamlit (Sprint 4.3)
│   ├── pages/
│   │   ├── home.py              # página inicial PT/EN + download do template
│   │   └── sapatas.py           # página de dimensionamento (shell fino)
│   ├── components/              # widgets reutilizáveis (planned)
│   └── i18n/                    # dicionários PT/EN centralizados (planned)
│
├── fundacao.py                  # compat shim (núcleo de FO + helpers GPR)
│                                # — track de deprecação descrito em ARCHITECTURE.md
│
├── tests/                       # suite pytest (162 testes, ~6 s)
│   ├── conftest.py
│   ├── test_engenharia.py
│   ├── test_avaliar_projeto.py
│   ├── test_domain.py
│   ├── test_api.py
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
`http://localhost:8501`). Fluxo de uso:

1. Definir parâmetros gerais (`fck`, cobrimento, dimensões mínima e
   máxima da sapata, número de gerações, tamanho da população).
2. Fazer upload da planilha Excel de entrada (template disponível
   para download na página inicial).
3. Clicar em **Dimensionar** — o método EGO+GPR+AG é executado em
   `n_rep = 5` repetições com seeds independentes (`base_seed + rep`).
4. Visualizar o arranjo otimizado em planta e exportar os resultados
   em Excel ou DXF.

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
# Toda a suite (162 testes em ~6 segundos)
pytest

# Por marker
pytest -m engineering   # NBR 6118/6122
pytest -m regression    # trava OF de referência (19.70604234767181)
pytest -m optimization  # contrato EGO/GPR + cache + recorder
pytest -m benchmark     # funções clássicas (sphere, rosenbrock, ...)
```

Distribuição (pós-Sprint 4.3):

| Arquivo                       | Testes |
|-------------------------------|--------|
| `tests/test_engenharia.py`    | 31     |
| `tests/test_domain.py`        | 15     |
| `tests/test_api.py`           | 26     |
| `tests/test_io.py`            | 21     |
| `tests/test_cache.py`         | 23     |
| `tests/test_experiments.py`   | 17     |
| `tests/test_ego_historico.py` |  8     |
| `tests/test_benchmark.py`     | 15     |
| `tests/test_avaliar_projeto.py`|  6     |
| **Total**                     | **162**|

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

---

## Próximos passos

1. **Sprint 4.4 — Logging estruturado**: emitir eventos JSON em
   paralelo às escritas do `ExperimentRecorder` para acompanhamento
   ao vivo (alinha com Fase 1 das sub-sprints "em paralelo" do
   roadmap).
2. **Frontend (Streamlit)** — popular `frontend/components/` com
   visualizadores 3D dos resultados de sapatas, gráfico
   *best-so-far* por iteração do EGO consumindo `ExperimentRun.history`,
   e diagnóstico de hiperparâmetros do GPR.
3. **Sprint 5.x — retire `fundacao.py`**: migrar `_avaliar_projeto`,
   `obj_*`, `constroi_kernel`, `gpr_pipelines`,
   `aprendizado_maquina_paralelo` e `treino_teste_para_processo_paralelo`
   para os módulos definitivos em `core/`.
4. **Frente de pesquisa principal** — incorporação do problema de
   **empacotamento (packing 2D)** ao processo de otimização,
   tratando as posições `(xg, yg)` como variáveis de projeto
   adicionais sob restrições rígidas de não sobreposição e fronteira
   do lote (sizing + layout acoplados).

Para questões técnicas ou colaborações, abrir uma *issue* neste
repositório.
