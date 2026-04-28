# Architecture

This document describes the **target architecture** of the FundaIA
codebase as it stands at the end of **Sprint 4.3** (April 2026).
Each layer states what it owns, what it depends on, and what it does
not depend on. The architecture has been incrementally materialised
across Sprints 3.1 through 4.3 with a single non-negotiable rule:
the regression test
``tests/test_avaliar_projeto.py::test_baseline_three_foundations_returns_19_706``
must keep locking ``of = 19.70604234767181`` after every commit.

> **Resumo em português.** Documento da arquitetura-alvo. As camadas
> isolam o domínio (sapatas, pilares, solos, combinações de carga) das
> bibliotecas de UI (Streamlit), de otimização (mealpy, scikit-learn)
> e de I/O (pandas, ezdxf, pyarrow). A migração foi incremental e
> preservou o baseline `of = 19,70604234767181` em todas as sprints.

---

## High-level dependency graph

```
                      ┌─────────────────────────────┐
                      │  app.py (Streamlit entry)   │
                      └───────────────┬─────────────┘
                                      │
                      ┌───────────────▼─────────────┐
                      │  frontend/                  │
                      │    pages/                   │  ← Streamlit only
                      │    components/              │
                      │    i18n/                    │
                      └───────────────┬─────────────┘
                                      │
                      ┌───────────────▼─────────────┐
                      │  core.api                   │  ← optimize / evaluate
                      └─┬───────┬─────────┬─────────┘
                        │       │         │
              ┌─────────┘       │         └──────────┐
              ▼                 ▼                    ▼
      ┌──────────────┐  ┌──────────────┐    ┌──────────────────┐
      │ core.optim.  │  │ core.io      │    │ core.engineering │
      │ (EGO+GPR+GA) │  │ (Excel/DXF/  │    │ (NBR 6118/6122)  │
      │              │  │  experiments)│    │                  │
      └──────┬───────┘  └──────┬───────┘    └────────┬─────────┘
             │                 │                     │
             └─────────────────┴─────────────────────┘
                               │
                       ┌───────▼────────┐
                       │  core.domain   │  ← entities (frozen dataclasses)
                       └────────────────┘
```

**Rules of dependency direction**

- `core.domain` depends on **nothing** inside the project. It is
  pure Python + standard library.
- `core.engineering`, `core.optimization` and `core.io` depend on
  `core.domain` only. They never import each other.
- `core.api` is the **only** layer allowed to wire `engineering`,
  `optimization` and `io` together.
- `frontend/` depends on `core.api` only. Engineering and
  optimisation logic must never appear inside `frontend/`.
- `app.py` declares the page graph (`st.Page(...)`) and nothing else.

---

## Layer responsibilities

| Layer                 | Owns                                                                 | Forbids                                          |
|-----------------------|----------------------------------------------------------------------|--------------------------------------------------|
| `core.domain`         | Frozen entities — `Solo`, `Pilar`, `Combinacao`, `Sapata`, `FundacaoProjeto` | Streamlit, pandas, sklearn, mealpy, pyarrow |
| `core.engineering`    | Pure analytical checks (`tensao_adm_solo`, `calcular_sigma_max_min`, `verificacao_puncao_sapata`, `checagem_geometria`, `sobreposicao_sapatas`, `sobreposicao_matrix`) | Streamlit, sklearn, mealpy, pyarrow |
| `core.optimization`   | EGO architecture, GPR pipelines, GA / GWO wrappers, benchmark functions, surrogate cache (`SurrogateCache`, `fit_or_get_cached`) | Streamlit |
| `core.io`             | Excel reader, DXF writer, experiment recorder/loader (`ExperimentRecorder`, `load_experiment`, `summarise_history`, `compute_metrics`) | Streamlit, sklearn, mealpy |
| `core.api`            | `optimize(project, config, *, recorder, cache)`, `evaluate(project, sapatas)`, public types `OptimisationConfig`, `OptimisationResult`, `EvaluationResult` | Streamlit |
| `frontend.pages`      | Streamlit widgets, session state, file uploads, page-level rendering | Direct imports of engineering or optimisation modules |
| `frontend.components` | Reusable Streamlit widgets (3D viewer, EGO curve chart, GPR plots) — **planned**; package scaffolded only | Engineering or optimisation logic |
| `frontend.i18n`       | Centralised PT/EN translation dictionaries — **planned**; package scaffolded only | Anything stateful |

---

## On-disk layout

```
fundaIA/
├── app.py                     ← Streamlit page graph entry point
├── ARCHITECTURE.md            ← (this file)
├── README.md
├── requirements.txt
├── pytest.ini
├── .gitignore
│
├── core/                      ← framework-free domain code
│   ├── domain/
│   ├── engineering/
│   ├── optimization/
│   │   ├── ego.py
│   │   ├── genetic_algorithm.py
│   │   ├── grey_wolf.py
│   │   ├── benchmark.py
│   │   ├── funcs.py
│   │   └── cache.py           ← Sprint 4.1
│   ├── io/
│   │   ├── excel.py
│   │   ├── cad_dxf.py
│   │   └── experiments.py     ← Sprint 4.2
│   └── api/
│       ├── evaluate.py
│       ├── optimize.py
│       ├── types.py
│       └── _adapter.py
│
├── frontend/                  ← Streamlit-only layer (Sprint 4.3)
│   ├── pages/{home,sapatas}.py
│   ├── components/            ← scaffold (3D viewer, EGO/GPR plots planned)
│   └── i18n/                  ← scaffold (centralised PT/EN dicts planned)
│
├── fundacao.py                ← legacy compat shim — see "Deprecation tracks" below
│
├── tests/                     ← pytest suite (regression net)
│   ├── test_avaliar_projeto.py
│   ├── test_engenharia.py
│   ├── test_domain.py
│   ├── test_api.py
│   ├── test_io.py
│   ├── test_cache.py
│   ├── test_experiments.py
│   ├── test_ego_historico.py
│   ├── test_benchmark.py
│   └── conftest.py
│
├── notebooks/                 ← exploratory & validation .ipynb (Sprint 4.3)
├── scripts/                   ← operational helpers (env_setup, wake_up bot)
├── archive/                   ← frozen pre-Sprint-0 codebase (do not import)
│
└── assets/
    ├── data/                  ← canonical input spreadsheets
    ├── tables/                ← exported summary tables (notebooks)
    ├── graphics/              ← exported plots (notebooks)
    └── legacy/                ← retired sample inputs
```

---

## Sprint history

| Sprint | Outcome                                                                                  | Tests |
|--------|------------------------------------------------------------------------------------------|-------|
| **0**  | Saneamento (`requirements`, `methods.py` morto, `obj_*` fundidas, `fundacao.py` deduplicada). | 0     |
| **1**  | EGO history correctness (`ITER`, `ID`), `n_rep` reusing populations, `seed` parameter.    | 0     |
| **2**  | Pytest suite + benchmark fixes + notebook path fixes.                                     | 55    |
| **3.1**| `core/{domain,engineering,optimization,io,api}` skeleton + this document.                 | 55    |
| **3.2**| 6 engineering checks moved into `core.engineering/*` (compat shim in `fundacao.py`).      | 55    |
| **3.3**| Domain entities (`Solo`, `Pilar`, `Combinacao`, `Sapata`, `FundacaoProjeto`).             | 70    |
| **3.4**| IO layer (`read_projeto_from_excel`, `sapatas_to_dxf_bytes`).                             | 91    |
| **3.5**| API layer (`evaluate`, `optimize`); `pages/sapatas.py` becomes a thin shell.              | 113   |
| **3.6**| `metapy_toolbox/` → `core/optimization/` via `git mv` (history preserved).                | 113   |
| **3.7**| `OptimisationConfig` becomes a Pydantic v2 model with rich validation + JSON schema.      | 117   |
| **3.8**| Vectorised AABB overlap (`sobreposicao_matrix`); ~100× speedup at N=200, baseline preserved. | 122   |
| **4.1**| `SurrogateCache` for the GPR pipeline (LRU + optional joblib disk).                       | 145   |
| **4.2**| `ExperimentRecorder` + `load_experiment`; per-run folder with manifest + Parquet history. | 162   |
| **4.3**| Repository reorganisation (frontend/, scripts/, notebooks/, archive/, assets/data/) + docs. | 162   |

---

## Acceptance criteria for every commit

1. `pytest` is green (162 tests at the end of Sprint 4.3).
2. The regression test
   `tests/test_avaliar_projeto.py::test_baseline_three_foundations_returns_19_706`
   keeps locking ``of = 19.70604234767181``.
3. The dependency direction declared above is preserved
   (no upward imports).
4. Every public new symbol has a docstring with `:param:` and
   `:return:` blocks (English; short Portuguese summary acceptable).

---

## Deprecation tracks

These are intentional debts kept under control until a dedicated sprint
retires them.

### `fundacao.py` (root)

After the Sprint 3 refactor, `fundacao.py` no longer hosts the
analytical checks (those live in `core.engineering`). It still
exposes:

- `_avaliar_projeto(x, args)` — the computational core that produces the
  regression baseline. Migrating it requires reshaping its DataFrame
  contract; planned for **Sprint 5.x — retire `fundacao.py`**.
- `obj_felipe_lucas`, `obj_teste` — wrappers used by `core.api` and
  legacy notebooks.
- `constroi_kernel`, `gpr_pipelines`, `aprendizado_maquina_paralelo`,
  `treino_teste_para_processo_paralelo` — GPR plumbing currently
  shared with the GPR sandbox notebooks. Will move to
  `core.optimization` (kernel construction) and a dedicated
  `core.training` module (parallel training).
- Re-exports of every `core.engineering.*` symbol so legacy
  `from fundacao import tensao_adm_solo` keeps working.

The file is **safe to import** but should not grow new functions.

### `frontend/components/` and `frontend/i18n/` (placeholders)

Scaffolded in Sprint 4.3 but not yet populated. Future moves:

- The PT/EN dict in `app.py` (`titulos_nav`) → `frontend/i18n/nav.py`.
- The plot helpers `_plot_layout` and `_plot_layout_3d` from
  `frontend/pages/sapatas.py` → `frontend/components/footings_2d.py`
  and `frontend/components/footings_3d.py`.
- A new `frontend/components/ego_charts.py` for the per-iteration
  best-so-far plot consuming `ExperimentRun.history`.
- A new `frontend/components/gpr_diagnostics.py` for GPR
  hyperparameter / residual visualisations.

### `archive/`

Strictly frozen. The folder exists to preserve provenance of the IC.
**Do not import from `archive/`** in any layer.

---

## Naming and language conventions

This codebase follows the project conventions registered in the vault
under `obsidian_vault/01_Projeto/Convenções do Projeto.md`:

- Commit messages and docstrings: **English** (Conventional Commits).
- Domain identifiers that mirror Brazilian standards (NBR 6118 / 6122):
  kept in **Portuguese** (e.g. `tensao_adm_solo`,
  `verificacao_puncao_sapata`, `checagem_geometria`).
- Inline comments: English preferred; short Portuguese summaries are
  acceptable as orientation.
