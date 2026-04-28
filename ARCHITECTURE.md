# Architecture

This document describes the **target architecture** that the FundaIA
codebase is migrating to during Sprint 3 of the refactor roadmap.
It is intentionally short and prescriptive: each layer states **what
it owns, what it depends on and what it does not depend on**.

> Resumo em português: este documento define a arquitetura-alvo do
> repositório. As camadas estão organizadas para que o código de
> domínio seja independente do framework de UI (Streamlit) e da
> camada de I/O. A migração é incremental e preserva o baseline de
> regressão `of = 19.70604234767181`.

---

## Layered package layout

```
fundaIA/
├── app.py                  Streamlit entry-point (kept thin)
├── pages/                  Streamlit pages (UI only)
│   ├── home.py
│   └── sapatas.py
│
├── core/                   Pure domain — no Streamlit, no top-level I/O
│   ├── domain/             Business entities (immutable dataclasses)
│   ├── engineering/        Pure analytical checks (NBR 6118 / 6122)
│   ├── optimization/       EGO / GA / GWO algorithms (incoming)
│   ├── io/                 Excel readers/writers, DXF export
│   └── api/                High-level entry points (optimize, evaluate)
│
├── metapy_toolbox/         Optimisation library (will fold into core/optimization)
├── fundacao.py             Engineering + GPR + objective (compat shim during migration)
│
├── tests/                  Pytest suite (regression safety net — 55 tests)
├── ops/                    Wake-up robot (Playwright)
└── assets/                 Spreadsheet templates and experimental artefacts
```

## Dependency direction

```
            ┌────────────────────────┐
            │  pages/  (Streamlit)   │
            └──────────┬─────────────┘
                       │ imports
                       ▼
            ┌────────────────────────┐
            │  core.api              │
            └──────────┬─────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌────────────┐  ┌────────────┐  ┌────────────┐
│ core.engin │  │ core.optim │  │ core.io    │
└─────┬──────┘  └─────┬──────┘  └────────────┘
      │               │
      └───────┬───────┘
              ▼
      ┌──────────────┐
      │ core.domain  │
      └──────────────┘
```

**Rules:**

- ``core.domain`` depends on **nothing inside the project**.
- ``core.engineering`` and ``core.optimization`` depend on ``core.domain`` only.
- ``core.io`` depends on ``core.domain`` only.
- ``core.api`` is the **only** layer allowed to wire the others together.
- ``pages/`` (Streamlit) depends on ``core.api`` only — no engineering or
  optimisation imports inside the UI.

## Layer responsibilities

| Layer | Owns | Forbids |
|---|---|---|
| **`core.domain`** | Entities (`Solo`, `Pilar`, `Combinacao`, `Sapata`, `FundacaoProjeto`) | Streamlit, pandas, sklearn, mealpy |
| **`core.engineering`** | NBR 6118/6122 checks (`tensao_adm_solo`, `sigma_max_min`, `puncao`, `geometria`, `packing`) | Streamlit, sklearn, mealpy |
| **`core.optimization`** | EGO architecture, GPR pipelines, GA/GWO wrappers | Streamlit |
| **`core.io`** | Excel readers/writers, DXF export | Streamlit, sklearn, mealpy |
| **`core.api`** | `optimize(project, config)`, `evaluate(project, x)` | Streamlit |
| **`pages/`** | Streamlit widgets, session state, file uploads | Direct imports of engineering/optimization |

## Migration plan (Sprint 3)

| Step | Status | Description |
|---|---|---|
| 3.1 — Skeleton | ✅ done | Create empty `core/` packages and the architectural document. No production code touched. |
| 3.2 — Engineering migration | ⏳ next | Move pure functions from `fundacao.py` to `core/engineering/`; keep `fundacao.py` as a backwards-compatible shim. |
| 3.3 — Domain entities | ⏳ planned | Introduce dataclasses in `core/domain/`. Engineering functions stay procedural; domain is built around them. |
| 3.4 — IO layer | ⏳ planned | Extract Excel and DXF logic from `pages/sapatas.py` into `core/io/`. |
| 3.5 — API layer | ⏳ planned | Implement `optimize(project, config)` in `core/api/`; `pages/sapatas.py` becomes a thin Streamlit shell. |
| 3.6 — Optimisation migration | ⏳ planned | Fold `metapy_toolbox` into `core/optimization/` (or keep as a separately published package). |
| 3.7 — Pydantic config | ⏳ planned | Replace loose variables in `pages/sapatas.py` with a validated `OptimisationConfig`. |
| 3.8 — Vectorisation | ⏳ planned | Replace nested `df.iterrows()` in the objective function with NumPy matrix operations. |

## Acceptance criteria for every commit during Sprint 3

1. `pytest` is green (55+ tests passing).
2. The regression test ``test_baseline_three_foundations_returns_19_706``
   keeps locking the canonical value ``of = 19.70604234767181``.
3. Public APIs of `fundacao.py` and `metapy_toolbox.ego_01_architecture`
   remain importable until the consumers (Streamlit pages, notebooks)
   have been migrated.

## Naming and language conventions

This refactor follows the project conventions registered in
``obsidian_vault/01_Projeto/Convenções do Projeto.md``:

- Commit messages and docstrings: **English** (Conventional Commits).
- Domain identifiers that mirror Brazilian standards (NBR 6118): kept
  in **Portuguese** (e.g. `tensao_adm_solo`, `verificacao_puncao_sapata`).
- Inline comments: English preferred; short Portuguese summaries are
  acceptable as orientation.
