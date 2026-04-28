# `archive/` — frozen pre-Sprint-0 codebase

Snapshot of the FundaIA repository as it stood before the refactor
sprints began (April 2026). Nothing in this folder is imported from
the live application or from `core/`; it is preserved as historical
context for the IC, not as a dependency.

## Why keep it?

- **Provenance**: the original objective function, kernels and
  notebooks belong to the foundational period of the IC; deleting
  them would erase the genesis of the engineering decisions.
- **Diff anchor**: when the refactor introduces a behavioural
  change, `git diff archive/fundacao.py core/...` makes the change
  visible side by side.
- **Legacy notebooks**: the `*.ipynb` files here document early
  punching-shear, geometry-overhang and overlap experiments. Worth
  reading when extending the corresponding modules in `core/engineering/`.

## Contents

| File                                           | Lineage                                                                  |
|------------------------------------------------|--------------------------------------------------------------------------|
| `app.py`, `app01.py`                            | Earliest single-file Streamlit prototypes.                                |
| `fundacao.py`                                   | Pre-Sprint-0 monolithic objective function. Compare against `core/engineering/`. |
| `fundacoes.ipynb`                               | First end-to-end notebook combining engineering checks.                   |
| `Funcoes_Tubulão.ipynb`                         | Out-of-scope tubulão study (pile foundations) kept for archeology.        |
| `Otimização funcionado.ipynb`, `GPR.ipynb`      | Earliest optimisation-loop and GPR-fit notebooks before the EGO formalisation. |
| `teste_*.ipynb`                                 | Per-check unit notebooks (geometry, punching, overlap, allowable soil pressure, applied stress). |
| `tabela_IC_*.xlsx`, `teste_*.xlsx`              | Historical input spreadsheets.                                             |

## Rules of engagement

- **Do not import** from `archive/`. Code in `core/`, `frontend/`,
  `tests/` and `notebooks/` must not reach into this folder.
- **Do not edit** the contents of this folder. If you need to revisit
  a legacy experiment, copy it into `notebooks/` (or `notebooks/scratch/`)
  and evolve the copy.
- **Do not link** to filenames here from `README.md` / docs as if they
  were canonical references; cite by sprint/date.

## Migration history

- **Sprint 4.3** — folder renamed from `old/` to `archive/` to make
  the "preserved on purpose, not pending deletion" intent explicit.
