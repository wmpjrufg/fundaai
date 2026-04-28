# `notebooks/` — exploratory and validation notebooks

Live experiments around the FundaIA stack. The four notebooks here are
the historical artefacts of the IC; they kept compatibility imports
through the Sprint 3 refactor and continue to work after Sprint 4.3
moved them out of the repo root.

## Contents

| Notebook                    | Purpose                                                                                |
|-----------------------------|----------------------------------------------------------------------------------------|
| `testes_fo_filipe.ipynb`    | Original objective-function smoke test from Filipe's TCC. Touches `fundacao.*`.        |
| `testes_otm.ipynb`          | First end-to-end optimisation experiments (LHS + EGO + GA).                             |
| `testes_otm_lucas.ipynb`    | Lucas's IC optimisation experiments (richer kernel comparisons + replication runs).     |
| `testes_gpr_lucas.ipynb`    | Lucas's IC GPR sandbox (kernel sweep, train/test splits, persisted models in `models/`).|

## How they work after Sprint 4.3

Each notebook starts with a **bootstrap cell** (tagged
`fundaia_bootstrap`) that resolves the repository root (one level up
from `notebooks/`), inserts it into `sys.path` and `chdir`s into it.
This means:

- `from core.optimization import *` works (the package is on the path).
- `from fundacao import *` works (the legacy compat shim is on the path).
- Historical relative paths like `assets/data/toy_problem.xlsx` and
  `assets/problema_fund_três.xlsx` resolve without modification.

You can launch Jupyter **from anywhere**:

```bash
# From repo root (recommended)
jupyter lab

# Or from notebooks/ directly — the bootstrap cell still moves cwd up
jupyter lab notebooks/
```

## Where to put new notebooks

- **Exploratory / scratch**: `notebooks/scratch/<your_topic>.ipynb`.
  Anything in `scratch/` is gitignored; nothing there is "blessed".
- **Reproducible experiments tied to a paper section**: keep them at
  the top level of `notebooks/` and link from
  `obsidian_vault/12_Auditoria/`.

## Migration history

- **Sprint 3.6** — `from metapy_toolbox import *` kept working through
  a single-line shim re-exporting `core.optimization`.
- **Sprint 4.3** — the shim was retired; every `from metapy_toolbox`
  import in these notebooks was rewritten to `from core.optimization`.
  The notebooks themselves were moved from the repo root into this
  folder, and the bootstrap cell was added at the top of each.
