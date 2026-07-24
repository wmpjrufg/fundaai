---
tags: [refactor, sprint, log, organizacao, docs, frontend]
data: 2026-04-28
branch: refactor/core-architecture
escopo: Sprint 4.3 — Reorganização de pastas + documentação completa
---

# Sprint 4.3 — Repository reorganization + docs

> Sprint dedicada a **arrumar a casa** depois de toda a Fase 2.
> Renomes coerentes (`pages/` → `frontend/`, `ops/` → `scripts/`,
> `old/` → `archive/`), notebooks consolidados em `notebooks/` com
> bootstrap, planilhas oficiais consolidadas em `assets/data/`,
> remoção do shim `metapy_toolbox` e reescrita completa de
> `README.md` e `ARCHITECTURE.md` para refletir o estado atual.

## TL;DR — O que mudou, em uma linha

> O repositório passou a ter **uma única convenção de pastas** e
> documentação que **bate com a realidade**: o frontend cresce
> isolado em `frontend/`, scripts operacionais em `scripts/`,
> notebooks em `notebooks/` com bootstrap, código pré-Sprint-0 em
> `archive/` e dados oficiais em `assets/data/`.

## Decisões executadas (todas confirmadas pelo usuário antes do touch)

1. **Shim `metapy_toolbox` removido**. 6 sites de import reescritos:
   `core/api/optimize.py`, `tests/test_ego_historico.py`,
   `tests/test_benchmark.py`, e os 3 notebooks que usavam
   `from metapy_toolbox import *`. Pasta `metapy_toolbox/`
   apagada via `git rm`.

2. **`fundacao.py` permanece na raiz**. Migrar `_avaliar_projeto`
   e o resto vira **Sprint 5.x — retire `fundacao.py`** (track de
   deprecação documentado em `ARCHITECTURE.md`).

3. **`pages/` → `frontend/{pages,components,i18n}/`**. `app.py`
   atualizado: `st.Page("frontend/pages/home.py", ...)`.
   `frontend/components/` e `frontend/i18n/` ficam scaffolded
   (com docstrings dirigentes) prontos para receber:
   - 3D viewer das sapatas otimizadas;
   - chart `best-so-far` por iteração do EGO consumindo
     `ExperimentRun.history`;
   - diagnósticos do GPR (resíduos, banda de incerteza,
     hiperparâmetros);
   - dicionários PT/EN centralizados.

4. **`ops/` → `scripts/`**. `env-setup.py` movido para
   `scripts/env_setup.py` (rename + snake_case por consistência).
   `scripts/README.md` reescrito explicando os dois entry points
   (env_setup e wake_up).

5. **Notebooks movidos para `notebooks/`**. 4 notebooks com
   `git mv`. Cada um recebeu um **bootstrap cell** (tag
   `fundaia_bootstrap`) que:
   - resolve repo root (uma pasta acima de `notebooks/`);
   - insere em `sys.path[0]`;
   - faz `os.chdir` para a raiz, garantindo que paths como
     `assets/data/...` continuem resolvendo.
   `notebooks/README.md` documenta como rodar.

6. **`old/` → `archive/`**. Renome semântico (intenção
   "preservado de propósito" vs. "candidato a remoção").
   `archive/README.md` deixa explícito que ninguém deve
   importar de lá.

7. **`assets/` consolidados**. `problema_fund_{um,dois,três}.xlsx`
   movidos da raiz de `assets/` para `assets/data/` (junto com os
   `toy_problem*.xlsx` que já estavam lá). `old_assets/` →
   `legacy/`. Refs atualizadas em:
   - `tests/conftest.py` (fixtures `df_problema_um`, `df_problema_tres`),
   - `tests/test_io.py` (parametrize), `tests/test_api.py`,
     `tests/test_experiments.py`, `tests/test_avaliar_projeto.py` docstring,
   - `frontend/pages/home.py` (download do template),
   - 2 notebooks (`testes_fo_filipe.ipynb`, `testes_otm.ipynb`).

8. **Circular import resolvido**. Após Sprint 4.2,
   `core.io.experiments` importava `OptimisationConfig` de
   `core.api.types`, e `core.api.optimize` importava
   `ExperimentRecorder`. Ordem de coleção do pytest mudou após o
   reorg e expôs o ciclo. Solução: import do
   `OptimisationConfig` colocado em `if TYPE_CHECKING:` (usado
   apenas como type hint).

9. **`.gitignore` atualizado**. Adicionados `/experiments/`
   (pastas de runs) e `/notebooks/scratch/` (área livre para
   esboços).

10. **`README.md` reescrito do zero** (470+ linhas). Agora cobre:
    visão geral, **pipeline atualizado** (com o
    `ExperimentRecorder` e o `SurrogateCache` no diagrama),
    **arquitetura em camadas** (com diagrama de dependências),
    **árvore de pastas atual com situação de cada pasta**, setup,
    como rodar via Streamlit, **como rodar otimização
    programaticamente**, **persistência de experimentos**,
    **cache do surrogate**, suite de testes (162),
    stack técnica e próximos passos.

11. **`ARCHITECTURE.md` reescrito**. Diagrama de dependências
    novo (incluindo `frontend/`, `cache.py`, `experiments.py`),
    tabela de responsabilidades por camada, **histórico de sprints
    0 → 4.3 com contagem de testes**, critérios de aceitação por
    commit e **deprecation tracks** explícitos para `fundacao.py`,
    `frontend/components/` e `frontend/i18n/`.

## Estrutura final do repositório

```
fundaIA/
├── app.py                        # Streamlit page graph
├── ARCHITECTURE.md
├── README.md
├── requirements.txt
├── pytest.ini
├── .gitignore
│
├── core/                         # framework-free core
│   ├── domain/
│   ├── engineering/
│   ├── optimization/             # ego, ga, gwo, benchmark, funcs, cache
│   ├── io/                       # excel, cad_dxf, experiments
│   └── api/                      # evaluate, optimize, types, _adapter
│
├── frontend/                     # Streamlit only
│   ├── pages/{home,sapatas}.py
│   ├── components/               # planned: 3D viewer, EGO chart, GPR plots
│   └── i18n/                     # planned: dicionários PT/EN
│
├── fundacao.py                   # compat shim (deprecation track Sprint 5.x)
│
├── tests/                        # 162 testes
├── notebooks/                    # 4 .ipynb com bootstrap cell
├── scripts/                      # env_setup, wake_up
├── archive/                      # codebase pré-Sprint-0 (não importar)
│
└── assets/
    ├── data/                     # planilhas oficiais + toy problems
    ├── tables/
    ├── graphics/
    └── legacy/
```

## Validação

```text
=== suite ===
  162 passed in ~5 s

=== imports limpos ===
  $ python -c "import core, frontend"      → ok
  $ python -c "import core.io.experiments" → ok (TYPE_CHECKING resolve circular)

=== caminho do baseline ===
  tests/test_avaliar_projeto.py::test_baseline_three_foundations_returns_19_706
  PASSED — of == 19.70604234767181 (rel=1e-12)
```

## Implicações práticas

### Para o frontend (próximas iterações)

- Adicionar visualizador 3D em `frontend/components/footings_3d.py`
  consumindo `OptimisationResult.sapatas` (cada sapata já carrega
  `vertices` e `centro`).
- Adicionar `frontend/components/ego_chart.py` que recebe um
  `ExperimentRun` e plota a curva *best-so-far* por repetição
  + faixa entre `worst_of` e `best_of`.
- Adicionar `frontend/components/gpr_diagnostics.py` que recebe
  uma `Pipeline` GPR + um split de teste e renderiza paired
  plots (resíduos, banda de incerteza, comparação real vs predito).
- Mover `titulos_nav` de `app.py` e os labels de
  `frontend/pages/sapatas.py` para `frontend/i18n/`.

### Para os notebooks

- O bootstrap cell garante portabilidade. Notebooks novos podem
  ser criados em `notebooks/` ou em `notebooks/scratch/`
  (gitignored).
- `from core.optimization import *` substitui o antigo
  `from metapy_toolbox import *` em todos os notebooks vivos.

### Para a colaboração

- `archive/README.md` deixa explícito que pull requests **não
  devem mexer** em `archive/`.
- `scripts/README.md` separa claramente os dois propósitos
  (setup do ambiente vs. wake-up do app deployado).

## Vínculos

- [[10_Melhorias/Roadmap Sugerido]]
- [[10_Melhorias/MOC - Melhorias]]
- [[12_Auditoria/Sprint 4.2 - Experiment persistence - 2026-04-28]] — sprint anterior
- [[01_Projeto/Convenções do Projeto]]
- [[01_Projeto/Stack Tecnológico]]
