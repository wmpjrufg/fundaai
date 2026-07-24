---
tags: [refactor, sprint, log, arquitetura, optimization]
data: 2026-04-28
branch: refactor/core-architecture
escopo: Sprint 3.6 — migração de metapy_toolbox para core/optimization
---

# Sprint 3.6 — Optimization migration

> Log da sexta sub-sprint da Sprint 3 (refactor estrutural). Move a
> biblioteca de otimização de `metapy_toolbox/` para
> `core/optimization/`. Comportamento numérico inalterado;
> `metapy_toolbox` permanece como camada de compatibilidade.

## Escopo executado

| # | Item | Status |
|---|---|---|
| 1 | Mapear imports internos do `metapy_toolbox` | ✅ |
| 2 | `git mv` dos 5 módulos para `core/optimization/` (preserva história) | ✅ |
| 3 | Reescrever `from metapy_toolbox import funcs` → `from core.optimization import funcs` em 4 arquivos | ✅ |
| 4 | Reexportar tudo em `core/optimization/__init__.py` | ✅ |
| 5 | Transformar `metapy_toolbox/__init__.py` em shim (`from core.optimization import *`) | ✅ |
| 6 | Validar imports legados + suite pytest | ✅ |

## Estrutura final

```
core/optimization/
├── __init__.py            # Re-exports * de cada módulo
├── funcs.py               # LHS, evaluation, fit_value, check_interval_01, mutation_01
├── benchmark.py           # sphere, rosenbrock, rastrigin, ackley, griewank, powell, ...
├── genetic_algorithm.py   # GA + 8 crossovers + roleta/torneio
├── grey_wolf.py           # GWO clássico
└── ego.py                 # EGO + GPR pipeline (parâmetro seed da Sprint 1 preservado)

metapy_toolbox/
└── __init__.py            # Camada de compat: from core.optimization import *
```

## Decisões de migração

- **`git mv` em vez de copiar e deletar**: preserva o histórico de cada arquivo (`git log --follow core/optimization/ego.py` agora mostra a evolução completa, incluindo Sprint 1).
- **Imports internos reescritos**: `from metapy_toolbox import funcs` em 4 arquivos (`funcs`, `genetic_algorithm`, `grey_wolf`, `ego`) virou `from core.optimization import funcs`. Sem outras mudanças no corpo dos módulos.
- **`metapy_toolbox` continua importável**: o `__init__.py` faz `from core.optimization import *`. Os notebooks históricos (`testes_otm.ipynb`, `testes_otm_lucas.ipynb`, `testes_gpr_lucas.ipynb`) e qualquer consumidor externo continuam funcionando sem alteração.
- **`core.api.optimize` ainda importa de `metapy_toolbox`**: sem impacto porque `metapy_toolbox` agora é só um redirect. Será trocado para `core.optimization` numa limpeza futura.

## Validação

```text
=== AST ===
  ✓ 43 arquivos Python OK (5 movidos + 1 shim alterado)

=== Imports legados (compat surface) ===
  ✓ todos os 14 símbolos públicos reexportados via metapy_toolbox:
      ego_01_architecture, genetic_algorithm_01, grey_wolf_optimizer_01,
      initial_population_01, sphere, rosenbrock, rastrigin, ackley,
      griewank, powell, evaluation, fit_value, check_interval_01,
      roulette_wheel_selection

=== Imports diretos (novo caminho) ===
  ✓ from core.optimization import ego_01_architecture, ... — todos OK

=== pytest ===
  113 passed in ~3 s
    test_api.py              22
    test_avaliar_projeto.py   6
    test_benchmark.py        15  (importa via metapy_toolbox)
    test_domain.py           15
    test_ego_historico.py     8  (importa via metapy_toolbox)
    test_engenharia.py       26
    test_io.py               21
```

A trava de regressão `of = 19,70604234767181` permanece intocada.
Os testes que importavam de `metapy_toolbox` (`test_benchmark.py`,
`test_ego_historico.py`) continuam verdes graças ao shim.

## Próxima sub-sprint (Sprint 3.7)

Configuração com Pydantic — substituir o dataclass simples
`OptimisationConfig` (Sprint 3.5) por um modelo Pydantic com validação
mais rica e geração automática de schema JSON. Isso prepara o terreno
para uma futura camada CLI/API web sem invalidar a API atual.

## Vínculos

- [[10_Melhorias/Roadmap Sugerido]]
- [[10_Melhorias/MOC - Melhorias]]
- [[10_Melhorias/Refactor - Empacotar metapy_toolbox]] — esta sprint executa a parte interna; a publicação como pacote PyPI continua sendo decisão futura
- [[12_Auditoria/Sprint 3 - Refactor estrutural - kickoff - 2026-04-27]]
- [[12_Auditoria/Sprint 3.5 - API layer - 2026-04-28]]
- [[01_Projeto/Convenções do Projeto]]
