---
tags: [refactor, sprint, log, arquitetura, engineering]
data: 2026-04-28
branch: refactor/core-architecture
escopo: Sprint 3.2 — migração da camada de engenharia
---

# Sprint 3.2 — Engineering migration

> Log da segunda sub-sprint da Sprint 3 (refactor estrutural). Move
> as 6 funções analíticas puras de `fundacao.py` para
> `core/engineering/`, mantendo `fundacao.py` como camada de
> compatibilidade. Nenhuma mudança de comportamento numérico.

## Escopo executado

| # | Função | Origem | Destino | Status |
|---|---|---|---|---|
| 1 | `tensao_adm_solo` | `fundacao.py` | `core/engineering/solo.py` | ✅ |
| 2 | `calcular_sigma_max_min` | `fundacao.py` | `core/engineering/tensao.py` | ✅ |
| 3 | `checagem_tensao_max_min` | `fundacao.py` | `core/engineering/tensao.py` | ✅ |
| 4 | `checagem_geometria` | `fundacao.py` | `core/engineering/geometria.py` | ✅ |
| 5 | `verificacao_puncao_sapata` | `fundacao.py` | `core/engineering/puncao.py` | ✅ |
| 6 | `sobreposicao_sapatas` | `fundacao.py` | `core/engineering/packing.py` | ✅ |

`core/engineering/__init__.py` reexporta as seis funções via `__all__`.

## Estratégia de retrocompatibilidade

`fundacao.py` continua sendo um ponto de import válido para qualquer
consumidor antigo (notebooks, `pages/sapatas.py`, testes). O cabeçalho
do arquivo agora reexporta explicitamente:

```python
from core.engineering import (
    tensao_adm_solo,
    calcular_sigma_max_min,
    checagem_tensao_max_min,
    checagem_geometria,
    verificacao_puncao_sapata,
    sobreposicao_sapatas,
)
```

As definições legadas (173 linhas) foram removidas do corpo de
`fundacao.py`. O módulo permanece com:

- `download_template` (utilitário Streamlit)
- `_PENALTY_DEFAULT`, `_unpack_args`, `_avaliar_projeto`, `obj_felipe_lucas`, `obj_teste` (FO)
- `constroi_kernel`, `gpr_pipelines`, `aprendizado_maquina_paralelo`, `treino_teste_para_processo_paralelo` (GPR)

## Convenções aplicadas

Todos os módulos novos seguem [[01_Projeto/Convenções do Projeto]]:

- Docstrings em inglês (`This function ...`, `:param:`, `:return:`).
- Resumo em português permitido como linha auxiliar de localização.
- Identificadores de domínio NBR mantidos em PT.
- `__all__` explícito no `__init__.py`.

## Validação

```text
=== AST de tudo ===
  ✓ 29 arquivos Python (12 produção + 11 metapy + 6 tests + ARCHITECTURE.md skipped)

=== Backwards compat: imports legados ===
  ✓ from fundacao import tensao_adm_solo, calcular_sigma_max_min, ...
    obj_felipe_lucas, obj_teste, _avaliar_projeto, _PENALTY_DEFAULT,
    constroi_kernel, gpr_pipelines, aprendizado_maquina_paralelo,
    download_template — todos OK.

=== Imports diretos via core ===
  ✓ from core.engineering import tensao_adm_solo, ... — todos OK.

=== pytest (regression safety net) ===
  55 passed in 3.x s
```

A trava de regressão `of = 19,70604234767181` permanece intocada (a
função `_avaliar_projeto` em `fundacao.py` ainda é o consumidor das
funções engineering — agora importadas de `core.engineering`).

## Próxima sub-sprint (Sprint 3.3)

Introduzir as **entidades de domínio** (`Solo`, `Pilar`, `Combinacao`,
`Sapata`, `FundacaoProjeto`) em `core/domain/`. As funções de
`core/engineering/` permanecem procedurais; a camada de domínio é
construída ao redor delas, sem reescrever lógica analítica.

## Vínculos

- [[10_Melhorias/Roadmap Sugerido]]
- [[10_Melhorias/MOC - Melhorias]]
- [[12_Auditoria/Sprint 3 - Refactor estrutural - kickoff - 2026-04-27]]
- [[01_Projeto/Convenções do Projeto]]
