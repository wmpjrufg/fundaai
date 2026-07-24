---
tags: [refactor, sprint, log, arquitetura, kickoff]
data: 2026-04-27
branch: refactor/core-architecture
escopo: Sprint 3.1 — bootstrap da arquitetura
---

# Sprint 3 — Refactor estrutural (kickoff)

> Log de abertura da Sprint 3, dedicada à reorganização da arquitetura
> do FundaIA. Diferente das Sprints 0/1/2 (correções e testes), aqui
> a meta é **mover** a lógica para uma estrutura de camadas que torne
> o código testável, manutenível e pronto para a próxima frente de
> pesquisa.

## Sprint 3.1 — Skeleton (concluída em 2026-04-27)

### Escopo

| # | Item | Status |
|---|---|---|
| 1 | Criar pasta `core/` com subpacotes (`domain/`, `engineering/`, `optimization/`, `io/`, `api/`) | ✅ |
| 2 | `__init__.py` documentados em inglês para cada subpacote | ✅ |
| 3 | `ARCHITECTURE.md` na raiz descrevendo arquitetura-alvo, regras de dependência e plano de migração | ✅ |
| 4 | Smoke test: `pytest` continua verde (55 testes) | ✅ |
| 5 | AST de todos os arquivos Python (24 ao todo) | ✅ |

### Decisão arquitetural

A estrutura adotada segue a recomendação de [[10_Melhorias/Refactor - Plano Geral]]:

```
core/
├── domain/         entities (Solo, Pilar, Combinacao, Sapata, Projeto)
├── engineering/    pure NBR 6118/6122 checks
├── optimization/   EGO/GA/GWO (eventually absorbs metapy_toolbox)
├── io/             Excel readers/writers, DXF export
└── api/            high-level entry points (optimize, evaluate)
```

Regras de dependência (também em `ARCHITECTURE.md`):

- `core.domain` depende de **nada** dentro do projeto.
- `core.engineering` e `core.optimization` dependem só de `core.domain`.
- `core.io` depende só de `core.domain`.
- `core.api` é a única camada que pode amarrar todas.
- `pages/` (Streamlit) só pode importar de `core.api`.

### Por que skeleton primeiro

A intenção é **separar a mudança estrutural da mudança de comportamento**:

- Sprint 3.1 muda a estrutura de pastas, sem tocar em código de produção.
- Sprints 3.2–3.8 movem a lógica progressivamente, com `pytest` verde após cada commit.

Isso garante que se algo quebrar nas próximas sprints, o `git bisect`
identifica imediatamente qual migração introduziu o bug.

### Plano da Sprint 3 completa

| Sub-sprint | Escopo | Status |
|---|---|---|
| 3.1 | Skeleton + ARCHITECTURE.md | ✅ |
| 3.2 | Migrar engenharia para `core/engineering/` (com shim em `fundacao.py`) | ⏳ próxima |
| 3.3 | Domain entities (`core/domain/`) | ⏳ planejada |
| 3.4 | IO layer (`core/io/`) | ⏳ planejada |
| 3.5 | API layer (`core/api/optimize.py`) | ⏳ planejada |
| 3.6 | Migrar `metapy_toolbox` para `core/optimization/` | ⏳ planejada |
| 3.7 | Pydantic config | ⏳ planejada |
| 3.8 | Vetorização da FO | ⏳ planejada |

### Critério de aceite (constante em toda a Sprint 3)

1. `pytest` verde (55+ testes).
2. `test_baseline_three_foundations_returns_19_706` continua travando `of = 19.70604234767181`.
3. APIs públicas de `fundacao.py` e `metapy_toolbox.ego_01_architecture` permanecem importáveis até que os consumidores (Streamlit, notebooks) tenham sido migrados.

### Convenções aplicadas (a partir desta sprint)

Esta é a primeira sprint a aplicar 100% da convenção registrada em
[[01_Projeto/Convenções do Projeto]]:

- Mensagens de commit em **inglês** (Conventional Commits).
- Docstrings em **inglês** (`This function ...`, `:param:`, `:return:`).
- Resumo em português permitido como linha auxiliar de localização.
- Identificadores de domínio NBR (`tensao_adm_solo`, etc.) permanecem em PT.

## Vínculos

- [[10_Melhorias/Roadmap Sugerido]]
- [[10_Melhorias/Refactor - Plano Geral]]
- [[01_Projeto/Convenções do Projeto]]
- [[12_Auditoria/Sprint 0 - Saneamento - 2026-04-27]]
- [[12_Auditoria/Sprint 1 - Ciencia (EGO + n_rep) - 2026-04-27]]
- [[12_Auditoria/Sprint 2 - Testes e Saneamento Experimental - 2026-04-27]]
