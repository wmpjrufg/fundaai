---
tags: [melhorias, refactor, sugestao]
---

# Refactor — Plano Geral

> [!note] Sugestão
> O código atual cresceu por adição: hoje há **scripts procedurais** (`fundacao.py`, `pages/sapatas.py`) misturando engenharia, surrogate, UI e I/O. A separação por **camadas** abaixo facilita testes, paraleliza desenvolvimento e abre porta para a Fase 4.

## Arquitetura proposta (camadas)

```
core/
  domain/        # entidades de negócio (POO) — ver [[10_Melhorias/Refactor - POO Domain Model]]
    sapata.py
    pilar.py
    solo.py
    combinacao.py
    projeto.py
  engineering/   # verificações puras (funções)
    tensao.py    # σ_max, σ_min, σ_adm
    puncao.py    # NBR 6118
    geometria.py
    packing.py   # AABB / NFP futuro
  optimization/  # algoritmos
    ego.py
    surrogate/   # GPR, PI-GPR (futuro)
    metaheuristics/  # GA, GWO, PSO (wrappers mealpy)
    constraints/ # penalização, Deb, AL
  io/
    excel.py
    cad_dxf.py
  api/
    optimize.py  # função pura: optimize(projeto, config) -> resultado
ui/
  streamlit/
    app.py
    pages/
experiments/
  runs/<timestamp>/   # config.json + results.parquet
tests/
docs/
```

## Princípios

1. **Domínio puro**: classes em `core/domain` não importam streamlit, pandas ou numpy diretamente — só tipos primitivos. Facilita teste.
2. **Engenharia stateless**: `tensao`, `puncao`, etc. recebem dados explícitos (não `df.apply` mágico).
3. **UI fina**: `pages/sapatas.py` apenas chama `core.api.optimize` e renderiza.
4. **Algoritmos plugáveis**: `optimize(projeto, algorithm=EGO_GA)` aceita registro de novos otimizadores.
5. **Configuração validada**: ver [[10_Melhorias/Refactor - Configuração com Pydantic]].

## Por que isso ajuda a pesquisa

- Trocar surrogate (GPR → [[11_Frentes_de_Pesquisa/Physics-Informed Surrogates]]) vira plugin.
- Trocar tratamento de restrições (penalização → Deb → AL) vira plugin.
- Comparar metaheurísticas em benchmark fica trivial (ver [[10_Melhorias/Validação contra problema-benchmark]]).

## Migração incremental

Não jogar tudo fora. Um caminho conservador:

1. Criar `core/engineering/tensao.py` com as funções **idênticas** ao atual `fundacao.py`. Manter o `fundacao.py` como compat shim (`from core.engineering.tensao import *`).
2. Mover testes para `tests/engineering/`.
3. Quando estiver coberto por testes, refatorar internamente sem medo.

## Vínculos

- [[10_Melhorias/Refactor - POO Domain Model]]
- [[10_Melhorias/Refactor - Separar UI de Domínio]]
- [[10_Melhorias/Roadmap Sugerido]]
