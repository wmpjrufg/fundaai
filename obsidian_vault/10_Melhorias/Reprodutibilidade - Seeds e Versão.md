---
tags: [melhorias, ciencia, sugestao]
---

# Reprodutibilidade — Seeds e Versão

> [!note] Sugestão
> Em IC, cada figura no relatório precisa ser **reprodutível**. Hoje o EGO fixa `random_state=42` no GPR mas o `mealpy.GA` e `np.random` em `mutation_01_random_walk` não têm seed controlado.

## Checklist

1. Aceitar `seed: int | None` em todo nível: `Config.seed → propaga para LHS, GA, GPR, GWO, mutações`.
2. Registrar em cada run:
   - `git rev-parse HEAD` (ou `git describe --dirty`)
   - Hash SHA-256 do `requirements.txt`
   - `seed` usado
   - Configuração completa (ver [[10_Melhorias/Refactor - Configuração com Pydantic]])
3. Salvar em `experiments/<timestamp>_<gitsha>/manifest.json` (ver [[10_Melhorias/Persistência de Experimentos]]).

## Cuidados

- `mealpy` aceita `seed` no construtor de cada algoritmo? Verificar e propagar.
- Threads paralelas em `multiprocessing.Pool` ⇒ usar `np.random.default_rng(seed_filho)` por worker, não `np.random.seed` global.

## Vínculos

- [[10_Melhorias/Persistência de Experimentos]]
- [[10_Melhorias/Logging Estruturado]]
