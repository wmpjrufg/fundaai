---
tags: [codigo, engenharia, surrogate]
file: fundacao.py
loc: 449
sprint_atual: 3.9
---

# `fundacao.py`

**Shim de compatibilidade** — não deve crescer novas funções (regra arquitetural). As verificações puras foram movidas para `core/engineering/` (Sprint 3.2); a FO vetorizada foi movida para `core/api/objective.py` (Sprint 3.9). Este módulo reexporta engenharia e mantém os kernels GPR e treino paralelo, pendentes de migração para Sprint 5.

## 1. Engenharia (reexportadas de `core/engineering/`)

| Função | Conceito |
|---|---|
| `tensao_adm_solo(solo, spt)` | [[02_Engenharia/Tensão Admissível do Solo]] |
| `calcular_sigma_max_min(F_z, M_x, M_y, h_x, h_y)` | [[02_Engenharia/Flexão Composta - Sigma Max e Min]] |
| `checagem_tensao_max_min(σ, σ_adm)` | g_tensao |
| `checagem_geometria(dim_sapata, dim_pilar, balanco_min)` | [[02_Engenharia/Restrição de Geometria]] |
| `verificacao_puncao_sapata(h_z, f_ck, a_p, b_p, F_zk, cob)` | [[02_Engenharia/Verificação à Punção]] (seção C) |
| `sobreposicao_sapatas(...)` | [[03_Otimizacao/Problema de Empacotamento]] — AABB escalar |
| `sobreposicao_matrix(...)` | AABB vetorizado N×N (Sprint 3.8) |

## 2. Função-objetivo

### Núcleo interno (em `fundacao.py`)

| Função | Descrição |
|---|---|
| `_unpack_args(args)` | Extrai `(df, n_comb, f_ck, cob_m, penalty)` da tupla |
| `_avaliar_projeto(x, args)` | Versão original — pandas/`df.apply`. Retorna `(of, df_anotado)`. Usada por `obj_teste`. ~6–13 ms/chamada |

> ⚠️ **Sprint 3.9**: `_avaliar_projeto_fast` foi movida para [[04_Codigo/core-api-objective.py]] como `avaliar_projeto_fast`. **Não está mais em `fundacao.py`.**

### Wrappers públicos

| Função | Chama | Retorno | Uso |
|---|---|---|---|
| `obj_felipe_lucas(x, args)` | `core.api.objective.avaliar_projeto_fast` (import lazy) | escalar `of` | Loop de otimização (EGO, GA, benchmarks) |
| `obj_teste(x, args)` | `_avaliar_projeto` | `(of, df_anotado)` | Diagnóstico, pós-processamento UI |

> **`obj_felipe_lucas`** usa import deferido (`from core.api.objective import avaliar_projeto_fast`) dentro do corpo da função para evitar import circular: `fundacao → core.api → core.api.benchmark → fundacao`.

> **`obj_felipe_lucas_legacy`** foi removida de `fundacao.py`. A versão legacy agora vive em `core/api/objective.py` como `avaliar_projeto_legacy`.

> **Regra:** use `obj_felipe_lucas` (ou diretamente `core.api.objective.avaliar_projeto_fast`) em qualquer loop de otimização. Use `obj_teste` apenas para inspecionar restrições/tensões de uma solução específica.

## 3. Surrogate (GPR)

| Função | Conceito |
|---|---|
| `constroi_kernel(ls0)` | [[03_Otimizacao/Kernels GPR]] — 21 kernels (k00–k20) |
| `gpr_pipelines(...)` | Pipeline `StandardScaler → GPR` |
| `aprendizado_maquina_paralelo(...)` | Treino paralelo (`mp.Pool`) |
| `treino_teste_para_processo_paralelo(...)` | Worker: treina, calcula R², MAE, RMSE, salva `.pkl` |

Pendente de migração para `core.optimization` / `core.training` no Sprint 5.

## Misc

- `download_template(path, label, filename)` — botão Streamlit para baixar Excel.

## Sprints relevantes

| Sprint | Mudança |
|---|---|
| 3.2 | Funções de engenharia movidas para `core/engineering/`; `fundacao.py` reexporta |
| 3.8 | `sobreposicao_matrix` substitui laço `iterrows()` — primeiro `df.apply` vetorizado |
| 3.9 | FO vetorizada (`avaliar_projeto_fast` + `avaliar_projeto_legacy`) movida para `core/api/objective.py`. `obj_felipe_lucas` vira shim com import lazy. `fundacao.py` não cresce (regra respeitada). |

## Links

- [[04_Codigo/core-api-objective.py]]
- [[03_Otimizacao/EGO - Efficient Global Optimization]]
- [[10_Melhorias/Questao Aberta - Custo da FO e Justificativa do EGO]]
- [[10_Melhorias/Refactor - Vetorização da FO]]
