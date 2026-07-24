---
tags: [refactor, sprint, log, arquitetura, api]
data: 2026-04-28
branch: refactor/core-architecture
escopo: Sprint 3.5 — camada de API + migração da página Streamlit
---

# Sprint 3.5 — API layer + thin Streamlit shell

> Log da quinta sub-sprint da Sprint 3 (refactor estrutural). Conecta
> todas as camadas (`domain`, `engineering`, `io`) numa API pública
> framework-free e migra `pages/sapatas.py` para um shell fino sobre
> ela. Esta é a sprint que **fecha o ciclo** da arquitetura proposta
> em [[10_Melhorias/Refactor - Plano Geral]].

## Escopo executado

| # | Item | Arquivo | Status |
|---|---|---|---|
| 1 | Tipos públicos `OptimisationConfig`, `OptimisationResult`, `EvaluationResult` | `core/api/types.py` | ✅ |
| 2 | Adaptador `domain ↔ DataFrame` (compat bridge) | `core/api/_adapter.py` | ✅ |
| 3 | Função `evaluate(projeto, sapatas)` | `core/api/evaluate.py` | ✅ |
| 4 | Função `optimize(projeto, config)` | `core/api/optimize.py` | ✅ |
| 5 | Re-exports `core.api` | `core/api/__init__.py` | ✅ |
| 6 | Testes (22 casos) | `tests/test_api.py` | ✅ |
| 7 | Migração de `pages/sapatas.py` para shell fino | `pages/sapatas.py` | ✅ |
| 8 | Issue [[07_Issues/Issue - DXF tempfile não removido]] resolvida | — | ✅ |

## Decisões de design

### `OptimisationConfig` (frozen dataclass)

Todos os parâmetros que a UI/CLI exporta vão por aqui. Defaults
espelham a configuração histórica da página Streamlit (`n_rep=5`,
`base_seed=42`, `kernel_index=-1` que aponta para Matérn ν=2.5).
Validação cross-field no `__post_init__`:

- `h_min_m` e `h_max_m` positivos, `h_min_m < h_max_m`.
- `n_gen ≥ 1`, `n_pop ≥ 2`, `n_rep ≥ 1`.
- `ga_epoch ≥ 1`, `ga_pop_size ≥ 2`.
- `penalty` (quando especificado) > 0.

### `OptimisationResult` (frozen dataclass)

```python
@dataclass(frozen=True, slots=True)
class OptimisationResult:
    sapatas: Sequence[Sapata]
    best_of: float
    best_seed: int
    per_rep_of: tuple[float, ...]
```

Inclui `per_rep_of` para reportar `mean ± std` honesto em relatórios
futuros.

### `EvaluationResult` (frozen dataclass)

Devolvida por `evaluate(projeto, sapatas)`. Carrega o `of_total`, as
sapatas avaliadas e a tabela `constraints[rotulo][nome] -> g`.
Constraint names: `g sobreposicao`, `g punção secao C`, `g tensao`,
`g geometria` — convenção `g <= 0 ⇒ factível`.

### Adaptador domain ↔ DataFrame (`_adapter.py`)

Estratégia para preservar a regressão `of = 19.70604234767181`:
**não reescrever `_avaliar_projeto`**. Em vez disso, o adapter
reconstrói o DataFrame que `_avaliar_projeto` espera (mesmo nome de
colunas que o Excel oficial gera) e a `evaluate`/`optimize` chamam
`_avaliar_projeto` por baixo. A partir da Sprint 3.8 (vetorização) o
adapter pode ser eliminado.

Três funções públicas (módulo privado, mas testadas):
- `projeto_to_dataframe(projeto) -> pd.DataFrame`
- `design_vector_to_sapatas(x, projeto) -> list[Sapata]`
- `sapatas_to_design_vector(sapatas) -> list[float]`

### `optimize(projeto, config) -> OptimisationResult`

Replica a orquestração que estava inline em `pages/sapatas.py`:

```python
for rep in range(config.n_rep):
    rep_seed = config.base_seed + rep
    x_ini = initial_population_01(config.n_pop, dim, x_lower, x_upper,
                                  seed=rep_seed, use_lhs=True)
    x_new, of_rep, _ = ego_01_architecture(
        obj_felipe_lucas, config.n_gen, x_ini, x_lower, x_upper,
        paras_opt, paras_kernel, args=args_obj, seed=rep_seed,
    )
    if of_rep < best_of:
        best_of = of_rep
        best_x = x_new
        best_seed = rep_seed
```

Cada repetição parte de uma população LHS independente (Sprint 1) e
o `seed` é propagado ao GPR/SciPy/mealpy.

### `pages/sapatas.py` agora é shell fino

- Antes: 325 linhas com engenharia, GPR, save_dxf com tempfile órfão.
- Depois: 299 linhas, **só Streamlit**, sem `tempfile`, sem `ezdxf` direto, sem `pandas` para construir o DataFrame da FO.
- Imports principais: `core.api.{OptimisationConfig, OptimisationResult, evaluate, optimize}`, `core.domain.{FundacaoProjeto, Sapata}`, `core.io.{read_projeto_from_excel, sapatas_to_dxf_bytes}`.
- Helpers locais (`_plot_layout`, `_result_to_dataframe`, `_build_results_xlsx`) — todos puros, todos sobre entidades de domínio.

## Testes (22 casos em `tests/test_api.py`)

### Adapter (3 testes)
- DataFrame reconstruído tem mesmas colunas que o Excel.
- `sapatas → vector → sapatas` round-trip lossless.
- Vector com tamanho errado levanta `ValueError`.

### `OptimisationConfig` (12 cenários)
- Defaults válidos.
- 10 combinações inválidas rejeitadas (bounds, contagens, penalty).
- `penalty=None` é aceito (delega ao engineering default).

### `evaluate` (4 testes)
- **Reproduz `of = 19.70604234767181` end-to-end via API** (Excel reader → domain → evaluate).
- Tabela de restrições tem entrada por pilar com 4 chaves.
- `penalty` override altera o `of` em ordens de grandeza.
- Mismatch `len(sapatas) != n_fund` levanta `ValueError`.

### `optimize` (3 testes — plumbing only)
- Retorna `OptimisationResult` com shape correto.
- Reprodutível: duas chamadas com mesma config → mesmo `best_of`.
- Sapatas resultantes respeitam os bounds do config.

## Issue resolvida nesta sprint

- [[07_Issues/Issue - DXF tempfile não removido]] — `pages/sapatas.py` não usa mais `tempfile`; chama `sapatas_to_dxf_bytes` da camada IO. Teste regressivo
  `test_dxf_writer_has_no_tempfile_side_effect` em `tests/test_io.py`
  garante que a regressão não volta.

## Validação

```text
=== AST ===
  ✓ 41 arquivos Python OK

=== suite completa ===
tests/test_adapter (impl. dentro de test_api.py)
tests/test_api.py            22
tests/test_avaliar_projeto.py 6
tests/test_benchmark.py      15
tests/test_domain.py         15
tests/test_ego_historico.py   8
tests/test_engenharia.py     26
tests/test_io.py             21
─────────────────────────────────
                            113 passed
```

A trava de regressão `of = 19.70604234767181` permanece intocada e agora
é exercitada **duas vezes** (uma direto via `_avaliar_projeto`, outra
end-to-end via `core.api.evaluate`).

## Próxima sub-sprint (Sprint 3.6)

Migrar `metapy_toolbox` → `core/optimization/`. Como `core.api.optimize`
ainda importa de `metapy_toolbox` e `fundacao`, podemos fazer essa
migração de forma incremental sem quebrar a API pública.

## Vínculos

- [[10_Melhorias/Roadmap Sugerido]]
- [[10_Melhorias/MOC - Melhorias]]
- [[12_Auditoria/Sprint 3 - Refactor estrutural - kickoff - 2026-04-27]]
- [[12_Auditoria/Sprint 3.2 - Engineering migration - 2026-04-28]]
- [[12_Auditoria/Sprint 3.3 - Domain entities - 2026-04-28]]
- [[12_Auditoria/Sprint 3.4 - IO layer - 2026-04-28]]
- [[10_Melhorias/Refactor - Separar UI de Domínio]]
- [[01_Projeto/Convenções do Projeto]]
