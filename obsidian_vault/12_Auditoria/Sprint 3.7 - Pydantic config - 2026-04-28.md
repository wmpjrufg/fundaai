---
tags: [refactor, sprint, log, arquitetura, pydantic, validacao]
data: 2026-04-28
branch: refactor/core-architecture
escopo: Sprint 3.7 — Pydantic config + validação rica
---

# Sprint 3.7 — Pydantic config

> Log da sétima sub-sprint da Sprint 3 (refactor estrutural). Substitui
> o dataclass simples `OptimisationConfig` por um modelo Pydantic v2,
> ganhando validação rica, JSON schema gerado, serialização round-trip
> e rejeição de campos extras — sem quebrar a API pública.

## Escopo executado

| # | Item | Status |
|---|---|---|
| 1 | Adicionar `pydantic>=2.0,<3.0` ao `requirements.txt` | ✅ |
| 2 | Reescrever `OptimisationConfig` em `core/api/types.py` como `BaseModel` | ✅ |
| 3 | Manter `OptimisationResult` e `EvaluationResult` como dataclasses | ✅ |
| 4 | Estender `tests/test_api.py` com 4 novos casos Pydantic | ✅ |
| 5 | Validar suite completa (117 testes) | ✅ |

## Decisões de design

### Por que Pydantic apenas para `OptimisationConfig`?

- **`OptimisationConfig`** é o único tipo da API que recebe input
  *de fora* (Streamlit, CLI, notebook). Cada chamada nova é uma
  oportunidade para um typo silencioso no nome do campo, um valor
  inconsistente ou uma combinação inválida. Validação rica vale o custo.
- **`OptimisationResult` e `EvaluationResult`** são produzidos pela
  *própria API*. Não há input externo a validar. Manter como dataclass
  é mais leve e evita serialização desnecessária.

### Recursos Pydantic adotados

```python
class OptimisationConfig(BaseModel):
    model_config = ConfigDict(
        frozen=True,                # Imutabilidade (igual ao dataclass anterior)
        extra="forbid",             # Typo em nome de campo é erro, não silencioso
        str_strip_whitespace=True,  # Defesa em profundidade
    )

    h_min_m: float = Field(default=0.60, gt=0.0, description="...")
    h_max_m: float = Field(default=1.50, gt=0.0, description="...")
    # ... (Field com gt/ge/description em todos)
    penalty: float | None = Field(default=None, gt=0.0, description="...")

    @model_validator(mode="after")
    def _check_bounds_order(self) -> "OptimisationConfig":
        if self.h_min_m >= self.h_max_m:
            raise ValueError(...)
        return self
```

### Compatibilidade com testes existentes

Pydantic v2 lança `pydantic.ValidationError`, que **herda de `ValueError`**.
Os 12 testes existentes que usam `with pytest.raises(ValueError)`
continuam passando sem alteração. Zero breakage.

## Testes adicionados (4 novos casos em `tests/test_api.py`)

| Teste | O que valida |
|---|---|
| `test_extra_fields_are_forbidden` | `OptimisationConfig(unknown_field=1)` levanta erro — pega typos do tipo `pop_size` em vez de `ga_pop_size` |
| `test_model_is_frozen` | Atribuição em instância existente é rejeitada (mesma garantia do `frozen=True` do dataclass) |
| `test_model_dump_round_trip` | `model_dump()` + `OptimisationConfig(**dump)` reproduz a instância original — viabiliza persistência em parquet/json/MLflow |
| `test_json_schema_is_self_describing` | `model_json_schema()` produz objeto JSON Schema com todos os 10 campos, todos com `description` preenchida — abre porta para FastAPI/OpenAPI/docs gerados automaticamente |

## Validação

```text
=== suite ===
  117 passed in ~3 s
    test_api.py              26  (22 + 4 novos)
    test_avaliar_projeto.py   6
    test_benchmark.py        15
    test_domain.py           15
    test_ego_historico.py     8
    test_engenharia.py       26
    test_io.py               21

=== smoke pydantic ===
  cfg = OptimisationConfig(h_min_m=0.6, ..., penalty=None)
  model_dump keys: 10 (todos os campos)
  json schema title: 'OptimisationConfig'
```

A trava de regressão `of = 19,70604234767181` permanece intocada;
`evaluate` e `optimize` consomem `OptimisationConfig` exatamente da
mesma forma que antes (acesso por atributo).

## Implicação prática

A configuração da otimização agora é **autodocumentada**: qualquer
ferramenta capaz de ler JSON Schema (FastAPI, Pydantic + uv, geradores
de docs como `pydantic-doc`) consegue produzir documentação de
referência sem código adicional. Isso será útil quando a API exposer
HTTP for considerada (Sprint futura, fora deste roadmap).

## Próxima sub-sprint (Sprint 3.8 — última da Sprint 3)

Vetorização da função objetivo. O laço duplo `df.iterrows()` na
verificação de sobreposição (`O(N²)` em pandas) será substituído por
uma matriz `N×N` numpy. Como bônus, depois disso o adapter
`projeto_to_dataframe` em `core/api/_adapter.py` pode ser eliminado
e `_avaliar_projeto` consumirá `FundacaoProjeto` diretamente.

## Vínculos

- [[10_Melhorias/Roadmap Sugerido]]
- [[10_Melhorias/MOC - Melhorias]]
- [[10_Melhorias/Refactor - Configuração com Pydantic]] — esta sprint executa
- [[12_Auditoria/Sprint 3 - Refactor estrutural - kickoff - 2026-04-27]]
- [[12_Auditoria/Sprint 3.5 - API layer - 2026-04-28]] — define a `OptimisationConfig` original
- [[01_Projeto/Convenções do Projeto]]
