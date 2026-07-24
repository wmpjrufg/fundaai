---
tags: [melhorias, refactor, validacao, sugestao]
---

# Refactor — Configuração com Pydantic

> [!note] Sugestão
> Hoje os parâmetros (`n_comb`, `f_ck`, `cob`, `h_min/h_max`, `n_gen`, `n_pop`) circulam como variáveis soltas em `pages/sapatas.py` e como tuples `args` em `obj_felipe_lucas`. Validar uma vez, no entry, evita bug em camada profunda.

## Modelo proposto (Pydantic v2)

```python
from pydantic import BaseModel, Field, model_validator

class ConfigOtimizacao(BaseModel):
    n_comb: int = Field(ge=1, le=10)
    f_ck_mpa: float = Field(ge=15, le=90)
    cobrimento_cm: float = Field(ge=2.0, le=10.0)
    h_min_cm: float = Field(ge=30, le=300)
    h_max_cm: float = Field(ge=30, le=500)
    n_gen: int = Field(ge=1, le=500)
    n_pop: int = Field(ge=10, le=5000)
    n_rep: int = Field(default=5, ge=1, le=100)
    seed: int | None = None
    surrogate_kernel: str = "matern_2_5"
    optimizer: str = "ga_mealpy"
    penalty_factor: float = 10.0   # hoje hardcoded

    @model_validator(mode="after")
    def _check_bounds(self):
        if self.h_min_cm >= self.h_max_cm:
            raise ValueError("h_min deve ser < h_max")
        return self

    @property
    def f_ck_kpa(self) -> float: return self.f_ck_mpa * 1000
    @property
    def cobrimento_m(self) -> float: return self.cobrimento_cm / 100
```

## Benefícios

- Mensagem de erro clara para o usuário antes de gastar 5 minutos otimizando.
- Documentação automática (`Config.model_json_schema()`).
- Serialização para `experiments/<run>/config.json` (ver [[10_Melhorias/Persistência de Experimentos]]).
- Type hints honestas em toda a stack.

## Resolve naturalmente

- [[07_Issues/Issue - Args extras em obj_teste]] — `penalty_factor` vira atributo, não 5º elemento ignorado de tuple.

## Vínculos

- [[10_Melhorias/Refactor - Plano Geral]]
- [[10_Melhorias/Refactor - POO Domain Model]]
- [[10_Melhorias/Persistência de Experimentos]]
