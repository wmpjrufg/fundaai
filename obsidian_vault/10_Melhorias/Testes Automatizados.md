---
tags: [melhorias, testes, qualidade, sugestao]
---

# Testes Automatizados

> [!note] Sugestão
> Sem suíte de testes, qualquer refatoração é perigosa. Começar pelos pontos onde a engenharia é mais sensível.

## Estrutura mínima

```
tests/
  engineering/
    test_tensao_admissivel.py
    test_sigma_max_min.py
    test_puncao.py
    test_geometria.py
    test_packing.py
  optimization/
    test_ego_smoke.py        # roda EGO em sphere() e checa convergência
    test_ga_smoke.py
    test_lhs_distribution.py # cobertura Latin Hypercube
  integration/
    test_pipeline_end_to_end.py
```

## Casos críticos

### `test_tensao_admissivel.py`
- Argila com SPT=10 ⇒ 200 kPa (10/50·1000).
- Areia com SPT=20 ⇒ 500 kPa.
- Pedregulho com SPT=30 ⇒ 1000 kPa.

### `test_sigma_max_min.py`
- Com `M_x = M_y = 0` ⇒ σ_max = σ_min = `(F + γ_c h_x h_y h_z)/(h_x h_y)`.
- Momento puro na convenção FundaIA ⇒ `Mx` varia a pressão ao longo de `h_x`; `My` varia ao longo de `h_y`.

### `test_puncao.py`
- Confrontar contra exemplo resolvido manualmente da [[02_Engenharia/NBR 6118]].

### `test_packing.py`
- Sapatas afastadas ⇒ overlap = 0.
- Sapatas idênticas no mesmo ponto ⇒ overlap = h_x · h_y.
- Sapatas com canto encostando ⇒ overlap = 0 (não é > 0).

### `test_ego_smoke.py`
- `sphere(x)` com bounds `[-5,5]^2`, n_gen=10, n_pop=20 ⇒ best_of < 1e-2.

## Frameworks

- `pytest` + `pytest-cov` + `hypothesis` (testes baseados em propriedades para casos edge).

## CI

Ver [[10_Melhorias/CI-CD - Lint Test Build]].

## Vínculos

- [[10_Melhorias/Refactor - Plano Geral]]
- [[02_Engenharia/Verificação à Punção]]
