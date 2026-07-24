---
tags: [issue, medio, frontend, benchmark, resolvido]
file: frontend/pages/experimentos.py
severity: medio
status: resolvido
data: 2026-06-05
---

# Issue — Cap de pop_size no Streamlit (página Experimentos)

## Sintoma

Na página Experimentos, ao digitar valores de população como 5.000 ou 50.000 para GA/PSO/GWO, o campo retornava 500 silenciosamente (o valor digitado era ignorado).

## Por que é problema

Impede comparações com populações maiores e confunde o usuário, que não recebe feedback de que o valor foi cortado.

## Diagnóstico (causa)

Dois pontos independentes:

**1. Hard cap no widget** (`frontend/pages/experimentos.py`):
```python
meta_pop_size = st.number_input(
    "GA/PSO/GWO · tamanho da população",
    min_value=4, max_value=500,   # ← cap de 500
    ...
)
```
O `max_value=500` do Streamlit clipa qualquer valor acima para 500 sem aviso. O mesmo vale para `budget_evals` que tinha `max_value=5000`.

**2. Mismatch semântico** (independente do cap):
O benchmark usa **orçamento fixo de avaliações** (`budget_evals`). Com `meta_pop_size > budget_evals`, o algoritmo não consegue completar nem uma geração antes do corte — resultado degenerado. Isso não é um bug, mas precisa de aviso explícito.

## Correção aplicada (Sprint 3.9 — 2026-06-05)

- `max_value` de `meta_pop_size`: `500 → 50_000`
- `max_value` de `budget_evals`: `5_000 → 100_000`
- Adicionado aviso visual quando `meta_pop_size > budget_evals`:
  ```
  ⚠️ pop_size (X) > budget (Y): o GA/PSO/GWO não completa nem uma geração.
  ```
- Help text atualizado com regra prática: `pop_size ≤ budget_evals / 4`

## Notas

Com a FO vetorizada (`_avaliar_projeto_fast`, ~0,1 ms/eval), `budget=50_000` leva ~5 s — viável para experimentos sérios. Com a versão legacy (~10 ms), o mesmo budget levaria ~500 s por rep.

## Vínculo

- [[10_Melhorias/Questao Aberta - Custo da FO e Justificativa do EGO]]
- [[07_Issues/Lista Mestre de Issues]]
