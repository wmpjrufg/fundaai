---
tags: [codigo, testes, qualidade]
folder: tests/
---

# Pasta `tests/`

Suite **pytest** introduzida na Sprint 2 (2026-04-27, branch `fix/code-sanitization-and-tests`). Trava o comportamento atual antes da refatoração estrutural prevista para a próxima fase.

## Estrutura

```
tests/
├── __init__.py
├── conftest.py                # path setup + fixtures compartilhadas
├── test_engenharia.py         # 26 testes (NBR 6118 + NBR 6122)
├── test_avaliar_projeto.py    #  6 testes de regressão numérica + wrappers
├── test_ego_historico.py      #  8 testes do contrato do EGO
└── test_benchmark.py          # 15 testes das funções benchmark
```

E `pytest.ini` na raiz, com markers `engineering`, `regression`, `optimization`, `benchmark`, `smoke`.

## Resultado atual

```text
55 passed in 3.44s
```

## Cobertura por área

### `test_engenharia.py` (26 testes)
- `tensao_adm_solo` — 3 ramos (pedregulho/areia/silte/argila) + case-insensitive.
- `calcular_sigma_max_min` — 4 cenários (sem momento, excentricidade pura, módulo, tração).
- `checagem_tensao_max_min` — 4 limiares (igual, dentro, acima, tração).
- `checagem_geometria` — 4 cenários (limite, maior, menor, balanço personalizado).
- `verificacao_puncao_sapata` — 4 invariantes (perímetro, independência da carga, monotonicidade, fórmula α_v2).
- `sobreposicao_sapatas` — 5 casos (afastados, encostando, idênticos, parcial, simetria).

### `test_avaliar_projeto.py` (6 testes)
- **Trava de regressão**: `_avaliar_projeto(x_seed42, args=(df_3fund, 3, 25e3, 0.04))` deve devolver `of = 19,70604234767181`.
- `_PENALTY_DEFAULT == 10.0` (constante).
- `penalty=10` reproduz o default.
- `penalty=1e6` aumenta OF em ordens de grandeza (parametrização funcional).
- `obj_felipe_lucas` ≡ `obj_teste` (escalar concorda).
- `args` aceita 4 ou 5 elementos (retrocompatibilidade).

### `test_ego_historico.py` (8 testes)
- `df['ITER']` cobre `0..n_gen`.
- Total de linhas == `n_pop + n_gen`.
- IDs únicos em todo o histórico.
- Cada iteração `t > 0` adiciona exatamente uma linha.
- LHS reproduzível com mesma seed.
- LHS diferente com seeds diferentes.
- EGO com mesma seed → mesmo `best_of`.
- Parâmetro `seed` está na assinatura pública.

### `test_benchmark.py` (15 testes)
- 8 mínimos conhecidos (sphere, rosenbrock, rastrigin, ackley, zakharov, easom, dixon_price, goldstein_price).
- 3 testes específicos de `griewank` (mínimo, produto efetivo, simetria).
- 4 testes específicos de `powell` (mínimo d=4, mínimo d=8, raise para d não múltiplo de 4, valor pinado em (1,2,3,4) = 1512).

## Como rodar

```bash
.venv/bin/pytest                          # toda a suite
.venv/bin/pytest -m engineering           # só engenharia
.venv/bin/pytest -m regression            # só regressão numérica
.venv/bin/pytest tests/test_ego_historico.py -v
```

## Por que isso importa

Antes da Sprint 2, qualquer refatoração futura (fase POO, separar UI, vetorização, Pydantic) corria o risco de mudar comportamento numérico em silêncio. Com a suite, qualquer alteração que mude `_avaliar_projeto`, o histórico do EGO ou as restrições normativas dispara teste vermelho imediatamente.

## Vínculos

- [[04_Codigo/fundacao.py]]
- [[04_Codigo/metapy_toolbox - ego.py]]
- [[04_Codigo/metapy_toolbox - benchmark.py]]
- [[10_Melhorias/Testes Automatizados]]
- [[12_Auditoria/Sprint 2 - Testes e Saneamento Experimental - 2026-04-27]]
