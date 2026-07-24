---
tags: [refactor, sprint, log, testes, qualidade]
data: 2026-04-27
branch: fix/code-sanitization-and-tests
escopo: Fase 0 — itens 8 e 10 + suite pytest
---

# Sprint 2 — Testes e Saneamento Experimental

> Log da terceira sprint da nova rodada de refatoração. Cria a **rede
> de segurança** (suite `pytest`) que faltava antes da refatoração
> estrutural, e fecha as duas pendências experimentais que poderiam
> invalidar resultados publicados (notebooks com path quebrado,
> benchmarks com bug numérico).

## Escopo executado

| # | Item | Issue / artefato | Status |
|---|---|---|---|
| 1 | Setup `pytest` (`pytest.ini`, `tests/`, `conftest.py`) | infraestrutura | ✅ |
| 2 | Testes de engenharia (NBR 6118 / 6122) | `tests/test_engenharia.py` | ✅ (26) |
| 3 | Testes de regressão numérica do `_avaliar_projeto` | `tests/test_avaliar_projeto.py` | ✅ (6) |
| 4 | Testes do contrato do EGO (histórico, IDs, seeds) | `tests/test_ego_historico.py` | ✅ (8) |
| 5 | Sanear benchmarks (`griewank`, `powell`) | [[07_Issues/Issue - Benchmarks suspeitos]] + `tests/test_benchmark.py` | ✅ (15) |
| 6 | Corrigir paths quebrados em notebooks | [[07_Issues/Issue - Notebooks com paths quebrados]] | ✅ |

## Itens não executados (intencionalmente adiados)

| # | Item | Motivo |
|---|---|---|
| Sobreposição contada 2× | [[07_Issues/Issue - Sobreposição contada duas vezes]] | Decisão de modelagem, requer alinhamento com orientador. |
| 20 vs 21 kernels | [[03_Otimizacao/Kernels GPR]] | Decisão de nomenclatura, requer alinhamento com orientador. |
| Punção C' | [[07_Issues/Issue - Punção seção C linha comentada]] | Requer revisão técnica completa da NBR 6118 + tabela 19.2. Sprint própria. |
| Diversidade GWO | [[07_Issues/Issue - Placeholder Diversidade GWO]] | GWO não é usado em produção; baixa prioridade. |
| DXF tempfile | [[07_Issues/Issue - DXF tempfile não removido]] | Trivial; será absorvido na refatoração de UI. |

## Resultado da suite

```text
============================= test session starts ==============================
platform darwin -- Python 3.12.12, pytest-9.0.3, pluggy-1.6.0
rootdir: /Users/lucasteixeiracorreia/Documents/IC/fundaIA
configfile: pytest.ini
testpaths: tests
collected 55 items

tests/test_avaliar_projeto.py ......                                     [ 10%]
tests/test_benchmark.py ...............                                  [ 38%]
tests/test_ego_historico.py ........                                     [ 52%]
tests/test_engenharia.py ..........................                      [100%]

============================== 55 passed in 3.44s ==============================
```

## Decisões técnicas

### Convenção de docstring
Todos os testes seguem o estilo do professor:

```python
def test_xxx(...):
    """This test ensures <invariante em ingles>.

    Detalhamento opcional em portugues quando o contexto exige.

    :param fixture_a: ...
    :return: Nada (assert interno)
    """
```

### Markers
`pytest.ini` define `engineering`, `regression`, `optimization`, `benchmark`, `smoke`. Permite execução parcial:

```bash
pytest -m regression           # só os testes que travam o comportamento atual
pytest -m benchmark            # só os benchmarks
```

### Fixtures compartilhadas (`conftest.py`)
- `repo_root`, `assets_dir`: caminhos absolutos.
- `df_problema_um`, `df_problema_tres`: DataFrames carregados das planilhas oficiais.
- `cfg_calibracao`: defaults históricos da UI (`f_ck=25 MPa`, `cob=0,04`, `h_min=0,60`, `h_max=3,00`, `n_comb=3`).

### Trava de regressão numérica
O baseline `of = 19,70604234767181` foi capturado em 2026-04-27 com:
- Caso `assets/problema_fund_três.xlsx` (3 fundações, argila, SPT 35/45/43).
- `np.random.seed(42)` + `np.random.uniform(0.6, 3.0, 9)`.
- `args = (df, 3, 25 000 kPa, 0,04 m)` — penalty implícito = `_PENALTY_DEFAULT = 10`.

Qualquer refatoração futura que altere esse valor terá que justificar a mudança.

## Correções de código incluídas na Sprint 2

### `metapy_toolbox/benchmark.py`

**`griewank`**: produto movido para dentro do loop. Antes:
```python
for i in range(n_dim): sum += (x_i**2)/4000
prod *= np.cos(x_i / np.sqrt(i+1))   # FORA do loop, usa só último x_i
```
Depois:
```python
for i in range(n_dim):
    soma += (x_i**2)/4000
    produto *= np.cos(x_i / np.sqrt(i+1))   # DENTRO do loop
```

**`powell`**: indexação 1-based substituída pelo equivalente 0-based, com `ValueError` explícito quando `len(x) % 4 != 0`:
```python
if n_dimensions % 4 != 0:
    raise ValueError(f"powell exige d multiplo de 4; recebeu d={n_dimensions}.")
for i in range(n_blocks):
    a, b, c, d = x[4*i+0], x[4*i+1], x[4*i+2], x[4*i+3]
    ...
```

### Notebooks
Substituição direta nas duas células:
```diff
- pd.read_excel(r"assets\el08.xlsx")
+ pd.read_excel(r"assets/problema_fund_três.xlsx")
```
JSON dos notebooks reescrito via `json.load`/`json.dumps` preservando estrutura.

## Próxima sprint sugerida (Sprint 3 — Refactor estrutural)

Agora que existe rede de segurança, a refatoração estrutural pode avançar sem medo de regressão silenciosa:

1. [[10_Melhorias/Refactor - Separar UI de Domínio]] — extrair lógica de `pages/sapatas.py` para `core/`.
2. [[10_Melhorias/Refactor - POO Domain Model]] — classes `Sapata`, `Pilar`, `Solo`, `Combinacao`, `Projeto`.
3. [[10_Melhorias/Refactor - Configuração com Pydantic]] — validação de inputs.
4. [[10_Melhorias/Refactor - Vetorização da FO]] — eliminar `df.iterrows()` aninhado.
5. [[10_Melhorias/Logging Estruturado]] — substituir `print` por `logging`.

Antes de cada commit nessa fase: `pytest` deve continuar verde. Trava de regressão `of = 19,70604234767181` é a referência principal.

## Vínculos

- [[10_Melhorias/Roadmap Sugerido]]
- [[10_Melhorias/MOC - Melhorias]]
- [[07_Issues/Lista Mestre de Issues]]
- [[04_Codigo/tests]]
- [[12_Auditoria/Sprint 0 - Saneamento - 2026-04-27]]
- [[12_Auditoria/Sprint 1 - Ciencia (EGO + n_rep) - 2026-04-27]]
