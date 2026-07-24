---
tags: [refactor, sprint, log, saneamento]
data: 2026-04-27
branch: fix/code-sanitization-and-tests
escopo: Fase 0 — Saneamento (parcial)
---

# Sprint 0 — Saneamento (parcial)

> Log de execução da primeira sprint da nova rodada de refatoração, conforme [[10_Melhorias/Roadmap Sugerido]].

## Escopo executado

Itens 1–5 da Fase 0 do roadmap, escolhidos por baixo risco e alto impacto/esforço favorável:

| # | Item | Issue origem | Status |
|---|---|---|---|
| 1 | Recriar `requirements.txt` em UTF-8 | [[07_Issues/Issue - requirements.txt UTF-16]] | ✅ |
| 2 | Remover bloco duplicado em `pages/sapatas.py` | [[07_Issues/Issue - Duplicação em sapatas.py]] | ✅ |
| 3 | Deletar `metapy_toolbox/methods.py` | [[07_Issues/Issue - methods.py morto]] | ✅ |
| 4 | Fundir `obj_felipe_lucas` e `obj_teste` | [[07_Issues/Issue - obj_felipe_lucas vs obj_teste]] | ✅ |
| 5 | Parametrizar fator de penalidade | [[07_Issues/Issue - Args extras em obj_teste]] | ✅ |

## Itens não executados (intencionalmente adiados)

| # | Item | Motivo de adiar |
|---|---|---|
| 6 | Histórico do EGO (ITER/ID) | Risco médio: mexe no laço do EGO. Sprint própria. |
| 7 | `n_rep` reusa LHS | Idem. Casa com 6 numa Sprint 1. |
| 8 | Notebooks paths quebrados | Carrega validação experimental dos resultados. |
| 9 | Sobreposição 2× | Decisão de modelagem — discutir com orientador. |
| 10 | Benchmarks suspeitos | Requer validação contra referência (Surjanovic & Bingham). |
| 11 | 20 vs 21 kernels | Decisão semântica — discutir com orientador. |
| 12 | Branches dispersos | Operação destrutiva — exige confirmação. |

## Detalhes técnicos por item

### 1. `requirements.txt` UTF-16 → UTF-8
- Confirmado encoding original via `file requirements.txt` (`UTF-16, little-endian, BOM`).
- Recriado via `Path.write_text(..., encoding='utf-8', newline='\n')`.
- Adicionados pacotes que o código usa mas não declarava: `numpy`, `pandas`, `scipy`, `matplotlib`, `joblib`.
- Reorganizado por seções comentadas (Núcleo / Otimização / UI / I/O / Notebooks / Auxiliares / Manutenção).
- 20 pacotes legíveis após a operação.

### 2. Duplicação em `pages/sapatas.py`
- Confirmação prévia: `diff` entre linhas 120–325 e 326–531 retornou apenas o terminador de arquivo.
- Corte feito linha-a-linha (Python `splitlines(keepends=True)`), mantendo 0..324 (1-based: 1..325).
- Resultado: 531 → 325 linhas. AST validado. `ast.walk` mostra 4 funções, 0 duplicatas.

### 3. `metapy_toolbox/methods.py`
- Verificado: 51 linhas, 0 com código (todas comentadas).
- `git rm metapy_toolbox/methods.py`.
- Em `__init__.py`, removida a linha `from .methods import *` e adicionada docstring com nota histórica.
- Smoke test confirma que todos os símbolos públicos esperados continuam disponíveis.

### 4. + 5. Fusão `obj_felipe_lucas` ≡ `obj_teste` + parametrização do penalty
Estrutura final em `fundacao.py`:

```python
_PENALTY_DEFAULT = 1e1   # preserva valor histórico hardcoded

def _unpack_args(args):
    """Aceita 4 ou 5 elementos; quinto é penalty (default 10)."""

def _avaliar_projeto(x, args, *, penalty=None) -> tuple[float, pd.DataFrame]:
    """Núcleo computacional compartilhado."""

def obj_felipe_lucas(x, args) -> float:
    of, _ = _avaliar_projeto(x, args)
    return of

def obj_teste(x, args) -> tuple[float, pd.DataFrame]:
    return _avaliar_projeto(x, args)
```

Smoke test (caso `problema_fund_três.xlsx`, `x` LHS com seed 42):

| Chamada | OF |
|---|---|
| `obj_felipe_lucas(x, (df, 3, 25e3, 0.04))` | `19,706042` |
| `obj_teste(x, (df, 3, 25e3, 0.04))` (escalar) | `19,706042` |
| `obj_felipe_lucas(x, (df, 3, 25e3, 0.04, 1e1))` | `19,706042` |
| `obj_felipe_lucas(x, (df, 3, 25e3, 0.04, 1e6))` | `354 645,545530` |

→ wrappers concordam, default preservado, penalty agora é parametrizável.

## Validação global

```text
=== AST de todos os arquivos Python ===
  ✓ app.py
  ✓ fundacao.py
  ✓ env-setup.py
  ✓ pages/home.py
  ✓ pages/sapatas.py
  ✓ metapy_toolbox/__init__.py
  ✓ metapy_toolbox/benchmark.py
  ✓ metapy_toolbox/ego.py
  ✓ metapy_toolbox/funcs.py
  ✓ metapy_toolbox/genetic_algorithm.py
  ✓ metapy_toolbox/grey_wolf.py
  ✓ ops/wake_up.py

=== Imports completos via .venv ===
  ✓ metapy_toolbox importa
  ✓ fundacao importa
  exports relevantes: ['obj_felipe_lucas', 'obj_teste']
  internals: ['_PENALTY_DEFAULT', '_avaliar_projeto', '_unpack_args']

=== sapatas.py: confirmar ausência de duplicação ===
  Funções definidas: 4 | duplicatas: ✓ nenhuma
```

## Implicações imediatas

- Todos os experimentos rotulados como `penalty=1e6` em
  [[06_Notebooks/testes_otm_lucas]] e [[06_Notebooks/testes_gpr_lucas]]
  precisam ser **reexecutados** para refletir a parametrização real.
  Os PNGs em `assets/graphics/z_GPR_*_pen_1e1_vs_1e6.png` e as tabelas
  em `assets/tables/` ficam marcados como **não confiáveis** até a
  reexecução.
- A próxima sprint deve atacar os itens 6 e 7 (ITER/ID + n_rep) antes
  de qualquer comparação de convergência ser reportada.

## Próxima Sprint sugerida (Sprint 1)

1. [[07_Issues/Issue - Histórico do EGO com ITER e ID incorretos]]
2. [[07_Issues/Issue - n_rep reusa população inicial]]
3. Reexecutar experimentos de penalidade (com seeds controladas).
4. Criar `tests/` com casos para `tensao_adm_solo`, `calcular_sigma_max_min`, `verificacao_puncao_sapata`, `sobreposicao_sapatas` e `_avaliar_projeto`.

## Vínculos

- [[10_Melhorias/Roadmap Sugerido]]
- [[10_Melhorias/MOC - Melhorias]]
- [[07_Issues/Lista Mestre de Issues]]
- [[12_Auditoria/Auditoria 2026-04-27 - Vault vs Projeto]]
