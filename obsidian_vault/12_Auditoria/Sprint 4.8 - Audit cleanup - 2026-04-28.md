---
tags: [refactor, sprint, log, auditoria, qualidade, docs]
data: 2026-04-28
branch: refactor/core-architecture
escopo: Sprint 4.8 — Audit cleanup
---

# Sprint 4.8 — Audit cleanup

> Sprint reativa à auditoria pós-4.7. Cada achado virou ação
> verificada por teste ou edição direta. Suite passou de **211
> → 221 testes**, baseline `of = 19,70604234767181` intacto.

## TL;DR

> Pureza arquitetural: `Solo` não importa mais `core.engineering`.
> Robustez: `best_avg_worst` usa `.loc` em vez de `.values[idx]`,
> imune a DataFrame com índice não-default. Guardrails de
> engenharia: testes de borda fixam o que acontece em regimes
> inválidos (spt=0, f_zk=0, h_z<=cob, solo desconhecido). UI:
> input morto `n_comb` removido e contagens auto-detectadas
> exibidas como métricas read-only. Docs: referências obsoletas
> a `metapy_toolbox` purgadas; `core/__init__.py` reescrito
> com a realidade pós-4.7. Setup: `env_setup.py` cria `.venv/`
> na raiz do repo.

## Achados resolvidos

### 1. `core.domain.solo` violava a pureza arquitetural

**Antes**: `core/domain/solo.py` importava `tensao_adm_solo` de
`core.engineering` para expor `Solo.sigma_adm_kpa` como property.

**Depois**: a property foi **removida**. Callers que precisam
de `sigma_adm` importam `from core.engineering import tensao_adm_solo`
diretamente. Teste novo:
- `TestSolo::test_pure_data_container_no_engineering_import`
  garante que `Solo` não tem mais `sigma_adm_kpa`.
- `TestSolo::test_admissible_pressure_delegated_to_engineering_helper`
  testa a correlação via helper.

### 2. `funcs.best_avg_worst` quebrava com índice não-default

**Antes**: `df['X_0'].values[best_idx]` falhava silenciosamente
quando `best_idx` era um label de índice ≠ posição
(`IndexError` ou retorno errado em DataFrames filtrados).

**Depois**: trocado para `df.loc[best_idx, col]`. Novo arquivo
de teste `tests/test_funcs.py` com 3 casos:
- `test_default_index` — RangeIndex 0..n-1 (regressão).
- `test_non_default_index_does_not_raise` — índice [5, 6, 7].
- `test_negative_indices_in_dataframe_label` — índices
  negativos.

### 3. Engineering — guardrails para casos-limite

Novos testes em `tests/test_engenharia.py::TestEngineeringEdgeCases`:

- `test_tensao_adm_solo_unknown_soil_falls_back_to_spt_over_50`
  → trava o comportamento histórico do fallback (sem
  alterar a numérica).
- `test_tensao_adm_solo_spt_zero_returns_zero`
  → documenta `spt=0 → sigma_adm=0`.
- `test_checagem_tensao_zero_admissible_is_undefined`
  → trava `ZeroDivisionError` quando `sigma_adm=0`.
- `test_sigma_max_min_zero_load_raises`
  → trava `ZeroDivisionError` quando `f_zk=0`.
- `test_puncao_h_z_equal_to_cover_raises` /
  `test_puncao_h_z_below_cover_yields_negative_stress`
  → travam `ZeroDivisionError` (h_z = cob) e `tau_sd2 < 0`
  (h_z < cob).

Não mudei a numérica de nenhum helper — os testes documentam
o comportamento atual; uma futura sprint pode introduzir
`ValueError` upfront e atualizar os testes.

### 4. Input `n_comb` morto na UI

**Antes**: `frontend/pages/sapatas.py` tinha
`n_comb_ui = st.number_input("Número de combinações", value=3)`
que **não entrava em** `OptimisationConfig` — `read_projeto_from_excel`
infere a partir das colunas. Mexer nele não fazia nada;
confundia o usuário.

**Depois**: input removido. Em vez disso, logo após o upload,
duas métricas read-only mostram `Pilares detectados` e
`Combinações detectadas`, lidas direto da `FundacaoProjeto`.
Chave `n_comb` removida dos dicts PT/EN.

### 5. Stale `metapy_toolbox` em docs / tests

- `core/optimization/__init__.py` — docstring antiga dizia
  "ainda existe como shim de compatibilidade"; reescrita
  para "retirado em Sprint 4.3".
- `tests/test_ego_historico.py`, `tests/test_benchmark.py`,
  `tests/conftest.py` — docstrings atualizadas para
  apontarem `core.optimization`.
- `core/__init__.py` — reescrito do zero. Antes era a
  docstring de Sprint 3.1 ("começa vazio, nada migrado");
  agora descreve a arquitetura final pós-4.7
  (`domain`/`engineering`/`optimization`/`io`/`api`/`observability`)
  e o critério de aceitação por commit.

### 6. README / ARCHITECTURE pós-4.7 desatualizados

- `frontend/components/` agora é descrito como
  "populated in Sprints 4.5–4.7" com a lista real
  (`footings_3d.py`, `ego_chart.py`, `result_export.py`).
- `frontend/theme/` adicionado à tabela de
  responsabilidades por camada e à árvore de pastas.
- Sprints 4.4 → 4.8 adicionadas à tabela "Sprint history".
- Total de testes ajustado de 162 → **221**.
- Distribuição por arquivo de teste atualizada no README.

### 7. `env_setup.py` divergia de README e `.gitignore`

**Antes**: criava `venv/` ao lado do diretório corrente;
quando rodado de `scripts/`, criava `scripts/venv/`. README
documentava `.venv/` na raiz.

**Depois**: o script resolve `repo_root` via
`Path(__file__).resolve().parent.parent`, cria
`<repo_root>/.venv/`, e instala o `requirements.txt`
da raiz. Mensagens de ativação atualizadas para
`source .venv/bin/activate` (POSIX) e
`.venv\Scripts\Activate.ps1` (Windows). Ajustada também a
linha do `requirements.txt` que ainda mencionava
`ops/requirements.txt` (path antigo).

## Validação

```text
=== suite ===
  221 passed in ~7 s

  novos arquivos de teste:
    tests/test_funcs.py            +3
    tests/test_engenharia.py       +6  (TestEngineeringEdgeCases)
    tests/test_domain.py           +1  (purity guard substituiu sigma_adm_kpa)
```

Baseline `of = 19,70604234767181` permanece intocado — nenhuma
alteração numérica.

## Não tratado (e por quê)

- **GA crossover incorreto / GWO placeholder de diversidade**:
  citados na lista de issues do vault e fora do escopo da
  auditoria desta sprint. Continuam abertos como issues no
  `obsidian_vault/07_Issues/`.
- **Punção seção C'**: também já é issue rastreada; deve entrar
  numa sprint de engenharia dedicada.

## Vínculos

- [[12_Auditoria/Sprint 4.7 - UI polish + live progress - 2026-04-28]]
- [[10_Melhorias/MOC - Melhorias]]
- [[07_Issues/Lista Mestre de Issues]]
