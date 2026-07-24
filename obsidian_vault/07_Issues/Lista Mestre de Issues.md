---
tags: [issue, indice]
---

# Lista Mestre de Issues

> Não corrigir nada sem aprovação do orientador (Prof. Wanderley) e da equipe do projeto. Para criar uma nova issue, use [[99_Templates/Template - Issue]].

> [!info] Última auditoria
> 2026-04-27 — leitura estática + parse AST + inspeção de planilhas/notebooks. Detalhes em [[12_Auditoria/Auditoria 2026-04-27 - Vault vs Projeto]].

> [!success] Sprint 0 (Saneamento) — concluída em 2026-04-27 na branch `fix/code-sanitization-and-tests`
> 5 issues resolvidas.

> [!success] Sprint 1 (Ciência) — concluída em 2026-04-27 na branch `fix/code-sanitization-and-tests`
> 2 issues científicas resolvidas + parâmetro `seed` no EGO.

> [!success] Sprint 2 (Testes + saneamento experimental) — concluída em 2026-04-27 na branch `fix/code-sanitization-and-tests`
> Suite `pytest` com **55 testes** (engenharia, regressão, EGO, benchmarks).
> 2 issues fechadas: notebooks paths quebrados, benchmarks suspeitos.

## ✅ Concluído

### Sprint 0 — Saneamento (2026-04-27)

- [x] [[07_Issues/Issue - Duplicação em sapatas.py]] — bloco duplicado removido (531 → 325 linhas).
- [x] [[07_Issues/Issue - requirements.txt UTF-16]] — recriado em UTF-8 + 5 deps adicionadas.
- [x] [[07_Issues/Issue - obj_felipe_lucas vs obj_teste]] — funções viraram wrappers de `_avaliar_projeto`.
- [x] [[07_Issues/Issue - Args extras em obj_teste]] — penalty agora parametrizável (default = 10).
- [x] [[07_Issues/Issue - methods.py morto]] — arquivo deletado, import limpo.

### Sprint 1 — Ciência (2026-04-27)

- [x] [[07_Issues/Issue - Histórico do EGO com ITER e ID incorretos]] — pontos novos com `ITER=t` e `ID = max(ID)+1`. EGO ganhou parâmetro `seed`.
- [x] [[07_Issues/Issue - n_rep reusa população inicial]] — `initial_population_01` agora dentro do laço, com seed propagada `base_seed + rep`.

### Sprint 2 — Testes + saneamento experimental (2026-04-27)

- [x] **Suite `pytest`** — 55 testes em 4 arquivos (`test_engenharia.py`, `test_avaliar_projeto.py`, `test_ego_historico.py`, `test_benchmark.py`). Trava regressão numérica `of = 19,70604234767181` no caso de 3 fundações + estrutura do histórico do EGO + reprodutibilidade por seed.
- [x] [[07_Issues/Issue - Notebooks com paths quebrados]] — `assets/el08.xlsx` substituído por `assets/problema_fund_três.xlsx` em `testes_fo_filipe.ipynb` e `testes_otm.ipynb`.
- [x] [[07_Issues/Issue - Benchmarks suspeitos]] — `griewank` e `powell` corrigidos contra Surjanovic & Bingham + 7 testes regressivos específicos.

### Sprint 3.5 — API layer (2026-04-28, branch `refactor/core-architecture`)

- [x] [[07_Issues/Issue - DXF tempfile não removido]] — `pages/sapatas.py` migrada para usar `core.io.sapatas_to_dxf_bytes` (in-memory `io.StringIO`); teste regressivo em `tests/test_io.py` previne reincidência.

### Sprint 5.2 — Punção C′ (2026-07-10)

- [x] [[07_Issues/Issue - Punção seção C linha comentada]] — C′ implementada em `core/engineering/puncao.py` com contorno a `2d`, Tabela 19.2, taxa mínima da Tabela 17.3 e integração na FO como `max(g_C, g_C')`.

## 🟠 Médio (pendente)

- [[07_Issues/Issue - Placeholder Diversidade GWO]] — string literal em vez de cálculo.
- [[07_Issues/Issue - Sobreposição contada duas vezes]] — **decisão pendente com orientador** sobre intenção (1× ou 2× como hoje).

## 🟡 Baixo (pendente)

- [[07_Issues/Issue - Branches dispersos]] — 16+ branches no remote.

## ❓ Decisões pendentes (orientador)

- **20 vs 21 kernels** ([[03_Otimizacao/Kernels GPR]]) — declarar oficialmente "20 experimentais + 1 produção" ou simplesmente "21 kernels".
- **Sobreposição contada duas vezes** — confirmar se é intencional ou bug.

## Triagem rápida (matriz pós-Sprint 2)

| Issue                            | Esforço | Impacto | Status      |
| -------------------------------- | ------- | ------- | ----------- |
| Duplicação `sapatas.py`          | baixo   | alto    | ✅ Sprint 0  |
| `requirements.txt` UTF-16        | trivial | alto    | ✅ Sprint 0  |
| Args extras `obj_teste`          | médio   | alto    | ✅ Sprint 0  |
| `obj_felipe_lucas` ≡ `obj_teste` | médio   | médio   | ✅ Sprint 0  |
| `methods.py`                     | trivial | baixo   | ✅ Sprint 0  |
| ITER/ID incorretos no EGO        | baixo   | alto    | ✅ Sprint 1  |
| `n_rep` reusa LHS                | trivial | alto    | ✅ Sprint 1  |
| Notebooks paths quebrados        | trivial | médio   | ✅ Sprint 2  |
| Benchmarks suspeitos             | médio   | médio   | ✅ Sprint 2  |
| Punção C'                        | alto    | médio   | ✅ Sprint 5.2 |
| Sobreposição 2×                  | trivial | médio   | ❓ orientador |
| 20 vs 21 kernels                 | trivial | médio   | ❓ orientador |
| Diversidade GWO                  | médio   | baixo   | ⏳ pendente  |
| DXF tempfile                     | trivial | baixo   | ✅ Sprint 3.5 |
| Cap pop_size Streamlit           | trivial | médio   | ✅ Sprint 3.9 |
| FO lenta (df.apply)              | médio   | alto    | ✅ Sprint 3.9 |
