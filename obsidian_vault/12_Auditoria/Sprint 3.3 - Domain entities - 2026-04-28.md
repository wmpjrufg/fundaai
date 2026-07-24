---
tags: [refactor, sprint, log, arquitetura, domain]
data: 2026-04-28
branch: refactor/core-architecture
escopo: Sprint 3.3 — entidades de domínio (POO)
---

# Sprint 3.3 — Domain entities (POO)

> Log da terceira sub-sprint da Sprint 3 (refactor estrutural).
> Introduz as entidades de negócio em `core/domain/` como dataclasses
> puras, sem dependência de pandas, sklearn, mealpy ou Streamlit.
> Nenhuma mudança de comportamento numérico — as entidades são
> escritas em paralelo ao código existente e ainda não são consumidas
> pelo `_avaliar_projeto`.

## Escopo executado

| # | Entidade | Arquivo | Mutabilidade | Helpers |
|---|---|---|---|---|
| 1 | `Solo` | `core/domain/solo.py` | frozen | `sigma_adm_kpa` (delega a `core.engineering.solo`) |
| 2 | `Pilar` | `core/domain/pilar.py` | frozen | — |
| 3 | `Combinacao` | `core/domain/combinacao.py` | frozen | — |
| 4 | `Sapata` | `core/domain/sapata.py` | mutable | `volume`, `vertices` |
| 5 | `FundacaoProjeto` | `core/domain/projeto.py` | frozen | `n_fund`, `n_comb` |

`core/domain/__init__.py` reexporta as cinco entidades (mais o alias
`SoilType`) via `__all__`.

## Decisões de design

- **Imutabilidade por padrão**: `Solo`, `Pilar`, `Combinacao` e
  `FundacaoProjeto` são `@dataclass(frozen=True, slots=True)`. Refletem
  dados de entrada que não devem mudar entre chamadas.
- **`Sapata` é mutável**: as três dimensões (`h_x`, `h_y`, `h_z`) são
  exatamente as variáveis de projeto que o otimizador altera durante
  a busca; um frozen aqui inviabilizaria o uso pelo loop EGO.
- **Validação no `__post_init__`**: `Solo` rejeita `spt < 0`, `Pilar` e
  `Sapata` rejeitam dimensões não positivas, `FundacaoProjeto` rejeita
  mapeamentos incompletos e parâmetros globais inválidos.
- **`Solo.sigma_adm_kpa` delega a `core.engineering.solo.tensao_adm_solo`**.
  A correlação empírica fica num único lugar (camada de engenharia);
  a entidade só expõe a propriedade.
- **`Sapata.vertices` retorna o AABB centrado em `(pilar.xg, pilar.yg)`**,
  espelhando a hipótese geométrica de
  `core.engineering.packing.sobreposicao_sapatas`.

## Testes adicionados

`tests/test_domain.py` — **15 novos testes** organizados em 5 classes:

| Classe | Testes | Cobertura |
|---|---|---|
| `TestSolo` | 3 | delegação a `sigma_adm_kpa`, invariante de SPT, frozen |
| `TestPilar` | 3 | round-trip de campos, dimensões positivas, frozen |
| `TestCombinacao` | 1 | round-trip de quatro campos |
| `TestSapata` | 4 | volume, vértices centrados, mutabilidade, dimensões positivas |
| `TestFundacaoProjeto` | 4 | `n_fund`/`n_comb`, mapas faltantes (solo e combinações), parâmetros globais inválidos |

A fixture interna `_three_element_project()` reproduz o caso canônico
`problema_fund_três.xlsx` (P04, P05, P16; argila; SPT 35/45/43).

## Convenções aplicadas

Tudo segue [[01_Projeto/Convenções do Projeto]]:

- Docstrings em inglês (`This class ...`, `:param:`, `:return:`).
- Resumo curto em português como linha auxiliar de localização.
- Identificadores de domínio NBR mantidos em PT (`tipo`, `spt`, `a_p`, `b_p`, `xg`, `yg`, `f_z`, `m_x`, `m_y`).
- `__all__` explícito no `__init__.py`.

## Validação

```text
=== pytest ===
tests/test_avaliar_projeto.py ......                                     [ 8%]
tests/test_benchmark.py ...............                                  [30%]
tests/test_domain.py ...............                                     [51%]
tests/test_ego_historico.py ........                                     [62%]
tests/test_engenharia.py ..........................                      [100%]

70 passed in 3.31s
```

A trava de regressão `of = 19,70604234767181` permanece intocada.
O domínio é puro e ainda não é consumido por `_avaliar_projeto`; a
integração acontece nas sub-sprints 3.4 (IO) e 3.5 (API).

## Próxima sub-sprint (Sprint 3.4)

Camada de IO: extrair leitura/escrita de planilhas Excel
(`core/io/excel.py`) e exportação DXF (`core/io/cad_dxf.py`) de
`pages/sapatas.py`. Fronteira do domínio: leitor Excel devolve
`FundacaoProjeto`; escritor recebe `OptimisationResult` (Sprint 3.5)
e produz Excel/DXF.

## Vínculos

- [[10_Melhorias/Roadmap Sugerido]]
- [[10_Melhorias/MOC - Melhorias]]
- [[12_Auditoria/Sprint 3 - Refactor estrutural - kickoff - 2026-04-27]]
- [[12_Auditoria/Sprint 3.2 - Engineering migration - 2026-04-28]]
- [[10_Melhorias/Refactor - POO Domain Model]]
- [[01_Projeto/Convenções do Projeto]]
