---
tags: [auditoria, vault, projeto]
data: 2026-04-27
escopo: leitura-estatica
---

# Auditoria — Vault Obsidian vs Projeto FundaIA

> Síntese das ações tomadas no vault em resposta à auditoria de 2026-04-27.
> Relatório original integral em [[09_Relatorios/Auditoria - Vault vs Projeto - 2026-04-27]].
> Auditoria feita por leitura estática + parse AST + inspeção de planilhas/notebooks na branch `refactor/code-base`. Sem rodar Streamlit nem otimizações.

## Resumo

Vault está **80–85% coerente** com o projeto. Captura bem arquitetura, conceitos de engenharia, pipeline EGO/GPR e principais débitos técnicos. Os 15–20% restantes são pontos que poderiam induzir interpretação errada e foram **endereçados nesta nota** com criação de issues novas e correção de descrições.

## Pontos confirmados (vault descreve corretamente)

- `pages/sapatas.py` duplicado em dois blocos quase idênticos.
- `requirements.txt` em UTF-16 LE com BOM.
- `obj_felipe_lucas` e `obj_teste` clones.
- Punção seção C' comentada.
- `metapy_toolbox/methods.py` 100% comentado e ainda importado.
- `grey_wolf.py` com placeholder `'aqui implementa função lucas'`.
- Notebooks passam 5º arg de penalidade que a função ignora.

## Pontos novos identificados na auditoria (e adicionados ao vault)

| # | Descoberta | Onde foi registrado |
|---|---|---|
| 1 | `ego.py` registra novo ponto com `ITER=0` e `ID` constante | [[07_Issues/Issue - Histórico do EGO com ITER e ID incorretos]] |
| 2 | `n_rep=5` reusa o mesmo `x_ini` LHS | [[07_Issues/Issue - n_rep reusa população inicial]] |
| 3 | `griewank`/`powell` em `benchmark.py` aparentam erros | [[07_Issues/Issue - Benchmarks suspeitos]] |
| 4 | Notebooks ainda apontam para `assets/el08.xlsx` (inexistente) | [[07_Issues/Issue - Notebooks com paths quebrados]] |
| 5 | `save_dxf` deixa tempfile órfão em `/tmp` | [[07_Issues/Issue - DXF tempfile não removido]] |
| 6 | Sobreposição contada duas vezes (`i→j` + `j→i`) — intencional? | [[07_Issues/Issue - Sobreposição contada duas vezes]] |

## Inconsistências documentais corrigidas no vault

| Inconsistência | Correção aplicada |
|---|---|
| Vault dizia "20 kernels"; código tem 21 (k00–k19 + k20 produção) | [[03_Otimizacao/Kernels GPR]] e [[04_Codigo/fundacao.py]] reescritos com a convenção "20 + 1" |
| `toy_problem_copy_3.xlsx` descrito como 3 fundações; tem 1 | [[05_Dados/Assets - Templates Excel]] corrigido |
| Nota [[03_Otimizacao/Problema de Empacotamento]] estava vazia | Preenchida (estado atual + roadmap completo de packing) |
| `wake_up.py` — confusão entre default da dataclass (None) e do CLI | [[04_Codigo/ops - wake_up.py]] esclarecido |
| Severidade "baixa" da issue de args ignorados subestimava o impacto | [[07_Issues/Issue - Args extras em obj_teste]] elevada para "alta" com aviso `[!danger]` |

## Estatísticas do vault após auditoria

- **107 notas** Markdown.
- **0 link real quebrado** (5 placeholders de template — esperados).
- **0 órfãos** além de README e templates.
- **Tags principais**: `#otimizacao` `#engenharia` `#codigo` `#issue` `#projeto` `#pesquisa` `#melhorias` `#packing` `#gpr` `#ego`.

## Recomendações antes da apresentação ao orientador

> [!check] Pré-envio
> - [x] Preencher nota de packing (feito nesta auditoria).
> - [x] Alinhar contagem de kernels (feito).
> - [x] Marcar paths quebrados de notebooks (feito).
> - [x] Issues novas para EGO/n_rep/penalidade/benchmarks (feito).
> - [ ] Definir, junto ao orientador, a prioridade dos itens P0 do roadmap.
> - [ ] Avaliar a inclusão do vault no controle de versão (hoje está untracked).
> - [ ] Registrar artigos lidos pela equipe em [[08_Artigos/Index de Artigos]].

## Prioridade recomendada (compactada)

**P0** (antes de qualquer resultado novo):
1. requirements.txt + dependências.
2. Duplicação `sapatas.py`.
3. Penalidade nos notebooks (decidir e parametrizar).
4. Validar/ajustar gráficos e tabelas existentes em `assets/graphics/` e `assets/tables/` em função do item 3.
5. Testes mínimos da engenharia.

**P1** (antes de defender):
1. Histórico do EGO (`ITER`/`ID`).
2. `n_rep` independentes.
3. Sementes controladas.
4. Validar benchmarks.
5. Validar engenharia contra exemplo de bibliografia.

**P2** (evolução do software):
- Refactor POO + UI separada + Pydantic + vetorização + persistência + CI.

**P3** (pesquisa original):
- Packing como variável (ver [[03_Otimizacao/Problema de Empacotamento]] §2).
- PI-GPR / Constrained BO (ver [[11_Frentes_de_Pesquisa/MOC - Frentes de Pesquisa]]).

## Veredito

Vault pronto para apresentação ao orientador como **mapa do projeto** e **proposta de direções**. Não constitui (ainda) fonte de verdade experimental — os pontos P0 e P1 acima precisam ser tratados antes da defesa de resultados quantitativos.

## Vínculos

- [[00_Mapa/MOC - Mapa Geral]]
- [[07_Issues/Lista Mestre de Issues]]
- [[10_Melhorias/Roadmap Sugerido]]
- [[01_Projeto/Escopo da IC]]
