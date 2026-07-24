---
tags: [auditoria, sprint, artigo, tensao, nbr6118, metodologia, reproducibilidade]
data: 2026-07-12
status: concluido
---

# Sprint 5.4 - Correcoes artigo e tensao (2026-07-12)

Sprint de fechamento motivada pela revisao critica do artigo em `docs/artigo_ic_lucas` e pela necessidade de alinhar codigo, metodologia, resultados e vault antes de nova rodada com a orientadora.

## Veredito curto

A lista de revisoes era majoritariamente correta. Os pontos mais graves eram: formula de tensao com coeficientes legados, peso proprio nao dependente do volume, convencao ambigua de momentos, aceitacao implicita de violacoes por penalidade, estatistica nao pareada e afirmacoes fortes demais sobre dimensionalidade/causalidade.

O projeto agora esta coerente como **pre-dimensionamento geometrico experimental**, nao como projeto executivo completo de fundacoes.

## Correcoes implementadas

- `core/engineering/tensao.py`: `calcular_sigma_max_min` agora calcula peso proprio explicitamente como `gamma_c * h_x * h_y * h_z`, com `gamma_c = 25 kN/m3`, e remove os coeficientes legados `1.05` e `1.30`.
- `core/api/objective.py`: nucleo vetorizado da funcao objetivo atualizado para a mesma formulacao escalar.
- `fundacao.py`: compatibilidade atualizada para passar `h_z` ao calculo de tensoes.
- `tests/test_engenharia.py`: testes reescritos para a nova formula e para a convencao de momentos do FundaIA.
- `core/domain/combinacao.py` e `core/api/objective.py`: documentacao corrigida; `Fz > 0` deixou de ser justificado por divisao por `Fz` e passou a ser escopo fisico do modelo de contato solo-sapata comprimido.
- Documentacao de punção: C' descrito como contorno a `2d`, em coerencia com a implementacao.

## Convencao de momentos

A convencao do FundaIA foi explicitada:

- `Mx = Fz * ex`: componente que produz excentricidade na direcao `x`, portanto entra com `h_x`.
- `My = Fz * ey`: componente que produz excentricidade na direcao `y`, portanto entra com `h_y`.

Se a origem dos esforcos reportar momentos estruturais em torno dos eixos globais, a conversao deve ser feita antes da importacao. Esta convencao foi registrada no artigo, nas docstrings e no vault.

## Artigo

Arquivos principais revisados:

- `docs/artigo_ic_lucas/main.tex`
- `docs/artigo_ic_lucas/secoes/04_metodologia.tex`
- `docs/artigo_ic_lucas/secoes/06_resultados_parciais.tex`
- `docs/artigo_ic_lucas/secoes/07_discussao.tex`
- `docs/artigo_ic_lucas/secoes/08_conclusoes_parciais.tex`
- `docs/artigo_ic_lucas/README.md`

Mudancas centrais:

- Formula de tensao atualizada para peso proprio volumetrico e comparacao direta com `sigma_adm`.
- Texto reposicionado como pre-dimensionamento geometrico experimental.
- Violacoes residuais por penalidade passaram a ser tratadas como diagnostico metodologico, nao como solucao aceitavel de projeto.
- Comparacoes estatisticas descritas como Wilcoxon pareado com correcao de Holm.
- Afirmações causais sobre dimensao foram reduzidas: ha apenas uma instancia por dimensionalidade.
- CBO, EGO, buscas diretas e busca aleatoria agora sao comparados apenas pelos resultados factiveis/metricas coerentes do protocolo regenerado.
- NBR 6118:2026 mantida como referencia normativa citada; catalogo ABNT consultado para confirmar a existencia da versao 2026: <https://www.abntcatalogo.com.br/pnm.aspx?Q=cWxPU1p0MjNpTkI2cU1zTjVwS2dSclRyTFdxMW9vbVRMQUJUTERWST0=>.

## Resultados regenerados

Protocolos reexecutados apos a correcao da tensao:

- `scripts/run_final_benchmark.py`: protocolo final concluido em 87,0 min.
- `scripts/run_cbo_benchmark.py`: protocolo CBO concluido em 133,3 min.
- `scripts/run_gpr_kernel_study.py`: estudo com 126 ajustes concluido em 3,2 min.
- `scripts/make_paper_artifacts.py`: figuras, tabelas LaTeX e CSVs do artigo regenerados.

Numeros-chave agora refletidos no manuscrito:

- Melhor arquitetura assistida vs busca aleatoria: reducoes medianas de `24,7%`, `45,7%` e `71,4%`.
- EGO isolado vs busca aleatoria: `23,4%`, `38,8%` e `63,9%`.
- CBO melhora a media de `Theta` frente ao EGO em `1,5%`, `9,3%` e `21,7%`.
- CBO melhora o melhor volume estritamente factivel frente ao EGO em `0,8%`, `3,5%` e `15,3%`.
- Factibilidade estrita: EGO `83%` nos tres casos; CBO `63%`, `37%` e `83%`.
- Kernel de producao: `R2 = 0,931 +/- 0,011`; RMSE na regiao factivel `1,28 m3` com `alpha=10`, contra aproximadamente `1,2e5 m3` com `alpha=1e6`.

## Vault atualizado

Notas atualizadas ou alinhadas:

- [[01_Projeto/Convenções do Projeto]]
- [[02_Engenharia/Flexão Composta - Sigma Max e Min]]
- [[02_Engenharia/Guia Didatico - Dimensionamento de Sapatas Isoladas]]
- [[02_Engenharia/NBR 6118]]
- [[02_Engenharia/Tensão Admissível do Solo]]
- [[02_Engenharia/Verificação à Punção]]
- [[05_Dados/Schema das Planilhas]]
- [[07_Issues/Lista Mestre de Issues]]
- [[07_Issues/Issue - Punção seção C linha comentada]]
- [[10_Melhorias/Punção Seção C linha - completar]]
- [[10_Melhorias/Guia - Validação antes do Bin Packing]]
- [[10_Melhorias/MOC - Melhorias]]

## Validacao

- `.venv/bin/pytest` -> `264 passed in 11.04s`.
- `cd docs/artigo_ic_lucas && latexmk -pdf -g -interaction=nonstopmode main.tex` -> `main.pdf` gerado com 21 paginas.

Avisos de compilacao remanescentes:

- `Optional argument of \twocolumn too tall on page 1`.
- `Underfull`/`Overfull` pequenos e pagina com floats na pagina 16.

Nao ha erro de LaTeX, BibTeX, referencia ou figura ausente.

## Pendencias assumidas

Estas pendencias nao devem ser escondidas no artigo:

- Formalizar combinacoes de acoes separando servico e ELU.
- Implementar dimensionamento/verificacao de flexao, cisalhamento unidirecional, ancoragem/detalhamento e custo total.
- Validar ou substituir as correlacoes `Nspt/30`, `Nspt/40` e `Nspt/50`; por enquanto elas sao hipotese empirica de pre-dimensionamento.
- Adicionar baselines deterministicas, Differential Evolution, CMA-ES e decomposicao por sapata.
- Separar efeito de dimensao do efeito da instancia com mais casos por dimensionalidade.
- Ampliar o estudo de penalidade e aumentar replicas do estudo de kernels.
- Reportar tamanhos de efeito, intervalos de confianca e calibracao de incerteza do GP.

## Notas relacionadas

- [[12_Auditoria/Sprint 5.1 - Protocolo experimental final e casos-limite - 2026-07-10]]
- [[12_Auditoria/Sprint 5.2 - Puncao C linha e duas colunas - 2026-07-10]]
- [[12_Auditoria/Sprint 5.3 - Frente C CBO - 2026-07-11]]
