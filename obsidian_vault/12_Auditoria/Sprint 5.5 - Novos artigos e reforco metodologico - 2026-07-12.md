# Sprint 5.5 - Novos artigos e reforco metodologico - 2026-07-12

## Objetivo

Incorporar os novos PDFs soltos em `docs/articles`, detectar duplicatas, criar fichas no vault, revisar citacoes/referencias do artigo em `docs/artigo_ic_lucas` e fortalecer metodologicamente o manuscrito sem fugir do escopo: pre-dimensionamento geometrico experimental de sapatas isoladas com posicoes fixas.

## Triagem dos novos artigos

### Incorporados e citados

- [[Nigdeli et al. 2018 - Metaheuristic Optimization RC Footings]]
  - Uso no artigo: fundamenta metaheuristicas em sapatas de concreto armado e a pertinencia de Differential Evolution como referencia. Tambem reforca que estudos de projeto economico completo incluem armadura, custo e verificacoes estruturais adicionais.
  - DOI: `10.1007/s12205-018-2010-6`.

- [[Mathern et al. 2021 - Multiobjective Constrained BO Structural Design]]
  - Uso no artigo: reforca a discussao de otimizacao bayesiana com restricoes em projeto estrutural e a distincao entre objetivos baratos e restricoes potencialmente caras.
  - DOI: `10.1007/s00158-020-02720-2`.

- [[Khajehzadeh et al. 2022 - Hybrid Soft Computing Shallow Foundations]]
  - Uso no artigo: apoio bibliografico para modelos de dados e otimizacao em fundacoes superficiais; citado com cuidado para nao validar diretamente a correlacao simples `N_spt -> sigma_adm`.
  - DOI: `10.3390/su14031847`.

- [[Fattahi et al. 2025 - Settlement Prediction Intelligent Optimization]]
  - Uso no artigo: apoio para a frente futura de recalques e sensibilidade de modelos geotecnicos ao SPT; nao usado como validacao da correlacao `N_spt/30`, `N_spt/40`, `N_spt/50`.
  - DOI: `10.32604/cmes.2025.062390`.

- [[Yu et al. 2025 - PFN Constrained Engineering BO]]
  - Uso no artigo: citado apenas como possibilidade futura se o ajuste de multiplos GPs em CBO se tornar gargalo real.
  - DOI: `10.1007/s00158-025-03987-z`.

### Fichados, mas fora do escopo do artigo 1

- [[Chandra et al. 2021 - Bored Pile Cost Optimization]]
  - Motivo: fundacoes profundas (estacas escavadas), nao sapatas isoladas. Pode ser util em revisao futura ampla, mas nao entra no recorte atual.

- [[Jakubczyk-Galczynska et al. 2024 - Construction Management Bayesian Networks]]
  - Motivo: redes bayesianas para gestao da construcao; nao trata otimizacao bayesiana nem fundacoes superficiais.

### Duplicatas detectadas

- `buildings-12-00471-v2.pdf` era duplicata exata do artigo de Waheed et al. 2022 ja existente em `docs/articles/01_artigo_1_ego_gpr/`.
- `s00158-025-03987-z.pdf` e `s00158-025-03987-z-2.pdf` eram duplicatas exatas entre si; uma copia foi preservada como artigo classificado e a outra movida para `docs/articles/00_duplicados_exatos/`.
- Registro detalhado: [[Duplicatas detectadas - 2026-07-12]].

## Organizacao fisica em `docs/articles`

- Criada a pasta `00_duplicados_exatos/` para PDFs duplicados preservados sem poluir a base principal.
- Criada a pasta `06_novos_classificados/fora_escopo/` para artigos lidos e fichados, mas nao usados no artigo 1.
- Novos PDFs relevantes foram movidos para:
  - `01_artigo_1_ego_gpr/`
  - `02_apoio_tecnico_geotecnia/`
  - `05_frente_c_cbo/`
- `docs/articles/README.md` foi atualizado com a classificacao, DOIs e decisoes de uso.

## Fortalecimento metodologico implementado

### Baseline de decomposicao por sapata

Foi criado `scripts/run_decomposition_baseline.py` para quantificar a quase separabilidade dos tres casos congelados.

Procedimento:

1. Divide cada caso em subproblemas de uma sapata.
2. Otimiza cada subproblema tridimensional com Differential Evolution.
3. Concatena as solucoes locais.
4. Reavalia o vetor completo no mesmo `avaliar_projeto_componentes`, incluindo novamente sobreposicao.
5. Persiste artefatos em `experiments/protocolo_final/decomposicao_de/`.

Resultados reavaliados:

| Caso | Volume decomposto | Melhor protocolo | Ganho |
|---|---:|---:|---:|
| Caso 1 | 3,108824 m3 | 3,108826 m3 | <0,01% |
| Caso 2 | 4,750747 m3 | 4,787486 m3 | 0,77% |
| Caso 3 | 2,122252 m3 | 2,167259 m3 | 2,08% |

Interpretacao:

- O baseline confirma que os casos atuais sao quase separaveis por sapata.
- O artigo agora evita atribuir causalmente o aumento de ganho a dimensionalidade.
- O baseline nao entra nas matrizes de Wilcoxon porque usa orcamento diagnostico diferente; ele mede proximidade estrutural ao otimo das instancias simplificadas.

### Artefatos do artigo

- `scripts/make_paper_artifacts.py` agora gera `tabelas/tab_decomposicao.tex`.
- `assets/tables/protocolo_final/decomposicao_de_summary.csv` foi criado como espelho CSV.
- `docs/artigo_ic_lucas/secoes/06_resultados_parciais.tex` recebeu a subsecao "Diagnostico de quase separabilidade por decomposicao".
- `docs/artigo_ic_lucas/secoes/04_metodologia.tex` documenta o protocolo da auditoria.
- `docs/artigo_ic_lucas/secoes/07_discussao.tex` e `08_conclusoes_parciais.tex` foram ajustadas para incorporar a nova evidencia.
- `docs/artigo_ic_lucas/secoes/01_introducao.tex` e `README.md` foram atualizados para refletir a auditoria.

## Referencias e citacoes

Entradas adicionadas em `docs/artigo_ic_lucas/referencias.bib`:

- `nigdeli2018metaheuristic`
- `mathern2021multiobjective`
- `khajehzadeh2022effective`
- `fattahi2025settlement`
- `yu2025fast`

Total atual: 29 referencias.

As citacoes foram inseridas com cautela:

- Nigdeli et al. 2018: comparacao de metaheuristicas e escopo mais completo de projeto economico.
- Mathern et al. 2021: BO/CBO em projeto estrutural.
- Khajehzadeh et al. 2022 e Fattahi et al. 2025: geotecnia baseada em dados e necessidade de calibracao futura, sem validar diretamente a correlacao empirica atual.
- Yu et al. 2025: futuro, apenas se PFN/CBO fizer sentido por gargalo de GP.

## Validacao executada

- `scripts/run_decomposition_baseline.py` executado com sucesso.
- `scripts/make_paper_artifacts.py` executado com sucesso.
- `latexmk -pdf -g -interaction=nonstopmode main.tex` executado em `docs/artigo_ic_lucas`.
  - Resultado: `main.pdf` gerado com 22 paginas.
  - Sem citacoes ausentes ou erros de compilacao.
  - Restaram apenas avisos de layout (`underfull`, `balance`), esperados neste estagio.
- `.venv/bin/python -m compileall scripts/run_decomposition_baseline.py scripts/make_paper_artifacts.py` executado com sucesso.

## Pontos ainda pendentes

- Validar/substituir a correlacao empirica `N_spt/30`, `N_spt/40`, `N_spt/50` com fonte geotecnica apropriada ou manter explicitamente como hipotese preliminar.
- Reconstruir formalmente combinacoes de acoes de servico e ELU em etapa futura.
- Incluir verificacao de flexao, cisalhamento unidirecional, armadura, ancoragem/detalhamento e custo total para aproximar o escopo de projeto economico completo.
- Construir casos acoplados com posicoes variaveis/empacotamento para testar de fato o regime nao separavel.
- Definir revista-alvo e ajustar template, secoes obrigatorias, disponibilidade de dados/codigo e politica de referencias.
