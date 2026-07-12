# Artigo da IC — Lucas Teixeira Correia

Artigo científico em construção, vinculado ao plano de trabalho de Iniciação Científica intitulado **"Otimização do dimensionamento e posicionamento de fundações superficiais de edifícios considerando conceitos de empacotamento e mecânica das estruturas"** (orientação: Profa. Dra. Maria José Pereira Dantas, PUC Goiás).

## Recorte do artigo

Apresenta a primeira frente do plano de trabalho — o **pré-dimensionamento geométrico otimizado de sapatas isoladas** com posições de pilares fornecidas pelo projeto estrutural — por meio de uma arquitetura híbrida **EGO + GPR + Algoritmo Genético**. A frente complementar (incorporação explícita do problema de empacotamento) é apresentada como trabalho futuro.

Título adotado no manuscrito:

> **Otimização computacional do pré-dimensionamento geométrico de sapatas isoladas: uma abordagem com modelo substituto e metaheurística**

Esse título mantém aderência ao plano de trabalho maior, mas delimita o recorte atual do artigo ao pré-dimensionamento geométrico, sem antecipar a implementação futura de empacotamento/posicionamento nem sugerir projeto executivo completo.

> Status (2026-07-12): **manuscrito reposicionado como pré-dimensionamento geométrico experimental e polido para pré-submissão**. A punção cobre os dois contornos críticos da NBR 6118 (C e C′ a 2d, com Tabelas 19.2/17.3 e hipóteses declaradas; fonte de apoio: Santos, Lima Neto & Ferreira 2018, RIEM — PDF em `docs/articles/`). A formulação de tensão foi corrigida: o peso próprio agora é calculado como `gamma_c * h_x * h_y * h_z`, sem os coeficientes legados `1,05` e `1,30`, e a comparação com `sigma_adm` é direta. Após essa alteração, os protocolos `run_final_benchmark.py`, `run_cbo_benchmark.py`, `run_gpr_kernel_study.py`, `run_decomposition_baseline.py` e `make_paper_artifacts.py` foram executados; os números, tabelas e figuras do manuscrito refletem a formulação corrigida e incluem auditoria de quase separabilidade por sapata. A rodada de polish encurtou resumo/abstract, removeu placeholders editoriais, padronizou termos menos herméticos e fechou agradecimentos, conflitos de interesse e disponibilidade de dados/código sem depender ainda de revista-alvo.

## Estrutura de arquivos

```
artigo_ic_lucas/
├── main.tex                # documento principal (carrega seções via \input)
├── referencias.bib         # base BibTeX (estilo ABNT autor-data)
├── secoes/
│   ├── 01_introducao.tex
│   ├── 02_estado_da_arte.tex
│   ├── 03_fundamentacao_teorica.tex
│   ├── 04_metodologia.tex
│   ├── 05_implementacao_software.tex
│   ├── 06_resultados_parciais.tex
│   ├── 07_discussao.tex
│   ├── 08_conclusoes_parciais.tex
│   └── 09_agradecimentos.tex
├── figuras/                # figuras geradas deterministicamente
└── README.md
```

## Compilação

Requer uma distribuição LaTeX completa (TeX Live, MiKTeX) com `abntex2` instalado.

```bash
cd docs/artigo_ic_lucas
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Alternativa moderna:

```bash
latexmk -pdf main.tex
```

## Observações sobre o conteúdo

- **Formato**: o manuscrito está em **duas colunas** (10pt, título e resumos em largura total, margens 2,0 cm, `balance` na última página), formato universal para submissão internacional; quando a revista-alvo for definida, basta migrar para o template próprio.
- **Título**: o título prioriza o problema de engenharia e a família metodológica, em vez de listar EGO--GPR--AG. A arquitetura específica permanece no resumo, metodologia e discussão.
- **Citações e links**: as citações usam `abntex2cite` em estilo autor-data. Para aproximar a saída da NBR 10520:2023, as citações parentéticas foram padronizadas com macros locais (`\citepabnt`, `\citepabnttwo`, `\citepabntthree`, `\citepabntfour`), evitando sobrenomes em caixa alta. Os links foram configurados com `hidelinks`, sem citações azuis.
- **Notas visíveis de consolidação**: as marcações internas de revisão foram removidas do manuscrito. Pendências técnicas permanecem registradas neste README e na discussão como limitações ou trabalhos futuros.
- **Justificativa do EGO**: a função pseudo-objetivo atual tem baixo custo computacional. Por isso, o manuscrito apresenta o EGO--GPR--AG como arquitetura metodológica em investigação e preparação para extensões mais custosas, não como escolha automaticamente necessária para reduzir custo de avaliação.
- **Aderência normativa**: o texto separa prescrições normativas, correlações empíricas, hipóteses de pré-dimensionamento e simplificações internas. A punção cobre os dois contornos críticos da NBR 6118 (C e C′), com ρ mínimo declarado; antes da submissão, ainda é necessário validar/substituir a correlação baseada em `Nspt` e formalizar as combinações de ações.
- **Diferenciação em relação ao TCC do grupo (Pereira et al., em construção)**: o TCC apresenta a metodologia de forma abrangente e aplicada (3 estudos de caso, comparação completa Monte Carlo, exemplo numérico exaustivo). Este artigo recorta o **eixo metodológico** (otimização assistida por modelo substituto para fundações) e **integra revisão crítica do estado da arte** com posicionamento da pesquisa para a frente futura de empacotamento.
- **Sem plágio**: todas as citações usam paráfrase com atribuição. Equações da metodologia são reapresentadas porque pertencem ao núcleo formal do problema (verificações de pré-dimensionamento, punção e EGO clássico) e não constituem material original do TCC.
- **Métricas de kernels**: valores definitivos inseridos (21 kernels × 2 penalidades × 3 réplicas com seeds), incluindo o RMSE restrito à região factível — a métrica que expõe o que o R² global esconde.
- **Tabelas**: a tabela de parâmetros do AG reflete o protocolo final (`pop_size=50`, `epoch=30`, mealpy 3.0.3); todas as tabelas de resultados são geradas por script a partir dos artefatos persistidos (nunca editar `tabelas/*.tex` à mão). A tabela `tab_decomposicao.tex` vem de `experiments/protocolo_final/decomposicao_de/summary.csv` e documenta a linha de base de Differential Evolution por sapata.
- **Figura ausente**: ainda é recomendável inserir um fluxograma do EGO em `figuras/fluxo_ego.pdf` ou como TikZ antes da versão de submissão.

## Referências utilizadas

29 referências em `referencias.bib`, organizadas em seis grupos:

1. Otimização global e modelos substitutos (Jones, Snoek, Shahriari, Schulz, Williams).
2. Otimização aplicada a fundações superficiais (Wang & Kulhawy, Gandomi & Kashani, Nigdeli et al., Kashani et al., Waheed et al. ×2).
3. Modelos de dados em geotecnia e aprendizado ativo em projeto computacional (Ahmad et al., Khajehzadeh et al., Fattahi et al., Deng et al.).
4. Comparação de metaheurísticas (Morales-Castañeda, Abualigah, Gomes).
5. Engenharia de fundações + normas (NBR 6118, NBR 6122, Juang & Wang, Santos/Lima Neto/Ferreira 2018 — punção em sapatas; Bezerra et al. e Khan et al. permanecem no `.bib` como referências candidatas, mas não devem ser usadas no corpo sem nova validação de pertinência).
6. Bayesian optimization com restrições em engenharia estrutural (Gardner et al., Eriksson & Poloczek, Mathern et al., Yu et al.).

A maior parte dos artigos corresponde a PDFs presentes em `docs/articles/`. A triagem de 2026-07-12 moveu os novos PDFs soltos para subpastas, isolou duplicatas exatas e criou fichas no vault. Os artigos de Chandra et al. (2021) e Jakubczyk-Galczynska et al. (2024) foram fichados, mas classificados como fora do escopo do artigo 1 e não foram citados. As normas ABNT citadas (`NBR 6118` e `NBR 6122`) devem ser conferidas nas versões oficiais antes da submissão, pois não substituem a literatura científica indexada. Quando novos artigos forem incorporados, atualizar `referencias.bib` e criar a entrada `cite` correspondente nas seções.

## Próximas ações sugeridas

- [x] Inserir valores definitivos de R², MAE e RMSE em `06_resultados_parciais.tex`. *(2026-07-10 — estudo controlado: 21 kernels × 2 penalidades × 3 réplicas; inclui RMSE restrito à região factível)*
- [x] Equalizar busca aleatória, EGO e linhas de base pelo mesmo orçamento de avaliações e por múltiplas seeds. *(2026-07-12 — protocolo S1/S2, 30 seeds pareadas, Wilcoxon-Holm)*
- [x] Inserir figuras dos scatter plots observado×predito. *(2026-07-10 — `figuras/fig_gpr_obs_pred.pdf`, geradas do estudo com seeds)*
- [x] Comparar EGO--GPR--AG contra AG puro, PSO, GWO e busca aleatória sob protocolo equivalente. *(2026-07-10 — Seções 6.4–6.6; CMA-ES fica como trabalho futuro)*
- [x] Separar/qualificar a questão da sobreposição: nos 3 casos congelados a restrição é inativa por construção — declarado explicitamente na Seção 6.1.
- [x] Quantificar a quase separabilidade com linha de base de decomposição por sapata. *(2026-07-12 — `scripts/run_decomposition_baseline.py`, Tabela `tab_decomposicao`)*
- [ ] Inserir figura do fluxograma do EGO (`figuras/fluxo_ego.pdf`) — opcional; o Algoritmo 1 já cobre o fluxo.
- [x] Inserir figura do arranjo em planta dos casos de estudo. *(2026-07-10 — `fig_planta_casos`, melhores soluções factíveis reproduzidas por seed)*
- [ ] Substituir ou validar a origem das correlações `Nspt/30`, `Nspt/40` e `Nspt/50`; se mantidas, apresentá-las apenas como hipótese de pré-dimensionamento.
- [x] Reexecutar `scripts/run_final_benchmark.py`, `scripts/run_cbo_benchmark.py`, `scripts/run_gpr_kernel_study.py` e `scripts/make_paper_artifacts.py` após a correção da formulação de tensão. *(2026-07-12 — resultados, tabelas e figuras atualizados com a formulação corrigida)*
- [x] Completar a verificação de punção. *(2026-07-10 — contorno C′ a 2d implementado na FO com ρ_min declarado; nunca ativo nos casos: S/R ≤ 0,69)*
- [x] Remover todas as marcações internas de revisão do manuscrito.
- [x] Fazer polish de pré-submissão: resumo/abstract encurtados, tom técnico padronizado, placeholders removidos e declarações finais fechadas. *(2026-07-12 — `latexmk -pdf -g` compilou 22 páginas sem `Overfull` ou referências indefinidas)*
- [ ] Definir revista-alvo (sugestões: *Engineering Structures*, *Structural Concrete*, *Computers & Structures*, *IBRACON Structures and Materials Journal*) e ajustar template.
- [x] Declarar Agradecimentos, Conflitos de interesse e Disponibilidade dos dados/código de forma genérica e compatível com submissão preliminar.

## Reprodução dos resultados

```bash
# 1. Protocolo comparativo (≈ 87 min): experiments/protocolo_final/
.venv/bin/python scripts/run_final_benchmark.py
# 2. Estudo GPR kernels × penalidade (≈ 3 min): experiments/estudo_gpr/
.venv/bin/python scripts/run_gpr_kernel_study.py
# 3. Baseline de decomposição por sapata:
.venv/bin/python scripts/run_decomposition_baseline.py
# 4. Figuras + tabelas do manuscrito (determinístico):
.venv/bin/python scripts/make_paper_artifacts.py
# 5. Compilação:
cd docs/artigo_ic_lucas && latexmk -pdf main.tex
```
