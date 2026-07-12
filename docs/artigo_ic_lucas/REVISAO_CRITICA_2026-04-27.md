# Revisao critica do artigo IC Lucas - 2026-04-27

## Escopo da revisao

Revisao feita sobre o manuscrito LaTeX em `docs/artigo_ic_lucas`, considerando:

- contexto do FundaIA no vault;
- plano de trabalho da IC;
- relatorio parcial da IC;
- TCC/artigo em construcao de Filipe Amaral;
- PDFs disponiveis em `docs/articles`.

O objetivo foi verificar coerencia cientifica, adequacao das citacoes, tom academico, aderencia ao estado atual do software e riscos antes de futura submissao.

## Veredito

O manuscrito tem uma estrutura adequada para artigo cientifico e esta bem posicionado como trabalho parcial sobre **otimizacao assistida por surrogate para dimensionamento de sapatas isoladas**. O texto estava mais forte do que o estado atual do projeto permitia em alguns pontos; por isso, foram suavizadas afirmacoes sobre reprodutibilidade, otimalidade, aderencia normativa, Monte Carlo como pratica profissional, viabilidade dos resultados e maturidade da etapa de posicionamento/empacotamento.

A versao revisada esta mais segura para evoluir ate uma submissao: apresenta o FundaIA/EGO-GPR como contribuicao metodologica parcial, sem afirmar que o bin packing/layout completo ja foi resolvido. A versao atual contem marcacoes explicitas `[[[[ ... ]]]]` para deixar visiveis pontos que exigem pesquisa, validacao experimental ou decisao editorial. Essas marcacoes sao intencionais e devem ser removidas apenas quando cada pendencia for resolvida.

## Principais correcoes aplicadas

- Corrigida a referencia local de Processos Gaussianos para o PDF disponivel: Williams e Rasmussen (1995), em vez do livro de 2006 nao presente em `docs/articles`.
- Corrigido o DOI de Deng et al. (2026): `10.1007/s00158-025-04231-4`.
- Corrigida a referencia de Ahmad et al. (2021): titulo, autores, journal `Applied Sciences`, volume 11, numero 21, artigo 10317 e DOI `10.3390/app112110317`.
- Corrigida a referencia de Waheed et al. (2025): journal `Innovative Infrastructure Solutions`, volume 10, artigo 56 e DOI `10.1007/s41062-024-01823-9`.
- Adicionada referencia Khajehzadeh et al. (2012) sobre `Gravitational Search Algorithm` em fundacoes rasas, porque ela esta em `docs/articles` e fortalece o estado da arte de metaheuristicas em fundacoes.
- Removidas expressoes fortes como "solucoes otimas" sem qualificacao e "Monte Carlo como pratica tradicional" em favor de "baseline de busca aleatoria".
- Marcado que P01/P02 pertencem ao caso de proximidade severa e nao devem ser usados como evidencia de solucao globalmente factivel.
- Removida tabela com `Placeholder` e valores `--` do corpo compilavel.
- Removida referencia a figura ausente no corpo do artigo.
- Ajustado o preambulo LaTeX para carregar `english` no `babel`, pois o abstract em ingles usa `otherlanguage`.
- Movido `xspace` para antes da macro `\fundaai`.
- Removida do resumo, do abstract, da introducao e da conclusao a formulacao que apresentava o manuscrito como "vinculado a uma pesquisa de Iniciacao Cientifica em andamento"; o vinculo institucional foi mantido apenas em locais contextuais, como cabecalho, README e agradecimentos.
- Ajustado o formato para mais proximo de artigo cientifico: retirada da capa isolada, retirada do sumario, manutencao em uma coluna e links/citacoes sem cor azul.
- Compactados resumo e abstract para faixa compativel com artigo cientifico; palavras-chave foram separadas por ponto.
- Revisadas as tabelas: a tabela do AG agora declara somente parametros comprovados na implementacao, e a tabela de volumes foi marcada como media bruta diagnostica, com fonte e status dos casos de proximidade severa.
- Padronizadas as citacoes parenteticas para aproximacao com a NBR 10520:2023, evitando sobrenomes em caixa alta no corpo do texto. Para isso foram criadas macros locais `\citepabnt`, `\citepabnttwo`, `\citepabntthree` e `\citepabntfour`.
- Revisado o titulo do artigo para priorizar o problema de engenharia e a familia metodologica, em vez de restringir o manuscrito a EGO--GPR--AG no titulo.
- Reforcada a justificativa do EGO: como a funcao pseudo-objetivo atual e relativamente barata, o texto agora apresenta o EGO--GPR--AG como arquitetura metodologica em investigacao e preparacao para extensoes mais custosas, nao como necessidade computacional obvia nesta etapa.
- Separadas, na metodologia, prescricoes normativas, correlacoes empiricas, hipoteses de pre-dimensionamento e simplificacoes internas da implementacao.
- Rebaixada a verificacao de puncao para "verificacao parcial inspirada nos criterios da NBR 6118", limitada ao perimetro C, ate que C', momentos e coeficientes normativos sejam implementados ou delimitados formalmente.
- Inseridas notas visiveis no texto para: validar Nspt/30, Nspt/40, Nspt/50; validar coeficientes 1,05 e 1,30; inserir tabela quantitativa do GPR; formalizar protocolo experimental; separar casos de proximidade severa; e justificar melhor o uso de EGO. Atualizacao 2026-07-12: a pendencia dos coeficientes 1,05/1,30 foi resolvida por remocao da metodologia; o peso proprio agora entra por volume e a comparacao com `sigma_adm` e direta.
- Retiradas/estacionadas no corpo as referencias perifericas ou ainda pouco rastreadas (`khan2023python` e `bezerra2024elementos`), mantendo-as no `.bib` como candidatas que podem voltar apos validacao.
- Esvaziadas as secoes de Agradecimentos, Conflitos de interesse e Disponibilidade de dados/codigo, substituindo-as por notas de preenchimento futuro conforme revista-alvo, politica institucional e decisao de abertura do repositorio.

## Validacao das citacoes

| Chave | Validacao contra material local |
| --- | --- |
| `jones1998efficient` | Coerente para EGO, EI e otimizacao de funcoes black-box caras. |
| `snoek2012practical` | Coerente para Bayesian Optimization com GP e escolha de hiperparametros. |
| `shahriari2016review` | Coerente para revisao de BO, aquisicoes, restricoes e desafios praticos. |
| `schulz2018tutorial` | Coerente para GPR, kernels, exploracao/explotacao e interpretacao probabilistica. |
| `williams1995gaussian` | Coerente para fundamentos de GP em regressao; corresponde ao PDF local. |
| `wang2008economic` | Coerente para formular fundacoes como problema de custo com restricoes. |
| `khajehzadeh2012gsa` | Coerente para metaheuristica aplicada a fundacao rasa com restricoes e custo. |
| `gandomi2018construction` | Coerente para inteligencia de enxame e minimizacao de custo de fundacoes rasas. |
| `kashani2020optimum` | Coerente para algoritmos evolutivos e analises de sensibilidade em fundacoes rasas. |
| `waheed2022practical` | Coerente para ferramenta pratica de otimizacao de sapatas isoladas em concreto armado. |
| `waheed2025optimization` | Coerente para sapatas isoladas/escalonadas, economia, custo, energia incorporada e emissoes. |
| `ahmad2021gpr` | Coerente para GPR em capacidade de carga de fundacoes rasas; nao deve ser tratado como otimizacao de sapatas. |
| `deng2026metamaterial` | Coerente apenas como inspiracao metodologica futura sobre GPR, autoencoder e active learning; nao e referencia de fundacoes. |
| `morales2020balance` | Coerente para equilibrio exploracao/intensificacao em metaheuristicas. |
| `abualigah2021arithmetic` | Coerente como exemplo de metaheuristica moderna; nao e central para fundacoes. |
| `gomes2018probabilistic` | Coerente para comparacao estatistica/probabilistica de metaheuristicas. |
| `khan2023python` | Referencia periferica. Mantida no `.bib` como candidata, mas removida do corpo porque nao sustenta de forma central a escolha cientifica do metodo. Reintroduzir apenas se houver discussao explicita sobre automacao Python no setor AEC. |
| `bezerra2024elementos` | Referencia sob revisao. Mantida no `.bib` como candidata, mas removida do corpo ate validacao de rastreabilidade, qualidade editorial e pertinencia direta para o argumento usado. |
| `juang2013reliability` | Coerente para robustez, confiabilidade e incerteza como frente futura. |
| `abnt6118`, `abnt6122` | Usadas como normas; precisam ser conferidas em versao oficial antes de submissao. |

## Pontos ainda pendentes antes da versao final

- Conferir as versoes oficiais da NBR 6118 e NBR 6122.
- Validar a origem e a forma correta de apresentar as correlacoes `Nspt/30`, `Nspt/40` e `Nspt/50` como estimativas preliminares de tensao admissivel, sem atribui-las diretamente a NBR 6122 se nao forem prescricoes normativas.
- Superado em 2026-07-12: os coeficientes `1,05` e `1,30` foram removidos da formulacao atual; nao devem ser validados nem citados como metodologia corrente.
- Completar a verificacao de puncao ou declarar formalmente o escopo parcial limitado ao perimetro C.
- Fortalecer a justificativa do EGO para a funcao atual barata ou inserir comparacao com metodos mais simples sob mesmo orcamento.
- Inserir datas de submissao/aprovacao somente quando houver revista ou evento definido, se o template exigir.
- Reexecutar os experimentos com seeds registradas.
- Equalizar EGO, Monte Carlo e eventuais baselines pelo mesmo orcamento de avaliacoes.
- Separar resultados plenamente factiveis dos casos usados para diagnosticar sobreposicao severa.
- Preencher metricas definitivas de kernels: `R2`, MAE e RMSE.
- Inserir tabela quantitativa do GPR com kernel, penalidade, split, tamanho de treino/teste, seed, `R2`, MAE e RMSE.
- Apresentar os resultados de volume com media, desvio-padrao, intervalo de confianca, melhor/pior caso e taxa de factibilidade.
- Inserir figuras finais: fluxo EGO, observado vs predito, arranjos em planta e convergencia.
- Definir oficialmente a convencao de kernels do projeto: 20 experimentais + 1 producao, ou 21 kernels.
- Conferir se o kernel destacado no texto corresponde ao kernel efetivamente usado na versao final do software.
- Se o artigo for submetido em revista internacional, aumentar a revisao com bibliografia especifica de packing/layout apenas quando essa frente entrar no manuscrito.
- Remover todas as marcacoes `[[[[ ... ]]]]` antes de qualquer submissao formal, mantendo-as por enquanto como guia visual de pesquisa.

## Avaliacao de extensao

O manuscrito revisado tem aproximadamente 6,8 mil palavras antes das referencias, sem contar figuras e tabelas futuras. Para um artigo completo, a extensao ainda e aceitavel; para uma submissao curta ou para inserir muitos resultados finais, recomenda-se compactar principalmente o estado da arte e a discussao, movendo detalhes experimentais para tabelas/figuras.

## Limitacao desta revisao

Nao foi possivel compilar o LaTeX localmente porque `pdflatex`, `bibtex`, `latexmk`, `tectonic`, `xelatex` e `lualatex` nao estao instalados no ambiente atual. Foi feita uma checagem estrutural automatica de labels/referencias, contagem de ambientes LaTeX e consistencia das chaves BibTeX.
