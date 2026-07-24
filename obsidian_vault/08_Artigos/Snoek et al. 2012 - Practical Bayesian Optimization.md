---
tags: [artigo, bayesian-optimization, gpr, expected-improvement, artigo1]
autores: Jasper Snoek; Hugo Larochelle; Ryan P. Adams
ano: 2012
journal: NeurIPS / Advances in Neural Information Processing Systems
doi: confirmar
status: fichado-ia
prioridade: essencial
uso: artigo1
arquivo_pdf: docs/articles/01_artigo_1_ego_gpr/2012_snoek_larochelle_adams_practical_bayesian_optimization.pdf
validacao: dupla-varredura-texto-extraido
confianca: alta
---

# Snoek et al. (2012) - Practical Bayesian Optimization of Machine Learning Algorithms

> **Autores**: Jasper Snoek; Hugo Larochelle; Ryan P. Adams  
> **Ano**: 2012  
> **Journal/Conf**: NeurIPS / Advances in Neural Information Processing Systems  
> **DOI**: confirmar  
> **Fonte conferida**: `/tmp/fundaia_article_text/Snoek_et_al._-_2012_-_Practical_Bayesian_Optimization_of_Machine_Learning_Algorithms.txt`  
> **Confiança da ficha**: alta

## Arquivo local

- PDF: `docs/articles/01_artigo_1_ego_gpr/2012_snoek_larochelle_adams_practical_bayesian_optimization.pdf`
- Caminho absoluto: `/Users/lucasteixeiracorreia/Documents/IC/fundaIA/docs/articles/01_artigo_1_ego_gpr/2012_snoek_larochelle_adams_practical_bayesian_optimization.pdf`

## Resumo

Apresenta Bayesian Optimization como alternativa automatizada para ajuste de hiperparametros, usando Gaussian Processes e funcoes de aquisicao. O foco aplicado e machine learning, mas a estrutura metodologica e relevante para qualquer problema de caixa-preta com custo de avaliacao.

O artigo tambem discute tratamento bayesiano de hiperparametros do GP, custo variavel de experimentos e execucao paralela, pontos que ajudam a contextualizar boas praticas alem da implementacao atual do FundaIA.

## Pontos-chave

- Defende BO para substituir busca manual ou busca bruta por uma estrategia amostral eficiente.
- Discute Expected Improvement em contexto pratico.
- Mostra que escolhas do GP e de hiperparametros afetam fortemente o desempenho.

## Conexões com o FundaIA

- Apoia [[03_Otimizacao/Gaussian Process Regressor]].
- Apoia [[03_Otimizacao/Expected Improvement]].
- Ajuda a justificar escolhas e limites do surrogate no artigo 1.

## Uso recomendado no artigo 1

Usar para explicar por que BO/EGO e adequado a problemas em que se quer reduzir avaliacoes reais.

## Limitações e cuidados

- Nao trata fundacoes nem restricoes estruturais.
- Nao usar resultados numericos de ML como evidencia direta para sapatas.

## Possível uso futuro

Pode embasar custo de avaliacao, paralelizacao e melhoria de aquisicao.

## Checagem de coerência

- [x] Ficha criada a partir do texto extraído do PDF enviado.
- [x] Primeira validação: resumo confrontado com abstract/início e conclusão quando disponíveis.
- [x] Segunda validação: nota revisada para evitar extrapolar além do que o artigo sustenta.
- [ ] Conferir metadados finais (DOI, páginas, volume/número) antes de submissão acadêmica.