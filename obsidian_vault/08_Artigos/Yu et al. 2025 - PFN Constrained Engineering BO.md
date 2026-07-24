---
tags: [artigo, bayesian-optimization, restricoes, surrogate, transformers, futuro]
autores: Rosen Ting-Ying Yu; Cyril Picard; Faez Ahmed
ano: 2025
journal: Structural and Multidisciplinary Optimization
doi: 10.1007/s00158-025-03987-z
status: lido
prioridade: apoio
uso: futuro-cbo-escalavel
arquivo_pdf: docs/articles/05_frente_c_cbo/2025_yu_picard_ahmed_pfn_constrained_engineering_bo.pdf
---

# Yu et al. (2025) - Fast and Accurate Bayesian Optimization with Pre-trained Transformers for Constrained Engineering Problems

> **Autores**: Rosen Ting-Ying Yu, Cyril Picard, Faez Ahmed  
> **Ano**: 2025  
> **Journal**: Structural and Multidisciplinary Optimization, 68, artigo 66  
> **DOI**: 10.1007/s00158-025-03987-z

## Resumo

O artigo propõe substituir GPs tradicionais por Prior-data Fitted Networks (PFNs), um modelo transformer pre-treinado, em Bayesian optimization com restricoes. A motivacao e reduzir o custo de ajustar um GP para o objetivo e um GP por restricao a cada iteracao. O benchmark cobre problemas sinteticos e de engenharia, comparando penalizacao e CEI com surrogates GP e PFN.

Os resultados relatados indicam ganho de ordem de grandeza em tempo para PFN em relacao a GP nos cenarios testados, mantendo ou melhorando a qualidade de solucao, especialmente em problemas de engenharia com varias restricoes. A propria discussao reconhece limites importantes: dimensao atual limitada, necessidade de validacao em mais aplicacoes reais e diferencas quando o problema sai da distribuicao de treino.

## Pontos-chave

- CBO tradicional com GP escala mal quando ha muitas restricoes.
- CEI com GP exige surrogate separado para objetivo e restricoes.
- PFN permite avaliar objetivo e restricoes em uma passagem.
- A referencia e recente e de fronteira; ainda nao e pratica consolidada.

## Conexoes com o FundaIA

- Apoia a linha futura de CBO escalavel alem de [[08_Artigos/Eriksson e Poloczek 2021 - Scalable Constrained BO]].
- Reforca que, para packing + sizing com muitas restricoes, custo de surrogate pode virar gargalo.
- Nao deve substituir a fundamentacao atual baseada em GP/EGO.

## Uso recomendado no artigo

Usar no maximo em trabalhos futuros, como exemplo recente de caminhos para reduzir o custo de CBO em problemas de engenharia com muitas restricoes. Evitar citar na metodologia atual, pois o FundaIA usa GPR classico.

## Limites para o nosso escopo

- Metodo nao implementado no FundaIA.
- Validacao ainda e por benchmark, nao por fundacoes.
- Pode parecer desvio de foco se entrar cedo demais no estado da arte.
