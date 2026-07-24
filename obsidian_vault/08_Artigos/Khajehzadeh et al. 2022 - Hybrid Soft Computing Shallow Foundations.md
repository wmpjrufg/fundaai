---
tags: [artigo, sapatas, capacidade-de-carga, ann, metaheuristica, apoio]
autores: Mohammad Khajehzadeh; Suraparb Keawsawasvong; Moncef L. Nehdi
ano: 2022
journal: Sustainability
doi: 10.3390/su14031847
status: lido
prioridade: apoio
uso: artigo1-futuro-geotecnia
arquivo_pdf: docs/articles/02_apoio_tecnico_geotecnia/2022_khajehzadeh_keawsawasvong_nehdi_hybrid_soft_computing_shallow_foundations.pdf
---

# Khajehzadeh et al. (2022) - Effective Hybrid Soft Computing Approach for Optimum Design of Shallow Foundations

> **Autores**: Mohammad Khajehzadeh, Suraparb Keawsawasvong, Moncef L. Nehdi  
> **Ano**: 2022  
> **Journal**: Sustainability, 14(3), 1847  
> **DOI**: 10.3390/su14031847

## Resumo

O artigo combina ANN e Modified Rat Swarm Optimizer para estimar capacidade de carga ultima de fundacoes superficiais e aplicar essa estimativa em um problema de otimizacao de sapata. A base de dados foi montada com 97 ensaios de carga em sapatas de escala real e modelos reduzidos. O modelo ANN reportado tem arquitetura `5 x 10 x 1`, RMSE de 0,0249 e correlacao de 0,9908 no problema tratado.

A contribuicao relevante para o FundaIA e mostrar uma linha alternativa ao uso de correlacoes simples: predicao de capacidade de carga por modelo treinado em dados experimentais. Isso reforca a decisao de manter `Nspt/30/40/50` como hipotese empirica de pre-dimensionamento, nao como prescricao normativa.

## Pontos-chave

- Objetivo de otimizacao e custo de sapata, nao apenas volume.
- Design variables incluem dimensoes da fundacao, embutimento e armaduras.
- A capacidade de carga e prevista por ANN treinada com dados experimentais.
- O artigo enfatiza que capacidade de carga e recalque sao requisitos essenciais de fundacoes.

## Conexoes com o FundaIA

- Apoia [[02_Engenharia/Tensao Admissivel do Solo]] como ponto que precisa evoluir alem de correlacao simples.
- Ajuda a justificar trabalhos futuros com incerteza geotecnica e modelos preditivos.
- Dialoga com [[08_Artigos/Ahmad et al. 2021 - GPR Bearing Capacity Shallow Foundations]].

## Uso recomendado no artigo

Pode ser citado na discussao de limitacoes geotecnicas e na frente futura de substituicao/validacao da estimativa de capacidade do solo. Nao deve ser usado para sustentar diretamente o metodo atual do FundaIA.

## Limites para o nosso escopo

- Nao usa NBR.
- Mistura dados de escala real e reduzida; exige cuidado com efeito de escala.
- O otimizador MRSO nao e baseline atual do artigo.
