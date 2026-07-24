---
tags: [artigo, sapatas, metaheuristica, otimizacao, artigo1]
autores: Sinan Melih Nigdeli; Gebrail Bekdas; Xin-She Yang
ano: 2018
journal: KSCE Journal of Civil Engineering
doi: 10.1007/s12205-018-2010-6
status: lido
prioridade: alta
uso: artigo1
arquivo_pdf: docs/articles/01_artigo_1_ego_gpr/2018_nigdeli_bekdas_yang_metaheuristic_optimization_rc_footings.pdf
---

# Nigdeli et al. (2018) - Metaheuristic Optimization of Reinforced Concrete Footings

> **Autores**: Sinan Melih Nigdeli, Gebrail Bekdas, Xin-She Yang  
> **Ano**: 2018  
> **Journal**: KSCE Journal of Civil Engineering, 22(11), 4555-4563  
> **DOI**: 10.1007/s12205-018-2010-6

## Resumo

O artigo formula a otimizacao de sapatas de concreto armado como problema de custo com restricoes geotecnicas e estruturais, comparando DE, PSO, Harmony Search, Flower Pollination Algorithm e TLBO. A formulacao inclui pressoes no solo sob carga axial e momentos biaxiais, recalque elastico, verificacoes de flexao, cisalhamento unidirecional, puncao e detalhamento de armadura conforme ACI 318.

A contribuicao mais util para o FundaIA e dupla: (i) mostra que a literatura de sapatas normalmente trabalha com custo, armadura e verificacoes estruturais completas; (ii) reforca que comparacoes entre metaheuristicas precisam de muitas repeticoes e estatisticas de dispersao, pois algoritmos diferentes alcancam custos proximos com estabilidade e esforco computacional diferentes.

## Pontos-chave

- Otimiza dimensoes, excentricidade/posicao relativa do pilar e armaduras.
- Usa penalizacao para descartar solucoes que falham primeiro nas restricoes geotecnicas.
- Considera tensao admissivel/no-tension, recalque, flexao, cisalhamento unidirecional e puncao.
- Compara algoritmos por melhor custo, custo medio, desvio-padrao e numero de analises.
- Aponta DE e FPA como competitivos nos exemplos, com diferencas entre qualidade, rapidez e robustez.

## Conexoes com o FundaIA

- Reforca a lacuna assumida em [[12_Auditoria/Sprint 5.4 - Correcoes artigo e tensao - 2026-07-12]]: o FundaIA atual ainda nao dimensiona flexao, cisalhamento unidirecional, armadura e custo total.
- Apoia o estado da arte de otimizacao de sapatas isoladas em [[02_Engenharia/Sapatas Isoladas]].
- Justifica incluir Differential Evolution como baseline futuro em [[10_Melhorias/Guia - Validacao antes do Bin Packing]].

## Uso recomendado no artigo

Usar na revisao de literatura para mostrar que estudos de metaheuristica em sapatas ja tratam verificacoes estruturais completas e custo. Nao usar como prova de superioridade universal de nenhum algoritmo: os resultados sao por casos e codigo normativo especificos.

## Limites para o nosso escopo

- Base normativa ACI, nao NBR.
- Objetivo de custo, nao volume penalizado.
- Nao usa surrogate/BO.
- A otimizacao de posicao e local ao pilar/sapata, nao layout/packing entre varias sapatas.
