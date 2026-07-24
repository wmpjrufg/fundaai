---
tags: [artigo, bayesian-optimization, restricoes, multiobjetivo, estruturas, artigo1]
autores: Alexandre Mathern et al.
ano: 2021
journal: Structural and Multidisciplinary Optimization
doi: 10.1007/s00158-020-02720-2
status: lido
prioridade: alta
uso: artigo1-metodologia
arquivo_pdf: docs/articles/05_frente_c_cbo/2021_mathern_et_al_multiobjective_constrained_bo_structural_design.pdf
---

# Mathern et al. (2021) - Multi-objective Constrained Bayesian Optimization for Structural Design

> **Autores**: Alexandre Mathern, Olof Skogby Steinholtz, Anders Sjoberg, Magnus Onnheim, Kristine Ek, Rasmus Rempling, Emil Gustavsson, Mats Jirstrand  
> **Ano**: 2021  
> **Journal**: Structural and Multidisciplinary Optimization, 63(2), 689-701  
> **DOI**: 10.1007/s00158-020-02720-2

## Resumo

O artigo aplica Bayesian optimization com restricoes a projeto estrutural de concreto armado em contexto multiobjetivo. A ideia metodologica central e explorar uma assimetria comum em engenharia: objetivos como custo, impacto ambiental, tempo/construtibilidade e desempenho podem ser baratos de avaliar, enquanto restricoes normativas exigem calculos estruturais mais caros. O benchmark e uma viga de concreto armado com oito variaveis de projeto, cinco objetivos e restricoes de flexao/cisalhamento segundo codigo.

Os autores mostram que o algoritmo bayesiano encontrou conjuntos de Pareto de alta qualidade com poucas avaliacoes de restricoes, superando NSGA-II e busca aleatoria no problema testado. Para o FundaIA, esta e uma referencia metodologica forte: ela aproxima BO/CBO de projeto estrutural normativo e ajuda a justificar o uso de BO quando a continuidade da pesquisa incluir restricoes mais caras.

## Pontos-chave

- BO em engenharia estrutural com restricoes de codigo.
- Explora objetivos baratos e restricoes caras, em vez de assumir que tudo e caro.
- Compara contra NSGA-II e busca aleatoria.
- Conecta otimizacao com sustentabilidade, construtibilidade e desempenho.

## Conexoes com o FundaIA

- Fortalece a argumentacao de [[03_Otimizacao/EGO - Efficient Global Optimization]] quando o artigo admite que a FO atual e barata.
- Dialoga com a CBO ja implementada em [[12_Auditoria/Sprint 5.3 - Frente C CBO - 2026-07-11]].
- Ajuda a defender trabalhos futuros com custo total, flexao/cisalhamento e multiobjetivo.

## Uso recomendado no artigo

Usar na fundamentacao/discussao para mostrar que BO/CBO tem precedentes fortes em projeto estrutural com restricoes normativas, especialmente quando as restricoes se tornam caras. Nao apresentar como referencia direta de sapatas, pois o caso e viga de concreto armado.

## Limites para o nosso escopo

- Problema multiobjetivo de viga, nao sapata.
- Nao usa GPR para a mesma formulacao do FundaIA.
- Serve para metodologia e posicionamento futuro, nao para validar os resultados de volume do artigo.
