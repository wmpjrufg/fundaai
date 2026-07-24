---
tags: [artigo, geotecnia, recalque, spt, machine-learning, futuro]
autores: Hadi Fattahi; Hossein Ghaedi; Danial Jahed Armaghani
ano: 2025
journal: Computer Modeling in Engineering & Sciences
doi: 10.32604/cmes.2025.062390
status: lido
prioridade: apoio
uso: futuro-recalques
arquivo_pdf: docs/articles/02_apoio_tecnico_geotecnia/2025_fattahi_ghaedi_armaghani_settlement_prediction_intelligent_optimization.pdf
---

# Fattahi et al. (2025) - Improving Shallow Foundation Settlement Prediction through Intelligent Optimization Techniques

> **Autores**: Hadi Fattahi, Hossein Ghaedi, Danial Jahed Armaghani  
> **Ano**: 2025  
> **Journal**: Computer Modeling in Engineering & Sciences, 143(1), 747-766  
> **DOI**: 10.32604/cmes.2025.062390

## Resumo

O artigo usa Harmony Search e TLBO para prever recalque de fundacoes superficiais a partir de cinco variaveis: largura da sapata, pressao aplicada, numero de golpes SPT, relacao de embutimento `Df/B` e geometria `L/B`. A base tem 189 pontos, divididos em treino e teste. Os autores reportam `R2` acima de 0,94 e usam analise de sensibilidade para indicar que o SPT foi a variavel mais influente no conjunto estudado.

Para o FundaIA, a referencia nao valida diretamente a correlacao `Nspt/30/40/50`, mas reforca que o SPT tem papel preditivo relevante em modelos geotecnicos e que recalque deve entrar como frente futura quando a ferramenta sair do pre-dimensionamento geometrico.

## Pontos-chave

- Variaveis de entrada incluem `B`, `q`, `N`, `Df/B` e `L/B`.
- O foco e predicao de recalque, nao dimensionamento estrutural de sapatas.
- TLBO e HS foram tratados como tecnicas inteligentes de ajuste/predicao.
- Os autores recomendam validacao local/site-specific antes de uso pratico.

## Conexoes com o FundaIA

- Apoia [[02_Engenharia/SPT - Sondagem]] como variavel geotecnica relevante, mas nao como regra direta para tensao admissivel.
- Fortalece a pendencia de incorporar recalques em [[10_Melhorias/Guia - Validacao antes do Bin Packing]].
- Pode ser citado apenas como evidencia de que modelos de ML/geotecnia dependem de validacao local.

## Uso recomendado no artigo

Usar com cautela na discussao/trabalhos futuros, ao falar de recalques e incerteza geotecnica. Evitar usar como suporte para `sigma_adm = Nspt/k`, pois o artigo trata recalque e nao tensao admissivel.

## Limites para o nosso escopo

- Nao otimiza sapata de concreto armado.
- Nao verifica punção, flexao ou cisalhamento.
- Nao substitui norma geotecnica nem relatorio de sondagem.
