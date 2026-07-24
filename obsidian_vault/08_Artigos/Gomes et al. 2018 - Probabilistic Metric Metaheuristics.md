---
tags: [artigo, metaheuristica, validacao, estatistica, artigo1]
autores: Wellison J. S. Gomes; Andre T. Beck; Rafael H. Lopez; Leandro F. F. Miguel
ano: 2018
journal: Structural Safety
doi: confirmar
status: fichado-ia
prioridade: essencial
uso: artigo1
arquivo_pdf: docs/articles/01_artigo_1_ego_gpr/2018_gomes_et_al_probabilistic_metric_metaheuristics.pdf
validacao: dupla-varredura-texto-extraido
confianca: alta
---

# Gomes et al. (2018) - A probabilistic metric for comparing metaheuristic optimization algorithms

> **Autores**: Wellison J. S. Gomes; Andre T. Beck; Rafael H. Lopez; Leandro F. F. Miguel  
> **Ano**: 2018  
> **Journal/Conf**: Structural Safety  
> **DOI**: confirmar  
> **Fonte conferida**: `/tmp/fundaia_article_text/Gomes_et_al._-_2018_-_A_probabilistic_metric_for_comparing_metaheuristic_optimization_algorithms.txt`  
> **Confiança da ficha**: alta

## Arquivo local

- PDF: `docs/articles/01_artigo_1_ego_gpr/2018_gomes_et_al_probabilistic_metric_metaheuristics.pdf`
- Caminho absoluto: `/Users/lucasteixeiracorreia/Documents/IC/fundaIA/docs/articles/01_artigo_1_ego_gpr/2018_gomes_et_al_probabilistic_metric_metaheuristics.pdf`

## Resumo

Propõe uma metrica probabilistica para comparar algoritmos metaheuristicos considerando que execucoes estocasticas geram resultados diferentes. Em vez de depender apenas de media, desvio, melhor e pior, a metrica estima a probabilidade de um algoritmo produzir resultado melhor que outro em uma execucao.

E muito importante para desenhar a validacao experimental do FundaIA, especialmente se houver comparacao entre EGO-GPR, Monte Carlo e GA puro.

## Pontos-chave

- Reforca que uma unica execucao nao basta para comparar metaheuristicas.
- Mostra necessidade de multiplas runs/seeds.
- Ajuda a defender comparacao probabilistica ou pelo menos estatisticamente consciente.

## Conexões com o FundaIA

- Apoia [[10_Melhorias/Reprodutibilidade - Seeds e Versão]].
- Apoia [[10_Melhorias/Persistência de Experimentos]].
- Apoia o Gate 3 da [[10_Melhorias/Guia - Validação antes do Bin Packing]].

## Uso recomendado no artigo 1

Usar para justificar multiplas seeds, estatisticas e comparacao justa.

## Limitações e cuidados

- Nao e artigo de fundacoes.
- A metrica proposta pode ser mais sofisticada do que o necessario para a primeira versao do artigo; ainda assim, seus principios sao essenciais.

## Possível uso futuro

Implementar a metrica probabilistica na analise final se houver tempo.

## Checagem de coerência

- [x] Ficha criada a partir do texto extraído do PDF enviado.
- [x] Primeira validação: resumo confrontado com abstract/início e conclusão quando disponíveis.
- [x] Segunda validação: nota revisada para evitar extrapolar além do que o artigo sustenta.
- [ ] Conferir metadados finais (DOI, páginas, volume/número) antes de submissão acadêmica.