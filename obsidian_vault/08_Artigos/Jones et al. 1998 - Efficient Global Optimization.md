---
tags: [artigo, ego, expected-improvement, surrogate, artigo1]
autores: Donald R. Jones; Matthias Schonlau; William J. Welch
ano: 1998
journal: Journal of Global Optimization
doi: confirmar
status: fichado-ia
prioridade: essencial
uso: artigo1
arquivo_pdf: docs/articles/01_artigo_1_ego_gpr/1998_jones_schonlau_welch_efficient_global_optimization.pdf
validacao: dupla-varredura-texto-extraido
confianca: alta
---

# Jones et al. (1998) - Efficient Global Optimization of Expensive Black-Box Functions

> **Autores**: Donald R. Jones; Matthias Schonlau; William J. Welch  
> **Ano**: 1998  
> **Journal/Conf**: Journal of Global Optimization  
> **DOI**: confirmar  
> **Fonte conferida**: `/tmp/fundaia_article_text/Jones_and_Schonlau_-_Efficient_Global_Optimization_of_Expensive_Black-Box_Functions.txt`  
> **Confiança da ficha**: alta

## Arquivo local

- PDF: `docs/articles/01_artigo_1_ego_gpr/1998_jones_schonlau_welch_efficient_global_optimization.pdf`
- Caminho absoluto: `/Users/lucasteixeiracorreia/Documents/IC/fundaIA/docs/articles/01_artigo_1_ego_gpr/1998_jones_schonlau_welch_efficient_global_optimization.pdf`

## Resumo

Artigo-base do Efficient Global Optimization (EGO). O problema central tratado e a otimizacao global quando cada avaliacao da funcao objetivo e cara, o que torna inviavel usar metodos que exigem muitas chamadas diretas. A solucao proposta combina modelo substituto do tipo resposta/kriging com uma regra de amostragem que equilibra explorar regioes incertas e explorar regioes promissoras.

Para o FundaIA, e a principal referencia para justificar o loop EGO-GPR: amostra inicial, ajuste do surrogate, maximizacao da Expected Improvement e avaliacao iterativa de novos pontos reais.

## Pontos-chave

- Introduz a logica de usar superficies de resposta para otimizacao global com poucas avaliacoes.
- A Expected Improvement aparece como mecanismo para balancear busca local em regioes boas e exploracao de incerteza.
- Da base teorica direta para a arquitetura EGO descrita no relatorio parcial.

## Conexões com o FundaIA

- Base de [[03_Otimizacao/EGO - Efficient Global Optimization]].
- Base de [[03_Otimizacao/Expected Improvement]].
- Sustenta a metodologia do artigo 1 sobre EGO-GPR no FundaIA.

## Uso recomendado no artigo 1

Citar como referencia metodologica principal do EGO e da Expected Improvement.

## Limitações e cuidados

- Nao e artigo de fundacoes; deve ser usado para justificar o metodo de otimizacao, nao a engenharia da sapata.
- A implementacao do FundaIA usa `sklearn`/GA interno; nao assumir que reproduz todos os detalhes do artigo original.

## Possível uso futuro

Pode sustentar comparacoes com outras funcoes de aquisicao e constrained BO.

## Checagem de coerência

- [x] Ficha criada a partir do texto extraído do PDF enviado.
- [x] Primeira validação: resumo confrontado com abstract/início e conclusão quando disponíveis.
- [x] Segunda validação: nota revisada para evitar extrapolar além do que o artigo sustenta.
- [ ] Conferir metadados finais (DOI, páginas, volume/número) antes de submissão acadêmica.