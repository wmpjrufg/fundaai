---
tags: [artigo, gpr, surrogate, kernel, artigo1]
autores: Eric Schulz; Maarten Speekenbrink; Andreas Krause
ano: 2018
journal: Journal of Mathematical Psychology
doi: confirmar
status: fichado-ia
prioridade: essencial
uso: artigo1
arquivo_pdf: docs/articles/01_artigo_1_ego_gpr/2018_schulz_speekenbrink_krause_tutorial_gpr.pdf
validacao: dupla-varredura-texto-extraido
confianca: alta
---

# Schulz et al. (2018) - A tutorial on Gaussian process regression

> **Autores**: Eric Schulz; Maarten Speekenbrink; Andreas Krause  
> **Ano**: 2018  
> **Journal/Conf**: Journal of Mathematical Psychology  
> **DOI**: confirmar  
> **Fonte conferida**: `/tmp/fundaia_article_text/Schulz_et_al._-_A_tutorial_on_Gaussian_process_regression_Modelling_exploring_and_exploiting_functions.txt`  
> **Confiança da ficha**: alta

## Arquivo local

- PDF: `docs/articles/01_artigo_1_ego_gpr/2018_schulz_speekenbrink_krause_tutorial_gpr.pdf`
- Caminho absoluto: `/Users/lucasteixeiracorreia/Documents/IC/fundaIA/docs/articles/01_artigo_1_ego_gpr/2018_schulz_speekenbrink_krause_tutorial_gpr.pdf`

## Resumo

Tutorial de GPR como abordagem bayesiana nao parametrica para modelar funcoes desconhecidas. Explica como o GP fornece predicao e incerteza, como kernels codificam suposicoes sobre a funcao e como GPR pode apoiar exploracao e explotacao.

E util para explicar por que o FundaIA usa GPR como surrogate: alem de prever o valor da funcao penalizada, o modelo fornece incerteza, necessaria para a Expected Improvement.

## Pontos-chave

- Introduz GP/GPR de forma didatica.
- Mostra a importancia do kernel como suposicao previa sobre suavidade/estrutura da funcao.
- Conecta GPR a cenarios de exploracao e explotacao.

## Conexões com o FundaIA

- Apoia [[03_Otimizacao/Gaussian Process Regressor]].
- Apoia [[03_Otimizacao/Kernels GPR]].
- Justifica o estudo de kernels do FundaIA.

## Uso recomendado no artigo 1

Usar para fundamentar GPR, kernel e incerteza preditiva.

## Limitações e cuidados

- Tutorial geral; nao traz validacao em fundacoes.
- Nao usar como evidencia de desempenho do FundaIA, mas como base conceitual.

## Possível uso futuro

Base para PI-GPR e GPyTorch.

## Checagem de coerência

- [x] Ficha criada a partir do texto extraído do PDF enviado.
- [x] Primeira validação: resumo confrontado com abstract/início e conclusão quando disponíveis.
- [x] Segunda validação: nota revisada para evitar extrapolar além do que o artigo sustenta.
- [ ] Conferir metadados finais (DOI, páginas, volume/número) antes de submissão acadêmica.