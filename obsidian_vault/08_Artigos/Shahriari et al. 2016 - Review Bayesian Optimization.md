---
tags: [artigo, bayesian-optimization, review, artigo1]
autores: Bobak Shahriari; Kevin Swersky; Ziyu Wang; Ryan P. Adams; Nando de Freitas
ano: 2016
journal: Proceedings of the IEEE
doi: confirmar
status: fichado-ia
prioridade: essencial
uso: artigo1
arquivo_pdf: docs/articles/01_artigo_1_ego_gpr/2016_shahriari_et_al_review_bayesian_optimization.pdf
validacao: dupla-varredura-texto-extraido
confianca: alta
---

# Shahriari et al. (2016) - Taking the Human Out of the Loop: A Review of Bayesian Optimization

> **Autores**: Bobak Shahriari; Kevin Swersky; Ziyu Wang; Ryan P. Adams; Nando de Freitas  
> **Ano**: 2016  
> **Journal/Conf**: Proceedings of the IEEE  
> **DOI**: confirmar  
> **Fonte conferida**: `/tmp/fundaia_article_text/Shahriari_et_al._-_2016_-_Taking_the_Human_Out_of_the_Loop_A_Review_of_Bayesian_Optimization.txt`  
> **Confiança da ficha**: alta

## Arquivo local

- PDF: `docs/articles/01_artigo_1_ego_gpr/2016_shahriari_et_al_review_bayesian_optimization.pdf`
- Caminho absoluto: `/Users/lucasteixeiracorreia/Documents/IC/fundaIA/docs/articles/01_artigo_1_ego_gpr/2016_shahriari_et_al_review_bayesian_optimization.pdf`

## Resumo

Revisao ampla de Bayesian Optimization. Organiza o campo em torno de dois componentes: modelo probabilistico substituto e funcao de aquisicao para escolher novas avaliacoes. Tambem cobre extensoes, aplicacoes e desafios de escalabilidade.

No FundaIA, funciona como referencia de enquadramento: o projeto usa uma instancia especifica de BO/EGO com GPR e EI, aplicada a uma funcao objetivo penalizada de engenharia.

## Pontos-chave

- Boa fonte para definir BO em linguagem academica.
- Ajuda a situar Expected Improvement entre varias funcoes de aquisicao.
- Inclui discussao de constrained Bayesian optimization, util para frentes futuras.

## Conexões com o FundaIA

- Apoia [[11_Frentes_de_Pesquisa/Bayesian Optimization Constrained]].
- Apoia [[10_Melhorias/Acquisition Functions Modernas]].
- Referencia de revisao para a metodologia do artigo 1.

## Uso recomendado no artigo 1

Usar na revisao metodologica para apresentar BO e aquisicoes.

## Limitações e cuidados

- E uma revisao geral, nao um estudo de fundacoes.
- Nao substitui artigos especificos de sapatas na justificativa de engenharia.

## Possível uso futuro

Base para CBO, batch BO e aquisicoes modernas.

## Checagem de coerência

- [x] Ficha criada a partir do texto extraído do PDF enviado.
- [x] Primeira validação: resumo confrontado com abstract/início e conclusão quando disponíveis.
- [x] Segunda validação: nota revisada para evitar extrapolar além do que o artigo sustenta.
- [ ] Conferir metadados finais (DOI, páginas, volume/número) antes de submissão acadêmica.