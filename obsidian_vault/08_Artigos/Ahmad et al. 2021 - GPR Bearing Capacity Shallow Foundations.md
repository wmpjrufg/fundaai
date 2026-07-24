---
tags: [artigo, gpr, fundacoes, capacidade-carga, geotecnia, artigo1]
autores: Mahmood Ahmad et al.
ano: 2021
journal: Applied Sciences
doi: confirmar
status: fichado-ia
prioridade: alta
uso: artigo1
arquivo_pdf: docs/articles/02_apoio_tecnico_geotecnia/2021_ahmad_et_al_gpr_bearing_capacity_shallow_foundations.pdf
validacao: dupla-varredura-texto-extraido
confianca: alta
---

# Ahmad et al. (2021) - Prediction of Ultimate Bearing Capacity of Shallow Foundations on Cohesionless Soils: A Gaussian Process Regression Approach

> **Autores**: Mahmood Ahmad et al.  
> **Ano**: 2021  
> **Journal/Conf**: Applied Sciences  
> **DOI**: confirmar  
> **Fonte conferida**: `/tmp/fundaia_article_text/Ahmad_et_al._-_2021_-_Prediction_of_Ultimate_Bearing_Capacity_of_Shallow_Foundations_on_Cohesionless_Soils_A_Gaussian_Pr.txt`  
> **Confiança da ficha**: alta

## Arquivo local

- PDF: `docs/articles/02_apoio_tecnico_geotecnia/2021_ahmad_et_al_gpr_bearing_capacity_shallow_foundations.pdf`
- Caminho absoluto: `/Users/lucasteixeiracorreia/Documents/IC/fundaIA/docs/articles/02_apoio_tecnico_geotecnia/2021_ahmad_et_al_gpr_bearing_capacity_shallow_foundations.pdf`

## Resumo

Aplica GPR para prever capacidade de carga ultima de fundacoes rasas em solos nao coesivos. Usa como entradas largura, profundidade, geometria da fundacao, peso especifico da areia e angulo de atrito, comparando o GPR com abordagens teoricas da literatura.

E uma ponte importante entre GPR e geotecnia: embora o FundaIA use GPR como surrogate da funcao objetivo, este artigo mostra GPR aplicado diretamente a um problema de fundacoes rasas.

## Pontos-chave

- GPR e usado para problema geotecnico de fundacoes rasas.
- O artigo conclui que o GPR teve desempenho melhor que abordagens teoricas consideradas para o dataset.
- A analise de sensibilidade destaca parametros influentes no dataset estudado.

## Conexões com o FundaIA

- Apoia [[03_Otimizacao/Gaussian Process Regressor]].
- Apoia [[02_Engenharia/Tensão Admissível do Solo]].
- Ajuda a justificar GPR em contexto de fundacoes.

## Uso recomendado no artigo 1

Usar para mostrar que GPR ja e aceito em aplicacoes geotecnicas de fundacoes.

## Limitações e cuidados

- Prediz capacidade de carga, nao otimiza sapatas.
- Dataset e formulacao diferem do FundaIA; nao usar como validacao direta do modelo atual.

## Possível uso futuro

Pode inspirar surrogate geotecnico mais fisico, separado da FO penalizada.

## Checagem de coerência

- [x] Ficha criada a partir do texto extraído do PDF enviado.
- [x] Primeira validação: resumo confrontado com abstract/início e conclusão quando disponíveis.
- [x] Segunda validação: nota revisada para evitar extrapolar além do que o artigo sustenta.
- [ ] Conferir metadados finais (DOI, páginas, volume/número) antes de submissão acadêmica.