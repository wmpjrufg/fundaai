---
tags: [auditoria, artigo, submission-polish, sprint-5-7]
data: 2026-07-12
---

# Sprint 5.7 - Submission polish artigo 1 - 2026-07-12

## Objetivo

Preparar o artigo 1 para uma versão madura de pré-submissão, ainda sem template de revista/congresso definido, mantendo o escopo como pré-dimensionamento geométrico experimental de sapatas isoladas e evitando linguagem hermética ou marcas internas de desenvolvimento.

## Mudanças aplicadas

- `docs/artigo_ic_lucas/main.tex`
  - Resumo e abstract foram encurtados e reescritos com foco em: problema, formulação, protocolo, resultados principais, auditoria de decomposição, estudo de penalidade e frente futura de empacotamento.
  - A redução corrigiu o excesso de conteúdo no bloco inicial de duas colunas.
- `docs/artigo_ic_lucas/secoes/01_introducao.tex`
  - Removido o tom de "resultado parcial" e substituído por apresentação do protocolo experimental.
  - `surrogate` foi trocado por "modelo substituto" no corpo em português, mantendo termos internacionais apenas quando ajudam a indexação.
- `docs/artigo_ic_lucas/secoes/02_estado_da_arte.tex`
  - Padronização de "modelos substitutos" e "baixo custo computacional".
  - Mantido o termo `surrogate-assisted optimization` apenas como expressão técnica internacional na revisão.
- `docs/artigo_ic_lucas/secoes/03_fundamentacao_teorica.tex`
  - Substituições editoriais para reduzir dependência de jargão em inglês no corpo do texto.
- `docs/artigo_ic_lucas/secoes/04_metodologia.tex`
  - Troca de "penhasco artificial" por "variação artificial abrupta".
  - Padronização de "parâmetros" no lugar de "alavancas" e "modelo substituto" no lugar de `surrogate`.
- `docs/artigo_ic_lucas/secoes/05_implementacao_software.tex`
  - Reprodutibilidade foi reescrita sem referência ao vault Obsidian no corpo do artigo.
  - A seção agora descreve aplicação, artefatos e documentação de forma mais publicável.
- `docs/artigo_ic_lucas/secoes/06_resultados_parciais.tex`
  - Caption de convergência reescrita sem "mergulha".
  - Causalidade sobre dimensionalidade/CBO foi suavizada: o texto agora atribui o achado às instâncias avaliadas e remete problemas realmente acoplados à frente de posicionamento conjunto.
  - "Avaliação barata" foi substituída por "avaliação de baixo custo".
- `docs/artigo_ic_lucas/secoes/07_discussao.tex`
  - "black-box" foi traduzido para "caixa-preta".
  - "baseline" foi substituído por "linha de base".
  - Corrigido "Quatro limitações" para "Cinco limitações".
- `docs/artigo_ic_lucas/secoes/08_conclusoes_parciais.tex`
  - Padronização de termos e fechamento mais coeso da conclusão.
- `docs/artigo_ic_lucas/secoes/09_agradecimentos.tex`
  - Removidos placeholders.
  - Agradecimentos, conflitos de interesse e disponibilidade de dados/código foram preenchidos em forma genérica, compatível com submissão preliminar.
- `docs/artigo_ic_lucas/README.md`
  - Atualizado status do manuscrito, checklist e notas de submissão.

## Validação

- Comando executado:

```bash
cd docs/artigo_ic_lucas
latexmk -pdf -g -interaction=nonstopmode main.tex
```

- Resultado:
  - Compilação concluída com sucesso.
  - PDF gerado com 22 páginas.
  - Sem `Overfull`.
  - Sem citações ou referências indefinidas.
  - Sem os avisos anteriores de resumo/abstract altos demais na primeira página.
  - Permanecem apenas avisos `Underfull`, esperados em texto de duas colunas com tabelas e floats.

## Decisão editorial

O artigo 1 deve continuar sem resultados da Fase B no corpo principal. A Fase B foi iniciada e documentada, mas ainda deve entrar como trabalho futuro até haver protocolo próprio para problemas acoplados de posicionamento/empacotamento. Inserir resultados exploratórios de packing agora diluiria o artigo 1 e exigiria uma nova rodada de validação estatística.

## Pendências antes de submissão real

- Escolher veículo-alvo e migrar para template específico.
- Validar/substituir a correlação empírica `N_spt`--tensão admissível.
- Formalizar combinações de serviço/ELU se o veículo exigir leitura mais próxima de projeto executivo.
- Decidir se o fluxograma do EGO será incluído como figura adicional ou se o algoritmo já é suficiente.
- Quando houver repositório público, licença ou DOI, substituir a declaração genérica de disponibilidade por link permanente.
