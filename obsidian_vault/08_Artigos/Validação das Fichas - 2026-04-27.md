---
tags: [artigo, validacao, auditoria, fichamento]
data: 2026-04-27
escopo: fichas-artigos
---

# Validação das Fichas - 2026-04-27

## Resultado

Foram criadas **23 fichas** em [[08_Artigos/Index de Artigos]], cobrindo os PDFs/textos extraídos da biblioteca enviada.

Os PDFs correspondentes foram reorganizados em `docs/articles`, com manifesto em `docs/articles/README.md`. Cada ficha recebeu o campo `arquivo_pdf` e uma seção `Arquivo local` apontando para o PDF canônico usado como referência.

As fichas foram organizadas em três grupos:

- **Essenciais para o artigo 1**: EGO, GPR, Bayesian Optimization, otimização de sapatas e comparação experimental.
- **Apoio técnico / contexto**: geotecnia, fundações, Python/AEC e metaheurísticas secundárias.
- **Próximas frentes / uso futuro**: bin packing/layout, confiabilidade, recalque, active learning e tópicos periféricos.

## Método de validação

### Passada 1 - Conferência artigo a artigo

Cada ficha foi criada a partir do texto extraído do PDF enviado e confrontada com:

- título/início do artigo;
- abstract/resumo quando disponível;
- conclusão/considerações finais quando disponível;
- termos técnicos principais do artigo.

### Passada 2 - Coerência e segurança acadêmica

Depois da criação das fichas, foi feita uma segunda varredura para:

- verificar se cada nota aponta para uma fonte extraída existente;
- verificar se cada ficha possui caminho local para PDF existente em `docs/articles`;
- checar presença dos termos centrais no texto extraído;
- procurar frases fortes demais ou extrapolações indevidas;
- marcar explicitamente limitações de uso;
- separar o que serve para o **artigo 1** do que deve ficar para **frentes futuras**.

## Pontos de atenção

- A ficha [[The application of Bayesian methods - OCR pendente]] tem **confiança baixa**: o texto extraído não trouxe conteúdo útil, apenas marcadores de páginas. Precisa OCR ou leitura manual antes de uso acadêmico.
- A ficha [[NBR 6122 1996 - Projeto e Execucao de Fundacoes]] usa uma versão antiga da norma. Serve para orientação conceitual no vault, mas a versão vigente deve ser conferida antes de citar em artigo.
- A ficha [[Rasheed et al. 2017 - Optimization Shallow Foundation GSA]] tem metadado temporal ambíguo no arquivo extraído. Confirmar ano e referência bibliográfica final antes de citar.
- Vários DOIs ficaram como `confirmar` porque eu não assumi metadados que não estavam claros no texto extraído. Isso é intencional para evitar erro bibliográfico.

## Validação por grupo

### Essenciais para o artigo 1

- [[Jones et al. 1998 - Efficient Global Optimization]] - coerente com EGO, response surface e Expected Improvement.
- [[Snoek et al. 2012 - Practical Bayesian Optimization]] - coerente com Bayesian Optimization, GP e Expected Improvement.
- [[Shahriari et al. 2016 - Review Bayesian Optimization]] - coerente com revisão de BO, surrogate e aquisição.
- [[Schulz et al. 2018 - Tutorial Gaussian Process Regression]] - coerente com GPR, kernels, exploração e explotação.
- [[Williams e Rasmussen - Gaussian Processes for Regression]] - coerente com GP para regressão; metadados ainda precisam conferência.
- [[Wang e Kulhawy 2008 - Economic Design Optimization of Foundations]] - coerente com otimização econômica de fundações, ULS/SLS e custo.
- [[Gandomi e Kashani 2018 - Cost Minimization Shallow Foundation]] - coerente com swarm intelligence em fundações rasas e comparação de algoritmos.
- [[Kashani et al. 2020 - Optimum Design of Shallow Foundation]] - coerente com algoritmos evolutivos e sensibilidade em fundações rasas.
- [[Waheed et al. 2022 - Practical Tool RC Isolated Footings]] - coerente com ferramenta prática, sapatas isoladas, GA/EA e economia reportada.
- [[Waheed et al. 2025 - Economical Design RC Isolated Footings]] - coerente com sapatas escalonadas, GA e comparações econômicas.
- [[Gomes et al. 2018 - Probabilistic Metric Metaheuristics]] - coerente com comparação probabilística de metaheurísticas.

### Apoio técnico / contexto

- [[Ahmad et al. 2021 - GPR Bearing Capacity Shallow Foundations]] - coerente com GPR aplicado a capacidade de carga.
- [[Bezerra et al. 2024 - Elementos de Fundacao]] - coerente com escolha preliminar de elementos de fundação.
- [[Khan et al. 2023 - Python Automation AEC]] - coerente com automação Python no setor AEC.
- [[Rasheed et al. 2017 - Optimization Shallow Foundation GSA]] - coerente com GSA, fundações rasas e comparação com GA; metadado precisa conferência.
- [[Morales-Castaneda et al. 2020 - Balance in Metaheuristics]] - coerente com exploração, explotação e diversidade.
- [[Abualigah et al. 2021 - Arithmetic Optimization Algorithm]] - coerente com AOA e metaheurísticas modernas.

### Próximas frentes / uso futuro

- [[Juang e Wang 2013 - Reliability Robust Spread Foundations]] - coerente com confiabilidade, robustez, Pareto e NSGA-II.
- [[G09-002 - Geotechnical Engineering Shallow Foundations]] - coerente com fundações rasas, recalque, sapatas combinadas e distribuição de tensões.
- [[Mbock et al. 2019 - Optimal Forms Shallow Foundations]] - coerente com forma ótima e otimização estrutural evolutiva.
- [[Deng et al. 2026 - Metamaterial Autoencoder Active Learning]] - coerente com autoencoder, GPR e active learning; uso apenas futuro.
- [[The application of Bayesian methods - OCR pendente]] - registrada como pendência, não como ficha validada de conteúdo.

## Veredito

As fichas estão coerentes com os textos extraídos dos artigos enviados dentro do limite da extração disponível. Elas devem ser usadas como **biblioteca comentada do projeto**, não como substituto da leitura/citação final na escrita acadêmica.

Antes de submissão de artigo, ainda é necessário:

- conferir DOI, páginas, volume e número nos PDFs ou Zotero;
- checar versão vigente das normas brasileiras;
- abrir manualmente qualquer artigo marcado com confiança média/baixa;
- confirmar citações diretas apenas no PDF original, com página.

## Vínculos

- [[08_Artigos/Index de Artigos]]
- [[10_Melhorias/Guia - Validação antes do Bin Packing]]
- [[09_Relatorios/Analise - Roadmap Artigo IC - 2026-04-27]]
