---
tags: [projeto, contexto-academico, ic, tcc, artigo, fundaia, ego, gpr]
data: 2026-04-27
status: ativo
---

# Contexto Acadêmico - IC Lucas e TCC Filipe Amaral

## Função desta nota

Esta nota integra três documentos-base do projeto:

- plano de trabalho da IC do Lucas;
- relatório parcial da IC;
- TCC/artigo em construção de Filipe Amaral Pereira, baseado no FundaIA.

Ela deve servir como contexto rápido para escrita acadêmica, revisão do manuscrito, alinhamento com o software e decisões de roadmap. O ponto mais importante: **a etapa atual deve fechar e validar o FundaIA + EGO-GPR antes de avançar para bin packing completo**.

## Arquivos locais

| Documento | Caminho relativo | Papel |
| --- | --- | --- |
| Plano de trabalho da IC | `docs/contexto_academico/ic_lucas/plano_de_trabalho_ic_lucas.pdf` | Escopo formal aprovado: otimização do dimensionamento e posicionamento de fundações superficiais, com empacotamento e mecânica das estruturas. |
| Relatório parcial da IC | `docs/contexto_academico/ic_lucas/relatorio_parcial_ic_lucas.pdf` | Registro da evolução metodológica para EGO-GPR, testes de kernels, penalização e resultados preliminares. |
| TCC/artigo Filipe Amaral | `docs/contexto_academico/tcc_filipe_amaral/filipe_amaral_tcc_fundaia_sapatas_ego_gpr_em_construcao.pdf` | Manuscrito em construção baseado no FundaIA, com formulação, metodologia, interface e resultados. |

## Caminhos absolutos

- `/Users/lucasteixeiracorreia/Documents/IC/fundaIA/docs/contexto_academico/ic_lucas/plano_de_trabalho_ic_lucas.pdf`
- `/Users/lucasteixeiracorreia/Documents/IC/fundaIA/docs/contexto_academico/ic_lucas/relatorio_parcial_ic_lucas.pdf`
- `/Users/lucasteixeiracorreia/Documents/IC/fundaIA/docs/contexto_academico/tcc_filipe_amaral/filipe_amaral_tcc_fundaia_sapatas_ego_gpr_em_construcao.pdf`

## Leitura realizada

- O TCC/artigo do Filipe foi lido a partir de extração textual de 37 páginas.
- O plano de trabalho foi lido a partir de extração textual de 5 páginas.
- O relatório parcial foi lido a partir de extração textual de 3 páginas.

## Encaixe entre os documentos

| Eixo | Plano de trabalho | Relatório parcial | TCC/artigo Filipe | Estado no FundaIA |
| --- | --- | --- | --- | --- |
| Objetivo geral | Dimensionamento e posicionamento otimizado de fundações superficiais | Abordagem computacional para dimensionamento e posicionamento, com Python/Streamlit | Dimensionamento automatizado e otimizado de sapatas isoladas | Implementado para dimensões `(hx, hy, hz)` com posições `xg, yg` fornecidas |
| Método | Metaheurísticas, empacotamento e mecânica das estruturas | EGO-GPR com GA interno e Expected Improvement | EGO, GPR/Kriging, EI e AG interno | Coerente com [[03_Otimizacao/EGO - Efficient Global Optimization]] e [[03_Otimizacao/Gaussian Process Regressor]] |
| Engenharia | Critérios geométricos, geotécnicos e estruturais | Restrições de geometria, tensão admissível, punção e sobreposição | NBR 6118, NBR 6122, tensão no solo, geometria, sobreposição e punção C | Coerente com [[02_Engenharia/Sapatas Isoladas]], [[02_Engenharia/Tensão Admissível do Solo]] e [[02_Engenharia/Verificação à Punção]] |
| Empacotamento/layout | Escopo declarado no plano | Ainda tratado como frente futura/limite | Verificação de sobreposição em planta; sapatas associadas ficam para futuro | Ainda não é bin packing completo; ver [[10_Melhorias/Guia - Validação antes do Bin Packing]] |
| Artigo | Previsto como saída | Já citado como manuscrito em coautoria | Manuscrito em construção | Base forte para artigo 1, desde que resultados sejam revalidados |

## Síntese do TCC/artigo do Filipe

Título extraído: **Modelagem paramétrica para projeto inteligente e automatizado de fundações superficiais do tipo sapata: Pré dimensionamento, projeto automatizado e confiabilidade**.

Autores extraídos: Filipe Amaral Pereira; Wanderlei M. Pereira Junior; Lucas Teixeira Correia.

O manuscrito apresenta o FundaIA como plataforma computacional para dimensionamento automatizado de sapatas isoladas. A função objetivo minimiza o volume de concreto e usa penalização para restrições de tensão no solo, geometria mínima, sobreposição entre sapatas e punção. A otimização é descrita como EGO, com GPR/Kriging como modelo substituto, Expected Improvement como aquisição e algoritmo genético como otimizador interno.

Os resultados descritos incluem:

- estudo de penalização no GPR, comparando `alpha = 1e1` e `alpha = 1e6`;
- estudo de kernels, com destaque para configuração `K06`;
- exemplos com uma fundação, três fundações e duas fundações próximas;
- comparação com Monte Carlo/tentativa aleatória;
- redução média reportada de 40,79% no volume de concreto, com máximo de 85,87% em caso específico;
- identificação de que a sobreposição severa não é resolvida apenas reduzindo dimensões, apontando necessidade futura de sapatas associadas.

## Parte do Lucas no manuscrito

A contribuição do Lucas fica especialmente ligada a:

- fundamentação teórica de [[03_Otimizacao/EGO - Efficient Global Optimization]];
- fundamentação de [[03_Otimizacao/Gaussian Process Regressor]];
- explicação de [[03_Otimizacao/Expected Improvement]];
- estudo de [[03_Otimizacao/Kernels GPR]];
- análise de penalização da função objetivo;
- conexão entre os testes dos notebooks e a escrita acadêmica;
- delimitação honesta entre restrição de sobreposição e bin packing completo.

## Pontos fortes já aproveitáveis

- O manuscrito conecta bem a motivação prática: tentativa e erro em escritórios, consumo elevado de concreto e necessidade de racionalização.
- A formulação de restrições está alinhada com o software: tensão solo-sapata, geometria mínima, sobreposição e punção.
- A narrativa EGO-GPR está coerente com o relatório parcial da IC.
- A conclusão reconhece corretamente que sobreposição severa exige etapa futura, possivelmente sapatas associadas.
- O estudo de kernels e penalização dá identidade metodológica ao artigo, não deixando o FundaIA apenas como ferramenta de interface.

## Pontos que precisam revisão antes da versão final

### Escopo e linguagem científica

- Evitar dizer que o método **garante** ótimo global. Melhor: "favorece busca global", "busca soluções de boa qualidade" ou "reduz avaliações diretas ao guiar a busca pelo surrogate".
- Padronizar o termo principal: o texto alterna entre metaheurística evolutiva, otimização híbrida, EGO e aprendizado de máquina. A formulação mais segura é: **otimização global assistida por modelo substituto, com EGO-GPR e AG interno para maximização da aquisição**.
- Cuidado ao chamar Monte Carlo de prática tradicional de escritório. Melhor dizer que é uma **aproximação computacional de tentativa e erro**, usada como baseline simples.

### Kernels e experimentos

- O TCC menciona 18 configurações de kernel, enquanto o vault/código registra [[03_Otimizacao/Kernels GPR|20 kernels experimentais + 1 kernel de produção]]. Antes de versão final, decidir a convenção oficial.
- O manuscrito destaca `K06`, mas o código de produção usa atualmente `constroi_kernel()[-1]`, registrado no vault como `k20`. Isso precisa ser explicado ou unificado.
- Recriar tabelas/figuras finais depois de corrigir pendências de reprodutibilidade listadas em [[10_Melhorias/Guia - Validação antes do Bin Packing]].

### Resultados e tabelas

- Existem marcadores de referência incompletos como `Tabela??` e `Seção??`.
- Na tabela de restrições de tensão aparece `P06` onde o contexto indica `P16`.
- Na conta de sobreposição P04-P05, conferir o termo textual `34,10 - 22,95`, pois os dados da tabela indicam coordenadas próximas de `33,95`.
- Conferir todos os volumes, restrições e razões S/R contra execução atual do código antes de congelar os números.
- A comparação com Monte Carlo deve registrar orçamento de avaliações, seeds, taxa de factibilidade e critério de seleção da melhor solução.

### Engenharia e limitações

- O texto está correto em tratar sapatas associadas como trabalho futuro. Não transformar isso em entrega atual.
- Deixar explícito que `xg` e `yg` são dados de entrada, não variáveis otimizadas no estágio atual.
- Sobreposição atual é uma penalização geométrica AABB com retângulos alinhados. Isso ainda não é bin packing formal.
- Punção está descrita para seção C. Se a seção C' não entrar, declarar como limite metodológico ou trabalho futuro.

## Como usar este documento daqui para frente

### Para fechar a etapa atual

Seguir [[10_Melhorias/Guia - Validação antes do Bin Packing]]:

- sanear duplicações e reprodutibilidade;
- validar funções de engenharia;
- congelar casos de 1, 2 e 3 sapatas;
- refazer experimentos EGO-GPR com seeds registradas;
- comparar com Monte Carlo/random search e, se couber, GA puro.

### Para o artigo 1

Tema recomendado:

> FundaIA: otimização assistida por processos gaussianos para dimensionamento de sapatas isoladas de concreto armado.

O TCC/artigo do Filipe pode ser usado como base textual, mas a versão final deve manter o escopo:

- dimensionamento otimizado de sapatas isoladas;
- posições fornecidas como entrada;
- restrição preliminar de sobreposição;
- EGO-GPR como contribuição metodológica;
- bin packing/layout completo como trabalho futuro.

### Para o bin packing

Não iniciar como foco principal antes de validar o núcleo atual. Quando iniciar, separar como nova frente:

- posições `xg`, `yg` como variáveis;
- fronteira do lote;
- margens construtivas;
- decisão de sapata isolada versus associada;
- bulbo de tensão e interação geotécnica;
- recalque;
- packing 2D formal.

## Vínculos importantes

- [[01_Projeto/Visão Geral do Projeto]]
- [[01_Projeto/Escopo da IC]]
- [[01_Projeto/Atores e Histórico]]
- [[03_Otimizacao/EGO - Efficient Global Optimization]]
- [[03_Otimizacao/Gaussian Process Regressor]]
- [[03_Otimizacao/Kernels GPR]]
- [[03_Otimizacao/Expected Improvement]]
- [[06_Notebooks/testes_gpr_lucas]]
- [[06_Notebooks/testes_otm_lucas]]
- [[09_Relatorios/Analise - Roadmap Artigo IC - 2026-04-27]]
- [[10_Melhorias/Guia - Validação antes do Bin Packing]]
- [[08_Artigos/Index de Artigos]]
