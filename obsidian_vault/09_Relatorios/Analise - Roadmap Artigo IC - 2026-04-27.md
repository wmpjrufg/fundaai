---
tags: [relatorio, roadmap, artigo, ic, pesquisa, packing, gpr, ego]
data: 2026-04-27
escopo: leitura-estatica
---

# Analise - Roadmap, artigo e frentes de pesquisa

## Veredito curto

Sim: o roadmap mais coerente agora e **fechar, sanear e validar o que ja foi construido** antes de testar uma fila grande de metodos novos.

O projeto ja tem uma entrega defendavel: uma ferramenta em Python/Streamlit para dimensionamento otimizado de sapatas isoladas, com funcao objetivo penalizada, EGO, GPR, Expected Improvement e GA interno. Mas essa entrega ainda precisa de reprodutibilidade, testes minimos, saneamento de inconsistencias e validacao experimental antes de sustentar um artigo.

Minha recomendacao e nao tentar colocar o **bin packing completo** dentro do primeiro artigo. O que existe hoje e uma restricao geometrica de sobreposicao AABB com posicoes fixas dos pilares/sapatas. Isso nao equivale ainda a otimizar layout, decidir agrupamento de sapatas, considerar bulbo de tensao ou transformar sapatas proximas em sapatas associadas. Essa parte e uma frente de pesquisa otima, mas deve vir depois que o nucleo atual estiver tecnicamente fechado.

## Materiais verificados

- Plano original: `docs/contexto_academico/ic_lucas/plano_de_trabalho_ic_lucas.pdf`.
- Relatorio parcial: `docs/contexto_academico/ic_lucas/relatorio_parcial_ic_lucas.pdf`.
- TCC/artigo Filipe Amaral em construcao: `docs/contexto_academico/tcc_filipe_amaral/filipe_amaral_tcc_fundaia_sapatas_ego_gpr_em_construcao.pdf`.
- Biblioteca de artigos: `/Users/lucasteixeiracorreia/Documents/Minha biblioteca.zip`.
- Vault Obsidian em `obsidian_vault/`.
- Codigo principal do projeto: `fundacao.py`, `pages/sapatas.py`, `metapy_toolbox/ego.py`, `metapy_toolbox/funcs.py`, `metapy_toolbox/benchmark.py`.
- Dados e artefatos: `assets/`, `models/`, notebooks na raiz e notas de auditoria do vault.

Nao executei Streamlit nem otimizacoes novas. A analise aqui e de coerencia documental, leitura estatica de codigo, leitura dos PDFs e triagem tecnica da biblioteca.

## O que o plano original prometia

O plano de trabalho tem um escopo ambicioso: **dimensionamento e posicionamento otimizado de fundacoes superficiais**, com conceitos de empacotamento e mecanica das estruturas. Os objetivos especificos incluem:

- revisar otimizacao aplicada a engenharia civil;
- modelar o problema de empacotamento aplicado a fundacoes;
- implementar algoritmo para gerar solucoes otimizadas de posicionamento;
- desenvolver software;
- comparar com metodos tradicionais;
- validar por estudos de caso;
- produzir artigo.

O cronograma tambem deixa claro que abril a julho de 2026 e o periodo de implementacao, testes e relatorios, e junho a agosto de 2026 e o periodo natural para artigo. Entao faz sentido que, neste momento, o foco seja **consolidar resultado**, nao abrir muitas frentes novas.

## O que o relatorio parcial consolidou

O relatorio parcial ja desloca o centro metodologico para uma arquitetura **EGO-GPR**:

- EGO como estrategia principal;
- GPR como modelo substituto;
- Expected Improvement como aquisicao;
- GA como otimizador interno da aquisicao;
- funcao objetivo mono-objetivo, minimizando volume de concreto;
- restricoes de geometria, tensao admissivel no solo e verificacoes estruturais;
- testes com kernels, penalizacoes e tamanho de base de treino;
- exemplos com um, dois e tres elementos de fundacao;
- comparacao preliminar com Monte Carlo.

Esse deslocamento e coerente com o plano porque ainda esta dentro de otimizacao computacional aplicada a fundacoes, mas ele muda o artigo provavel: o primeiro artigo fica mais forte se for sobre **surrogate-assisted optimization para dimensionamento de sapatas**, e nao sobre bin packing completo.

Ponto de cuidado: o relatorio parcial menciona reducao media de 40,79% no volume de concreto e ganho maximo de 85,87% em relacao a simulacoes Monte Carlo. Esses numeros sao promissores, mas o codigo/vault ainda tem pendencias que precisam ser resolvidas antes de publicar esses resultados como definitivos.

## Estado real do projeto hoje

### O que esta implementado de fato

O nucleo atual faz isto:

- le uma planilha de entrada com cargas, dimensoes de pilares, SPT, solo e coordenadas `xg (m)`, `yg (m)`;
- otimiza, para cada sapata, as dimensoes `(h_x, h_y, h_z)`;
- calcula volume;
- calcula tensao admissivel por regra simplificada baseada em tipo de solo e SPT;
- calcula `sigma_max` e `sigma_min`;
- verifica geometria minima;
- verifica puncao na secao C;
- calcula sobreposicao retangular entre sapatas;
- soma penalizacoes positivas a funcao objetivo;
- roda EGO com GPR e GA interno;
- apresenta resultado no Streamlit;
- gera tabela de dimensoes, verificacoes, plot em planta e DXF.

Isso e suficiente para um primeiro artigo se o escopo for nomeado com precisao.

### O que ainda nao esta implementado

Ainda nao existe, no codigo atual:

- otimizacao das coordenadas `xg`, `yg`;
- restricao de lote/fronteira do terreno;
- margem minima construtiva entre sapatas;
- packing hard com garantia de factibilidade;
- decoder ou reparo de layout;
- bin packing/strip packing formal;
- rotacao de sapatas;
- no-fit polygon;
- verificacao de bulbo de tensao entre fundacoes proximas;
- criterio automatico para transformar sapatas isoladas em associadas/combinadas;
- dimensionamento estrutural de sapata associada;
- modelo de recalque/interacao solo-estrutura;
- verificacao da secao C' de puncao, que esta comentada.

Logo, o projeto nao deve afirmar ainda que resolve "posicionamento otimizado completo" ou "bin packing de fundacoes" como entrega implementada. O correto e afirmar que ha uma **restricao preliminar de nao sobreposicao em planta com posicoes fixas**, e que o posicionamento completo e frente futura.

## Aderencia plano x relatorio x codigo x vault

| Item | Plano original | Relatorio parcial | Codigo atual | Veredito |
|---|---|---|---|---|
| Software Python/Streamlit | Previsto | Reportado | Implementado | Coerente, mas precisa higiene/reprodutibilidade |
| Metaheuristicas | Previsto | GA interno + EGO | GA interno no EGO | Coerente |
| GPR/EGO | Nao era o foco inicial | Consolidado | Implementado | Evolucao coerente do escopo |
| Dimensionamento de sapatas | Previsto | Consolidado | Implementado | Principal entrega atual |
| Posicionamento/layout | Previsto | Citado | Apenas visualizacao e coordenadas fixas | Parcial |
| Empacotamento/bin packing | Previsto | Pouco consolidado | Apenas sobreposicao AABB penalizada | Ainda nao implementado |
| Estudos de caso | Previsto | 1, 2 e 3 fundacoes | Dados existem em `assets/` | Precisa revalidar experimentalmente |
| Comparacao tradicional/Monte Carlo | Previsto | Reportada | Nao encontrei pipeline robusto consolidado | Precisa fechar antes do artigo |
| Artigo | Previsto | Em coautoria | Viavel | Escopo deve ser ajustado |

## Sobre o roadmap atual do vault

O [[10_Melhorias/Roadmap Sugerido]] esta bem montado: saneamento, refatoracao minima, robustez, ganhos algoritmicos e frentes de pesquisa. O ajuste que eu faria e **mudar a enfase temporal**.

Hoje o risco nao e faltar ideia. O risco e pular para metodos novos com o nucleo experimental ainda instavel. Portanto:

1. Fase 0 e Fase 1 devem virar prioridade absoluta.
2. Fase 2 deve ser feita apenas no que for necessario para reprodutibilidade e desempenho dos experimentos.
3. Fase 3, de metodos novos, deve ficar pequena antes do artigo: no maximo um baseline adicional, como GA puro ou random/Monte Carlo bem controlado.
4. Fase 4 deve ser tratada como roadmap de pesquisa futura, nao como compromisso do artigo atual.

## Pendencias que afetam publicacao

Estas pendencias nao impedem o uso interno do app, mas enfraquecem um artigo se nao forem resolvidas:

| Pendencia | Impacto cientifico |
|---|---|
| `requirements.txt` em UTF-16 | Dificulta reproducao por terceiros |
| `pages/sapatas.py` duplicado | Risco de manutencao e confusao |
| `obj_felipe_lucas` e `obj_teste` duplicadas | Risco de divergencia entre otimizacao e relatorio final |
| Penalidade extra ignorada nos notebooks | Pode invalidar comparacoes `1e1` vs `1e6` se figuras foram geradas com codigo atual |
| Historico do EGO com `ITER=0` e `ID` constante | Invalida curva de convergencia por iteracao |
| `n_rep=5` reutiliza a mesma populacao inicial | As repeticoes nao sao independentes como experimento estocastico |
| `griewank` e `powell` suspeitos | Benchmarks podem dar validacao falsa |
| Puncao C' comentada | Limita alegacao de aderencia estrutural completa |
| Posicao fixa `xg`, `yg` | Impede afirmar otimizacao real de layout |
| Sobreposicao penalizada e duplicada por par | Precisa decisao/documentacao antes de defender packing |

## Biblioteca de artigos: como usar no artigo

A biblioteca esta bem alinhada para sustentar tres blocos teoricos.

### Bloco 1 - Otimizacao de fundacoes rasas

Artigos centrais:

- Wang e Kulhawy (2008): base forte para otimizacao economica de fundacoes e discussao de estados limites/servico.
- Gandomi e Kashani (2018): minimizacao de custo de fundacao rasa com algoritmos de inteligencia de enxame; bom benchmark conceitual para mostrar que a literatura compara muitos algoritmos.
- Kashani et al. (2020): projeto otimo de fundacao rasa com algoritmos evolutivos; util para situar metaheuristicas classicas.
- Waheed et al. (2022): ferramenta pratica para sapatas isoladas em concreto armado; muito relevante para justificar o carater de software aplicado.
- Waheed et al. (2025): evolucao com sapatas isoladas escalonadas, custo, emissoes e energia incorporada; bom gancho para sustentabilidade e economia.
- Rasheed/Khajehzadeh: otimizacao de fundacao tipo spread footing com gravitational search.
- Juang e Wang (2013): desenho robusto e confiabilidade com GA multiobjetivo; melhor para trabalhos futuros do que para o primeiro artigo.

Uso recomendado: abrir a introducao mostrando que otimizar fundacoes e uma linha estabelecida, mas geralmente com foco em custo/dimensoes e metaheuristicas diretas. O FundaIA entra como contribuicao aplicada com surrogate EGO-GPR em um fluxo interativo.

### Bloco 2 - EGO, GPR e Bayesian Optimization

Artigos centrais:

- Jones, Schonlau e Welch (1998): base do EGO e Expected Improvement.
- Rasmussen/Williams e Schulz et al.: base teorica de GPR.
- Snoek et al. (2012): boas praticas de Bayesian Optimization com GP.
- Shahriari et al. (2016): revisao ampla de Bayesian Optimization.
- Ahmad et al. (2021): GPR em capacidade de carga de fundacoes rasas; ajuda a conectar GPR com geotecnia.

Uso recomendado: justificar que o problema de fundacoes tem FO com restricoes e custo computacional crescente, e que surrogate-assisted optimization pode reduzir avaliacoes reais ou organizar melhor a busca.

### Bloco 3 - Comparacao de metaheuristicas e rigor experimental

Artigos centrais:

- Gomes et al. (2018): bom para discutir comparacao probabilistica de metaheuristicas.
- Morales-Castaneda et al. (2020): exploracao/explotacao; util para evitar narrativa simplista de "algoritmo X e melhor".
- Abualigah et al. (2021): AOA entra como exemplo de metaheuristica moderna, mas nao precisa virar metodo prioritario agora.

Uso recomendado: defender o desenho experimental com multiplas seeds, media/desvio, melhor/pior, taxa de factibilidade e comparacao justa por numero de avaliacoes.

### Bloco 4 - Empacotamento, sapatas associadas e interacao

A biblioteca atual tem alguns apoios, mas ainda nao fecha totalmente essa frente:

- NBR 6122 e Bezerra et al. ajudam a conceituar sapata isolada, associada, corrida e radier.
- O manual de fundacoes rasas `G09-002` traz discussao util sobre combined footings, largura, recalque, estados limites, pressao equivalente e zona de influencia.
- O plano original cita bin packing/DRL, mas a biblioteca enviada ainda parece mais forte em fundacoes/otimizacao/GPR do que em packing 2D formal.

Para um artigo especifico de packing, ainda faltaria bibliografia dedicada: strip packing 2D, no-fit polygon, phi-functions, layout optimization, Boussinesq/Newmark/2:1 stress distribution, interacao entre bulbos de tensao e criterios para sapatas combinadas.

## Minha decisao sobre bin packing

Eu nao colocaria bin packing completo como entrega obrigatoria antes do primeiro artigo.

Motivo: o problema muda bastante. Hoje as variaveis sao apenas dimensoes das sapatas. Quando entra packing real, as variaveis passam a incluir posicao, excentricidade, possivel rotacao, margem de escavacao, fronteira do lote e talvez decisao topologica de quais pilares compartilham uma fundacao. Quando entra bulbo de tensao, o problema deixa de ser geometrico puro e passa a envolver interacao geotecnica e recalque. Quando duas sapatas proximas viram sapata associada, o tipo estrutural muda.

Isso e grande o suficiente para ser **um segundo artigo** ou uma segunda fase da IC.

### O que pode entrar no artigo atual

Pode entrar:

- restricao de sobreposicao em planta como simplificacao geometrica;
- discussao de que `xg`, `yg` sao dados de entrada;
- resultado para uma, duas e tres sapatas;
- identificacao de packing completo como extensao natural.

Nao deve entrar como afirmacao principal:

- bin packing implementado;
- layout otimizado;
- agrupamento automatico de sapatas;
- bulbo de tensao modelado;
- sapata associada dimensionada automaticamente.

## Roadmap revisado recomendado

### Fase A - Fechar base factual e reprodutibilidade

Objetivo: conseguir rodar o projeto em maquina limpa e confiar nos logs.

Entregas:

- corrigir `requirements.txt`;
- remover duplicacao de `pages/sapatas.py`;
- unificar `obj_felipe_lucas` e `obj_teste` ou documentar claramente a diferenca;
- corrigir/decidir parametrizacao de penalidade;
- corrigir `ITER` e `ID` no historico do EGO;
- fazer `n_rep` gerar populacoes iniciais independentes;
- registrar seeds em LHS, GA, GPR e numpy;
- registrar config completa de cada experimento;
- decidir oficialmente "20 kernels experimentais + 1 de producao" ou "21 kernels".

### Fase B - Validar engenharia minima

Objetivo: provar que a funcao objetivo calcula certo antes de provar que otimiza bem.

Entregas:

- testes para `tensao_adm_solo`;
- testes para `calcular_sigma_max_min`;
- testes para `checagem_geometria`;
- testes para `verificacao_puncao_sapata` na secao C;
- decisao explicita sobre secao C';
- testes para `sobreposicao_sapatas`;
- um exemplo manual pequeno com resultado calculado fora do otimizador.

Essa fase e mais importante para artigo do que adicionar outro algoritmo.

### Fase C - Validar otimizacao atual

Objetivo: defender que EGO-GPR entrega boas solucoes sob mesmo orcamento de avaliacoes.

Entregas:

- casos de 1, 2 e 3 sapatas com dados congelados;
- baseline Monte Carlo/random search com mesmo numero de avaliacoes reais;
- opcional: GA puro como baseline unico de metaheuristica;
- 20 a 30 seeds por caso, se o tempo permitir;
- metricas: melhor volume factivel, volume medio, desvio, taxa de factibilidade, violacao maxima, tempo, numero de avaliacoes;
- curva de convergencia com `ITER` correto;
- tabela final comparando EGO-GPR, Monte Carlo e GA puro;
- figuras recriadas depois da correcao da penalidade.

### Fase D - Escrever o artigo 1

Tema sugerido:

> Otimizacao assistida por surrogate para dimensionamento de sapatas isoladas em concreto armado com restricoes geotecnicas e estruturais.

Contribuicoes defendaveis:

- formulacao computacional do dimensionamento de sapatas como problema penalizado;
- integracao de verificacoes de engenharia em uma FO unica;
- arquitetura EGO-GPR com Expected Improvement e GA interno;
- ferramenta interativa FundaIA em Streamlit;
- estudo de kernels/penalizacao;
- validacao em pequenos estudos de caso;
- comparacao com Monte Carlo/random search e, opcionalmente, GA puro.

### Fase E - Abrir a frente packing/layout

So depois do artigo 1 ou em paralelo com escopo pequeno.

Entregas conceituais antes de codigo:

- definir se `g_sob` sera hard ou soft;
- definir margem minima entre sapatas;
- definir fronteira do lote;
- decidir se posicao livre sera centro da sapata ou deslocamento relativo ao pilar;
- incorporar excentricidade adicional quando a sapata sai do centro do pilar;
- escolher criterio para proximidade: overlap geometrico, distancia minima, bulbo de tensao, recalque, ou combinacao;
- definir quando sapatas proximas continuam isoladas e quando viram associadas;
- separar casos: sapata isolada, associada, corrida/radier parcial.

### Fase F - Metodos avancados

Somente depois da base validada:

- Constrained Bayesian Optimization, modelando restricoes separadamente;
- Physics-Informed GPR, usando estrutura fisica nas restricoes;
- multiobjetivo volume/custo/padronizacao/robustez;
- otimizacao sob incerteza para SPT, carga e resistencia;
- layout + sizing como problema conjunto.

## Proposta de artigo 1

### Titulo possivel

**FundaIA: otimizacao assistida por processos gaussianos para dimensionamento de sapatas isoladas de concreto armado**

Alternativa mais internacional:

**Surrogate-assisted optimization of reinforced concrete isolated footings using Efficient Global Optimization and Gaussian Process Regression**

### Pergunta de pesquisa

Uma arquitetura EGO-GPR consegue reduzir o volume de concreto de sapatas isoladas, mantendo factibilidade frente a restricoes geometricas, geotecnicas e estruturais, com menos avaliacoes diretas do que busca aleatoria/Monte Carlo?

### Escopo honesto

O artigo deve dizer que:

- as posicoes dos pilares/sapatas sao fornecidas como entrada;
- o otimizador altera dimensoes, nao layout completo;
- a sobreposicao e tratada como restricao geometrica preliminar;
- packing completo e trabalho futuro.

Isso nao enfraquece o artigo. Pelo contrario: deixa a contribuicao limpa e menos vulneravel a critica.

### Estrutura sugerida

1. Introducao: custo, seguranca e necessidade de automacao em fundacoes.
2. Revisao: otimizacao de sapatas, metaheuristicas, EGO/GPR.
3. Formulacao: variaveis, FO, restricoes, penalizacao.
4. Metodo: LHS, GPR, EI, GA interno, loop EGO.
5. Software: FundaIA, entrada Excel, saidas, DXF, visualizacao.
6. Experimentos: casos 1/2/3, seeds, baselines.
7. Resultados: volume, factibilidade, convergencia, sensibilidade a kernel/penalidade.
8. Discussao: limites, reprodutibilidade, packing como extensao.
9. Conclusao.

## Proposta de artigo 2

Tema:

**Dimensionamento e posicionamento conjunto de sapatas com restricoes de empacotamento e criterios de interacao geotecnica**

Esse sim pode tratar:

- variaveis `(h_x, h_y, h_z, xg, yg)`;
- lote;
- margem construtiva;
- packing hard;
- excentricidade;
- bulbo de tensao;
- decisao isolada vs associada;
- multiobjetivo volume/custo/margem/recalque.

Mas para esse artigo, e necessario montar bibliografia adicional e uma formulacao nova. Nao e apenas "adicionar bin packing" ao codigo atual.

## Frentes de pesquisa priorizadas

### Prioridade 1 - Validacao experimental do FundaIA

Mais urgente e mais publicavel no curto prazo. Entrega o artigo 1.

### Prioridade 2 - Constrained Bayesian Optimization

Muito coerente com o problema atual porque o FundaIA tem varias restricoes. Em vez de aprender a FO ja deformada por penalizacao, pode-se modelar volume e restricoes separadamente.

### Prioridade 3 - Packing/layout + sizing

Original e muito interessante, mas so deve virar frente principal quando a formulacao de engenharia estiver madura.

### Prioridade 4 - Otimizacao sob incerteza

Boa continuidade com Juang/Wang e com geotecnia real: SPT, cargas e resistencia do solo sao incertos. Pode render artigo forte depois.

### Prioridade 5 - Physics-Informed Surrogates

Tem potencial, mas eu colocaria depois de CBO. Como a FO atual e relativamente barata e analitica, PI-GPR so vira muito necessario quando o modelo incluir recalque, interacao solo-estrutura, FEM ou restricoes mais caras.

## Recomendacoes praticas

1. Nao abrir muitos algoritmos agora. Escolher EGO-GPR + Monte Carlo + no maximo GA puro.
2. Congelar os datasets de estudo de caso e versionar os resultados.
3. Refazer graficos/tabelas somente depois de corrigir penalidade, seeds e historico.
4. Nomear o escopo atual como "dimensionamento otimizado com restricao de sobreposicao", nao como "bin packing completo".
5. Atualizar o [[08_Artigos/Index de Artigos]] com notas individuais dos artigos centrais.
6. Separar claramente no texto: "implementado", "validado", "em desenvolvimento" e "trabalho futuro".
7. Conversar com a orientadora sobre uma frase oficial de escopo para o artigo 1.

## Decisoes para levar a orientadora

- O primeiro artigo sera sobre EGO-GPR aplicado ao dimensionamento de sapatas isoladas?
- A comparacao principal sera contra Monte Carlo/random search, GA puro ou ambos?
- O artigo pode assumir posicoes fixas dos pilares/sapatas?
- A secao C' de puncao precisa estar implementada para publicacao ou pode ser limite declarado?
- O valor de penalidade sera fixo, calibrado ou comparado?
- O numero oficial de kernels sera 20 experimentais + 1 de producao?
- Packing completo entra como trabalho futuro ou como uma secao conceitual curta?

## Conclusao

O melhor caminho e **terminar o que ja existe com rigor**, porque ja ha material suficiente para um artigo bom se o escopo for honesto. O FundaIA atual nao precisa prometer resolver todo o problema de posicionamento para ser publicavel. Ele precisa provar, com experimentos reprodutiveis, que a arquitetura EGO-GPR melhora o dimensionamento de sapatas isoladas sob restricoes de engenharia.

Depois disso, o projeto pode crescer para a contribuicao mais original: layout + sizing com packing, bulbo de tensao e decisao entre sapatas isoladas e associadas. Essa segunda frente e excelente, mas merece uma formulacao propria.

## Vínculos

- [[10_Melhorias/Roadmap Sugerido]]
- [[03_Otimizacao/Problema de Empacotamento]]
- [[11_Frentes_de_Pesquisa/Posicionamento Conjunto - Layout + Sizing]]
- [[11_Frentes_de_Pesquisa/Bayesian Optimization Constrained]]
- [[11_Frentes_de_Pesquisa/Physics-Informed Surrogates]]
- [[12_Auditoria/Auditoria 2026-04-27 - Vault vs Projeto]]
- [[09_Relatorios/Auditoria - Vault vs Projeto - 2026-04-27]]
