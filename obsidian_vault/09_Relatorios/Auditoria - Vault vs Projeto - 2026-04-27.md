---
tags: [relatorio, auditoria, vault, projeto]
data: 2026-04-27
escopo: leitura-estatica
---

# Auditoria — Vault Obsidian vs Projeto FundaIA

## Escopo

Auditoria feita em 2026-04-27, na branch `refactor/code-base`, com foco em:

- ler o vault `obsidian_vault/`;
- ler a estrutura real do projeto, codigos, dados, notebooks e arquivos de apoio;
- verificar se o vault esta coerente com o estado atual do projeto;
- validar se as melhorias propostas fazem sentido para a situacao atual;
- nao implementar nenhuma melhoria de codigo.

Nao rodei a aplicacao Streamlit nem otimizacoes completas. Fiz leitura estatica, parse AST dos arquivos Python e inspecao de planilhas/notebooks para validar schema e referencias.

## Resumo Executivo

O vault esta majoritariamente coerente com a situacao do projeto. Ele descreve corretamente a arquitetura principal: app Streamlit com `app.py`, paginas em `pages/`, nucleo de engenharia/GPR em `fundacao.py`, toolbox interna em `metapy_toolbox/`, assets Excel, modelos GPR e notebooks experimentais.

As principais issues documentadas no vault foram confirmadas no codigo:

- `pages/sapatas.py` esta duplicado em dois blocos quase identicos.
- `requirements.txt` esta em UTF-16 LE com BOM.
- `obj_felipe_lucas` e `obj_teste` sao clones com diferenca apenas no retorno.
- a verificacao de puncao na secao C' esta comentada.
- `metapy_toolbox/methods.py` esta 100% comentado e ainda e importado.
- `metapy_toolbox/grey_wolf.py` contem placeholder literal de diversidade.
- notebooks passam um quinto argumento de penalidade para `obj_teste`, mas a funcao atual ignora esse argumento.

Tambem encontrei pontos relevantes que o vault ainda nao registra ou registra de forma incompleta:

- `metapy_toolbox/ego.py` adiciona novos pontos com `ITER=0` e `ID` herdado do ultimo indice da populacao inicial; isso torna o historico do EGO pouco confiavel.
- `pages/sapatas.py` executa `n_rep=5`, mas reutiliza a mesma populacao inicial em todas as repeticoes.
- `03_Otimizacao/Problema de Empacotamento.md` esta vazio, apesar de ser central no mapa.
- Ha inconsistencia entre "20 kernels" e a implementacao real de `constroi_kernel`, que retorna 21 kernels; os `.pkl` persistidos cobrem somente k00-k19, enquanto a UI usa `k[-1]`, ou seja, o 21o kernel.
- `assets/data/toy_problem_copy_3.xlsx` tem 1 fundacao, mas o vault sugere que as copias do toy problem sao variacoes de 3 fundacoes.
- `testes_fo_filipe.ipynb` e `testes_otm.ipynb` ainda referenciam `assets/el08.xlsx`, arquivo que nao existe no estado atual.
- `metapy_toolbox/benchmark.py` tem pelo menos uma funcao benchmark suspeita: `griewank` multiplica o produto fora do loop, usando so a ultima dimensao. `powell` tambem merece revisao por indexacao.

Conclusao: o vault serve muito bem como mapa do projeto e como base de continuidade, mas precisa de uma rodada de saneamento documental para nao induzir decisoes erradas em experimentos, principalmente sobre penalidade, kernels, packing e notebooks.

## Inventario Validado

### Estado Git

- Branch atual: `refactor/code-base`.
- Alteracoes preexistentes:
  - `ops/wake_up.py` modificado.
  - `testes_gpr_lucas.ipynb` modificado.
  - `testes_otm_lucas.ipynb` modificado.
  - `obsidian_vault/` inteiro nao rastreado.
- O vault ainda nao faz parte do historico git.

### Tamanho e estrutura

- Repositorio total: aproximadamente 2.3 GB.
- `.venv/`: aproximadamente 1.1 GB.
- `models/`: aproximadamente 553 MB.
- `assets/`: aproximadamente 33 MB.
- `old/`: aproximadamente 700 KB.
- `obsidian_vault/`: aproximadamente 420 KB.

### Vault Obsidian

- 100 arquivos Markdown.
- 5 arquivos de configuracao Obsidian em `.obsidian/`.
- 4.169 linhas Markdown.
- 568 wikilinks encontrados.
- 5 links quebrados, todos placeholders/template:
  - `[[Nota]]` no README.
  - `[[Nome do Artigo]]` no index de artigos.
  - `[[08_Artigos/…]]` no template de conceito.

O grafo interno esta praticamente saudavel. O problema nao e link quebrado; e desatualizacao/incompletude em algumas notas.

### Codigo Python

Arquivos Python lidos e parseados sem erro de sintaxe:

- `app.py`
- `env-setup.py`
- `fundacao.py`
- `pages/home.py`
- `pages/sapatas.py`
- `ops/wake_up.py`
- `metapy_toolbox/__init__.py`
- `metapy_toolbox/benchmark.py`
- `metapy_toolbox/ego.py`
- `metapy_toolbox/funcs.py`
- `metapy_toolbox/genetic_algorithm.py`
- `metapy_toolbox/grey_wolf.py`
- `metapy_toolbox/methods.py`

Nao ha pasta `tests/`, `pyproject.toml`, config de pytest, pre-commit ou workflow `.github/`.

## Validacao Por Area

### 1. App e UI

O vault descreve corretamente `app.py`: o arquivo configura Streamlit wide layout, inicializa `st.session_state["lang"]`, cria duas paginas via `st.Page` e roda `st.navigation`.

`pages/home.py` tambem esta coerente com o vault:

- seletor PT/EN;
- texto explicativo;
- download do template `assets/problema_fund_três.xlsx`.

`pages/sapatas.py` esta corretamente descrito como pagina principal de dimensionamento, mas a issue de duplicacao e ainda mais critica na pratica:

- linhas 120-325 contem o fluxo completo;
- linhas 326-531 repetem o mesmo fluxo;
- apos upload bem-sucedido, o script tende a continuar ate o segundo bloco e recriar widgets com mesmos labels/keys;
- isso pode gerar erro de widget duplicado e torna a pagina dificil de manter.

Ponto adicional nao registrado no vault:

- `save_dxf` cria arquivo temporario com `NamedTemporaryFile(delete=False)` e nao remove depois. Em execucoes repetidas pode deixar lixo em `/tmp`.

### 2. Nucleo de engenharia em `fundacao.py`

O vault esta muito coerente com as formulas implementadas:

- `tensao_adm_solo`: pedregulho `SPT/30*1000`, areia `SPT/40*1000`, demais `SPT/50*1000`.
- `calcular_sigma_max_min`: usa `1.05` na parcela axial e `1.30` para tensao compressiva positiva.
- `checagem_tensao_max_min`: retorna `sigma/sigma_adm - 1` para compressao e `-sigma/sigma_adm` para tracao.
- `checagem_geometria`: codifica `h_sapata >= dim_pilar + 2*balanco_min`.
- `verificacao_puncao_sapata`: implementa somente a secao critica C.

A issue de puncao C' esta validada. O bloco esta comentado e depende de funcoes/variaveis ausentes:

- `rho_minimo_fck`;
- `tabela_19_2`;
- `sigma_cp`;
- contribuicoes de `m_xk` e `m_yk`, que nem entram na assinatura atual.

A formulacao do vault sobre restricoes penalizadas tambem bate com o codigo:

- volume bruto;
- `g sobreposicao`;
- `g punção secao C`;
- `g tensao`;
- `g geometria`;
- fator fixo `1E1`.

Ponto tecnico para investigar:

- a sobreposicao e computada por sapata e depois somada no volume final. Como cada par aparece para `i -> j` e `j -> i`, a penalidade global de sobreposicao pode estar contando cada intersecao duas vezes. Isso talvez seja intencional por normalizacao por area de cada sapata, mas deve ser documentado.

### 3. Packing

O tema de empacotamento aparece corretamente como parte central do problema, mas a nota `03_Otimizacao/Problema de Empacotamento.md` esta vazia.

Isso e uma lacuna importante porque varias notas apontam para ela:

- visao geral do projeto;
- formulacao do problema;
- MOC de otimizacao;
- posicionamento como variavel de projeto;
- layout + sizing.

Sugestao documental: preencher essa nota antes de qualquer frente de pesquisa sobre layout. Ela deveria explicar pelo menos:

- modelo atual AABB sem rotacao;
- centragem da sapata no pilar via `xg`, `yg`;
- formula do overlap;
- normalizacao `area_overlap / area_sapata`;
- limitacoes: nao ha terreno, margem, rotacao, NFP, nem posicao como variavel.

### 4. GPR, kernels e modelos persistidos

O vault acerta a arquitetura geral do GPR:

- `Pipeline(StandardScaler, GaussianProcessRegressor)`;
- `normalize_y=True`;
- `n_restarts_optimizer=5`;
- `joblib.dump` para persistencia;
- modelos em `models/`;
- graficos em `assets/graphics/`;
- tabelas em `assets/tables/`.

Validado:

- `models/` contem 118 arquivos `.pkl`.
- `assets/graphics/` contem 40 PNGs.
- `assets/tables/` contem as duas tabelas documentadas.

Distribuicao dos modelos:

- k00, k01 e k02: pops 180, 210, 270, 500, 600, 700, 800, 900, 1200, 1400, 1800.
- k03-k19: pops 500, 600, 700, 800, 900.

Inconsistencia importante:

- varias notas dizem "20 kernels";
- `constroi_kernel()` retorna 21 entradas;
- a nota `Kernels GPR.md` reconhece isso como k20;
- os modelos persistidos vao so ate k19;
- a UI usa `constroi_kernel()[-1]`, ou seja, o kernel k20, nao persistido nos modelos.

Recomendacao: decidir se o projeto tem 20 kernels experimentais + 1 kernel de producao, ou se oficialmente sao 21. Hoje as duas leituras coexistem.

### 5. EGO e otimizacao

O vault descreve corretamente a ideia do EGO:

- avalia populacao inicial;
- treina GPR;
- otimiza Expected Improvement;
- avalia novo ponto real;
- repete por `n_gen`;
- retorna melhor solucao.

A implementacao real tem detalhes que merecem virar issue:

1. `metapy_toolbox/ego.py` registra o novo ponto com `ITER=0`, nao com `ITER=t`.
2. O `ID` usado no novo ponto e a variavel `n` que sobrou do loop da populacao inicial; na pratica, todos os pontos novos tendem a carregar o mesmo ID.
3. Isso nao necessariamente quebra a escolha do melhor `OF`, mas quebra analise historica, contagem por iteracao e graficos de convergencia.
4. Em `pages/sapatas.py`, `x_ini` e gerado uma unica vez antes de `for rep in range(n_rep)`. As 5 repeticoes reutilizam a mesma populacao inicial. Ha aleatoriedade no GA interno, mas nao sao 5 inicializacoes independentes do EGO.
5. `alpha` do GPR no EGO e `0.1`, enquanto `fundacao.gpr_pipelines` tem default `1e-4`; o vault ja registra essa diferenca.

Sugestao: adicionar uma issue nova no vault para "Historico do EGO com ITER/ID incorretos".

### 6. GA, GWO e benchmarks

O vault esta coerente ao dizer que:

- a UI atual usa `mealpy.GA.BaseGA`, nao `genetic_algorithm_01`;
- `genetic_algorithm_01` existe como implementacao propria;
- `grey_wolf_optimizer_01` existe, mas nao e chamado pela UI;
- `grey_wolf.py` tem placeholder `df['DIVERSITY'] = 'aqui implementa função lucas'`;
- `methods.py` e arquivo morto.

Pontos adicionais:

- `benchmark.py` deveria ser revisado antes de ser usado como validacao cientifica. A funcao `griewank` aparenta ter erro de indentacao: o produto fica fora do loop e usa somente o ultimo `x_i`. Isso invalida o benchmark.
- `powell` tambem parece suspeita por usar indices como `x[4*i]`; em Python isso pode estourar para vetores de tamanho multiplo de 4.
- Como o roadmap propoe validar EGO/GA em benchmarks, estes benchmarks precisam ser saneados primeiro.

### 7. Dados e planilhas

O schema documentado no vault bate com as planilhas atuais:

- `Elemento`
- `ap (m)`
- `bp (m)`
- `spt`
- `solo`
- `xg (m)`
- `yg (m)`
- `Fz-c1`, `Mx-c1`, `My-c1`
- `Fz-c2`, `Mx-c2`, `My-c2`
- `Fz-c3`, `Mx-c3`, `My-c3`

Planilhas validadas:

- `assets/problema_fund_um.xlsx`: 1 fundacao.
- `assets/problema_fund_dois.xlsx`: 2 fundacoes.
- `assets/problema_fund_três.xlsx`: 3 fundacoes.
- `assets/data/toy_problem.xlsx`: 3 fundacoes.
- `assets/data/toy_problem_copy.xlsx`: 3 fundacoes.
- `assets/data/toy_problem_copy_2.xlsx`: 3 fundacoes.
- `assets/data/toy_problem_copy_3.xlsx`: 1 fundacao.
- `assets/old_assets/*.xlsx`: schemas compativeis.

Inconsistencias:

- o vault sugere que `toy_problem_copy{,_2,_3}` sao variacoes de 3 fundacoes, mas `toy_problem_copy_3.xlsx` tem 1 fundacao.
- `testes_fo_filipe.ipynb` e `testes_otm.ipynb` ainda apontam para `assets/el08.xlsx`, que nao existe.

### 8. Notebooks

As notas dos notebooks estao boas como resumo historico, mas alguns pontos precisam de alerta:

- `testes_gpr_lucas.ipynb`: atualmente usa `Path("assets/data/toy_problem_copy.xlsx")`, coerente com a reorganizacao de `assets/data/`.
- `testes_otm_lucas.ipynb`: foi ajustado para `assets/data/toy_problem_copy.xlsx` e para exportar tabelas em `assets/tables/`, coerente com o vault.
- `testes_fo_filipe.ipynb`: aponta para `assets/el08.xlsx`, inexistente.
- `testes_otm.ipynb`: tambem aponta para `assets\el08.xlsx`, inexistente.

A issue "Args extras em obj_teste" e critica para interpretar resultados:

- os notebooks rotulam experimentos como `penalty=1e1` e `penalty=1e6`;
- a funcao atual `obj_teste` so le `args[0]` a `args[3]`;
- o quinto argumento e ignorado;
- se os graficos/tabelas forem regenerados com o codigo atual, os cenarios 1e1 e 1e6 nao representam penalidades diferentes.

Isso afeta diretamente qualquer conclusao experimental sobre "penalidade leve vs pesada".

### 9. Dependencias e ambiente

Confirmado:

- `requirements.txt` esta em UTF-16 LE com BOM.
- `env-setup.py` depende dele para instalar o ambiente.
- Isso justifica a issue de alta prioridade no vault.

Conteudo decodificado de `requirements.txt`:

- `fqdn`
- `isoduration`
- `jsonpointer`
- `jupyter`
- `mealpy`
- `openpyxl`
- `pip-chill`
- `rfc3987-syntax`
- `scikit-learn`
- `streamlit`
- `tinycss2`
- `uri-template`
- `webcolors`
- `xlsxwriter`
- `ezdxf`

Imports usados diretamente pelo codigo que nao aparecem como dependencias diretas:

- `pandas`
- `numpy`
- `scipy`
- `matplotlib`
- `joblib`
- `playwright` fica em `ops/requirements.txt`, mas sem versao pinada.

Observacoes:

- algumas dessas dependencias chegam transitivamente por `streamlit` ou `scikit-learn`, mas e fragil depender disso;
- `ops/requirements.txt` tem `requests==2.31.0`, que nao e usado em `ops/wake_up.py`;
- `.gitignore` ignora `*.txt`, com excecao para `requirements.txt`; isso pode esconder documentacao ou dados `.txt` futuros.

### 10. Ops

`ops/wake_up.py` esta coerente com a nota do vault:

- usa Playwright;
- abre URL;
- tenta selector do botao de wake-up;
- fallback por role/texto;
- assume sucesso se nao encontrar o botao.

Detalhe:

- `WakeUpConfig.button_selector` no codigo tem default `None`, mas `main` injeta `"button[data-testid='wakeup-button-viewer']`. A nota do vault descreve como default do campo; melhor ajustar a linguagem para "default usado no CLI".
- ha alteracao local apenas em comentario nesse arquivo.

## Validacao das Melhorias Propostas

As melhorias do vault estao alinhadas com a situacao real do projeto. A ordem geral tambem faz sentido:

1. saneamento;
2. refatoracao minima;
3. performance/reprodutibilidade;
4. ganhos algoritmicos;
5. frentes de pesquisa.

Mas eu ajustaria a Fase 0 para incluir tambem:

- criar issue para `ego.py` registrar `ITER/ID` incorretos;
- decidir oficialmente "20 vs 21 kernels";
- revisar a validade dos experimentos de penalidade antes de usar as tabelas;
- preencher `Problema de Empacotamento.md`;
- atualizar notebooks que apontam para `assets/el08.xlsx`;
- revisar `benchmark.py` antes de usar em validacao.

### Melhorias de maior impacto imediato

1. Corrigir codificacao e dependencias do ambiente.
2. Remover duplicacao de `pages/sapatas.py`.
3. Corrigir/parametrizar penalidade para nao invalidar experimentos.
4. Corrigir historico do EGO (`ITER`, `ID`, seeds).
5. Criar testes unitarios minimos da engenharia.
6. Preencher packing e alinhar kernel count no vault.

### Melhorias coerentes, mas de fase posterior

- POO domain model.
- Separar UI de dominio.
- Pydantic para config.
- Vetorizacao da FO.
- Persistencia de experimentos.
- Logging estruturado.
- CI/CD.

Essas sao muito boas, mas devem vir depois de travar comportamento com testes.

### Frentes de pesquisa

As frentes de pesquisa estao coerentes, principalmente:

- Physics-Informed Surrogates;
- Constrained Bayesian Optimization;
- Hibridizacao memetica;
- Otimizacao sob incerteza;
- Posicionamento conjunto layout + sizing.

Minha recomendacao cientifica:

- curto prazo: resolver baseline e reproducibilidade;
- linha principal: Physics-Informed ou Constrained BO;
- linha complementar: tratamento de restricoes ou active learning;
- evitar LLM/RL como foco principal agora, deixar como trabalhos futuros.

## Prioridade Recomendada

### P0 — Antes de qualquer resultado novo

- Corrigir `requirements.txt` e dependencias diretas.
- Remover duplicacao de `pages/sapatas.py`.
- Parametrizar ou remover o quinto argumento de penalidade nos notebooks.
- Criar testes para as formulas base.
- Verificar se os graficos/tabelas atuais de penalidade sao validos com o codigo que os gerou.

### P1 — Antes de defender resultados no relatorio/artigo

- Corrigir historico do EGO.
- Controlar seeds.
- Registrar configs e git SHA.
- Revisar benchmarks.
- Validar engenharia contra exemplo manual/bibliografia.

### P2 — Para evoluir o software

- Separar UI de dominio.
- Criar `core/engineering`, `core/optimization`, `core/io`.
- Vetorizar FO.
- Cache/persistencia de experimentos.
- CI com pytest/ruff.

### P3 — Pesquisa original

- CBO com restricoes modeladas separadamente.
- PI-GPR ou multi-output GP.
- Comparacao estatistica com 30 seeds.
- Layout + sizing somente depois de formalizar packing.

## Veredito

O vault esta de acordo com o projeto em aproximadamente 80-85% do conteudo tecnico. Ele captura muito bem a arquitetura, os conceitos de engenharia, o pipeline Streamlit/EGO/GPR e as principais dividas tecnicas.

Os 15-20% restantes sao pontos que podem causar erro de interpretacao:

- nota vazia de packing;
- contagem inconsistente de kernels;
- experimentos de penalidade possivelmente invalidos;
- paths antigos em notebooks;
- `toy_problem_copy_3` descrito como 3 fundacoes, mas contendo 1;
- ausencia de issue para historico do EGO;
- benchmarks ainda nao confiaveis para validacao.

Minha sugestao e tratar o vault como mapa confiavel, mas nao como fonte final de verdade experimental ate sanearem esses pontos.

