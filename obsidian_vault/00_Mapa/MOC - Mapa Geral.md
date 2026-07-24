---
tags: [moc, indice]
aliases: [Índice, Home, MOC]
---

# 🗺️ MOC — Mapa Geral do FundaIA

Índice mestre. Toda nota do vault deve ser acessível a partir daqui em até 2 cliques.

## 🎯 Projeto

- [[01_Projeto/Visão Geral do Projeto]]
- [[01_Projeto/Escopo da IC]]
- [[01_Projeto/Contexto Acadêmico - IC Lucas e TCC Filipe Amaral]]
- [[01_Projeto/Atores e Histórico]]
- [[01_Projeto/Stack Tecnológico]]
- [[01_Projeto/Pipeline de Execução]]
- [[01_Projeto/Convenções do Projeto]] — padrão de commits, docstrings, branches

## 🏗️ Engenharia (mecânica das estruturas)

- [[00_Mapa/MOC - Engenharia]]
- [[02_Engenharia/Guia Didatico - Dimensionamento de Sapatas Isoladas]] ⭐ guia didático com fórmulas explicadas (núcleo: Waheed 2025)
- [[02_Engenharia/Sapatas Isoladas]]
- [[02_Engenharia/NBR 6118]]
- [[02_Engenharia/Tensão Admissível do Solo]]
- [[02_Engenharia/Flexão Composta - Sigma Max e Min]]
- [[02_Engenharia/Verificação à Punção]]
- [[02_Engenharia/Restrição de Geometria]]
- [[02_Engenharia/SPT - Sondagem]]

## 🧮 Otimização

- [[00_Mapa/MOC - Otimização]]
- [[03_Otimizacao/Guia Didatico - EGO e GPR]] ⭐ guia didático para não-estatísticos (núcleo: SMT v2.9.3 EGO docs)
- [[03_Otimizacao/Formulação do Problema]]
- [[03_Otimizacao/EGO - Efficient Global Optimization]]
- [[03_Otimizacao/Gaussian Process Regressor]]
- [[03_Otimizacao/Expected Improvement]]
- [[03_Otimizacao/Algoritmo Genético]]
- [[03_Otimizacao/Grey Wolf Optimizer]]
- [[03_Otimizacao/Latin Hypercube Sampling]]
- [[03_Otimizacao/Problema de Empacotamento]]
- [[03_Otimizacao/Penalização de Restrições]]

## 💻 Código

- [[00_Mapa/MOC - Código]]
- [[04_Codigo/app.py]]
- [[04_Codigo/fundacao.py]]
- [[04_Codigo/pages - sapatas.py]]
- [[04_Codigo/pages - home.py]]
- [[04_Codigo/metapy_toolbox - ego.py]]
- [[04_Codigo/metapy_toolbox - genetic_algorithm.py]]
- [[04_Codigo/metapy_toolbox - grey_wolf.py]]
- [[04_Codigo/metapy_toolbox - funcs.py]]
- [[04_Codigo/metapy_toolbox - benchmark.py]]
- [[04_Codigo/ops - wake_up.py]]

## 📊 Dados e Modelos

- [[05_Dados/Schema das Planilhas]]
- [[05_Dados/Modelos GPR Treinados]]
- [[05_Dados/Assets - Templates Excel]]
- [[05_Dados/Assets - Gráficos GPR]]

## 📓 Notebooks

- [[06_Notebooks/testes_fo_filipe]]
- [[06_Notebooks/testes_otm]]
- [[06_Notebooks/testes_gpr_lucas]]
- [[06_Notebooks/testes_otm_lucas]]

## ⚠️ Issues e Débito Técnico

- [[07_Issues/Lista Mestre de Issues]]
- [[07_Issues/Issue - Duplicação em sapatas.py]]
- [[07_Issues/Issue - requirements.txt UTF-16]]
- [[07_Issues/Issue - Histórico do EGO com ITER e ID incorretos]]
- [[07_Issues/Issue - n_rep reusa população inicial]]
- [[07_Issues/Issue - Args extras em obj_teste]]
- [[07_Issues/Issue - Sobreposição contada duas vezes]]
- [[07_Issues/Issue - obj_felipe_lucas vs obj_teste]]
- [[07_Issues/Issue - Punção seção C linha comentada]]
- [[07_Issues/Issue - Placeholder Diversidade GWO]]
- [[07_Issues/Issue - Notebooks com paths quebrados]]
- [[07_Issues/Issue - Benchmarks suspeitos]]
- [[07_Issues/Issue - DXF tempfile não removido]]
- [[07_Issues/Issue - methods.py morto]]
- [[07_Issues/Issue - Branches dispersos]]

## 🛠️ Sugestões de melhoria

- [[10_Melhorias/MOC - Melhorias]]
- [[10_Melhorias/Roadmap Sugerido]]
- [[10_Melhorias/Guia - Validação antes do Bin Packing]] — trilha prática atual: validar FundaIA/EGO-GPR antes de avançar para packing

## 🔬 Frentes de pesquisa (não implementar)

- [[11_Frentes_de_Pesquisa/MOC - Frentes de Pesquisa]]
- [[11_Frentes_de_Pesquisa/Physics-Informed Surrogates]] ⭐ frente declarada de interesse

## 🔍 Auditorias, Relatórios e Logs de Refactor

- [[09_Relatorios/Analise - Roadmap Artigo IC - 2026-04-27]] — análise do plano, relatório parcial, biblioteca de artigos e roadmap para artigo
- [[09_Relatorios/Auditoria - Vault vs Projeto - 2026-04-27]] — auditoria recebida (relatório de leitura estática)
- [[12_Auditoria/Auditoria 2026-04-27 - Vault vs Projeto]] — síntese das ações que tomei no vault em resposta à auditoria
- [[12_Auditoria/Sprint 0 - Saneamento - 2026-04-27]] — log da primeira sprint de refactor (5 issues resolvidas)
- [[12_Auditoria/Sprint 1 - Ciencia (EGO + n_rep) - 2026-04-27]] — log da segunda sprint (2 issues científicas resolvidas + bônus `seed`)
- [[12_Auditoria/Sprint 2 - Testes e Saneamento Experimental - 2026-04-27]] — log da terceira sprint (suite pytest 55 testes + 2 issues experimentais)
- [[12_Auditoria/Sprint 3 - Refactor estrutural - kickoff - 2026-04-27]] — kickoff da quarta sprint (skeleton `core/` + ARCHITECTURE.md)
- [[12_Auditoria/Sprint 3.2 - Engineering migration - 2026-04-28]] — migração das 6 funções de engenharia para `core/engineering/`
- [[12_Auditoria/Sprint 3.3 - Domain entities - 2026-04-28]] — entidades de domínio (Solo, Pilar, Combinacao, Sapata, FundacaoProjeto) em `core/domain/` + 15 testes
- [[12_Auditoria/Sprint 3.4 - IO layer - 2026-04-28]] — leitor Excel (entrada) e writer DXF em `core/io/` + 21 testes (entry point validado)
- [[12_Auditoria/Sprint 3.5 - API layer - 2026-04-28]] — `core/api/{evaluate,optimize}` + migração de `pages/sapatas.py` para shell fino + 22 testes
- [[12_Auditoria/Sprint 3.6 - Optimization migration - 2026-04-28]] — `metapy_toolbox/` → `core/optimization/` (shim de compat preservado)
- [[12_Auditoria/Sprint 3.7 - Pydantic config - 2026-04-28]] — `OptimisationConfig` migrada para Pydantic v2 (validação rica, JSON schema, 4 testes novos)
- [[12_Auditoria/Sprint 3.8 - Vectorized FO - 2026-04-28]] — laço duplo `df.iterrows()` na sobreposição substituído por matriz N×N numpy (5 testes novos, 100× speedup, baseline preservado)
- [[12_Auditoria/Sprint 4.1 - Surrogate cache - 2026-04-28]] — cache LRU+disco do GPR para EGO (`SurrogateCache`, `fit_or_get_cached`, 23 testes novos, hit retorna fit bit-exato sem retreinar)
- [[12_Auditoria/Sprint 4.2 - Experiment persistence - 2026-04-28]] — pasta autodescritiva por run (`ExperimentRecorder`, manifest+config+env+project+history Parquet+summary CSV+metrics JSON, 17 testes novos)
- [[12_Auditoria/Sprint 4.3 - Reorg + docs - 2026-04-28]] — reorganização do repositório (`frontend/`, `scripts/`, `notebooks/`, `archive/`, `assets/data/`), remoção do shim `metapy_toolbox` e reescrita completa de `README.md` + `ARCHITECTURE.md`
- [[12_Auditoria/Sprint 4.4 - Structured logging - 2026-04-28]] — `core/observability/` com JSON-line logger, `run_context`, eventos nomeados em `optimize`/`ego`/`cache`/`experiments` (9 testes novos)
- [[12_Auditoria/Sprint 4.5 - 3D footings viewer - 2026-04-28]] — visualizador 3D Plotly em `frontend/components/footings_3d.py` (sapatas enterradas, pilares acima, plano de solo, hover, abas 2D/3D na página Streamlit, 12 testes novos)
- [[12_Auditoria/Sprint 4.6 - Premium UI - 2026-04-28]] — tema dark + CSS injetado + Plotly tematizado + EGO history chart + 3D polido (presets/lighting/terreno) + painel unificado de export (DXF/JSON/HTML/PNG); recorder ligado por padrão na UI; 22 testes novos
- [[12_Auditoria/Sprint 4.7 - UI polish + live progress - 2026-04-28]] — progresso ao vivo via `progress=` callback, hover closest (não mais ribbon), 3D em seção própria full-width (sem flicker), eixos travados em `>=0`, input `n_rep` exposto, default `n_gen=20`, 6 testes novos
- [[12_Auditoria/Sprint 4.8 - Audit cleanup - 2026-04-28]] — `Solo` deixa de importar engineering, `best_avg_worst` index-safe, guardrails de engenharia em testes de borda, input morto `n_comb` removido, docs e env_setup alinhados (10 testes novos)
- [[12_Auditoria/Sprint 4.9 - Rotation, progress and cancel - 2026-04-28]] — 3D travado em turntable (azimuth+elevação, `up=+z`), progress bar coerente com `n_rep × (n_gen + 1)` etapas, eventos `lhs.start/eval/done` + `optimize.recording`, cancel cooperativo via `should_stop` + `OptimisationCancelled` + thread runner na UI (3 testes novos)
- [[12_Auditoria/Sprint 4.10 - 3D elevation lock - 2026-04-28]] — 3D agora trava elevação por default (mouse-drag não rotaciona; sliders de azimuth/elevação controlam a câmera); toggle "🔓 Rotação livre (mouse)" libera o turntable; `axis_lock="elevation"` parametriza o componente (3 testes novos)
- [[12_Auditoria/Sprint 4.11 - Restore free 3D rotation - 2026-04-28]] — reverte 4.10 a pedido do usuário; 3D volta a rotacionar livremente com o mouse (Z permanece como "up", sem roll), sliders e toggle removidos, `axis_lock` apagado da API
- [[12_Auditoria/Sprint 5.1 - Protocolo experimental final e casos-limite - 2026-07-10]] — protocolo final S1/S2, seeds pareadas, métricas de factibilidade, estudo GPR e artefatos do artigo
- [[12_Auditoria/Sprint 5.2 - Puncao C linha e duas colunas - 2026-07-10]] — punção C′ a 2d, artigo em duas colunas e figura dos casos em planta
- [[12_Auditoria/Sprint 5.3 - Frente C CBO - 2026-07-11]] — Constrained Bayesian Optimization integrada ao protocolo e ao manuscrito
- [[12_Auditoria/Sprint 5.4 - Correcoes artigo e tensao - 2026-07-12]] — correção da tensão, convenção de momentos, reruns completos, revisão metodológica e atualização do vault
- [[12_Auditoria/Sprint 5.5 - Novos artigos e reforco metodologico - 2026-07-12]] — triagem de novos artigos, fichas no vault, baseline de decomposição e reforço metodológico do manuscrito
- [[12_Auditoria/Sprint 5.6 - Correcoes pos-avaliacao e piloto Fase B - 2026-07-12]] — ajustes pós-avaliação externa, checagem NBR 6118 e primeiro piloto de packing/layout 5N
- [[12_Auditoria/Sprint 5.7 - Submission polish artigo 1 - 2026-07-12]] — polish editorial de pré-submissão, fechamento de declarações e validação LaTeX sem avisos críticos

## 📚 Artigos (preencher)

- [[08_Artigos/Index de Artigos]]

## 🧰 Templates (use ao criar notas novas)

- [[99_Templates/Template - Artigo]] — para entradas em [[08_Artigos/Index de Artigos]]
- [[99_Templates/Template - Conceito]] — para novas notas em `02_Engenharia/` ou `03_Otimizacao/`
- [[99_Templates/Template - Issue]] — para novas entradas em [[07_Issues/Lista Mestre de Issues]]

---

> [!tip] Ponto de partida recomendado
> Sequência de leitura sugerida para quem chega ao projeto: [[01_Projeto/Visão Geral do Projeto]] → [[03_Otimizacao/Formulação do Problema]] → [[04_Codigo/fundacao.py]] → [[04_Codigo/metapy_toolbox - ego.py]] → [[10_Melhorias/Guia - Validação antes do Bin Packing]] → [[12_Auditoria/Auditoria 2026-04-27 - Vault vs Projeto]] → [[07_Issues/Lista Mestre de Issues]].
