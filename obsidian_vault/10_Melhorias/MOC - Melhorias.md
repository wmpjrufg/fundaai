---
tags: [moc, melhorias, sugestao]
---

# 🛠️ MOC — Melhorias (Sugestões)

> [!warning] Sugestões, não execução
> Tudo aqui é **proposta**. Nada deve ser implementado sem aprovação do orientador (Prof. Wanderley) e da equipe do projeto.
> O foco é registrar **caminhos coerentes com o escopo da IC** ([[01_Projeto/Escopo da IC]]) — uso de metaheurísticas / hibridizações para o problema acoplado de dimensionamento + posicionamento de fundações rasas.

> [!info] Distinção importante
> Este MOC trata de **melhorias/refatorações**. Bugs e débito técnico
> historicamente identificados ficam em [[07_Issues/Lista Mestre de Issues]];
> os logs detalhados de cada sprint ficam em `12_Auditoria/`.
> As Sprints 0/1/2 (correções e testes, branch `fix/code-sanitization-and-tests`)
> estão lá. A Sprint 3 (refactor estrutural propriamente dita,
> branch `refactor/core-architecture`) é a primeira a ser tratada
> diretamente sob este MOC.

> [!success] Sprint 3.1 — Skeleton da arquitetura `core/` (2026-04-27, branch `refactor/core-architecture`)
> Criada a estrutura de pacotes `core/{domain,engineering,optimization,io,api}`
> com `__init__.py` documentados. `ARCHITECTURE.md` na raiz descreve a
> arquitetura-alvo, regras de dependência e plano de migração das
> sub-sprints 3.2–3.8. Nenhum código de produção movido nesta etapa;
> `pytest` continua verde (55 testes).
> Detalhes em [[12_Auditoria/Sprint 3 - Refactor estrutural - kickoff - 2026-04-27]].

> [!success] Sprint 3.2 — Engineering migration (2026-04-28, branch `refactor/core-architecture`)
> 6 funções analíticas puras (`tensao_adm_solo`, `calcular_sigma_max_min`,
> `checagem_tensao_max_min`, `checagem_geometria`, `verificacao_puncao_sapata`,
> `sobreposicao_sapatas`) movidas de `fundacao.py` para
> `core/engineering/{solo,tensao,geometria,puncao,packing}.py`.
> `fundacao.py` mantido como camada de compatibilidade (re-exports).
> Comportamento numérico inalterado; `pytest` verde (55 testes).
> Detalhes em [[12_Auditoria/Sprint 3.2 - Engineering migration - 2026-04-28]].

> [!success] Sprint 3.3 — Domain entities (2026-04-28, branch `refactor/core-architecture`)
> 5 entidades de negócio (`Solo`, `Pilar`, `Combinacao`, `Sapata`,
> `FundacaoProjeto`) introduzidas em `core/domain/` como dataclasses
> puras (`frozen` exceto `Sapata`, que é mutável por ser variável de
> projeto). 15 testes unitários novos cobrem invariantes e helpers.
> Comportamento existente intacto (entidades ainda não consumidas pelo
> `_avaliar_projeto`); `pytest` verde (**70 testes** ao todo).
> Detalhes em [[12_Auditoria/Sprint 3.3 - Domain entities - 2026-04-28]].

> [!success] Sprint 3.4 — IO layer (2026-04-28, branch `refactor/core-architecture`)
> Camada de IO introduzida em `core/io/` com `read_projeto_from_excel`
> (entrada única do projeto, schema rigorosamente validado, sanitização
> de vírgula decimal, mensagens de erro claras) e `sapatas_to_dxf_bytes`
> (escritor DXF em memória, sem `tempfile` órfão). 21 testes novos
> cobrem round-trip dos 3 templates oficiais, validação de schema,
> integridade de domínio, sanitização e estabilidade semântica do DXF.
> `pytest` verde (**91 testes** ao todo).
> Detalhes em [[12_Auditoria/Sprint 3.4 - IO layer - 2026-04-28]].

> [!success] Sprint 3.5 — API layer + shell Streamlit (2026-04-28, branch `refactor/core-architecture`)
> Camada API em `core/api/` com `evaluate(projeto, sapatas)` e
> `optimize(projeto, config)` (puras, framework-free). `pages/sapatas.py`
> migrada para shell fino que apenas chama `read_projeto_from_excel`,
> `optimize` e `sapatas_to_dxf_bytes`. Issue
> [[07_Issues/Issue - DXF tempfile não removido]] resolvida.
> 22 testes novos validam adapter, config, evaluate (regressão
> `of = 19,70604234767181` reproduzida via API end-to-end) e o
> plumbing de `optimize`. `pytest` verde (**113 testes** ao todo).
> Detalhes em [[12_Auditoria/Sprint 3.5 - API layer - 2026-04-28]].

> [!success] Sprint 3.6 — Optimization migration (2026-04-28, branch `refactor/core-architecture`)
> Os 5 módulos de `metapy_toolbox/` foram movidos para `core/optimization/`
> via `git mv` (histórico preservado). Imports internos reescritos para
> `from core.optimization import funcs`. `metapy_toolbox/__init__.py`
> virou shim de compatibilidade (`from core.optimization import *`),
> garantindo que notebooks legados (`testes_otm.ipynb`,
> `testes_otm_lucas.ipynb`, `testes_gpr_lucas.ipynb`) continuem
> funcionando sem alteração. Comportamento numérico inalterado;
> `pytest` verde (**113 testes** ao todo).
> Detalhes em [[12_Auditoria/Sprint 3.6 - Optimization migration - 2026-04-28]].

> [!success] Sprint 3.7 — Pydantic config (2026-04-28, branch `refactor/core-architecture`)
> `OptimisationConfig` reescrita como `pydantic.BaseModel` (v2): cada
> campo com `Field(...)` rico (descrição, `gt`, `ge`), validador
> cross-field para `h_min_m < h_max_m`, `extra="forbid"` para pegar
> typos, `frozen=True` para imutabilidade. JSON Schema gerado
> automaticamente; serialização round-trip via `model_dump()`.
> `OptimisationResult` e `EvaluationResult` mantidos como dataclasses
> (não são input externo). Compatibilidade preservada
> (`pydantic.ValidationError` herda de `ValueError`). 4 testes novos
> em `tests/test_api.py`; `pytest` verde (**117 testes** ao todo).
> Detalhes em [[12_Auditoria/Sprint 3.7 - Pydantic config - 2026-04-28]].

> [!success] Sprint 3.8 — Vectorized FO (2026-04-28, branch `refactor/core-architecture`)
> O laço duplo `df.iterrows()` da verificação de sobreposição em
> `_avaliar_projeto` foi substituído por uma matriz N×N inteiramente
> em numpy via nova função `sobreposicao_matrix(xmin, xmax, ymin, ymax)`
> em `core/engineering/packing.py`. Cálculo bit-exato vs. versão escalar
> (`atol=0, rtol=0`); 100× mais rápido a N=200; baseline
> `of = 19,70604234767181` intacto. 5 testes novos em
> `tests/test_engenharia.py`; `pytest` verde (**122 testes** ao todo).
> Detalhes em [[12_Auditoria/Sprint 3.8 - Vectorized FO - 2026-04-28]].

> [!success] Sprint 4.1 — Surrogate cache (2026-04-28, branch `refactor/core-architecture`)
> Cache LRU (em memória + disco opcional via joblib) do modelo GPR
> usado pelo EGO. Chave = SHA-256 de `(X bytes, y bytes, assinatura
> do pipeline)`. Novo módulo `core/optimization/cache.py` com
> `SurrogateCache`, `pipeline_signature`, `fingerprint` e
> `fit_or_get_cached`. `ego_01_architecture` ganha parâmetro opcional
> `cache=None` (default mantém comportamento histórico bit-a-bit).
> Em duas execuções idênticas a segunda só dá hits → ~1,4× em caso
> realista (Matern ν=2.5, d=6, n_pop=60); ganho cresce com n_rep e
> com a complexidade do kernel. 23 testes novos em
> `tests/test_cache.py`; `pytest` verde (**145 testes** ao todo).
> Detalhes em [[12_Auditoria/Sprint 4.1 - Surrogate cache - 2026-04-28]].

> [!success] Sprint 4.3 — Reorg + docs (2026-04-28, branch `refactor/core-architecture`)
> Reorganização final do repositório: `pages/` → `frontend/{pages,components,i18n}/`,
> `ops/` → `scripts/` (`env_setup.py` movido), `old/` → `archive/` (com README
> dirigente), 4 notebooks consolidados em `notebooks/` com bootstrap cell, 3
> planilhas oficiais consolidadas em `assets/data/`, shim `metapy_toolbox`
> apagado (6 sites de import reescritos para `core.optimization`).
> Circular `core.io ↔ core.api` resolvido via `TYPE_CHECKING`. `README.md`
> reescrito do zero (470+ linhas) com pipeline atualizado, árvore de pastas
> com situação atual, blocos para uso programático, persistência de
> experimentos e cache do surrogate. `ARCHITECTURE.md` reescrito com
> diagrama de dependências novo, histórico de sprints 0→4.3 e deprecation
> tracks explícitos. `pytest` verde (**162 testes**). Detalhes em
> [[12_Auditoria/Sprint 4.3 - Reorg + docs - 2026-04-28]].

> [!success] Sprint 4.2 — Experiment persistence (2026-04-28, branch `refactor/core-architecture`)
> Persistência completa por run: pasta `experiments/<run_id>/` com
> `manifest.json`, `config.json` (Pydantic round-trip), `env.json`
> (Python+libs+git), `project.json` (hash+sumário do `FundacaoProjeto`),
> `history/rep_NNN.parquet` (DataFrame inteiro por repetição),
> `summary.csv` ergonômico, `metrics.json` paper-grade
> (`best/mean/std/median`, `convergence_iter`, `auc_best_so_far`,
> `improvement_rel`, `wall_time_*`) e `artifacts/` para DXF/plots.
> Novo módulo `core/io/experiments.py` com `ExperimentRecorder`,
> `ExperimentRun`, `summarise_history`, `compute_metrics` e
> `load_experiment`. `optimize()` ganha `recorder=None` (e `cache=None`)
> opt-in; default mantém comportamento histórico. Schema versionado
> (`SCHEMA_VERSION = "1.0"`); escritas atômicas (temp+rename).
> 17 testes novos em `tests/test_experiments.py`; `pytest` verde
> (**162 testes** ao todo).
> Detalhes em [[12_Auditoria/Sprint 4.2 - Experiment persistence - 2026-04-28]].

## 🧱 Refatoração de código (qualidade)

- [[10_Melhorias/Refactor - Plano Geral]] — ✅ arquitetura-alvo `core/{domain,engineering,optimization,io,api}` materializada nas Sprints 3.1–3.8.
- [[10_Melhorias/Refactor - POO Domain Model]] — ✅ entregue na Sprint 3.3 (`Solo`, `Pilar`, `Combinacao`, `Sapata`, `FundacaoProjeto`).
- [[10_Melhorias/Refactor - Separar UI de Domínio]] — ✅ entregue nas Sprints 3.4 (IO) + 3.5 (`pages/sapatas.py` virou shell fino).
- [[10_Melhorias/Refactor - Vetorização da FO]] — ✅ entregue na Sprint 3.8 (`sobreposicao_matrix` N×N, ~100× speedup, baseline bit-exato).
- [[10_Melhorias/Refactor - Configuração com Pydantic]] — ✅ entregue na Sprint 3.7 (`OptimisationConfig` em Pydantic v2 com JSON Schema).
- [[10_Melhorias/Refactor - Empacotar metapy_toolbox]] — ✅ entregue na Sprint 3.6 (movido para `core/optimization/`, shim de compat preservado).
- [[10_Melhorias/Testes Automatizados]] — ✅ entregue como rede de segurança na Sprint 2; expandida nas Sprints 3.3–3.8 (**122 testes** atualmente).
- [[10_Melhorias/Logging Estruturado]] — ✅ entregue na Sprint 4.4 (`core/observability/` com JSON-line logger, `run_context`, eventos nomeados em `optimize`/`ego`/`cache`/`experiments`).
- [[10_Melhorias/Reprodutibilidade - Seeds e Versão]] — 🟡 parcialmente entregue na Sprint 1 (parâmetro `seed` em `ego_01_architecture`); falta versionamento de runs.

## 🚀 Engenharia de software / DevEx

- [[10_Melhorias/Higiene - requirements e venv]] — ✅ entregue na Sprint 0 (`requirements.txt` recriado em UTF-8 + 5 deps adicionadas; `pydantic` adicionado na Sprint 3.7).
- [[10_Melhorias/CI-CD - Lint Test Build]] — ⏳ pendente; suíte `pytest` (122 testes) já está pronta para ser plugada num workflow.
- [[10_Melhorias/Cache de Surrogate]] — ✅ entregue na Sprint 4.1 (`SurrogateCache` LRU + joblib em `core/optimization/cache.py`; opt-in via parâmetro `cache=` do `ego_01_architecture`).
- [[10_Melhorias/Persistência de Experimentos]] — ✅ entregue na Sprint 4.2 (`ExperimentRecorder`+`load_experiment` em `core/io/experiments.py`; manifest+config+env+project+history Parquet+summary CSV+metrics JSON+artifacts/).

## 🧮 Otimização (ganhos algorítmicos)

- [[10_Melhorias/Penalização Adaptativa]] — ⏳ pendente (Fase 3 do Roadmap).
- [[10_Melhorias/Tratamento de Restrições - Deb e Augmented Lagrangian]] — ⏳ pendente (Fase 3 do Roadmap).
- [[10_Melhorias/Acquisition Functions Modernas]] — ⏳ pendente (Fase 3 do Roadmap).
- [[10_Melhorias/Hibridização Memética]] — ⏳ pendente (Fase 3 do Roadmap).
- [[10_Melhorias/Multi-Objetivo - Volume vs Custo vs Reuso]] — ⏳ pendente (frente de pesquisa).
- [[10_Melhorias/Variáveis Discretas - Família de Sapatas]] — ⏳ pendente (frente de pesquisa).
- [[10_Melhorias/Posicionamento como Variável de Projeto]] — ⏳ pendente (frente de pesquisa, ligada ao bin packing).

## 🏗️ Engenharia (precisão normativa)

- [[10_Melhorias/Punção Seção C linha - completar]] — ✅ implementada na Sprint 5.2; manter como registro histórico.
- [[10_Melhorias/Métodos modernos de capacidade do solo]] — ⏳ pendente.
- [[10_Melhorias/Validação contra problema-benchmark]] — ⏳ pendente (parte da trilha "Validação antes do Bin Packing").

## ⚙️ Passo a passo recomendado

- [[10_Melhorias/Guia - Validação antes do Bin Packing]] — 🟡 trilha atual em andamento; refactor estrutural já concluído (Sprint 3.1–3.8), faltam etapas de validação experimental.
- [[10_Melhorias/Roadmap Sugerido]] — 🟡 Fases 0 e 1 concluídas; Fase 2 (Cache + Persistência) é a próxima.
- [[09_Relatorios/Analise - Roadmap Artigo IC - 2026-04-27]] — relatório completo que justifica a decisão de validar primeiro.

## Links

- [[07_Issues/Lista Mestre de Issues]] — bugs/débito existente (Sprints 0/1/2 fechadas lá).
- [[11_Frentes_de_Pesquisa/MOC - Frentes de Pesquisa]] — onde levar o projeto cientificamente.
- [[01_Projeto/Escopo da IC]] — alvo a manter.
