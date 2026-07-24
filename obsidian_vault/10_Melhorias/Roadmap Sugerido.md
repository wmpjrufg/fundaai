---
tags: [melhorias, roadmap, sugestao]
---

# Roadmap Sugerido — Passo a Passo

> [!note] Apenas sugestão
> Sequência priorizada por relação **esforço × impacto × risco**. Cada etapa entrega valor mesmo se a próxima não acontecer.

> [!important] Decisão de escopo atual
> A trilha ativa é validar o FundaIA/EGO-GPR antes de iniciar o bin packing completo. Use [[10_Melhorias/Guia - Validação antes do Bin Packing]] como checklist operacional e [[09_Relatorios/Analise - Roadmap Artigo IC - 2026-04-27]] como justificativa detalhada.

## Fase 0 — Saneamento (1–2 dias)

> Antes de qualquer pesquisa nova, deixar a base funcionando bem.
> Esta fase foi **revisada após [[12_Auditoria/Auditoria 2026-04-27 - Vault vs Projeto]]** com itens novos.
> **Sprint 0**: itens 1–5 concluídos em 2026-04-27 (`fix/code-sanitization-and-tests`).
> **Sprint 1**: itens 6 e 7 concluídos em 2026-04-27 (`fix/code-sanitization-and-tests`).
> **Sprint 2**: itens 8 e 10 concluídos + suite `pytest` adicionada em 2026-04-27 (`fix/code-sanitization-and-tests`).

1. ✅ [[07_Issues/Issue - requirements.txt UTF-16]] — recriado em UTF-8 + 5 deps adicionadas. *(Sprint 0)*
2. ✅ [[07_Issues/Issue - Duplicação em sapatas.py]] — bloco 326–531 removido (531 → 325 linhas). *(Sprint 0)*
3. ✅ [[07_Issues/Issue - methods.py morto]] — arquivo deletado, import limpo. *(Sprint 0)*
4. ✅ [[07_Issues/Issue - obj_felipe_lucas vs obj_teste]] — fundidas via `_avaliar_projeto`. *(Sprint 0)*
5. ✅ [[07_Issues/Issue - Args extras em obj_teste]] — penalty parametrizável (default 10). **Pendente: revalidar gráficos/tabelas existentes em `assets/graphics/` e `assets/tables/` à luz dessa decisão.** *(Sprint 0)*
6. ✅ [[07_Issues/Issue - Histórico do EGO com ITER e ID incorretos]] — `ITER=t` e `ID=max(ID)+1`; bônus: parâmetro `seed` no EGO. *(Sprint 1)*
7. ✅ [[07_Issues/Issue - n_rep reusa população inicial]] — `initial_population_01` dentro do laço, com `seed = base_seed + rep`. *(Sprint 1)*
8. ✅ [[07_Issues/Issue - Notebooks com paths quebrados]] — `assets/el08.xlsx` substituído por `assets/problema_fund_três.xlsx` em ambos. *(Sprint 2)*
9. ❓ Decidir intenção em [[07_Issues/Issue - Sobreposição contada duas vezes]] e documentar — **decisão pendente com orientador**.
10. ✅ [[07_Issues/Issue - Benchmarks suspeitos]] — `griewank` e `powell` corrigidos contra Surjanovic & Bingham + 7 testes regressivos. *(Sprint 2)*
11. ❓ Decisão oficial sobre **20 vs 21 kernels** ([[03_Otimizacao/Kernels GPR]]) — **decisão pendente com orientador**.
12. ⏳ Auditoria de branches ([[07_Issues/Issue - Branches dispersos]]).

**Bônus Sprint 2**: suite `tests/` com 55 testes pytest cobrindo engenharia, regressão numérica (`of = 19,70604234767181` travado), contrato do EGO e benchmarks. Arquivo `pytest.ini` configurado com markers `engineering`, `regression`, `optimization`, `benchmark`.

**Saída**: app limpo, instalável em uma máquina nova com `pip install -r requirements.txt`.

## Fase 1 — Refatoração estrutural (Sprint 3, em andamento)

> Habilita evolução. Mantém comportamento (`pytest` verde a cada commit).
> **Sprint 3 — em andamento na branch `refactor/core-architecture`**.

1. ✅ **Sprint 3.1** — Skeleton `core/` (`domain`, `engineering`, `optimization`, `io`, `api`) + `ARCHITECTURE.md`. *(2026-04-27)*
2. ✅ **Sprint 3.2** — 6 funções de engenharia migradas para `core/engineering/{solo,tensao,geometria,puncao,packing}.py` com shim em `fundacao.py`. *(2026-04-28)*
3. ✅ **Sprint 3.3** — Entidades de domínio (`Solo`, `Pilar`, `Combinacao`, `Sapata`, `FundacaoProjeto`) em `core/domain/` + 15 testes. *(2026-04-28)*
4. ✅ **Sprint 3.4** — IO layer (`core/io/excel.py`, `core/io/cad_dxf.py`) + 21 testes (entry point validado). *(2026-04-28)*
5. ✅ **Sprint 3.5** — API layer (`core/api/{evaluate,optimize,types}`) + `pages/sapatas.py` virou shell fino + 22 testes; baseline `of = 19,70604234767181` agora reproduzido end-to-end. *(2026-04-28)*
6. ✅ **Sprint 3.6** — `metapy_toolbox/` → `core/optimization/` (5 módulos via `git mv`); shim de compat preservado. *(2026-04-28)*
7. ✅ **Sprint 3.7** — `OptimisationConfig` reescrita em Pydantic v2 (validação rica + JSON schema); 117 testes verdes. *(2026-04-28)*
8. ✅ **Sprint 3.8** — Verificação de sobreposição vetorizada (matriz N×N numpy via `sobreposicao_matrix`); baseline preservado bit-a-bit; 122 testes verdes. *(2026-04-28)*
9. ⏳ **Em paralelo (qualquer sprint)** — [[10_Melhorias/Logging Estruturado]] e [[10_Melhorias/Reprodutibilidade - Seeds e Versão]].

**Saída**: `pytest` verde; app continua funcionando idêntico ao usuário.

## Fase 2 — Performance e robustez (1–2 semanas)

1. ✅ [[10_Melhorias/Refactor - Vetorização da FO]] — entregue na Sprint 3.8 (matriz N×N numpy, ~100× speedup). *(2026-04-28)*
2. ✅ [[10_Melhorias/Cache de Surrogate]] — entregue na Sprint 4.1 (`SurrogateCache` em `core/optimization/cache.py`; opt-in via `ego_01_architecture(..., cache=...)`). *(2026-04-28)*
3. ✅ [[10_Melhorias/Refactor - Configuração com Pydantic]] — entregue na Sprint 3.7 (`OptimisationConfig` em Pydantic v2). *(2026-04-28)*
4. ✅ [[10_Melhorias/Persistência de Experimentos]] — entregue na Sprint 4.2 (pasta `experiments/<run_id>/` com manifest+config+env+project+history Parquet+summary CSV+metrics JSON+artifacts/, `schema_version="1.0"`, escritas atômicas, round-trip via `load_experiment`). *(2026-04-28)*

**Saída**: rodar otimizações com `n_pop=2000` em tempo aceitável; experimentos rastreáveis.

## Fase 3 — Ganhos algorítmicos (2–4 semanas)

1. [[10_Melhorias/Penalização Adaptativa]] — fator crescente ao longo das iterações.
2. [[10_Melhorias/Tratamento de Restrições - Deb e Augmented Lagrangian]] — comparar contra penalização atual.
3. [[10_Melhorias/Acquisition Functions Modernas]] — UCB, PI, qEI (batch), CB-MOEA.
4. [[10_Melhorias/Hibridização Memética]] — GA da `mealpy` + busca local SLSQP no melhor agente.

**Saída**: tabela comparativa de algoritmos × benchmarks (ver [[10_Melhorias/Validação contra problema-benchmark]]).

## Fase 4 — Frentes de pesquisa (semestre)

1. [[11_Frentes_de_Pesquisa/Physics-Informed Surrogates]] — frente prioritária de pesquisa.
2. [[11_Frentes_de_Pesquisa/Posicionamento Conjunto - Layout + Sizing]].
3. [[11_Frentes_de_Pesquisa/Otimização Multi-objetivo]].
4. [[11_Frentes_de_Pesquisa/Surrogate Multifidelidade]].

**Saída**: contribuição original publicável.

## Trilha enxuta (foco mínimo até a pesquisa)

Caso o objetivo seja o **caminho mais curto até a frente de pesquisa**:

```
Fase 0 → Fase 1 (apenas POO + testes) → Fase 4 (Physics-Informed)
```

Pulando Fases 2 e 3, com registro em [[01_Projeto/Pipeline de Execução]] de que ficaram pendentes.

## Trilha atual — artigo antes do bin packing

Esta é a trilha recomendada para o momento atual:

```
Saneamento P0 → Validação de engenharia → Validação experimental EGO-GPR → Artigo 1 → Frente bin packing/layout
```

Critério para avançar ao bin packing: resultados atuais reproduzíveis, escopo do artigo 1 aprovado e pendências críticas de penalidade, seeds, histórico do EGO e validações mínimas resolvidas ou declaradas como limites.

## Vínculos

- [[10_Melhorias/Guia - Validação antes do Bin Packing]]
- [[09_Relatorios/Analise - Roadmap Artigo IC - 2026-04-27]]
- [[10_Melhorias/MOC - Melhorias]]
- [[11_Frentes_de_Pesquisa/MOC - Frentes de Pesquisa]]
