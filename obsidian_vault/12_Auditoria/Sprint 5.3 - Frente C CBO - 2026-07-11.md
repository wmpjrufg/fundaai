---
tags: [auditoria, sprint, cbo, bayesian-optimization, restricoes, frente-c, artigo]
data: 2026-07-11
status: concluido
---

# Sprint 5.3 — Frente C: Constrained Bayesian Optimization (2026-07-11)

Avanço metodológico do otimizador (opção C do plano combinado com o Lucas), executado de ponta a ponta: fontes, implementação, integração à bancada, protocolo completo com seeds, artigo, artefatos e preparação do terreno da Fase B. Iniciada pelo Fable 5, finalizada pelo Opus 4.8 (mesma linha de raciocínio).

## 0. Motivação (por que CBO, e por que agora)

Resposta metodológica direta aos DOIS achados empíricos do protocolo final (Sprint 5.1):
1. **α=10⁶ preserva o R² global do surrogate mas explode o RMSE na região factível em 5 ordens de grandeza** — o GPR gasta capacidade aprendendo o "penhasco" artificial da penalização exatamente onde a aquisição decide.
2. **A penalização linear α=10 admite violações residuais** (10⁻³–10⁻¹) nas soluções finais de todos os métodos — o ótimo penalizado é intrinsecamente deslocado para fora da região viável.

A CBO (Gardner et al., 2014) remove a penalização de TODOS os alvos de regressão: um GP para o volume + um GP por grupo de restrição, e a aquisição incorpora a probabilidade de factibilidade. É também a máquina de restrições que a Fase B (packing, sobreposição rígida e ativa) vai exigir.

## 1. Fontes (baixadas antes de citar — regra do projeto)

Nova pasta `docs/articles/05_frente_c_cbo/`:
- **Gardner, Kusner, Xu, Weinberger, Cunningham (2014)** — "Bayesian Optimization with Inequality Constraints", ICML/PMLR v32. **Fonte primária** da aquisição implementada. Ficha [[08_Artigos/Gardner et al. 2014 - Bayesian Optimization with Inequality Constraints]].
- **Eriksson & Poloczek (2021)** — SCBO (AISTATS/PMLR v130), variante escalável com trust regions. Para a Fase B (dim alta). Ficha [[08_Artigos/Eriksson e Poloczek 2021 - Scalable Constrained BO]]. **Correção de autoria**: a nota antiga citava "Eriksson & Jankowiak" — o correto é Poloczek.
- Schonlau et al. (1998) é a origem histórica do ECI, mas o PDF não foi obtido (Euclid bloqueou o download) → **não citado no artigo**; a citação do corpo é Gardner (2014).

`.bib`: `gardner2014bayesian`, `eriksson2021scbo`. Mapa `docs/articles/README.md` e [[08_Artigos/Index de Artigos]] atualizados.

## 2. Implementação

### FO por componentes (paridade bit a bit)
`core/api/objective.py`: extraí o núcleo numérico para `_nucleo_componentes(x, args)`, compartilhado por:
- `avaliar_projeto_fast` (expressão final inalterada → **Θ bit-idêntico por construção**);
- `avaliar_projeto_componentes(x, args) → (θ, volume, g[4])` (novo), com `g` = pior valor por grupo (sob/pun/ten/geo) sobre elementos e combinações.

Teste de regressão: `θ == avaliar_projeto_fast` exato no baseline **19.70604234767181** e em 200 pontos aleatórios. **Baseline intacto.**

### Motor CBO
`core/optimization/cbo.py::cbo_01_architecture` — interface espelha `ego_01_architecture`. A cada iteração: GP do volume + GP por grupo de restrição (pipeline de produção StandardScaler+GPR normalize_y, alpha=0.1); aquisição `ECI = EI(V | melhor factível observado) · Π Φ(−μ_k/σ_k)`; **fase só-PoF quando nenhum ponto factível foi observado** (Gardner §3.2); grupos de variância nula (sobreposição nos casos congelados) → `_ConstantConstraint` determinístico (prob 0/1). Histórico compatível com o EGO + colunas VOLUME e G_SOB..G_GEO. `OF` guarda Θ penalizado (métrica de comparação idêntica à dos demais).

### Bancada
`core/api/benchmark.py`: algoritmo `cbo` sob as MESMAS alavancas do EGO (`_TracedComponents` mantém o trace de Θ; `cbo_constraint_restarts` no config; orçamento = `ego_budget_evals`). Paleta de 6 algoritmos (`ALGORITHM_LABELS`).

### Testes
`tests/test_cbo.py` (17): EI/PoF calculados à mão, restrição constante degenerada, contrato da arquitetura (orçamento/reprodutibilidade), integração à bancada. Suite: **264 testes verdes**.

## 3. Protocolo completo (3 casos × 30 seeds pareadas 42–71)

Script `scripts/run_cbo_benchmark.py` (reusa as alavancas congeladas de `run_final_benchmark.py`), 133 min, persistido em `experiments/protocolo_final/<caso>/S1_cbo/`.

| Caso (dim) | CBO melhor Θ | CBO média±DP | EGO média±DP | p (pareado) | Fact. CBO/EGO | V_feas CBO/EGO |
|---|---|---|---|---|---|---|
| 1 (3) | 3,880 | 3,942±0,075 | 3,944±0,049 | 0,206 (empate) | 50% / 80% | 3,894 / 3,882 |
| 2 (6) | 6,432 | 6,924±0,398 | 6,883±0,360 | 0,900 (empate) | 23% / 53% | 6,432 / 6,158 |
| 3 (9) | **3,066** | **3,426±0,217** | 4,238±0,502 | **5,6e-10** | **83% / 50%** | **3,066 / 3,533** |

**Leitura**: o ganho do tratamento explícito CRESCE COM A DIMENSÃO. Em dim baixa é empate (e a factibilidade estrita do CBO até piora — no caso 2 o ótimo encosta na fronteira de tensão e, com base curta, o GP da restrição suaviza o cruzamento do zero e a PoF superestima a margem). Em dim 9 o CBO domina com folga esmagadora e entrega o melhor volume factível de todo o estudo (a 12,7% do teto prático que o GWO só alcança com 20× mais avaliações). Custo: 1,5–1,6× o tempo de parede do EGO (5 surrogates/iteração).

**Interpretação mecânica** (no artigo, §7): penalização = erro DE FORMULAÇÃO (ótimo deslocado, independe da dim); CBO = erro DE MODELO (suavização do GP, amortiza quando a dim cresce e a paisagem penalizada fica difícil para um único surrogate). Isso é a ponte direta para a Fase B.

## 4. Artigo

- **Metodologia §4.5 nova**: "Tratamento explícito de restrições" com equações ECI e PoF, citações Gardner/Eriksson, 3 decisões de implementação declaradas.
- **Resultados §6.1/6.4/6.6**: CBO integrado ao protocolo; 4º padrão (ganho×dimensão) com os números; factibilidade recontextualizada.
- **Discussão §7 novo parágrafo**: "Tratamento explícito de restrições: quando compensa" (erro de formulação × erro de modelo).
- **Conclusões §8**: novo item consolidado; "Frente 2" reescrita de "investigar CBO" → "aprofundar CBO já implementada" (SCBO, aquisições com risco de infactibilidade).
- **Resumos PT/EN + intro**: frase do CBO e do resultado dim-9.
- **Figuras/tabelas** regeneradas por `make_paper_artifacts.py` com 6 algoritmos (paleta validada, CBO=vermelho #e34948, pior par CVD ΔE 21,2 PASS); Tabela de p-valores virou `table*` (6 colunas). **Compila: 21 páginas, 0 erros, 0 refs indefinidas, 0 overfull>10pt.**

## 5. Terreno da Fase B preparado

[[11_Frentes_de_Pesquisa/Fase B - Kickoff Packing + Sizing - 2026-07-11]]: formulação `(hx,hy,hz,dx,dy)` com excentricidade→momentos efetivos, restrições novas (lote, margem via a mesma matriz AABB), plano experimental espelhando o protocolo, o que a Frente C já deixou pronto (FO por componentes estende para g[5..6]; a fase só-PoF do CBO é exatamente o que packing precisa quando o LHS inicial é ~todo infactível), riscos, bibliografia a adquirir e 7 decisões para a orientadora.

## 6. Arquivos tocados

- `core/api/objective.py` (núcleo componentes + `avaliar_projeto_componentes`), `core/optimization/cbo.py` (novo), `core/optimization/__init__.py`, `core/api/benchmark.py` (algoritmo cbo + `_TracedComponents`), `tests/test_cbo.py` (novo), `tests/test_avaliar_projeto.py` (paridade) — **commits `1dff82c2`, `47c19f62`** (sessão Fable).
- `scripts/run_cbo_benchmark.py` (novo), `scripts/make_paper_artifacts.py` (fusão CBO, paleta 6, pvalues table*) — **a commitar na sessão Opus**.
- `docs/artigo_ic_lucas/` (gitignored): metodologia, resultados, discussão, conclusões, resumos, `.bib`, figuras, tabelas.
- Vault: esta nota, [[11_Frentes_de_Pesquisa/Bayesian Optimization Constrained]], fichas dos 2 artigos, [[11_Frentes_de_Pesquisa/Fase B - Kickoff Packing + Sizing - 2026-07-11]].

## Pendências que permanecem

- [ ] Nspt/30-40-50: fonte citável ainda precisa ser decidida com a orientadora. Os coeficientes 1,05/1,30 foram removidos na Sprint 5.4 e não fazem parte da metodologia atual.
- [ ] Revista-alvo + tradução para inglês.
- [ ] Fase B: 7 decisões com a orientadora antes de codificar.

## Vínculos

- [[12_Auditoria/Sprint 5.2 - Puncao C linha e duas colunas - 2026-07-10]]
- [[11_Frentes_de_Pesquisa/Bayesian Optimization Constrained]]
- [[11_Frentes_de_Pesquisa/Fase B - Kickoff Packing + Sizing - 2026-07-11]]
- [[10_Melhorias/Questao Aberta - Custo da FO e Justificativa do EGO]]
- [[08_Artigos/Gardner et al. 2014 - Bayesian Optimization with Inequality Constraints]]
