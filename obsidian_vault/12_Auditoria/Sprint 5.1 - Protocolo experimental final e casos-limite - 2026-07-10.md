---
tags: [auditoria, sprint, protocolo, artigo, ego, gpr, benchmark]
data: 2026-07-10
status: concluido
---

# Sprint 5.1 — Protocolo experimental final e casos-limite (2026-07-10)

Sprint executada para fechar o artigo 1: auditoria dos casos-limite da formulação, correções com testes, baseline de busca aleatória + métricas de factibilidade na bancada, execução do protocolo experimental final com seeds, estudo GPR controlado e regeneração completa das figuras/tabelas do manuscrito.

## 1. Auditoria dos 5 pontos de atenção (veredito)

| Ponto | Veredito | Ação implementada |
|---|---|---|
| `Fz = 0` divide por zero em `6·M/(Fz·h)` | Risco real de fronteira (sem validação em `Combinacao`/Excel) | `Combinacao.__post_init__` exige `f_z > 0` com mensagem explícita (cargas nulas/tração fora do escopo declarado) |
| `hz ≤ cob` → `d ≤ 0` inverte o sinal da punção (parece viável) | Risco real de configuração (nada cruzava `h_min_m` × `cobrimento_m`) | Validação de entrada em `optimize()` e `run_benchmark()` (`h_min_m > cobrimento_m`) + guarda `d > 0` em `verificacao_puncao_sapata` e em `avaliar_projeto_fast` (fast e legacy falham identicamente) |
| Penalidade linear α=10, p=1 | Escolha de projeto, não bug | Declarada no artigo (metodologia); consequências **quantificadas** (ver §4 abaixo); factibilidade estrita agora reportada pela bancada |
| `f_ck` em kPa dividido por 1000 em `α_v2` | Risco real silencioso (25 MPa como `25` → τ_rd2 ≈ 4,8 kPa) | Range de plausibilidade em `FundacaoProjeto`: `10_000 ≤ f_ck_kpa ≤ 90_000` (C10–C90) com mensagem que explica a unidade |
| Punção só na seção C | Limitação declarada (docstring, artigo, vault) | Mantida como limite formal; C′ segue para frente futura |

Testes novos cobrindo cada guarda; suite **249 testes verdes**; baseline `of = 19.70604234767181` preservado bit a bit. Bônus: teste desatualizado `test_budget_is_respected_per_rep` corrigido para o contrato de orçamento separado (`ego_budget_evals`).

## 2. Bancada: baseline aleatório + factibilidade

- Novo algoritmo `random` em `core.api.benchmark` (amostragem uniforme sob `TracedObjective`, mesma seed/orçamento) — é o "Monte Carlo" do artigo, agora controlado.
- Pós-processamento da melhor solução de cada repetição via avaliação anotada: `volume_m3`, `feasible` (tol `g_k ≤ 1e-9`), `max_violation` e violações por grupo (`viol_sob/pun/ten/geo`).
- `BenchmarkResult.per_rep` exposto; `summary` ganhou `feasibility_rate`, `best_feasible_volume_m3`, `mean_max_violation`.

## 3. Protocolo experimental final (congelado)

Script: `scripts/run_final_benchmark.py` → `experiments/protocolo_final/<caso>/<cenario>/` (history.parquet, per_rep.csv, summary.csv, pvalues.csv, config.json, meta.json com git rev + versões).

- Casos: um (P08, dim 3), dois (P01–P02, dim 6), três (P04/P05/P16, dim 9); f_ck 25 MPa; cob 0,04 m; **h ∈ [0,60; 3,00] m** (3,00 é obrigatório: P08 tem ap=2,10 → g_geo exige hx ≥ 2,30; com h_max=1,50 o caso 1 é infactível por construção — bug de protocolo detectado e corrigido no piloto).
- **S1 (orçamento igual)**: EGO/GA/PSO/GWO/aleatória, 150 avaliações reais/rep, 30 reps, seeds 42–71, LHS 10·dim, AG interno 50×30, kernel produção k20.
- **S2 (orçamento estendido)**: GA/PSO/GWO/aleatória com 3.000 avaliações, mesmas seeds.
- Nota de escopo: nos 3 casos congelados a **sobreposição é inativa por construção** (distância mínima entre pilares 3,20 m > alcance máximo 2×1,50 m) — declarado no artigo.

### Resultados-chave (S1)

| Caso | EGO média±DP | Melhor concorrente | p EGO vs GWO | Redução mediana vs aleatória |
|---|---|---|---|---|
| 1 (dim 3) | 3,944 ± 0,049 | GWO 3,995 ± 0,075 | **0,004** | 21,9% |
| 2 (dim 6) | 6,883 ± 0,360 | GWO 6,933 ± 0,404 | 0,784 (empate) | 43,0% |
| 3 (dim 9) | 4,238 ± 0,502 | GWO 4,551 ± 0,589 | **0,015** | 56,3% |

EGO vs GA/PSO/aleatória: p < 0,001 em todos os casos. EGO tem as menores violações residuais e nunca é superado. **A vantagem cresce com a dimensionalidade.**

### Resultados-chave (S2)

Com 3.000 avaliações (~0,3 s de parede), GWO/PSO superam o EGO-150 (caso 3: GWO 2,72 ± 0,04 vs EGO 4,24 ± 0,50) em ~1% do tempo (EGO: 40–73 s/rep, dominado pelo próprio método). Os dois regimes delimitam o domínio de adequação do EGO — resposta empírica da [[10_Melhorias/Questao Aberta - Custo da FO e Justificativa do EGO]].

## 4. Estudo GPR kernels × penalidade (controlado)

Script: `scripts/run_gpr_kernel_study.py` → `experiments/estudo_gpr/` (metrics.csv, predictions.parquet, meta.json). 900 amostras LHS (caso 3), split 70/30, seeds 101–103, 21 kernels, α ∈ {10¹, 10⁶}.

- **R² global NÃO degrada com α=10⁶** (~0,92 em ambos) — a variância do alvo passa a ser a da penalidade e o R² a "explica". A formulação antiga ("penalidade alta degrada o R²") era imprecisa.
- **A métrica decisiva é o RMSE na região factível** (18,6% do teste; rótulos idênticos entre penalidades): ~1,5 m³ com α=10 vs ~1,5×10⁵ m³ com α=10⁶ — **cinco ordens de grandeza acima da escala do volume**. É o argumento quantitativo para penalidade moderada e para a frente futura de CBO.
- **Kernel é secundário**: 19 das 21 configurações empatam (R² ≈ 0,92 ± 0,013); falham só ExpSine puro (k14) e DotProduct linear (k13). k20 (produção, Matérn ν=2,5) está na banda superior (0,922 ± 0,014; melhor: k10 com 0,926). Convenção factual: 21 configurações k00–k20, sendo k20 a de produção.

## 5. Artigo

- Seção 6 reescrita com os resultados finais; seções 1, 4, 5, 7, 8 e resumos atualizados; tabela AG corrigida (50×30); α=10/p=1 declarado; notas `[[[[...]]]]` resolvidas quando havia base (permanecem: fontes Nspt/coeficientes, C′, revista-alvo).
- Figuras/tabelas 100% geradas por `scripts/make_paper_artifacts.py` (paleta categórica validada, 1 cor fixa por algoritmo).
- **Compila limpo: 30 páginas, 0 erros, 0 referências/citações indefinidas** (TeX Live local).

## 6. Ambiente

- `.venv` e `venv` estavam **copiados de outra máquina** (shebangs para `/Users/lucasteixeiracorreia/...`, dylibs com assinatura inválida). `.venv` recriado com Python 3.12 (Homebrew) + `requirements.txt` + pytest.

## Pendências que saem desta sprint

- [ ] Decidir com a orientadora: fontes citáveis para Nspt/30-40-50. Os coeficientes `1,05`/`1,30` deixaram de ser pendência na Sprint 5.4: foram removidos e substituídos por peso próprio explícito `γ_c h_x h_y h_z` e comparação direta com `σ_adm`.
- [ ] Figura opcional: arranjo em planta dos casos; fluxograma EGO.
- [ ] Revista-alvo e template.
- [ ] CMA-ES/memético no mesmo protocolo (Fase 3 do roadmap) e CBO (frente 2 de pesquisa).

## Vínculos

- [[10_Melhorias/Questao Aberta - Custo da FO e Justificativa do EGO]]
- [[10_Melhorias/Guia - Validação antes do Bin Packing]]
- [[09_Relatorios/Analise - Roadmap Artigo IC - 2026-04-27]]
- [[03_Otimizacao/Penalização de Restrições]]
- [[03_Otimizacao/Kernels GPR]]
