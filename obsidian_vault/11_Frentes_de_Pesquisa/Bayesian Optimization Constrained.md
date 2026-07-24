---
tags: [pesquisa, bo, restricoes]
aliases: [CBO, Constrained BO]
---

# Bayesian Optimization Constrained (CBO)

> [!success] IMPLEMENTADA E REANALISADA — Sprints 5.3–5.5 (2026-07-11/12)
> `core/optimization/cbo.py` implementa a aquisição **ECI de Gardner et al. (2014)**: GP do volume + 1 GP por grupo de restrição (nenhum alvo vê a penalização), `ECI = EI(V | melhor factível) · Π Φ(−μ_k/σ_k)`, fase só-PoF quando não há factível, grupos de variância nula tratados como determinísticos. Integrada à bancada como algoritmo `cbo` sob as MESMAS alavancas do EGO (LHS 10d, kernel k20, GA interno 50×30, orçamento 150, seeds 42–71) — a comparação isola o tratamento de restrições. Após a correção da tensão e a auditoria de decomposição, a leitura correta é mais cautelosa: **CBO melhora a média de Θ nos três casos** frente ao EGO (1,5% / 9,3% / 21,7%) e melhora o melhor volume estritamente factível (0,8% / 3,5% / 15,3%), com Wilcoxon-Holm pareado `p=0,014`, `<0,001`, `<0,001`; porém perde factibilidade estrita nos casos 1 e 2 (63% / 37% / 83% contra 83% / 83% / 83% do EGO). A auditoria por decomposição mostrou que os casos atuais são quase separáveis, então **não se deve atribuir causalmente o ganho à dimensionalidade**. Resultados consolidados: [[12_Auditoria/Sprint 5.5 - Novos artigos e reforco metodologico - 2026-07-12]]. Fontes em `docs/articles/05_frente_c_cbo/`.

> [!note] Motivação empírica (do protocolo final)
> (i) α=10⁶ preserva o R² global do surrogate mas explode o RMSE na região factível em 5 ordens de grandeza; (ii) a penalização linear α=10 admite violações residuais 10⁻³–10⁻¹ nas soluções finais de todos os métodos. A CBO remove o penhasco da penalização de TODOS os alvos de regressão.

## Funções de aquisição com restrições

### Expected Constrained Improvement (Schonlau, 1998; Gardner et al., 2014)
$$
\text{ECI}(x) = \text{EI}(x) \cdot \prod_k P(g_k(x) \le 0 \mid \mathcal{D})
$$

### Predictive Entropy Search with Constraints (PESC) — Hernández-Lobato et al. (2016)
Mais sofisticada. Funciona melhor quando muitas restrições são ativas.

### SCBO — Eriksson & Poloczek (2021)
"Scalable Constrained BO" — extensão de [TuRBO](https://botorch.org/) (Trust Region BO) para restrições. **Estado-da-arte para D moderado-alto** (até ~50 dim).

## Implementação prática

**BoTorch** já oferece:
- `qNoisyExpectedImprovement` com `inequality_constraints`.
- `qLogExpectedHypervolumeImprovement` (multi-objetivo + restrições).

## Vantagem para o FundaIA

Hoje a penalização (×10) deforma a FO **mesmo em regiões factíveis**. CBO mantém o surrogate da FO **limpo** e modela viabilidade separadamente.

## Possível experimento

| Estratégia | RMSE do surrogate da FO | Best_of factível em N evals |
|---|---|---|
| Atual (penalização ×10) | baseline | baseline |
| ECI com 4 GPRs (1 obj + 3 restr) | ? | ? |
| SCBO/TuRBO | ? | ? |

## Conexões

- Combina muito bem com [[11_Frentes_de_Pesquisa/Physics-Informed Surrogates]] — cada GPR de restrição pode ser PI-GPR informado pela equação correspondente (σ_adm, NBR 6118 punção, geometria).

## Vínculos

- [[10_Melhorias/Acquisition Functions Modernas]]
- [[03_Otimizacao/Penalização de Restrições]]
- [[11_Frentes_de_Pesquisa/Physics-Informed Surrogates]]
