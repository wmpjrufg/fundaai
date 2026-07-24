---
tags: [pesquisa, robustez, incerteza]
aliases: [RBO, Robust Optimization, RDO]
---

# Otimização sob Incerteza

> [!note] Frente
> Hoje todas as variáveis são determinísticas. Em prática:
> - SPT tem **dispersão** (CV ~30% típico).
> - Cargas `F_z`, `M_x`, `M_y` têm incerteza.
> - Geometria do pilar tem tolerância de execução.

## Sabores

### 1. **Robust Design Optimization (RDO)**
Otimizar `E[f(x, ξ)] + λ · Std[f(x, ξ)]`.
- Já existe esqueleto em [[04_Codigo/metapy_toolbox - genetic_algorithm.py]] (parâmetro `robustness=dict(n_evals, perturbation_pct)`).

### 2. **Reliability-Based Design Optimization (RBDO)**
Restrições probabilísticas: `P(g_k(x, ξ) ≤ 0) ≥ 1 - p_f`. Usa FORM, SORM, Monte Carlo, ou métodos importance-sampling.

### 3. **Worst-case (minimax)**
$\min_x \max_\xi f(x, \xi)$. Conservador.

### 4. **Distributionally Robust Optimization (DRO)**
Otimiza pior caso sobre uma família de distribuições. Estado da arte 2020+.

## Aplicação direta no FundaIA

- Modelar SPT como variável aleatória `N ~ Normal(N̄, CV·N̄)`.
- Calcular `P(σ_max > σ_adm) ≤ p_f = 1e-4` via Monte Carlo.
- Trocar `g_tensao ≤ 0` por essa restrição probabilística.

## Conexão forte com a NBR

A norma já incorpora coeficientes parciais de segurança (γ_f, γ_c, γ_s) que **são** uma forma simplificada de RBDO. Trocar por RBDO **explícita** é avanço acadêmico claro.

## Referências

- Schuëller & Jensen (2008) "Computational methods in optimization considering uncertainties".
- Beyer & Sendhoff (2007) "Robust optimization — A comprehensive survey".

## Vínculos

- [[04_Codigo/metapy_toolbox - genetic_algorithm.py]] (já tem `robustness` dict)
- [[02_Engenharia/SPT - Sondagem]]
- [[02_Engenharia/NBR 6118]]
