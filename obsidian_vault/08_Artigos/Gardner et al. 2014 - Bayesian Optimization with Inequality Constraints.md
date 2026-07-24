---
tags: [artigo, cbo, bayesian-optimization, restricoes, frente-c]
arquivo_pdf: docs/articles/05_frente_c_cbo/2014_gardner_et_al_bayesian_optimization_inequality_constraints.pdf
adicionado: 2026-07-11
---

# Gardner et al. (2014) — Bayesian Optimization with Inequality Constraints

**ICML 2014, PMLR v32(2), p. 937–945. Open access (PMLR).**
Autores: Gardner, Kusner, Xu, Weinberger, Cunningham.

## Por que está na biblioteca

**Fonte primária da Frente C** (Sprint 5.3): define a aquisição com restrições implementada em `core/optimization/cbo.py`:

- Objetivo e cada restrição modelados por **GPs independentes**;
- Aquisição: `ECI(x) = EI(x | melhor factível) · Π_k P(g_k(x) ≤ 0)`, com `P(g≤0) = Φ(−μ_k/σ_k)`;
- **Sem ponto factível observado**: maximiza só o produto das probabilidades de factibilidade (§3.2) — comportamento implementado literalmente.

## O que sustenta no artigo da IC

Seção 4.5 da metodologia (equações ECI e PoF) e a comparação CBO×EGO da seção de resultados. Chave BibTeX: `gardner2014bayesian`.

## Conexão com os achados do projeto

É a resposta direta aos dois resultados empíricos do protocolo: (i) penalidade α=10⁶ preserva R² global mas explode o RMSE na região factível (o GP da CBO nunca vê o penhasco da penalidade); (ii) penalização linear α=10 admite violações residuais (a CBO otimiza o volume condicionado à factibilidade).

## Vínculos

- [[11_Frentes_de_Pesquisa/Bayesian Optimization Constrained]]
- [[08_Artigos/Eriksson e Poloczek 2021 - Scalable Constrained BO]]
- [[08_Artigos/Index de Artigos]]
