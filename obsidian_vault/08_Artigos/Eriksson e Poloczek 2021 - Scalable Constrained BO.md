---
tags: [artigo, cbo, scbo, trust-region, frente-c, futuro]
arquivo_pdf: docs/articles/05_frente_c_cbo/2021_eriksson_poloczek_scalable_constrained_bo_scbo.pdf
adicionado: 2026-07-11
---

# Eriksson e Poloczek (2021) — Scalable Constrained Bayesian Optimization (SCBO)

**AISTATS 2021, PMLR v130, p. 730–738. Open access (arXiv 2002.08526 / PMLR).**

> [!warning] Correção de autoria
> A nota antiga da frente citava "Eriksson & **Jankowiak**" — o correto é Eriksson & **Poloczek** (Jankowiak é coautor de outro trabalho do Eriksson). Corrigido em 2026-07-11.

## Por que está na biblioteca

Estado da arte da família CBO para **dimensionalidades moderadas-altas** (até ~dezenas de variáveis): estende TuRBO (regiões de confiança) com transformação bilog das restrições e seleção por factibilidade. É a variante natural para a **Fase B** (packing: 5 variáveis por sapata → 50+ dimensões em projetos reais), quando a ECI global de Gardner tende a diluir.

## O que sustenta no artigo da IC

Citado na metodologia (§4.5) como extensão prevista para a frente de empacotamento. Chave BibTeX: `eriksson2021scbo`. Não implementado nesta etapa (decisão de escopo: ECI primeiro, num problema onde d ≤ 9).

## Vínculos

- [[08_Artigos/Gardner et al. 2014 - Bayesian Optimization with Inequality Constraints]]
- [[11_Frentes_de_Pesquisa/Bayesian Optimization Constrained]]
- [[11_Frentes_de_Pesquisa/Posicionamento Conjunto - Layout + Sizing]]
