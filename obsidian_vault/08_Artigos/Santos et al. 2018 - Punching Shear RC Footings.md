---
tags: [artigo, puncao, sapatas, nbr6118, aci318, ec2]
arquivo_pdf: docs/articles/02_apoio_tecnico_geotecnia/2018_santos_et_al_punching_shear_rc_footings_design_codes.pdf
doi: 10.1590/S1983-41952018000200011
adicionado: 2026-07-10
---

# Santos, Lima Neto e Ferreira (2018) — Punching shear resistance of reinforced concrete footings: evaluation of design codes

**Revista IBRACON de Estruturas e Materiais (RIEM), v. 11, n. 2, p. 432–454. Open access (SciELO).**

## Por que está na biblioteca

Fonte de apoio da implementação da **punção no contorno C′** do FundaIA (Sprint 5.2). Avalia ACI 318 (2014), NBR 6118 (2014) e Eurocode 2 (2010) contra **216 ensaios** de sapatas em concreto armado.

## O que sustenta no artigo da IC

- Confirma que a NBR 6118 manda verificar sapatas com **as mesmas recomendações de ligação laje–pilar** (§3.3 do paper), com contorno de controle a **2d** da face (Figura 7 — atenção: o texto do §3.3 repete "d/2" por cochilo de edição herdado da seção do ACI; a figura e a norma dizem 2d).
- Eq. (10) do paper = τ_R1 característica `0,182(1+√(200/d))(100ρf_c)^{1/3}` — nossa implementação usa a versão de projeto `0,13(...)` (γc embutido).
- **Só o EC2 permite abater a reação do solo** dentro do contorno (Eqs. 4–5); a NBR não prevê — implementamos sem abatimento (conservador).
- Conclusão citável: pela classificação de Collins, a NBR 6118 é o código **mais conservador** para sapatas — usada como caveat honesto na metodologia.

## Onde é citado

`secoes/04_metodologia.tex` (parágrafo da punção, formulação C′ + hipóteses) e `secoes/07_discussao.tex` (limitação da armadura não dimensionada). Chave BibTeX: `santos2018punching`.

## Vínculos

- [[02_Engenharia/Verificação à Punção]]
- [[08_Artigos/Index de Artigos]]
- [[12_Auditoria/Sprint 5.2 - Puncao C linha e duas colunas - 2026-07-10]]
