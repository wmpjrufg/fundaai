---
tags: [engenharia, geotecnia, spt]
aliases: [Nspt, SPT]
---

# SPT — Sondagem à Percussão

Ensaio de campo (NBR 6484) que mede a resistência do solo à penetração de um amostrador padronizado. Resultado: `N_SPT` (número de golpes para 30 cm finais de cravação).

## Uso no FundaIA

- Coluna `spt` da planilha de entrada (ver [[05_Dados/Schema das Planilhas]]).
- Alimenta [[02_Engenharia/Tensão Admissível do Solo]] junto com o tipo de solo.

## Tipos de solo aceitos

`pedregulho`, `areia`, `silte`, `argila` (case-insensitive). Default (não-pedregulho/areia) cai em `SPT/50 · 1000`.

## Limitações

- O método dos práticos é simplificado. Outros métodos (Décourt-Quaresma, Aoki-Velloso, Teixeira) podem ser estudados em [[08_Artigos/Index de Artigos]].
- Não há tratamento explícito de **NA** (nível d'água) ou camadas estratificadas.
