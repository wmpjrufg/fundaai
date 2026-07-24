---
tags: [artigo, biblioteca, duplicatas, auditoria]
data: 2026-07-12
status: concluido
---

# Duplicatas detectadas - 2026-07-12

Durante a triagem dos PDFs soltos em `docs/articles`, foram identificadas duas duplicatas exatas por SHA-256.

## Duplicatas

- `docs/articles/buildings-12-00471-v2.pdf` era duplicata exata de `docs/articles/01_artigo_1_ego_gpr/2022_waheed_et_al_practical_tool_rc_isolated_footings.pdf`.
  - Mantido como canonico: `docs/articles/01_artigo_1_ego_gpr/2022_waheed_et_al_practical_tool_rc_isolated_footings.pdf`.
  - Movido para duplicados: `docs/articles/00_duplicados_exatos/2022_waheed_practical_tool_DUPLICADO_EXATO.pdf`.

- `docs/articles/s00158-025-03987-z.pdf` e `docs/articles/s00158-025-03987-z-2.pdf` eram identicos.
  - Mantido como canonico: `docs/articles/05_frente_c_cbo/2025_yu_picard_ahmed_pfn_constrained_engineering_bo.pdf`.
  - Movido para duplicados: `docs/articles/00_duplicados_exatos/2025_yu_picard_ahmed_pfn_constrained_engineering_bo_DUPLICADO_EXATO.pdf`.

## Criterio

Comparacao por `sha256` do arquivo PDF, nao apenas por titulo. Portanto, estes casos sao duplicatas binarias exatas.
