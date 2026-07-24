---
status: resolvida (Sprint 5.2, 2026-07-10)
tags: [issue, medio, engenharia]
file: fundacao.py
severity: medio
---

# Issue — Verificação à punção (seção C') comentada

## Sintoma

No legado de `verificacao_puncao_sapata` ([[04_Codigo/fundacao.py]] linhas 149–167) havia um bloco extenso comentado que implementaria a punção na **seção crítica C'**. O texto antigo falava em `d/2`, mas a formulação corrigida usa perímetro a `2d` da face do pilar:

- `rho_x, rho_y` (taxas de armadura)
- `k_e = min(1 + √(20/(d·100)), 2)`
- `tau_rd1 = 1000 · (0,13 · k_e · (100·ρ·f_ck/1000)^(1/3) + 0,1·σ_cp)`
- `u_rd1 = 2(a + b) + 4π · h_z/2`
- contribuição de momentos via `kx`, `ky` (tabela 19.2 NBR)

## Por que é problema

A [[02_Engenharia/NBR 6118]] (item 19.5.3.4) **exige** as duas verificações (C e C'). A versão original entregava só C. Uma sapata pode passar em C e falhar em C'.

## Bloqueio para reativar

- Requer `rho_minimo_fck(f_ck)` (não definida).
- Requer `tabela_19_2(c_1/c_2)` (não definida).
- Requer `sigma_cp` (não passado como argumento).

## Vínculo

- [[02_Engenharia/Verificação à Punção]]
- [[02_Engenharia/NBR 6118]]
- [[04_Codigo/fundacao.py]]


> [!success] Resolvida em 2026-07-10 (Sprint 5.2)
> C' implementada em `core/engineering/puncao.py` com formulação normativa corrigida (contorno a 2d, não h_z/2). Ver [[12_Auditoria/Sprint 5.2 - Puncao C linha e duas colunas - 2026-07-10]].
