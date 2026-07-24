---
tags: [engenharia, puncao, nbr6118]
aliases: [Punção, Punção, g_puncao]
---

# Verificação à Punção

Mecanismo de **ruptura por cisalhamento de duas direções** ao redor do pilar — crítico em sapatas isoladas. Norma: [[02_Engenharia/NBR 6118]] item 19.5.

## Implementado: seção crítica C (face do pilar)

```python
def verificacao_puncao_sapata(h_z, f_ck, a_p, b_p, f_zk, cob=0.025):
    d = h_z - cob
    alpha_v2 = 1 - (f_ck/1000) / 250
    f_cd = f_ck / 1.4
    tau_rd2 = 0.27 * alpha_v2 * f_cd
    u_rd2 = 2 * (a_p + b_p)
    tau_sd2 = (1.4 * f_zk) / (u_rd2 * d)
    g_rd2 = tau_sd2 / tau_rd2 - 1
    return tau_sd2, tau_rd2, u_rd2, g_rd2
```

## Implementado (Sprint 5.2, 2026-07-10): seção crítica C' (contorno a `2d` da face)

`verificacao_puncao_sapata_c_linha` em `core/engineering/puncao.py`, conforme NBR 6118 item 19.5 e [[08_Artigos/Santos et al. 2018 - Punching Shear RC Footings]]:

- `u1' = 2(ap+bp) + 4πd` (contorno a **2d**, cantos circulares — o rascunho antigo usava `h_z/2` como aproximação de d e o artigo dizia "d/2"; ambos corrigidos).
- `W_px/W_py` + transferência de momentos com K da Tabela 19.2 (interp. e saturado em [0,5; 3,0]); momentos em módulo.
- `τ_Rd1 = 0,13·k_e·(100ρ·fck_MPa)^{1/3}`, `k_e = min(1+√(20/d_cm), 2)`; **ρ = ρ_min da Tabela 17.3** (`rho_minimo_flexao`) — hipótese conservadora declarada (a ferramenta não dimensiona armadura).
- **Sem abatimento da reação do solo** no contorno (permissão só do EC2; omitir é a favor da segurança).
- FO usa `g punção = max(g_C, g_C')`. Paridade fast/legacy bit a bit; baseline 19.70604234767181 intacto (C' tem folga ≥31% em todo o domínio dos 3 casos congelados — S/R ≤ 0,69).

## Rascunho histórico (superado)

O bloco antigo que estava **comentado** em `fundacao.py` foi superado pela implementação em `core/engineering/puncao.py`. A issue histórica [[07_Issues/Issue - Punção seção C linha comentada]] deve ser lida como registro do problema original, não como estado atual. As variáveis previstas eram:

- ρ — taxa geométrica de armadura (mín. via `rho_minimo_fck`).
- `k_e = min(1 + √(20/(d·100)), 2.0)`
- `u_rd1 = 2(a_p + b_p) + 4π·(h_z/2)`
- contribuição de momentos via `kx`, `ky` (tabela 19.2 da NBR).

## FO

`g_punção = max(g_C, g_C')` por elemento e por combinação. A função objetivo penalizada soma apenas violações positivas, seguindo o fator de penalização configurado no experimento.

## Links

- [[02_Engenharia/NBR 6118]]
- [[04_Codigo/fundacao.py]]
- [[03_Otimizacao/Penalização de Restrições]]
