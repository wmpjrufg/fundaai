---
tags: [auditoria, sprint, puncao, nbr6118, artigo, duas-colunas]
data: 2026-07-10
status: concluido
---

# Sprint 5.2 — Punção C′ + artigo em duas colunas (2026-07-10)

Continuação da [[12_Auditoria/Sprint 5.1 - Protocolo experimental final e casos-limite - 2026-07-10]], conforme decisões do Lucas: (a) lacunas de Nspt permanecem marcadas para verificação de fontes; os coeficientes 1,05/1,30 foram removidos depois, na Sprint 5.4; (b) punção C′ implementada por ser necessária para periódico internacional; (c) artigo convertido para duas colunas (mantido em português por ora); (d) figura de arranjo em planta adicionada.

## 1. Punção no contorno C′ — implementada

**Fonte nova na biblioteca**: Santos, Lima Neto & Ferreira (2018), RIEM 11(2), DOI 10.1590/S1983-41952018000200011 (open access, PDF em `docs/articles/02_apoio_tecnico_geotecnia/`, ficha em [[08_Artigos/Santos et al. 2018 - Punching Shear RC Footings]]). Avalia ACI/NBR/EC2 com 216 ensaios de sapatas; confirma contorno da NBR a **2d** (Figura 7) e que só o EC2 abate a reação do solo.

**Correções sobre o rascunho comentado do grupo** (archive/fundacao.py):
- distância do contorno: **2d** (o rascunho usava `secao_critica = h_z/2`; o texto antigo do artigo dizia "d/2" — ambos imprecisos);
- momentos em **módulo** (o rascunho deixava sinal negativo REDUZIR τ_sd1);
- tabela ρ_min completa até C90 (0,256%);
- pareamento K/W_p mantido na convenção do projeto (Mx ↔ excentricidade em x, como nas fórmulas de σ).

**Formulação implementada** (`core/engineering/puncao.py::verificacao_puncao_sapata_c_linha`): u1′ = 2(ap+bp)+4πd; W_px/W_py; K por Tabela 19.2 (interp., saturado); τ_Sd1 com γf=1,4 e |M|; τ_Rd1 = 0,13·k_e·(100ρf_ck)^⅓ com k_e = min(1+√(20/d_cm), 2); **ρ = ρ_min (Tabela 17.3)** — hipótese conservadora declarada; **sem abatimento da reação do solo** (conservador, fiel à NBR). FO: `g punção = max(g_C, g_C′)` (fast e legacy em paridade bit a bit).

## 2. Conferência (a FO mudou?) — NÃO, provado em 3 camadas

1. **Prova analítica**: g_C′ é monotônico decrescente em h_z; no pior ponto do domínio (h_z = 0,60 m), o máximo sobre todos os elementos × combinações dos 3 casos é **−0,3126** → folga ≥ 31% (S/R ≤ 0,69) em TODO o espaço de busca.
2. **Amostragem**: 120k+ pontos confirmam (máx −0,31).
3. **Bit-diff**: re-execução de 2 reps EGO (caso 3, seeds 42–43) reproduz o parquet persistido do protocolo **bit a bit** (best rep0 = 3.794885921710).

→ Θ é ponto a ponto idêntica; **protocolo, estudo GPR, figuras e tabelas continuam válidos sem re-execução**. Baseline 19.70604234767181 intacto. Resultado vira texto do artigo: punção (C e C′) nunca governa nos casos estudados.

## 3. Artigo

- **Duas colunas**: documentclass 10pt twocolumn, título+resumos em largura total (`\twocolumn[{...}]` com chaves protegendo colchetes internos), margens 2,0 cm, `balance` na última página; tabelas geradas viraram `table*[!t]` (p-valores em coluna única); figuras `figure*[!t]`; equações largas reformatadas (EI, g_tensão, AABB em `gathered/split`, τ_Sd1 em duas linhas). **Compila: 19 páginas, 0 erros, 0 refs indefinidas, 0 overfull > 10pt.**
- **Metodologia**: parágrafo de punção reescrito com os dois contornos (Eqs. u1′, W_p, τ_Sd1, τ_Rd1), hipóteses declaradas, citações abnt6118 + santos2018punching; nota `[[[[...]]]]` do C′ removida.
- **Figura nova**: `fig_planta_casos` — arranjo em planta da melhor solução estritamente factível do EGO por caso, **reproduzida deterministicamente pela seed vencedora** (caso1 seed 65 V=3,882; caso2 seed 68 V=6,158; caso3 seed 51 V=3,533 m³; verificação automática contra o per_rep, cache em `experiments/protocolo_final/best_designs.json`).
- Discussão/conclusões atualizadas: limitação passa de "C′ ausente" para "armadura não dimensionada (ρ=ρ_min declarado)"; nota final sem o item do C′, com item de tradução para inglês.
- `.bib`: `santos2018punching` (sobrenome composto protegido: `{Lima Neto}`).

## 4. Commits

- `e9fd2288` feat(engineering): punching-shear check at the C' critical section (NBR 6118)
- `db7f508a` feat(scripts): plan-view layout figure + two-column float layout

Suite: **254 testes verdes** (novos: valor de referência à mão do C′, Tabelas 17.3/19.2, monotonicidade, guarda de altura útil).

## Pendências que permanecem (decisão do Lucas/orientadora)

- [ ] Fontes das correlações Nspt/30-40-50. Os coeficientes 1,05/1,30 foram removidos na Sprint 5.4 e não devem ser citados como metodologia atual.
- [ ] Revista-alvo + tradução para inglês (formato já em duas colunas).
- [ ] Dimensionamento da armadura de flexão (levantaria ρ real no C′ e habilitaria custo total) — extensão futura.

## Vínculos

- [[12_Auditoria/Sprint 5.1 - Protocolo experimental final e casos-limite - 2026-07-10]]
- [[02_Engenharia/Verificação à Punção]]
- [[08_Artigos/Santos et al. 2018 - Punching Shear RC Footings]]
- [[07_Issues/Issue - Punção seção C linha comentada]]
