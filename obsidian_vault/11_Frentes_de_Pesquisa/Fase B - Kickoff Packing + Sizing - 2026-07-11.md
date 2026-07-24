---
tags: [pesquisa, packing, layout, fase-b, kickoff, cbo]
data: 2026-07-11
status: piloto-inicial-executado
---

# Fase B — Kickoff: Posicionamento + Dimensionamento (Packing + Sizing)

Documento de preparação de terreno escrito ao fim da Frente C (Sprint 5.3). Consolida a formulação proposta, os acoplamentos de engenharia, as decisões pendentes, o plano experimental e o que a Frente C já entregou de infraestrutura para esta fase. Referências de contexto: [[03_Otimizacao/Problema de Empacotamento]] (roteiro 2.1–2.9), [[11_Frentes_de_Pesquisa/Posicionamento Conjunto - Layout + Sizing]], [[09_Relatorios/Analise - Roadmap Artigo IC - 2026-04-27]] (Fase E).

> [!success] Atualização — piloto executado em 2026-07-12
> `scripts/run_packing_phase_b_pilot.py` iniciou a Fase B com um caso sintético de duas sapatas próximas, derivado de `assets/data/problema_fund_dois.xlsx`. Resultado: os ótimos individuais centralizados têm `V=4,750747 m³`, mas violam sobreposição (`g_sob=0,2307`); redimensionar mantendo centros fixos torna o projeto factível com `V=4,929703 m³`; permitir deslocamentos `dx,dy` por sapata gera solução factível com `V=4,525122 m³`. Artefatos: `experiments/phase_b_packing_pilot/{summary.csv,designs.csv,config.json}`. Interpretação: a Fase B já tem um caso mínimo em que a decomposição falha por packing e a variável de posicionamento reduz volume factível.

## 1. Por que a Fase B é o destino natural (e por que agora)

- O **plano de trabalho da IC promete** dimensionamento **e posicionamento** com conceitos de empacotamento — o artigo 1 declarou honestamente que posições são entrada; a Fase B fecha a promessa.
- O protocolo final provou que, com FO barata, buscas diretas vencem em tempo de parede. **Na Fase B a FO deixa de ser barata e a restrição de sobreposição deixa de ser inativa** — é o regime onde a eficiência amostral demonstrada no S1 passa a dominar também o relógio (ponto de equilíbrio estimado: ~18–20 ms/avaliação).
- A Frente C entregou a peça que faltava: **máquina de restrições com surrogates separados (ECI)** — sobreposição em packing é restrição *governante e rígida*; penalização ×10 é comprovadamente frágil para isso ([[10_Melhorias/Questao Aberta - Custo da FO e Justificativa do EGO]] + resultados de factibilidade do protocolo).

## 2. Formulação proposta (Fase B-1: excentricidade limitada)

Recomendação: começar pela **Fase C do roteiro de packing** (sapata excêntrica ao pilar), não pela posição livre — é a versão com significado de projeto imediato e acoplamento mecânico bem definido.

**Variáveis** por sapata i: `(h_x, h_y, h_z, d_x, d_y)` → dim = 5·N (caso 3 → 15; projeto real 30 pilares → 150).
- `d_x, d_y` = excentricidade do CENTRO DA SAPATA em relação ao centro do pilar, com `|d_x| ≤ e_max·h_x`, `|d_y| ≤ e_max·h_y` (e_max a decidir; sugestão inicial 1/6 — núcleo central).

**Acoplamento mecânico obrigatório** (é o que faz a FO ficar "de verdade"):
- Momentos efetivos na interface solo-sapata: `M_x,ef = M_x + F_z·d_x` e `M_y,ef = M_y + F_z·d_y` (sinal conforme convenção do projeto: M_x ↔ excentricidade em x). σ_max/σ_min e punção C′ (K·M/W_p) passam a usar M_ef → **posição entra na mecânica**, não só na geometria.
- Vértices da sapata deslocados: `xg_sapata = xg_pilar + d_x` → `sobreposicao_matrix` recebe os bounds deslocados (mudança de 2 linhas em `_nucleo_componentes`).

**Restrições novas**:
- `g_sob` vira **candidata a governante** → tratamento pela CBO (GP próprio) e/ou hard (ver decisões).
- Fronteira do lote: `xg_sapata ± h_x/2 ∈ [lote_x_min, lote_x_max]` (idem y) — exige novo input (4 números por projeto na planilha; schema: colunas de lote ou aba de configuração).
- Margem construtiva mínima entre sapatas: `folga_ij ≥ m_min` (sugestão 0,30 m para escavação) — generaliza g_sob: `g_margem = m_min − dist_ij` via a MESMA matriz AABB com bounds inflados em m_min/2 (implementação trivial reusando `sobreposicao_matrix`).

## 3. Por que a FO fica cara (e quanto)

1. **Dimensionalidade**: 3N → 5N; LHS 10d e o custo do GA interno crescem.
2. **Margem/lote/sobreposição ativos**: paisagem multimodal real (dois vales: desviar para a esquerda ou direita).
3. **Extensões geotécnicas da Fase B-2** (bulbo de tensões/recalque — método 2:1 ou Boussinesq por camadas): custo por avaliação estimado em **dezenas a centenas de ms** → cruza o ponto de equilíbrio de ~20 ms e o CBO/EGO passa a vencer também em tempo de parede. Implementar recalque como módulo `core/engineering/recalque.py` puro (mesmo padrão das demais verificações).

## 4. O que a Frente C já deixou pronto para a B

| Peça | Onde | Uso na Fase B |
|---|---|---|
| `avaliar_projeto_componentes` (θ, V, g[4]) | `core/api/objective.py` | Estender para g[5..6] (margem, lote) — o CBO consome sem mudança estrutural |
| `cbo_01_architecture` (ECI/PoF, fase PoF-only) | `core/optimization/cbo.py` | A fase PoF-only é exatamente o que packing precisa quando o LHS inicial é ~todo infactível |
| Bancada com seeds pareadas + factibilidade estrita | `core/api/benchmark.py` | Mesmo protocolo (S1/S2) com os novos casos; CBO vs EGO vs metas |
| Prova de invariância por monotonicidade | Sprint 5.2 | Modelo de "conferência" a repetir a cada mudança de FO |
| SCBO (fonte em mãos) | `docs/articles/05_frente_c_cbo/` | Variante para dim ≥ ~30 (trust regions) quando N crescer |

## 5. Plano experimental (espelho do protocolo atual)

- **Casos**: reusar os 3 congelados COM lote sintético apertado (para ativar packing de verdade) + 1 caso novo denso (6–10 pilares próximos — construir planilha; é o caso que motiva a fase). Congelar antes de rodar.
- **Cenários**: S1 (orçamento igual, 250–300 avals por causa da dim maior) e S2 (estendido) — mesmas seeds 42–71.
- **Algoritmos**: CBO (protagonista), EGO penalizado, GWO (melhor meta do artigo 1), aleatória (piso). GA/PSO opcionais.
- **Métricas**: as do artigo 1 + taxa de utilização do lote, margem mínima observada, overlap residual (=0 exigido) — já listadas em [[03_Otimizacao/Problema de Empacotamento]] §2.6.
- **Validação exata em caso pequeno**: MILP (OR-Tools/HiGHS) com posições discretizadas como referência de ótimo — recomendação da nota de packing §"programação matemática".

## 6. Decisões a levar para a orientadora ANTES de codificar

1. `e_max` (limite de excentricidade): 1/6 do lado? valor absoluto? norma/critério citável?
2. Sobreposição/margem: **hard** (regras de Deb/decoder com reparo) ou **CBO-PoF** (nossa aposta) ou híbrido (PoF na busca + reparo final)?
3. Margem construtiva m_min: 0,30 m? fonte (prática de escavação)?
4. Lote: sempre retangular? entrada na planilha (novas colunas) ou na UI?
5. Caso denso novo: real (projeto de referência) ou sintético?
6. Fase B-2 (recalque/bulbo): entra já ou só após o packing geométrico fechar? (Recomendação: B-1 geométrica+mecânica primeiro; B-2 é o que encarece a FO e ativa o argumento de tempo.)
7. Este conteúdo é o **artigo 2** — título provisório da Análise de Roadmap: "Dimensionamento e posicionamento conjunto de sapatas com restrições de empacotamento".

## 7. Bibliografia a adquirir (ainda NÃO temos os PDFs — regra do projeto: baixar antes de citar)

- Lodi, Martello & Vigo (2002) — survey bin packing 2D (Discrete Appl. Math.).
- Wäscher, Haußner & Schumann (2007) — tipologia C&P (EJOR).
- Hopper & Turton (2001) — metaheurísticas em strip packing (AI Review).
- Burke et al. (2007) — No-Fit Polygon robusto (EJOR) [só se houver rotação].
- Método 2:1 / Boussinesq: capítulo de manual aberto (G09-002 já cobre parte — reler §combined footings).
- Verificar open access; o que for paywalled, buscar versão de autor/repositório institucional.

## 8. Riscos e mitigação

- **LHS ~100% infactível no caso denso** → fase PoF-only do CBO cobre; alternativa: inicialização Bottom-Left-Fill (nota de packing §2.3) como semente factível.
- **dim alta degrada ECI global** → SCBO (fonte em mãos) como plano B declarado.
- **Baseline quebra** (FO muda de assinatura com 5N variáveis) → versão nova da FO CONVIVE com a atual (função separada, casos antigos intactos); baseline 19.706 permanece como regressão da formulação 3N.

## Vínculos

- [[11_Frentes_de_Pesquisa/Bayesian Optimization Constrained]]
- [[03_Otimizacao/Problema de Empacotamento]]
- [[10_Melhorias/Posicionamento como Variável de Projeto]]
- [[12_Auditoria/Sprint 5.3 - Frente C CBO - 2026-07-11]]
- [[08_Artigos/Eriksson e Poloczek 2021 - Scalable Constrained BO]]
