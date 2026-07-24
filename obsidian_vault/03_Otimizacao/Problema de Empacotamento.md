---
tags: [otimizacao, packing]
aliases: [Packing, Empacotamento, g_sobreposicao, Bin Packing, Strip Packing]
---

# Problema de Empacotamento

Restrição que impede que duas [[02_Engenharia/Sapatas Isoladas]] se sobreponham em planta. É a porção "packing" do problema acoplado da IC e, segundo decisão atual do projeto, constitui o **próximo grande tema** de pesquisa após o saneamento de código e a validação dos modelos GPR.

Esta nota tem dois blocos:

1. **Estado atual** — o que está implementado hoje em [[04_Codigo/fundacao.py]].
2. **Sugestões / próxima etapa** — direções coerentes com o escopo, sem aplicação imediata em código. A roadmap geral está em [[10_Melhorias/Roadmap Sugerido]].

---

## 1. Estado atual

### Implementação

`sobreposicao_sapatas` em [[04_Codigo/fundacao.py]] calcula a **área de interseção AABB** (axis-aligned bounding box):

```python
overlap_x = max(0, min(xi_max, xj_max) - max(xi_min, xj_min))
overlap_y = max(0, min(yi_max, yj_max) - max(yi_min, yj_min))
area = overlap_x * overlap_y
```

E `obj_felipe_lucas` agrega por sapata `i`:

```python
df.loc[idx, 'g sobreposicao'] = soma_overlap_com_outras / (h_x_i * h_y_i)
```

ou seja, **fração de área** sobreposta em relação à própria sapata. Se `> 0`, é penalizada com fator 10 ([[03_Otimizacao/Penalização de Restrições]]).

### Hipóteses embutidas no modelo atual

| # | Hipótese | Comentário |
|---|---|---|
| 1 | Sapatas são **retângulos** | sem rotação, sem polígonos |
| 2 | Eixos da sapata estão **alinhados aos eixos globais** | AABB simples |
| 3 | A sapata é **centrada** no pilar (`xg, yg`) | posição não é variável de projeto |
| 4 | `xg, yg` vêm da planilha e são **fixos** | usuário fornece, otimizador não mexe |
| 5 | Não há **terreno** com fronteira | sapatas podem em tese sair do lote |
| 6 | Não há **margem mínima** entre sapatas | só ausência de sobreposição |
| 7 | Restrição é **soft** (penalizada) | otimizador pode aceitar sobreposição se compensar no volume |
| 8 | A iteração é `O(N²)` em pares | `for idx in df: for jdx in df` |

### Limitação numérica conhecida

Como o cálculo é feito **por sapata** (e a soma do par `(i,j)` aparece tanto em `i→j` quanto em `j→i`), a penalidade global de sobreposição **conta cada interseção duas vezes** ao somar `g_sobreposicao` no volume final. Isso pode ser intencional (cada sapata "carrega" a fração que invade ou é invadida), mas convém documentar e decidir explicitamente.

---

## 2. Sugestões para a próxima etapa de packing

> [!warning] Não implementar agora
> Esta seção é mapa de direções para quando packing virar a frente ativa de pesquisa, depois do saneamento ([[10_Melhorias/Roadmap Sugerido]] Fase 0–2) e da validação dos modelos GPR.

### 2.1 Vocabulário (operations research)

| Termo | Aplicação no FundaIA |
|---|---|
| **Strip Packing 2D** | empacotar retângulos em uma faixa de largura fixa, minimizando a altura usada — análogo: lote retangular |
| **Bin Packing 2D** | empacotar em "containers" de tamanho fixo, minimizando número de containers — análogo: vários terrenos |
| **Cutting Stock** | dual do bin packing; também caracteriza forma+posição |
| **Knapsack 2D / Pallet loading** | seleção + posicionamento |
| **Irregular Packing / Nesting** | retângulos/polígonos com rotação livre |
| **No-Fit Polygon (NFP)** | conjunto de posições do polígono `B` em que ele toca mas não invade `A` — primitiva para detectar viabilidade rápido |
| **Φ-functions** (Stoyan) | função analítica para distância entre primitivas; suporta gradiente |

### 2.2 Roadmap sugerido para a frente de packing

**Fase A — Formalizar o problema atual** *(documentação, sem código)*
- Decidir se a restrição é **hard** (`g_sob = 0` obrigatório) ou **soft** (penalizada como hoje).
- Decidir se há **margem mínima** entre sapatas (e.g. 30 cm para escavação).
- Decidir se há **fronteira do lote** (limites em `xg, yg`).
- Decidir se a duplicação `i→j` + `j→i` na penalidade é intencional.
- Atualizar [[03_Otimizacao/Formulação do Problema]] com a decisão.

**Fase B — Sapata centrada com posição livre** *(mais simples, ainda AABB)*
- Variáveis: `(h_x_i, h_y_i, h_z_i, xg_i, yg_i)`.
- Restrição rígida: `xg_i ∈ [xg_min, xg_max]` (margem do lote).
- Restrição rígida: AABB sem sobreposição (Deb's rules — ver [[10_Melhorias/Tratamento de Restrições - Deb e Augmented Lagrangian]]).
- Vetorizar AABB ([[10_Melhorias/Refactor - Vetorização da FO]]) — vira NumPy com matriz `N×N`.
- Conexão com mecânica: posição da sapata diferente do centro do pilar gera **excentricidade** ⇒ momento adicional `dx · F_z` em `M_y` e `dy · F_z` em `M_x`. Ver [[02_Engenharia/Flexão Composta - Sigma Max e Min]].

**Fase C — Sapata excêntrica em relação ao pilar** *(versão prática mais comum)*
- Pilar fica fixo em `(xg, yg)` (input de projeto estrutural).
- Sapata centrada em `(xg + dx_i, yg + dy_i)` com `|dx|, |dy| ≤ excentricidade_max`.
- Penalidade extra: aumenta `M_y_efetivo = M_y + dx · F_z`.
- Modelagem boa para terrenos com obstáculos (postes, divisas).

**Fase D — Sapatas rotacionadas / polígonos** *(requer NFP)*
- Variável adicional: `θ_i ∈ {0°, 90°}` (discreto) ou contínuo.
- AABB deixa de ser válida; usar **NFP** ou **Φ-functions** para detectar interferência.
- Implementação: bibliotecas `shapely` (geometria booleana) ou `pyclipper` (polygon clipping).

**Fase E — Sapatas combinadas / vigas baldrame** *(extensão estrutural)*
- Quando duas sapatas se "encostariam", uma sapata combinada pode ser mais econômica.
- Sai do escopo de packing puro e entra em **decisão topológica** (qual pilar com qual).
- Não recomendado para a IC atual — listado por completude.

### 2.3 Algoritmos candidatos

#### Heurísticas construtivas (rápidas, soluções iniciais)
- **Bottom-Left-Fill (BLF)** — coloca retângulos um a um na posição "mais baixa, mais à esquerda".
- **First-Fit Decreasing Height (FFDH)** — ordena por altura, encaixa.
- **Best-Fit / Worst-Fit** — variações de empacotamento.

Uso no FundaIA: gerar **populações iniciais factíveis** para o GA (em vez de LHS puro, que é quase sempre infactível em packing).

#### Metaheurísticas

| Algoritmo | Adequação | Comentário |
|---|---|---|
| **GA com codificação ordem + decoder BLF** | ⭐⭐⭐ | clássico em strip packing |
| **GA real-coded sobre `(xg, yg, h_x, h_y, h_z)`** | ⭐⭐ | precisa reparo de viabilidade |
| **Simulated Annealing** | ⭐⭐ | bom para layouts pequenos |
| **PSO / GWO com reparo** | ⭐⭐ | já há código no `metapy_toolbox` |
| **CMA-ES** | ⭐⭐ | bom para D moderado, mas precisa restrições explícitas |
| **Memetic GA + busca local de slide** | ⭐⭐⭐ | combina exploração com refino geométrico |

#### Surrogate-assisted

- **EGO** (já implementado) com restrições por penalidade.
- **CBO** (ver [[11_Frentes_de_Pesquisa/Bayesian Optimization Constrained]]) — modela `g_sob` como GPR separado.
- **PI-GPR** (ver [[11_Frentes_de_Pesquisa/Physics-Informed Surrogates]]) — kernel pode codificar simetrias do AABB.

#### Programação matemática (referência)

Útil como **upper bound** ou para validar metaheurística em casos pequenos:
- **MILP** com variáveis binárias `δ_ij ∈ {0,1}^4` (Beasley, 1985) para "qual lado de `j` está `i`": `i` está à esquerda, à direita, abaixo, acima.
- **Solver**: Gurobi, CBC, HiGHS.
- Escala mal (`N > 20`), mas dá referência exata para benchmarks.

### 2.4 Tratamento de restrições para packing

A penalização atual (fator 10) tende a ser **fraca** demais para packing puro: o otimizador pode achar barato pagar a penalidade. Alternativas:

1. **Deb's rules** ([[10_Melhorias/Tratamento de Restrições - Deb e Augmented Lagrangian]]) — sapatas que se sobrepõem perdem para qualquer factível, sem fator mágico.
2. **Penalização adaptativa** — fator cresce ao longo das gerações ([[10_Melhorias/Penalização Adaptativa]]).
3. **Decoder com reparo** — após cada movimento, "empurrar" sapatas para resolver overlap (slide).
4. **CBO com `g_sob` modelado por GPR** — ([[11_Frentes_de_Pesquisa/Bayesian Optimization Constrained]]).

### 2.5 Multi-objetivo natural em packing

Quando posição entra como variável, surgem objetivos conflitantes:

- **min volume de concreto** (já existe).
- **min área retangular envolvente** (use o lote, "compactar" o layout).
- **max margem mínima** entre sapatas (folga para escavação).
- **min máximo recalque diferencial** (interação solo-estrutura, vide [[02_Engenharia/Tensão Admissível do Solo]]).

Conecta a [[10_Melhorias/Multi-Objetivo - Volume vs Custo vs Reuso]] e [[11_Frentes_de_Pesquisa/Otimização Multi-objetivo]].

### 2.6 Métricas de qualidade da solução de packing

- **Taxa de utilização** = `Σ área_sapata / área_lote`.
- **Overlap residual** = `Σ max(g_sob, 0)` — deve ser zero em solução factível.
- **Margem mínima observada** = menor folga entre sapatas vizinhas.
- **Compacidade** = razão entre área das sapatas e o AABB envolvente.

### 2.7 Bibliotecas e ferramentas

| Ferramenta | Uso |
|---|---|
| **shapely** | geometria 2D, união, interseção, distância |
| **pyclipper** | polygon clipping (Vatti) — base de NFP |
| **rectpack** | bin packing 2D pronto (heurísticas) |
| **packing-solver** | benchmarks |
| **OR-Tools** (Google) | MILP/CP-SAT para versões pequenas |

### 2.8 Referências de partida

> Adicione cada paper que ler em [[08_Artigos/Index de Artigos]] usando [[99_Templates/Template - Artigo]].

- Lodi, Martello, Vigo (2002). *"Recent advances on two-dimensional bin packing problems"*. Discrete Applied Mathematics.
- Bortfeldt & Wäscher (2013). *"Constraints in container loading – A state-of-the-art review"*. EJOR.
- Stoyan, Romanova et al. — série sobre **Φ-functions** para packing irregular.
- Burke, Hellier, Kendall, Whitwell (2007). *"Complete and robust no-fit polygon generation for the irregular stock cutting problem"*. EJOR.
- Beasley, J.E. (1985). *"An exact two-dimensional non-guillotine cutting tree search procedure"*. Operations Research.
- Hopper & Turton (2001). *"A review of the application of meta-heuristic algorithms to 2D strip packing problems"*. Artificial Intelligence Review.
- Wäscher, Haußner, Schumann (2007). *"An improved typology of cutting and packing problems"*. EJOR.

### 2.9 Decisões pendentes (para discutir com orientador)

- [ ] Posição como variável de projeto? (Fase B/C/D acima)
- [ ] Restrição hard ou soft?
- [ ] Margem mínima entre sapatas?
- [ ] Sapatas com rotação? (sai do AABB)
- [ ] Multiplo lote / multi-bin?
- [ ] Tratar combinação (sapatas associadas) ou só isoladas?
- [ ] Validar com solver MILP em casos pequenos?

---

## Vínculos

- [[02_Engenharia/Sapatas Isoladas]]
- [[04_Codigo/fundacao.py]]
- [[03_Otimizacao/Penalização de Restrições]]
- [[03_Otimizacao/Formulação do Problema]]
- [[10_Melhorias/Posicionamento como Variável de Projeto]]
- [[10_Melhorias/Refactor - Vetorização da FO]]
- [[10_Melhorias/Tratamento de Restrições - Deb e Augmented Lagrangian]]
- [[11_Frentes_de_Pesquisa/Posicionamento Conjunto - Layout + Sizing]]
- [[11_Frentes_de_Pesquisa/Bayesian Optimization Constrained]]
- [[01_Projeto/Escopo da IC]]
