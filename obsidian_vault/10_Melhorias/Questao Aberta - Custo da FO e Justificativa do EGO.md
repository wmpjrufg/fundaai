---
tags: [otimizacao, ego, custo-computacional, questao-aberta]
aliases: [Custo da FO, Justificativa EGO]
data_criacao: 2026-06-05
status: respondida-empiricamente
data_resposta: 2026-07-10
---

> [!success] Respondida empiricamente em 2026-07-10
> O protocolo final (30 seeds, orçamento equalizado, Mann–Whitney; ver [[12_Auditoria/Sprint 5.1 - Protocolo experimental final e casos-limite - 2026-07-10]]) delimitou os dois regimes:
> **(1) Eficiência amostral (orçamento igual de 150 avaliações):** EGO tem a melhor média nos 3 casos e nunca é superado (p<0,001 vs GA/PSO/aleatória; p=0,004 e 0,015 vs GWO nos casos 1 e 3; empate no caso 2). Redução mediana vs aleatória cresce com a dimensão: 21,9% → 43,0% → 56,3%.
> **(2) Tempo de parede (FO barata, orçamento estendido):** GWO/PSO com 3.000 avaliações superam o EGO-150 em ~1% do tempo (0,3 s vs 40–73 s/rep).
> **Conclusão para o artigo:** o EGO é a escolha quando avaliações reais são o recurso escasso (regime das frentes futuras: packing, recalque, ISE) e um investimento metodológico — não uma necessidade — na formulação barata atual. O artigo (Seções 6–8) já incorpora essa posição com os números.

# Questão Aberta — Custo da FO e Justificativa do EGO

## Contexto

A nota [[03_Otimizacao/EGO - Efficient Global Optimization]] justifica o uso do EGO com a seguinte afirmação:

> *"A FO `obj_felipe_lucas` é cara: cada avaliação roda múltiplos `df.apply` sobre todas as fundações × combinações."*

Esta nota questiona essa justificativa com base no artigo original de Jones et al. (1998) e em medições diretas do código.

---

## O que Jones et al. (1998) entendem por "expensive"

A referência metodológica do EGO é Jones, Schonlau e Welch (1998), ver [[08_Artigos/Jones et al. 1998 - Efficient Global Optimization]].

O abstract do artigo define o problema tratado:

> *"In many engineering optimization problems, the number of function evaluations is severely limited by **time or cost**."*

E na introdução, o exemplo motivador é explícito:

> *"Computer models facilitate the exploration of alternative designs and reduce the need for expensive hardware prototypes. […] For example, **an automotive crash simulation may take twenty hours**."*

**Conclusão do artigo:** "expensive function" significa, no contexto do EGO, uma função cujo custo de avaliação é **irredutível** — imposto pela física ou pelo software externo (FEM, CFD, simulações de crash), tipicamente na ordem de **minutos a horas** por avaliação.

---

## Custo real da `obj_felipe_lucas`

### Natureza das operações

A função `_avaliar_projeto` em `fundacao.py` executa, para cada chamada com N fundações e n_comb combinações de carregamento:

| Operação                                   | O que faz                           | Custo matemático                   |
| ------------------------------------------ | ----------------------------------- | ---------------------------------- |
| `df.copy()` + atribuições                  | Cópia e montagem do DataFrame       | Overhead pandas, não computação    |
| `sobreposicao_matrix`                      | Matriz N×N de interseções AABB      | O(N²), operações numpy vetorizadas |
| `tensao_adm_solo` via `df.apply`           | 2 comparações de string + 1 divisão | Trivial por linha                  |
| `verificacao_puncao_sapata` via `df.apply` | ~6 ops aritméticas por linha        | Trivial por linha                  |
| `calcular_sigma_max_min` via `df.apply`    | ~8 ops aritméticas por linha        | Trivial por linha                  |
| `checagem_tensao_max_min` via `df.apply`   | 1 comparação + 1 divisão por linha  | Trivial por linha                  |
| `checagem_geometria` via `df.apply`        | 3 ops aritméticas por linha         | Trivial por linha                  |

Não há: I/O, simulação FEM, solver iterativo, rede neural, nenhum laço de convergência.

### Medições (benchmark executado em 2026-06-05)

Benchmark realizado chamando `_avaliar_projeto` 300–500 vezes e calculando a média. Código de referência: `fundacao.py` + `core/engineering/`.

| Cenário | Tempo médio / chamada | GA (150 ind × 100 épocas) | EGO (20 pop + 50 iter, FO real apenas) |
|---|---|---|---|
| 3 fund / 2 comb | **6,46 ms** | ~97 s | ~0,45 s |
| 10 fund / 4 comb | **10,63 ms** | ~160 s | ~0,74 s |
| 30 fund / 4 comb | **13,43 ms** | ~202 s | ~0,94 s |

> **Nota:** a coluna "EGO (FO real)" contabiliza apenas o tempo das avaliações reais da FO. O tempo total do EGO inclui ainda o treinamento do GPR (O(n³), onde n cresce a cada iteração) e a otimização do EI via GA interno (epoch=50, pop_size=150 → 7.500 predições do surrogate por passo).

### Decomposição: implementação vs. matemática

A mesma lógica foi reimplementada em numpy puro (sem pandas/apply) para isolar o overhead:

| Implementação                      | Tempo (10 fund / 4 comb) | Fator                 |
| ---------------------------------- | ------------------------ | --------------------- |
| `fundacao.py` atual (pandas/apply) | 10,63 ms                 | —                     |
| Numpy puro equivalente             | **0,097 ms**             | **~109× mais rápido** |

**~99% do tempo de execução é overhead do `df.apply`** — chamada Python pura a uma função de linha para cada linha do DataFrame. O custo matemático real da FO é da ordem de **0,1 ms**.

---

## Por que isso importa para a escolha do EGO

### Argumento contra a justificativa atual

O EGO é eficiente quando a FO é **fundamentalmente cara** (lentidão irredutível). No caso da `obj_felipe_lucas`, a lentidão vem de uma **escolha de implementação** (`df.apply`), não da complexidade matemática do problema.

Evidência: a Sprint 3.8 já vetorizou a restrição de sobreposição (`sobreposicao_matrix`), eliminando o laço `iterrows()` original. Se as demais chamadas `df.apply` forem vetorizadas da mesma forma, o tempo por avaliação cairia para < 1 ms, e o GA direto passaria a ser competitivo ou superior ao EGO em tempo total de parede.

### O EGO ainda pode ser útil?

Sim, com nuances:

1. **Enquanto `df.apply` não for vetorizado:** com ~10 ms/avaliação, um GA(150×100) leva ~100–200 s de FO real. O EGO, usando apenas ~70 avaliações reais, executa a FO em < 1 s. O overhead do GPR e do EI pode compensar em alguns regimes.
2. **Justificativa de qualidade de solução:** o EGO pode encontrar soluções melhores em menos avaliações reais — isso é testável empiricamente (ver [[10_Melhorias/Validação contra problema-benchmark]]).
3. **Após vetorização completa:** com < 1 ms/avaliação, a vantagem do EGO desaparece. Um GA com 50.000 avaliações leva ~50 s e explora o espaço de busca com muito mais diversidade.

### Questão a responder com o orientador

> **A `obj_felipe_lucas`, na sua forma atual (~10 ms/chamada), é cara o suficiente para justificar o EGO no sentido de Jones et al. (1998)? Ou o critério correto de comparação é a qualidade da solução encontrada com orçamento fixo de avaliações?**

A resposta tem impacto direto no posicionamento do artigo (ver [[09_Relatorios/Analise - Roadmap Artigo IC - 2026-04-27]]).

---

## Resumo para discussão com orientador

| Ponto | Situação |
|---|---|
| Definição de "expensive" em Jones et al. (1998) | Funções com custo irredutível (FEM/CFD, horas por avaliação) |
| Custo matemático real da FO | ~0,1 ms (numpy puro, confirmado com _avaliar_projeto_fast — medido) |
| Custo da FO implementada | ~6–13 ms (overhead `df.apply`, medido) |
| Origem do custo | Implementação Python por linha, não complexidade matemática |
| Tendência com refatoração | Custo tende a cair com vetorização |
| EGO útil na versão atual? | Possivelmente sim (reduz avaliações reais ~150×), mas margem é discutível |
| Questão aberta | Justificativa empírica (qualidade × tempo) ainda não testada sistematicamente |

---

## Links

- [[03_Otimizacao/EGO - Efficient Global Optimization]]
- [[08_Artigos/Jones et al. 1998 - Efficient Global Optimization]]
- [[04_Codigo/fundacao.py]]
- [[10_Melhorias/Refactor - Vetorização da FO]]
- [[09_Relatorios/Analise - Roadmap Artigo IC - 2026-04-27]]

---

## Atualização — Sprint 3.9 (2026-06-05)

`_avaliar_projeto_fast` foi implementada e integrada em `obj_felipe_lucas`. Benchmarks confirmados:

| Cenário | legacy (pandas/apply) | fast (numpy) | Speedup |
|---|---|---|---|
| 3 fund / 2 comb | 6,36 ms | 0,090 ms | ~70× |
| 10 fund / 4 comb | 10,29 ms | 0,132 ms | ~78× |
| 30 fund / 4 comb | 12,79 ms | 0,148 ms | ~86× |

Validação numérica: `diff = 0.00e+00` em todos os cenários (factíveis e infactíveis). 46 testes passando.

**Impacto na questão aberta:** com ~0,1 ms/eval, um GA(pop=150, epoch=100) agora leva **~1,5 s** — o EGO perdeu a vantagem de tempo de parede. A justificativa do EGO passa a ser exclusivamente qualidade da solução por orçamento fixo de avaliações, não velocidade.

Ver [[04_Codigo/fundacao.py]] para detalhes da implementação.
