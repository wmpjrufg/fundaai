---
tags: [pesquisa, llm, especulativo, tendencia]
---

# LLM como Meta-Otimizador

> [!note] Frente especulativa / tendência 2024–2025
> Distante do escopo principal mas é tópico quente. Pode entrar como "trabalhos futuros" no relatório.

## Linhas

### 1. **OPRO — "Large Language Models as Optimizers"** (Yang et al., DeepMind, 2024)
LLM recebe histórico `[(x_i, f(x_i))]` e propõe próximo `x` em linguagem natural. Surpreendentemente competitivo em problemas de baixa dimensão.

### 2. **EvoPrompt / FunSearch** (DeepMind, 2024)
LLM evolui **código** (não só números). Em FundaIA, poderia evoluir variantes do operador de crossover, ou da função de aquisição.

### 3. **AutoML híbrido**
LLM como camada de meta-decisão: escolhe kernel, algoritmo, hiperparâmetros baseando-se em descrição em linguagem natural do problema.

## Risco / realismo

- Custo computacional alto (chamadas a API).
- Resultado pouco interpretável.
- Para IC: provavelmente não é o caminho principal, mas leitura de OPRO + FunSearch dá repertório.

## Vínculos

- [[11_Frentes_de_Pesquisa/Hibridizações Modernas]]
- [[11_Frentes_de_Pesquisa/Reinforcement Learning para Otimização]]
