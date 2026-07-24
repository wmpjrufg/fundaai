---
tags: [pesquisa, rl, otimizacao, especulativo]
---

# Reinforcement Learning para Otimização

> [!note] Frente especulativa
> Mais distante do estado atual do projeto, mas é tendência forte na literatura 2022–2025. Vale como leitura.

## Sabores

### 1. **L2O — Learning to Optimize**
Aprender o **otimizador** com RL. Andrychowicz et al. (2016) "Learning to learn by gradient descent by gradient descent". Não é diretamente aplicável a metaheurísticas pop., mas:

### 2. **Operator Selection via RL**
Em GA, escolher *qual* operador (linear, BLX-α, SBX, ...) aplicar a cada agente baseado em estado da população. Q-learning ou bandits adaptativos. Conexão direta com [[03_Otimizacao/Algoritmo Genético]].

### 3. **Neural Architecture Search analogias**
NAS usa RL (ENAS, MnasNet) para buscar arquitetura. Ideias podem ser portadas para escolher topologia de GA/EGO.

### 4. **GFlowNets / Bayesian RL para BO**
Substituir aquisição pontual por uma política amostradora. Bengio et al.

## Realismo para a IC

- Implementar RL adiciona muita complexidade.
- Risco de o orientador achar que sai do escopo (mecânica + packing + metaheurística).
- Mas: pode ser uma seção breve "explorações futuras" no relatório final.

## Vínculos

- [[11_Frentes_de_Pesquisa/Hibridizações Modernas]]
- [[03_Otimizacao/Algoritmo Genético]]
