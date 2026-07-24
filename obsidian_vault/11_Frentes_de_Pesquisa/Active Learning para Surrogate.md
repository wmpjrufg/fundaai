---
tags: [pesquisa, active-learning, surrogate]
---

# Active Learning para Surrogate

> [!note] Frente
> O LHS atual é amostragem **passiva**: gera todos os pontos antes de avaliar. Active learning escolhe o **próximo** ponto baseado no que o surrogate já sabe.

## Critérios

- **MaxVar**: amostre onde `σ(x)` é máximo (reduz incerteza global).
- **MaxIMSE / IVPE**: minimiza erro médio integrado predito.
- **EI / UCB**: já é active learning para otimização (não para regressão pura).
- **Query-by-Committee**: vários surrogates discordam ⇒ amostre lá.

## Diferença do EI atual

EI minimiza a FO. Active learning para **regressão** quer um surrogate **uniformemente bom**, não só preciso no ótimo.

## Aplicação no FundaIA

- Quando o objetivo é **um surrogate fixo** que represente bem a FO em todo o domínio (por exemplo, para uso em CAD interativo): faz sentido AL puro.
- Se quer só achar o ótimo: EI já basta.

## Conexão com a IC

Os experimentos atuais ([[06_Notebooks/testes_otm_lucas]]) treinam GPR com **LHS gigante** (1800 pontos) e medem R² em teste. Substituir por AL:
- Começar com poucos (~50 LHS).
- Adicionar pontos por MaxVar até `R² > threshold`.
- Comparar **número de avaliações até R²=0.95** entre LHS e AL.

Pode virar uma seção do relatório/artigo.

## Vínculos

- [[03_Otimizacao/Latin Hypercube Sampling]]
- [[06_Notebooks/testes_otm_lucas]]
- [[11_Frentes_de_Pesquisa/Bayesian Optimization Constrained]]
