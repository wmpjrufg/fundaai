---
tags: [melhorias, otimizacao, ego, sugestao]
---

# Acquisition Functions Modernas

> [!note] Sugestão
> O EGO atual usa apenas Expected Improvement (ver [[03_Otimizacao/Expected Improvement]]). Existem alternativas que costumam render benchmarks melhores.

## Funções clássicas além de EI

### Probability of Improvement (PI)
$$\text{PI}(x) = \Phi\!\left(\frac{f_\min - \mu(x)}{\sigma(x)}\right)$$
Mais "exploit" que EI.

### Lower Confidence Bound (LCB)
$$\text{LCB}(x) = \mu(x) - \kappa\,\sigma(x)$$
`κ` controla explore/exploit explicitamente.

### Knowledge Gradient (KG)
Mais sofisticada. Mede o ganho esperado **sobre o melhor que será conhecido depois da próxima observação**.

## Funções para batch / paralelo

- **q-EI** (qEI) — escolhe `q` pontos simultaneamente.
- **Local Penalization** (González et al., 2016) — penaliza vizinhança de pontos já escolhidos.

## Constrained Bayesian Optimization

- **Expected Constrained Improvement (ECI)** — Gardner et al. (2014).
- Modela cada `g_k` com seu próprio GPR e mede `EI · ∏ P(g_k ≤ 0)`.
- ⚠️ Pode resolver melhor o problema do FundaIA (várias restrições) que penalização externa.

## Pacotes prontos

- **BoTorch** (PyTorch) — implementa qEI, KG, multi-objective EI, CB-MOEA, batch parallel.
- **scikit-optimize** — mais simples; suporta EI/PI/LCB.
- **Trieste** (TensorFlow) — focado em BO industrial.

## Vínculos

- [[03_Otimizacao/Expected Improvement]]
- [[03_Otimizacao/EGO - Efficient Global Optimization]]
- [[11_Frentes_de_Pesquisa/Bayesian Optimization Constrained]]
