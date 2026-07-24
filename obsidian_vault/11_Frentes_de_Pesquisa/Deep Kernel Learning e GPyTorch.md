---
tags: [pesquisa, dkl, gpr]
aliases: [DKL]
---

# Deep Kernel Learning e GPyTorch

> [!note] Frente
> O `sklearn.GaussianProcessRegressor` atual é simples mas limitado: kernel fixo, escala `O(n³)`. **GPyTorch** + DKL abrem porta para problemas com mais dados e mais dimensões.

## Deep Kernel Learning (Wilson et al., 2016)

Kernel composto:
$$k_{DKL}(x, x') = k_\text{base}(\phi_\theta(x), \phi_\theta(x'))$$

onde `φ_θ` é uma rede neural treinada juntamente com os hiperparâmetros do kernel. Ganha capacidade de representação **sem perder a calibração de incerteza** do GP.

## Quando vale a pena

- Variáveis com **estrutura** (ex.: hierárquica — pilar → grupo → fundação).
- Mais de 1000 pontos de treino (GP clássico fica lento).
- Bom encaixe com [[11_Frentes_de_Pesquisa/Physics-Informed Surrogates]] (a NN φ pode ser parcialmente fixa pela física).

## GPyTorch — vantagens sobre sklearn

| Aspecto | sklearn | GPyTorch |
|---|---|---|
| Suporte a GPU | ❌ | ✅ |
| Inferência variacional / SVGP | ❌ | ✅ |
| Kernels customizados | limitado | ilimitado |
| Treino em batches | ❌ | ✅ |
| Multi-output | rudimentar | nativo |

## Esforço de migração

Trocar `GaussianProcessRegressor` por `gpytorch.models.ExactGP` em [[04_Codigo/fundacao.py]] e [[04_Codigo/metapy_toolbox - ego.py]] é viável; API é diferente mas conceito é o mesmo. Ver "Exact GP Regression with Multiple GPUs" no tutorial oficial.

## Vínculos

- [[03_Otimizacao/Gaussian Process Regressor]]
- [[03_Otimizacao/Kernels GPR]]
- [[11_Frentes_de_Pesquisa/Physics-Informed Surrogates]]
