---
tags: [pesquisa, fisica-informada, surrogate, frente-principal]
aliases: [PINN, Physics-Informed, PI-GPR]
---

# Physics-Informed Surrogates

> [!info] Frente prioritária de pesquisa
> Linha de pesquisa apontada como prioritária para a evolução do projeto. **Coerente** com o estado atual: a FO é cara de avaliar e o GPR usado hoje ignora a estrutura física do problema, o que abre espaço para ganhos com surrogates fisicamente informados.

## A ideia em uma frase

Em vez de o surrogate aprender a FO **só pelos dados**, embutir nele **leis físicas conhecidas** (equilíbrio, equações constitutivas, restrições de norma) — o surrogate fica mais preciso com **menos amostras**.

## Sabores principais

### 1. **PINN — Physics-Informed Neural Networks** (Raissi, Perdikaris, Karniadakis 2019)

Treina uma rede neural minimizando:

$$
\mathcal{L} = \mathcal{L}_\text{data} + \lambda \cdot \mathcal{L}_\text{PDE}
$$

onde `L_PDE` é o resíduo da equação física avaliada em pontos de colocação.

No FundaIA: σ no solo, equilíbrio de momentos, equação de punção podem entrar como termos de penalização do treino.

### 2. **PI-GPR — Physics-Informed Gaussian Processes**

Mais direto e provavelmente o caminho certo dado que o projeto já usa GPR.

Modos:
- **Constrained kernel**: construir kernel que **automaticamente** satisfaz a equação física (ex.: solenoidal kernel para divergência nula). Para desigualdade (g ≤ 0), é preciso adaptação.
- **Latent force model** (Alvarez, Lawrence): GP modela a entrada de uma equação diferencial linear; saída satisfaz a EDO.
- **Virtual observations**: incorporar pontos onde a física é conhecida como observações sintéticas com `σ_obs ≈ 0`.

### 3. **Multi-output GP com correlação física**

Modelar `(volume, g_tensao, g_puncao, g_geo, g_sob)` como saídas correlacionadas via kernel de coregionalização — exploraria correlações implícitas pela mecânica.

## Por que faz sentido aqui

- A física do problema é **bem conhecida** (equações analíticas em [[04_Codigo/fundacao.py]]).
- O custo da FO real não é "simulação de elementos finitos" — é puro `df.apply` — então o ganho de PINN é menor que em CFD.
- Mas: se o problema for **ampliado** (sapatas com FEM real, ou solo modelado com Mohr-Coulomb), o PI-GPR torna-se essencial.

## Possíveis contribuições originais

1. **Kernel customizado** que codifica desigualdade σ ≤ σ_adm.
2. Comparativo PI-GPR × GPR clássico × Random Forest no problema do FundaIA.
3. Acoplar PI-GPR com [[10_Melhorias/Acquisition Functions Modernas]] (constrained EI).

## Referências de partida

- Raissi, Perdikaris, Karniadakis (2019). *J. Comput. Phys.* 378.
- Karniadakis et al. (2021) "Physics-informed machine learning". *Nature Reviews Physics*.
- Swiler et al. (2020) "A Survey of Constrained Gaussian Process Regression". *arXiv:2007.05543*.
- Cuomo et al. (2022) "Scientific Machine Learning Through PINNs: Where We Are and What's Next".
- Pförtner et al. (2022) "Physics-Informed Gaussian Process Regression".

(Cada referência lida deve ser registrada em [[08_Artigos/Index de Artigos]].)

## Frameworks

- **DeepXDE** — PINN end-to-end.
- **Modulus** (NVIDIA) — PINN industrial.
- **GPyTorch** — GP customizável (kernels físicos).
- **PyMC** — Bayesian programming, kernel custom.

## Vínculos

- [[11_Frentes_de_Pesquisa/MOC - Frentes de Pesquisa]]
- [[03_Otimizacao/Gaussian Process Regressor]]
- [[03_Otimizacao/EGO - Efficient Global Optimization]]
- [[11_Frentes_de_Pesquisa/Surrogate Multifidelidade]]
- [[11_Frentes_de_Pesquisa/Deep Kernel Learning e GPyTorch]]
