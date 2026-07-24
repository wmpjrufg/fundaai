---
tags: [pesquisa, multifidelity, surrogate]
aliases: [Multifidelity, MFGP]
---

# Surrogate Multifidelidade

> [!note] Frente
> Em muitos problemas de engenharia há **modelos baratos e imprecisos** + **modelos caros e precisos**. Multifidelity surrogates combinam os dois.

## Aplicação no FundaIA

### Cenário A: hoje
- FO é cara (~ms) mas analítica. Não há "alta fidelidade" diferente.

### Cenário B: futuro (próximo passo da IC)
- **Baixa fidelidade**: as fórmulas analíticas atuais.
- **Alta fidelidade**: simulação de elementos finitos (FEM) da sapata + solo (e.g. com `pyfeap` ou `FEniCS`), resolvendo Mohr-Coulomb no solo, ELU à punção real, recalques, etc.

Aí faz total sentido um surrogate multifidelidade que:
- Usa muitos pontos do modelo barato.
- Calibra com poucos pontos do modelo caro.

## Modelos clássicos

- **Co-Kriging** (Kennedy & O'Hagan, 2000) — GP hierárquico.
- **MFNet / MF-DNN** — neural networks multifidelity.
- **NARGP** (Nonlinear Auto-Regressive GP, Perdikaris et al. 2017) — usado quando relação entre fidelidades é não-linear.

## Pacotes

- **EmuKit** (Microsoft) — multifidelity em Python.
- **GPy / GPyTorch** com extensão.

## Vínculos

- [[11_Frentes_de_Pesquisa/Physics-Informed Surrogates]] — pode-se usar PINN como modelo de alta fidelidade calibrado por dados FEM.
- [[03_Otimizacao/Gaussian Process Regressor]]
