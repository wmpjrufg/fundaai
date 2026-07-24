---
tags: [otimizacao, ego, ei]
aliases: [EI]
---

# Expected Improvement (EI)

**Função de aquisição** padrão do [[03_Otimizacao/EGO - Efficient Global Optimization]]. Quantifica o ganho esperado em melhorar o melhor valor atual `f_min` se a FO for avaliada em um novo ponto `x`.

## Fórmula

$$
\text{EI}(x) = (f_\text{min} - \mu(x))\, \Phi(z) + \sigma(x)\, \phi(z),
\quad z = \frac{f_\text{min} - \mu(x)}{\sigma(x)}
$$

onde `μ` e `σ` vêm do [[03_Otimizacao/Gaussian Process Regressor]], `Φ` é a CDF normal e `φ` a PDF normal.

## Implementação no projeto

`ego_01_architecture.obj_ego` em [[04_Codigo/metapy_toolbox - ego.py]]:

```python
def obj_ego(x, coef):
    model, fmin = coef
    x_df = pd.DataFrame([x], columns=model.feature_names_in_)
    mu, sig = model.predict(x_df, return_std=True)
    sigma = max(sig[0], 1e-10)
    z = (fmin - mu[0]) / sigma
    of = (fmin - mu[0]) * norm.cdf(z) + sigma * norm.pdf(z)
    return -of            # minimiza ⇒ negar
```

## Trade-off explore/exploit

- **Exploit** (μ baixo) ⇒ EI alto onde a média predita é boa.
- **Explore** (σ alto) ⇒ EI alto onde há incerteza.

## Otimização da EI

Como `EI(x)` é multimodal, o projeto roda **GA da mealpy** (ou um SciPy minimizer) para maximizá-la a cada iteração do EGO.

## Links

- [[03_Otimizacao/EGO - Efficient Global Optimization]]
- [[03_Otimizacao/Gaussian Process Regressor]]
