---
tags: [otimizacao, gpr, surrogate, machine-learning]
aliases: [GPR, Gaussian Process, Processo Gaussiano]
---

# Gaussian Process Regressor (GPR)

Modelo de regressão **bayesiano não paramétrico** que assume que `y(x)` é uma realização de um processo gaussiano com média e covariância especificadas por um **kernel**.

Saída: `μ(x)` (média predita) **+** `σ(x)` (incerteza). Esta incerteza é o que viabiliza [[03_Otimizacao/Expected Improvement]].

## Configuração no projeto

`gpr_pipelines` em [[04_Codigo/fundacao.py]]:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("gp", GaussianProcessRegressor(
        kernel=ker,
        normalize_y=True,
        alpha=1e-4,            # jitter numérico
        n_restarts_optimizer=5,
        random_state=42))
])
```

- Padronização dos `X` (média 0, variância 1).
- Normalização de `y` para média 0.
- 5 reinicializações do otimizador interno → mais robustez na escolha dos hiperparâmetros do kernel.

## Kernels disponíveis

20 variantes geradas por `constroi_kernel(ls0=1.0)`. Ver [[03_Otimizacao/Kernels GPR]].

## Onde é usado

| Local | Função |
|---|---|
| `ego.py:ego_01_architecture` | Surrogate dentro do EGO |
| `fundacao.py:aprendizado_maquina_paralelo` | Treina os 20 kernels em paralelo (estudo) |

## Modelos persistidos

118 `.pkl` em `models/` — ver [[05_Dados/Modelos GPR Treinados]].

## Limitações

- GPR escala `O(n³)` no tamanho da base de treino.
- O `alpha` precisa ser ajustado se houver muitas avaliações repetidas.

## Links

- [[03_Otimizacao/Kernels GPR]]
- [[03_Otimizacao/Expected Improvement]]
- [[06_Notebooks/testes_otm_lucas]]
