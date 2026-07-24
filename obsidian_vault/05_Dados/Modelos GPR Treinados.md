---
tags: [dados, gpr, modelos]
folder: models/
---

# Modelos GPR Treinados

Pasta `models/` contém **118 arquivos `.pkl`** persistidos por [[04_Codigo/fundacao.py]] via `joblib.dump`.

## Convenção de nomes

```
gpr_com_kernel_k{NN}_pop_{POP}.pkl
```

- `NN` ∈ `{00, 01, ..., 19}` — índice do kernel (ver [[03_Otimizacao/Kernels GPR]]).
- `POP` ∈ `{180, 210, 270, 500, 600, 700, 800, 900, 1200, 1400, 1800}` — tamanho da base de treino.

## Nem todos os kernels têm todos os tamanhos

Os kernels k00–k02 chegam a pop=1800; do k03 em diante, a maioria tem só 500–900.

## Como foram gerados

`aprendizado_maquina_paralelo` em [[04_Codigo/fundacao.py]] roda `treino_teste_para_processo_paralelo` em paralelo (`mp.Pool`) — cada worker chama:

```python
modelo.fit(x_treino, y_treino)
joblib.dump(modelo, dir_modelos / f"{nome_limpo}_pop_{len(x_treino)}.pkl")
```

## Métricas associadas

`assets/tables/tabela_metricas_gpr_toy_problem_all_*.xlsx` — `R²_treino, R²_teste, MAE, RMSE` para todos os modelos.

## Status no app

A UI **não carrega** estes `.pkl` — usa apenas `constroi_kernel()[-1]` (cria um kernel novo a cada otimização). Os arquivos são para análise/comparação nos notebooks.

## Vínculos

- [[03_Otimizacao/Kernels GPR]]
- [[06_Notebooks/testes_otm_lucas]]
- [[06_Notebooks/testes_gpr_lucas]]
