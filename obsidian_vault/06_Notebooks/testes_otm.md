---
tags: [notebook]
file: testes_otm.ipynb
size: 18 KB
cells: 14
---

# `testes_otm.ipynb`

Exemplo mínimo do EGO+GA com 1 fundação. Reproduz em notebook o que [[04_Codigo/pages - sapatas.py]] faz na UI.

## Pipeline

```python
from mealpy import GA
from metapy_toolbox import *
from fundacao import *

n_comb=3; f_ck=25.; cob=4; h_min=60.; h_max=300.
n_gen=3; n_pop=300; n_fun=1
df = pd.read_excel("assets/problema_fund_três.xlsx")   # corrigido na Sprint 2 (antes: assets/el08.xlsx)

paras_opt = {'optimizer algorithm': GA.BaseGA(epoch=50, pop_size=100)}
k = constroi_kernel()
paras_kernel = {'kernel': k[2]}    # produto multi-escala RBF

for rep in range(3):
    x_ini = initial_population_01(n_pop, 3*n_fun, x_l, x_u, use_lhs=True)
    x_new, best_of, _ = ego_01_architecture(obj_felipe_lucas, n_gen, x_ini, x_l, x_u, paras_opt, paras_kernel, args=(df,n_comb,f_ck_kpa,cob_m))
```

## Diferença vs UI

- Aqui `n_rep=3`, na UI `n_rep=5`.
- Aqui `kernel = k[2]` (produto RBF), na UI `k[-1]` (Matern ν=2.5 com bounds estendidos).

## Vínculos

- [[04_Codigo/metapy_toolbox - ego.py]]
- [[03_Otimizacao/EGO - Efficient Global Optimization]]
