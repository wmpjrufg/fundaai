---
tags: [moc, codigo]
---

# 💻 MOC — Código

## Camada de UI (Streamlit)

- [[04_Codigo/app.py]] — entry-point
- [[04_Codigo/pages - home.py]] — página inicial
- [[04_Codigo/pages - sapatas.py]] — página de dimensionamento (✅ saneada nas Sprints 0/1)

## Núcleo de engenharia

- [[04_Codigo/fundacao.py]] — checks NBR 6118 + GPR + objetivo

## Biblioteca de otimização

- [[04_Codigo/metapy_toolbox - __init__.py]]
- [[04_Codigo/metapy_toolbox - ego.py]]
- [[04_Codigo/metapy_toolbox - genetic_algorithm.py]]
- [[04_Codigo/metapy_toolbox - grey_wolf.py]]
- [[04_Codigo/metapy_toolbox - funcs.py]]
- [[04_Codigo/metapy_toolbox - benchmark.py]] — funções clássicas (✅ griewank/powell corrigidos na Sprint 2)
- ~~[[04_Codigo/metapy_toolbox - methods.py]]~~ — removido na Sprint 0

## Testes

- [[04_Codigo/tests]] — suite pytest (55 testes, introduzida na Sprint 2)

## Operações

- [[04_Codigo/ops - wake_up.py]] — robô Playwright

## Bootstrap

- [[04_Codigo/env-setup.py]]

## Dependências externas relevantes

- `streamlit` — UI
- `scikit-learn` — GPR (`GaussianProcessRegressor`, kernels)
- `mealpy` — GA, PSO e demais metaheurísticas
- `scipy.stats.qmc` — Latin Hypercube
- `scipy.optimize` — minimizers (L-BFGS-B, SLSQP, TNC)
- `ezdxf` — exportação CAD
- `playwright` — automação web
