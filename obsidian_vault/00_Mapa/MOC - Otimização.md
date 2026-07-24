---
tags: [moc, otimizacao]
---

# 🧮 MOC — Otimização

## Formulação

- [[03_Otimizacao/Formulação do Problema]] — variáveis, FO, restrições, dimensionalidade.

## Algoritmo principal: EGO híbrido

- [[03_Otimizacao/EGO - Efficient Global Optimization]]
- [[03_Otimizacao/Gaussian Process Regressor]] — surrogate
- [[03_Otimizacao/Kernels GPR]] — 20 variantes implementadas
- [[03_Otimizacao/Expected Improvement]] — função de aquisição

## Otimizadores internos do EGO

- [[03_Otimizacao/Algoritmo Genético]] — via `mealpy.GA.BaseGA`
- [[03_Otimizacao/Grey Wolf Optimizer]] — disponível, não usado em produção

## População inicial

- [[03_Otimizacao/Latin Hypercube Sampling]]
- [[03_Otimizacao/Opposite e Quasi-Opposite Population]]

## Restrições

- [[03_Otimizacao/Penalização de Restrições]] — método estático com fator 10
- [[03_Otimizacao/Problema de Empacotamento]] — restrição de sobreposição

## Implementação

- [[04_Codigo/metapy_toolbox - ego.py]]
- [[04_Codigo/metapy_toolbox - genetic_algorithm.py]]
- [[04_Codigo/metapy_toolbox - grey_wolf.py]]
- [[04_Codigo/metapy_toolbox - funcs.py]]
