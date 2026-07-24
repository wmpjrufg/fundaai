---
tags: [issue, medio]
file: metapy_toolbox/grey_wolf.py
severity: medio
---

# Issue — Placeholder de diversidade no GWO

## Sintoma

`grey_wolf_optimizer_01` em [[04_Codigo/metapy_toolbox - grey_wolf.py]] linha 134:

```python
df['DIVERSITY'] = 'aqui implementa função lucas'
```

Atribui uma **string literal** em vez de calcular a métrica de diversidade da população.

## Por que é problema

- Se alguém tentar usar essa coluna ⇒ erro de tipo (string em coluna esperada numérica).
- Indica **trabalho não terminado** que pode ser auditado pelo orientador.

## Possíveis métricas de diversidade

- Distância média ao centróide.
- Desvio-padrão das `OF` na população.
- Volume do AABB da população em `R^d`.

## Status

GWO **não é chamado pela UI atual**, então o placeholder não causa falha em produção. Mas qualquer uso isolado quebra.

## Vínculo

- [[04_Codigo/metapy_toolbox - grey_wolf.py]]
- [[03_Otimizacao/Grey Wolf Optimizer]]
