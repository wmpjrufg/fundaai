---
tags: [issue, medio, ciencia]
file: fundacao.py
severity: medio
---

# Issue — Penalidade de sobreposição possivelmente contada duas vezes

## Sintoma

Em `obj_felipe_lucas` ([[04_Codigo/fundacao.py]]):

```python
for idx, row in df.iterrows():
    aux = 0
    for jdx, row_j in df.iterrows():
        if jdx != idx:
            aux += sobreposicao_sapatas(...)
    df.loc[idx, 'g sobreposicao'] = aux / (h_x_i * h_y_i)
```

O par `(i, j)` é visitado nas duas direções — `i→j` contribui para `g_sob_i`, e `j→i` contribui para `g_sob_j`.

Quando `volume_final = volume + 10·Σ g_sob_i`, a interseção entre `i` e `j` aparece **duas vezes** no somatório global (uma normalizada por área de `i`, outra por área de `j`).

## Por que é problema

- Pode ser **intencional** (cada sapata "carrega" a fração que invade ou é invadida).
- Pode ser **bug** (deveria contar só uma vez).
- Independente da intenção, **não está documentado** em lugar nenhum.

## Implicação prática

- Se for bug: a pressão para evitar sobreposição é **2× mais forte** que o desejado, distorcendo o trade-off com volume.
- Se for feature: precisa estar explícito no relatório/artigo, ou revisor questiona.

## Correção sugerida (a confirmar)

1. Decidir formal: contar uma vez ou duas?
2. Documentar em [[03_Otimizacao/Problema de Empacotamento]] — já há nota explícita lá.
3. Se for "uma vez": iterar só com `jdx > idx` ou dividir por 2.

## Vínculo

- [[04_Codigo/fundacao.py]]
- [[03_Otimizacao/Problema de Empacotamento]]
- [[03_Otimizacao/Penalização de Restrições]]
- [[07_Issues/Lista Mestre de Issues]]
