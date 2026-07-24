---
tags: [otimizacao, penalizacao, restricoes]
---

# Penalização de Restrições

O problema é tratado como **irrestrito penalizado**: cada violação é multiplicada por uma constante e somada à FO.

## Fórmula no `obj_felipe_lucas`

```python
df['volume final (m3)'] = (
    df['volume (m3)']
    + df['g sobreposicao'].clip(lower=0) * 1e1
    + df['g punção secao C'].clip(lower=0) * 1e1
    + df['g tensao'].clip(lower=0) * 1e1
    + df['g geometria'].clip(lower=0) * 1e1
)
of = df['volume final (m3)'].sum()
```

Cada `g_k` é normalizado (`σ/σ_adm − 1`, fração de área, etc.) e penalizado com **fator 10**.

## Estudo de sensibilidade

Os notebooks [[06_Notebooks/testes_otm_lucas]] e [[06_Notebooks/testes_gpr_lucas]] comparam **fator 1e1 vs 1e6** para entender o efeito sobre o GPR (qualidade de ajuste). Resultados em `assets/graphics/z_GPR_*_pen_1e1_vs_1e6.png`.

## Discussão

- Penalidades muito altas (1e6) deformam muito o landscape, dificultando o GPR.
- Penalidades muito baixas (1e1) podem deixar a otimização aceitar soluções inviáveis.
- O fator atual em produção é **10** (compromisso).

## Alternativas a investigar

- **Penalização adaptativa**: aumentar o fator ao longo das iterações.
- **Penalização exterior progressiva** (e.g. quadrática).
- **Métodos de barreira interior** (mas exige `g < 0` estritamente).
- Algoritmos com tratamento explícito de restrições (Deb's rules, ε-constraint, augmented Lagrangian).

## Links

- [[03_Otimizacao/Formulação do Problema]]
- [[03_Otimizacao/Problema de Empacotamento]]
