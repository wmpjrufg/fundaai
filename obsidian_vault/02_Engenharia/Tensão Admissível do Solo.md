---
tags: [engenharia, solo, geotecnia]
aliases: [σ_adm, sigma_adm, tensao admissivel]
---

# Tensão Admissível do Solo

Tensão máxima que o solo suporta sem ruptura ou recalque excessivo. No FundaIA atual ela é estimada por uma correlação empírica legada baseada em [[02_Engenharia/SPT - Sondagem]] e tipo de solo.

> [!warning] Escopo
> Esta correlação é hipótese de pré-dimensionamento. Ela não deve ser apresentada como prescrição direta da NBR 6122 sem validação bibliográfica ou geotécnica complementar. Para projeto executivo, a tensão admissível deve ser definida a partir de investigação geotécnica e critérios normativos completos.

## Fórmula (em `tensao_adm_solo`)

| Solo | σ_adm [kPa] |
|---|---|
| pedregulho | `SPT/30 · 1000` |
| areia | `SPT/40 · 1000` |
| silte ou argila (default) | `SPT/50 · 1000` |

```python
def tensao_adm_solo(solo, spt):
    if solo.lower() == 'pedregulho': return spt/30 * 1e3
    elif solo.lower() == 'areia':    return spt/40 * 1e3
    else:                            return spt/50 * 1e3
```

## Uso

Compara-se com σ_max e σ_min calculados em [[02_Engenharia/Flexão Composta - Sigma Max e Min]] para gerar `g_tensao`.

> [!note] Referência teórica
> Método dos práticos (Terzaghi & Peck simplificado), mantido por compatibilidade com o projeto. Antes de submissão ou uso executivo, validar/substituir por fonte citável e por métodos geotécnicos mais completos (Décourt-Quaresma, Aoki-Velloso etc.).

## Links

- [[02_Engenharia/SPT - Sondagem]]
- [[02_Engenharia/Flexão Composta - Sigma Max e Min]]
- [[04_Codigo/fundacao.py]]
