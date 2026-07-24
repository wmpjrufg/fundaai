---
tags: [engenharia, tensao, flexao-composta]
aliases: [σ_max, σ_min, g_tensao]
---

# Flexão Composta — σ_max e σ_min

Tensão atuante na base da sapata sob **carga axial + momentos nos dois eixos** (flexão composta oblíqua). Implementada em `calcular_sigma_max_min` ([[04_Codigo/fundacao.py]]).

## Fórmula

O peso próprio da sapata é calculado como função do volume:

$$
W_c = \gamma_c h_x h_y h_z
$$

com `gamma_c = 25 kN/m3` por padrão.

$$
\sigma_{\max} =
\frac{F_z + W_c}{h_x h_y}
+ \frac{6 |M_x|}{h_x h_y h_x}
+ \frac{6 |M_y|}{h_x h_y h_y}
$$

$$
\sigma_{\min} =
\frac{F_z + W_c}{h_x h_y}
- \frac{6 |M_x|}{h_x h_y h_x}
- \frac{6 |M_y|}{h_x h_y h_y}
$$

> [!note] Convenção `Mx/My`
> No FundaIA, `Mx` é a componente que gera variação de pressão ao longo de `h_x`, e `My` é a componente que gera variação ao longo de `h_y`. Se as cargas vierem de outro software com momentos definidos em torno dos eixos globais, conferir e converter antes de importar.

> [!warning] Formulação anterior
> A formulação antiga usava `1,05` como aproximação do peso próprio e `1,30` sobre tensões compressivas. Esses fatores foram removidos da função atual. Combinações de ações e coeficientes normativos devem ser tratados explicitamente fora desta função.

## Restrição de projeto

Em `checagem_tensao_max_min`:

$$
g = \begin{cases}
\sigma/\sigma_{adm} - 1 & \text{se } \sigma \geq 0 \\
-\sigma/\sigma_{adm}    & \text{se } \sigma < 0
\end{cases}
$$

`g ≤ 0` ⇒ restrição satisfeita.

A FO usa `g tensao = max(g_max, g_min)` por combinação e depois `max` entre todas as combinações.

## Links

- [[02_Engenharia/Tensão Admissível do Solo]]
- [[02_Engenharia/NBR 6118]]
- [[03_Otimizacao/Penalização de Restrições]]
