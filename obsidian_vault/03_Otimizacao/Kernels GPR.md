---
tags: [otimizacao, gpr, kernel]
---

# Kernels GPR

`constroi_kernel(ls0=1.0)` em [[04_Codigo/fundacao.py]] retorna **21 kernels** ao todo. Todos usam `ConstantKernel C(1, (1e-5, 1e10))` como amplitude.

> [!warning] Convenção do projeto: 20 + 1
> - **k00 a k19** (20 kernels) — varridos nos experimentos GPR; cada um tem arquivos persistidos em [[05_Dados/Modelos GPR Treinados]].
> - **k20** (extra) — Matern ν=2.5 com bounds estendidos `(1e-2, 1e3)`. **Não está nos `.pkl` persistidos**, mas é o usado em produção via `constroi_kernel()[-1]` em [[04_Codigo/pages - sapatas.py]].
>
> Pendente de definição oficial se o projeto declara "20 kernels experimentais + 1 de produção" ou "21 kernels". Atualmente as duas leituras coexistem — risco de confusão em relatórios e publicações.

## Lista (índice → composição)

| k | Composição |
|---|---|
| k00 | `C * RBF(ls0)` |
| k01 | `C * (RBF(ls0) + RBF(ls0·0.3))` — soma multi-escala |
| k02 | `C * (RBF(ls0) * RBF(ls0·0.5))` — produto |
| k03 | `C * Matern(ls0, ν=0.5)` — exponencial |
| k04 | `C * Matern(ls0, ν=1.5)` |
| k05 | `C * Matern(ls0, ν=2.5)` |
| k06 | `C * (Matern(ν=1.5) + Matern(ν=2.5))` — multi-escala Matern |
| k07 | `C * RationalQuadratic(α=1)` |
| k08 | `C * RationalQuadratic(α=0.1)` |
| k09 | `C * RationalQuadratic(α=10)` |
| k10 | `C * (DotProduct + RBF)` — linear + suave |
| k11 | `C * (DotProduct + Matern(ν=1.5))` |
| k12 | `C * (DotProduct(σ₀=0.1) + RBF)` |
| k13 | `C * DotProduct` — puramente linear |
| k14 | `C * ExpSineSquared` — periódico |
| k15 | `C * (RBF * ExpSineSquared)` — quase-periódico |
| k16 | `C * (Matern(ν=1.5) * ExpSineSquared)` |
| k17 | `C * RBF + WhiteKernel(1e-12)` — jitter mínimo |
| k18 | `C * Matern(ν=2.5) + WhiteKernel` |
| k19 | `C * RationalQuadratic + WhiteKernel` |
| k20¹ | `C * Matern(ls0, bounds=(1e-2, 1e3), ν=2.5)` |

¹ k20 é o **Matern ν=2.5 com bounds estendidos** — é justamente o usado em produção via `constroi_kernel()[-1]` em [[04_Codigo/pages - sapatas.py]]. Os comentários internos da função citam "18–20" (off-by-one residual) — pendente de definição interna se a intenção era ter 20 ou 21 kernels.

## Estudo experimental

Os notebooks [[06_Notebooks/testes_gpr_lucas]] e [[06_Notebooks/testes_otm_lucas]] varrem **escala de penalidade** (1e1 vs 1e6) e **split treino/teste** (10/20/30/40/50%) para cada kernel. Resultados em `assets/graphics/z_GPR_*.png` (40 figuras).

## Heurística atual

- Matern ν=2.5 escolhido em produção: bom equilíbrio entre suavidade e capacidade de ajuste.
- WhiteKernel adicionado quando há ruído numérico.
