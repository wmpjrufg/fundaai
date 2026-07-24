---
tags: [melhorias, engenharia, nbr6118, sugestao]
status: implementada
---

# Punção Seção C' — completar

> [!success] Implementada
> A verificação C′ foi implementada na Sprint 5.2 em `core/engineering/puncao.py`, com contorno a `2d` da face do pilar. Esta nota permanece como registro da melhoria concluída.

## Bloqueios identificados

1. `rho_minimo_fck(f_ck)` — resolvida como `rho_minimo_flexao`.
2. `tabela_19_2(c_1/c_2)` — resolvida com interpolação/saturação dos coeficientes `kx`, `ky`.
3. `sigma_cp` — mantida como zero/sem contribuição extra nesta etapa, hipótese declarada.

## Plano para completar

1. [x] Implementar taxa mínima de armadura de flexão.
2. [x] Implementar Tabela 19.2 com interpolação linear nos valores tabelados.
3. [x] Adicionar a contribuição dos momentos `M_x`, `M_y` ao τ_sd1 conforme equação 19.5.3.4.

## Validação

Comparar com **exemplo resolvido** de bibliografia (Carvalho & Pinheiro, "Cálculo e Detalhamento de Estruturas Usuais de Concreto Armado" — vol. 2).

Adicionar caso ao [[10_Melhorias/Testes Automatizados]].

## Vínculos

- [[02_Engenharia/Verificação à Punção]]
- [[02_Engenharia/NBR 6118]]
- [[07_Issues/Issue - Punção seção C linha comentada]]
