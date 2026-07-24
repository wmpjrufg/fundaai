---
tags: [engenharia, sapata, fundacao-rasa]
aliases: [Sapata, Footing]
---

# Sapatas Isoladas

Elemento estrutural que transmite a carga do **pilar** ao **solo**. É o elemento de fundação rasa otimizado pelo FundaIA.

## Geometria (variáveis de projeto)

- `h_x` — dimensão em planta na direção X [m]
- `h_y` — dimensão em planta na direção Y [m]
- `h_z` — altura (espessura) [m]

Volume: `V = h_x · h_y · h_z`.

## Dados de entrada por sapata

- `ap, bp` — dimensões do pilar [m]
- `xg, yg` — coordenadas do centróide do pilar [m]
- `spt` — índice de [[02_Engenharia/SPT - Sondagem]]
- `solo` — pedregulho / areia / silte / argila
- `Fz-c{i}, Mx-c{i}, My-c{i}` — esforços por combinação

Ver [[05_Dados/Schema das Planilhas]].

## Verificações associadas (ver MOC)

- [[02_Engenharia/Tensão Admissível do Solo]]
- [[02_Engenharia/Flexão Composta - Sigma Max e Min]]
- [[02_Engenharia/Verificação à Punção]]
- [[02_Engenharia/Restrição de Geometria]]

## Posicionamento e packing

Várias sapatas no mesmo terreno não podem **se sobrepor**. Esta restrição transforma o problema em um caso de [[03_Otimizacao/Problema de Empacotamento]].
