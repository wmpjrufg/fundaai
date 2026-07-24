---
tags: [projeto, ic]
---

# Escopo da IC

Iniciação Científica conduzida sob orientação do **Prof. Wanderley**, voltada ao desenvolvimento de um **software de otimização de fundações rasas** que combine:

- **Mecânica de estruturas**: garantir que cada sapata respeite as verificações de tensão no solo, punção e geometria mínima conforme [[02_Engenharia/NBR 6118]].
- **Problema do empacotamento**: posicionar/dimensionar as sapatas no plano de modo a **minimizar volume total** sem que vizinhas se sobreponham.

## Pergunta de pesquisa central

> [!todo] A formalizar
> A formulação definitiva será consolidada pela equipe do projeto sob orientação do Prof. Wanderley. A direção geral é:
>
> **Foco**: identificar um método — baseado em **metaheurísticas** e/ou **hibridizações** (ex.: surrogate-assisted, memético, multi-início, etc.) — capaz de obter uma **solução de boa qualidade** para o problema acoplado de **dimensionamento + posicionamento de fundações rasas** sob restrições de mecânica das estruturas e empacotamento.
>
> _Este bloco será substituído pela formulação definitiva da pergunta._

## Subobjetivos atuais

1. Construir uma **FO penalizada** que exprima as 4 restrições. ✅ feito em [[04_Codigo/fundacao.py]].
2. Implementar um motor **EGO híbrido** com surrogate intercambiável. ✅ feito em [[04_Codigo/metapy_toolbox - ego.py]].
3. Estudar a **escolha de kernels** GPR. 🔄 ver [[06_Notebooks/testes_otm_lucas]] e [[05_Dados/Modelos GPR Treinados]].
4. Estudar o **efeito da escala de penalidade** (1e1 vs 1e6). 🔄 ver [[06_Notebooks/testes_otm_lucas]].
5. Disponibilizar uma **interface web** acessível ao engenheiro. ✅ feito em [[04_Codigo/pages - sapatas.py]].

## Saídas esperadas

- Conjunto de dimensões `(h_x, h_y, h_z)` ótimas por fundação.
- Visualização do **arranjo em planta** (matplotlib) e exportação **DXF** para CAD.
- Tabela de verificações detalhadas (tensões, punção, sobreposição, geometria).
- Contexto acadêmico consolidado em [[01_Projeto/Contexto Acadêmico - IC Lucas e TCC Filipe Amaral]].

## Links

- [[01_Projeto/Visão Geral do Projeto]]
- [[03_Otimizacao/Formulação do Problema]]
