# Sprint B0 - Plano Frente 2 empacotamento e bulbo

**Data:** 2026-07-12  
**Branch:** `codex/frente-2-binpacking`  
**Base preservada:** `codex/lucas-frente-1-artigo`, commit `fdba0ed90`.

## Decisão

A Frente 2 foi separada da Frente 1. A Frente 1 fica preservada como artigo de pré-dimensionamento geométrico experimental. A Frente 2 passa a estudar posicionamento conjunto de sapatas, empacotamento em planta e interação geotécnica por bulbo de tensões.

## Documento principal

- `docs/relatorios/plano_frente_2_empacotamento_bulbo_2026-07-12.md`

## Pontos técnicos principais

- Não usar diretamente a API do projeto `bin_packing_3d` como motor do FundaIA, porque o problema externo é 3D-BPP clássico e não contempla as verificações estruturais/geotécnicas das sapatas.
- Reaproveitar conceitos do projeto externo:
  - validação AABB;
  - restrição de distância mínima;
  - registro de convergência;
  - separação domínio/validador/solver/experimento;
  - referências de Extreme Points e Simulated Annealing.
- Formular a Frente 2 com variáveis `h_x, h_y, h_z, dx, dy`.
- Manter o avaliador da Frente 1 intacto e criar um avaliador novo para layout.
- Incluir bulbo de tensões como restrição/índice opcional e paramétrico, não como distância fixa arbitrária.

## Bulbo de tensões

Decisão metodológica:

```text
começar com aproximação 2V:1H para triagem
depois implementar Boussinesq/Fadum por superposição
estudar R_lim em valores paramétricos, por exemplo 0.10, 0.20 e 0.30
```

A restrição recomendada não é "distância mínima fixa"; é uma métrica de acréscimo de tensão vertical induzido por sapatas vizinhas em pontos e profundidades de controle.

## Próximo passo

Sprint B1:

- criar `core/engineering/layout.py`;
- extrair contenção, limites de lote, distância e sobreposição;
- escrever testes unitários geométricos;
- manter artigo 1 sem alterações.

