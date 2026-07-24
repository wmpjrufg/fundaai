---
tags: [melhorias, ciencia, sugestao]
---

# Persistência de Experimentos

> [!note] Sugestão
> Cada execução do otimizador é um **experimento** com config, ambiente e resultado. Sem registro estruturado, comparar 20 runs vira tarefa manual.

## Layout sugerido

```
experiments/
  2026-04-27T14-30_<gitsha>/
    config.json            # Pydantic dump
    manifest.json          # seed, gitsha, hash dos inputs
    pop_inicial.parquet
    history.parquet        # toda a trajetória de avaliações
    resultado.parquet      # melhor x e métricas
    plot_layout.png
    log.jsonl              # logs estruturados
```

## Ferramentas

- **MLflow** — rastreamento de runs, métricas, artefatos. Roda local sem servidor.
- **Weights & Biases** — opcional, dá UI bonita mas é cloud.
- **DVC** — para versionamento de dados grandes (não necessário agora).
- **Caminho minimalista**: só `parquet` + `json` na pasta — sem dependência extra.

## Métricas sugeridas

Por run:
- `best_of_final`
- `best_of_trajectory(t)` — converge?
- `n_evals_para_atingir_X`
- `tempo_total`
- `max_violacao_g_*` na solução final
- Hash do `projeto` (input)

## Comparação cross-run

Notebook `analise_experimentos.ipynb` que:
1. Lê todas as pastas em `experiments/`.
2. Concatena `manifest + resultado` em DataFrame.
3. Plot `best_of` × algoritmo, kernel, penalidade, etc.

## Vínculos

- [[10_Melhorias/Reprodutibilidade - Seeds e Versão]]
- [[10_Melhorias/Logging Estruturado]]
- [[10_Melhorias/Validação contra problema-benchmark]]
