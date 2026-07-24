---
tags: [melhorias, observabilidade, sugestao]
---

# Logging Estruturado

> [!note] Sugestão
> Hoje há `print` solto e `report_move` em strings concatenadas. Usar `logging` (ou `loguru`/`structlog`) facilita debug e geração de relatórios.

## Esquema sugerido

```python
import logging
log = logging.getLogger("fundaia.optimization.ego")
log.info("ego_iter", extra={"t": t, "best_of": df['OF'].min(), "n_evals": len(df)})
```

## Por que importa para a IC

- Reproduzir resultados ⇒ logs com seed, versão e config.
- Debug de convergência ⇒ trajetória `best_of(t)` em CSV.
- Geração de tabelas de resultados em LaTeX ⇒ direto do log estruturado.

## Alternativa moderna

`structlog` + saída JSON ⇒ análise posterior com `jq`/pandas.

## Vínculos

- [[10_Melhorias/Persistência de Experimentos]]
- [[10_Melhorias/Reprodutibilidade - Seeds e Versão]]
