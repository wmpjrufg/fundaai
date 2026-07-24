---
tags: [melhorias, devops, sugestao]
---

# Higiene — `requirements` e venv

> [!note] Sugestão
> Vários problemas pequenos somados; nenhum é difícil isolado.

## Pendências

1. Resolver [[07_Issues/Issue - requirements.txt UTF-16]] (re-salvar em UTF-8).
2. Adicionar imports faltantes que o código de fato usa: `pandas`, `numpy`, `scipy`, `matplotlib`, `joblib`.
3. Migrar para **`pyproject.toml`** (PEP 621) em vez de `requirements.txt`. Vantagens:
   - Define versão do Python.
   - Suporta `[project.optional-dependencies]` para `dev` (pytest, ruff) e `ops` (playwright).
4. Trocar `pip-chill` por `uv pip compile` ou `pip-tools` (`pip-compile`) — gera `requirements.lock` reprodutível.
5. Adotar **`uv`** (Astral) ou **`hatch`** para gerenciar venvs — bem mais rápido que `python -m venv` + `pip install`.
6. `.gitignore` linha 1 ignora `*.txt` — pode mascarar arquivos importantes; revisar regra.

## Layout sugerido

```
pyproject.toml
requirements.lock          # gerado, commitado para reproducibilidade
.python-version            # e.g. 3.11
ops/
  pyproject.toml           # ou requirements separado
.venv/                     # já no .gitignore
```

## Vínculos

- [[07_Issues/Issue - requirements.txt UTF-16]]
- [[04_Codigo/env-setup.py]]
