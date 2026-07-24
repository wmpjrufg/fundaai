---
tags: [melhorias, devops, ci, sugestao]
---

# CI/CD — Lint, Test, Build

> [!note] Sugestão
> Adicionar GitHub Actions com pipeline mínimo: rodar testes a cada PR.

## Workflow proposto (`.github/workflows/ci.yml`)

```yaml
name: CI
on: [push, pull_request]
jobs:
  lint-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --extra dev
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run mypy core/
      - run: uv run pytest --cov=core --cov-report=xml
      - uses: codecov/codecov-action@v4
```

## Ferramentas modernas

- **ruff** (lint + format) — substitui `flake8` + `black` + `isort`.
- **mypy** ou **pyright** — tipagem estática.
- **pytest** + **coverage**.
- **pre-commit** — roda lint local antes de commitar.

## Por que importa

- Pega [[07_Issues/Issue - Duplicação em sapatas.py]] automaticamente (ruff `F811` para nomes duplicados).
- Bloqueia regressão de testes (ver [[10_Melhorias/Testes Automatizados]]).
- Garante que `requirements.txt` está sintaticamente válido (resolve [[07_Issues/Issue - requirements.txt UTF-16]] na primeira execução).

## Vínculos

- [[10_Melhorias/Testes Automatizados]]
- [[10_Melhorias/Higiene - requirements e venv]]
