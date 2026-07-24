---
tags: [melhorias, refactor, packaging, sugestao]
---

# Refactor — Empacotar `metapy_toolbox`

> [!note] Sugestão
> A biblioteca `metapy_toolbox/` parece ter potencial para ser **um pacote independente** (já existe um `metapy` em PyPI — provavelmente diferente, conferir naming). Se for produto do laboratório, vale formalizar.

## Mínimo

- Adicionar `pyproject.toml` na raiz da `metapy_toolbox/`.
- Versionamento semântico (`__version__`).
- README com exemplos de cada algoritmo.
- Testes contra benchmark de [[04_Codigo/metapy_toolbox - benchmark.py]].

## Mudanças simbólicas

- `metapy_toolbox/methods.py` (todo comentado) — deletar (ver [[07_Issues/Issue - methods.py morto]]).
- API pública via `__all__` em vez de `from X import *` agressivo.
- Tipos em todas as funções públicas.

## Distribuir?

Avaliar, sob orientação do Prof. Wanderley, a pertinência de publicar a biblioteca como pacote do laboratório, sob licença MIT/BSD. Se aprovado:

- CI no GitHub publica em PyPI a cada tag.
- O FundaIA passa a depender dele em `requirements.txt` em vez de tê-lo embutido.

## Vínculos

- [[04_Codigo/metapy_toolbox - __init__.py]]
- [[10_Melhorias/Refactor - Plano Geral]]
