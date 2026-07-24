---
tags: [issue, baixo, dead-code]
file: metapy_toolbox/methods.py
severity: baixo
---

# Issue — `methods.py` 100% comentado

## Sintoma

Todo o conteúdo de `metapy_toolbox/methods.py` está comentado (versão antiga de `initial_population_01`). Mesmo assim, é importado por `__init__.py`:

```python
from .methods import *
```

## Por que é problema

- Confunde leitores (parece haver código).
- Polui exports (felizmente, sem efeito porque o arquivo não exporta nada).

## Correção sugerida

Deletar `methods.py` e remover a linha `from .methods import *` do `__init__.py`.

## Vínculo

- [[04_Codigo/metapy_toolbox - methods.py]]
- [[04_Codigo/metapy_toolbox - __init__.py]]
