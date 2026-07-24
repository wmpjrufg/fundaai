---
tags: [codigo, biblioteca]
file: metapy_toolbox/__init__.py
---

# `metapy_toolbox/__init__.py`

Re-export `*` de todos os submódulos:

```python
from .methods import *           # vazio (arquivo todo comentado)
from .funcs import *
from .benchmark import *
from .genetic_algorithm import *
from .grey_wolf import *
from .ego import *
```

Nota: `from .methods import *` não erra porque o arquivo só contém comentários, mas é semântica suspeita. Ver [[07_Issues/Issue - methods.py morto]].

## Submódulos

- [[04_Codigo/metapy_toolbox - funcs.py]]
- [[04_Codigo/metapy_toolbox - benchmark.py]]
- [[04_Codigo/metapy_toolbox - genetic_algorithm.py]]
- [[04_Codigo/metapy_toolbox - grey_wolf.py]]
- [[04_Codigo/metapy_toolbox - ego.py]]
- [[04_Codigo/metapy_toolbox - methods.py]] ⚠️
