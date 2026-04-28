"""Backwards-compatibility shim for the legacy ``metapy_toolbox`` import path.

The optimisation library was migrated to ``core.optimization`` in
Sprint 3.6. This package now exists solely to preserve the public
import surface used by historical notebooks (``testes_otm.ipynb``,
``testes_otm_lucas.ipynb``, ``testes_gpr_lucas.ipynb``) and by any
external consumer that imported from ``metapy_toolbox``.

Resumo em português:
    Camada de compatibilidade. A biblioteca de otimização vive agora em
    ``core.optimization``; este pacote apenas reexporta tudo para que
    notebooks legados (``from metapy_toolbox import ego_01_architecture``)
    continuem funcionando sem alteração.
"""

from core.optimization import *   # noqa: F401, F403  (intentional re-export)
