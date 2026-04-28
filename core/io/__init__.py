"""I/O layer — reading project data and writing results.

This subpackage will expose adapters between the persistence formats
used in practice (Excel spreadsheets, DXF drawings) and the in-memory
domain objects from ``core.domain``. Streamlit-specific adapters are
not part of this layer; they live in ``pages/``.

Resumo em português:
    Camada de I/O. Adaptadores entre formatos de arquivo (Excel, DXF)
    e os objetos de domínio (``core.domain``). Adaptadores específicos
    do Streamlit ficam em ``pages/``.
"""
