"""I/O layer — adapters between persistence formats and the domain layer.

Exposes the spreadsheet reader (``read_projeto_from_excel``) and the
DXF writer (``sapatas_to_dxf_bytes``). These are the only two functions
that the UI/CLI should need in order to talk to disk.

Resumo em português:
    Camada de I/O. Reúne o leitor de planilhas Excel (``read_projeto_from_excel``)
    e o gerador de DXF (``sapatas_to_dxf_bytes``).
"""

from .excel import read_projeto_from_excel
from .cad_dxf import sapatas_to_dxf_bytes

__all__ = [
    "read_projeto_from_excel",
    "sapatas_to_dxf_bytes",
]
