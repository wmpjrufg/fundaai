"""I/O layer — adapters between persistence formats and the domain layer.

Exposes the spreadsheet reader (``read_projeto_from_excel``), the DXF
writer (``sapatas_to_dxf_bytes``) and the experiment persistence
front-end (``ExperimentRecorder`` / ``load_experiment``).

Resumo em português:
    Camada de I/O. Reúne o leitor de planilhas Excel
    (``read_projeto_from_excel``), o gerador de DXF
    (``sapatas_to_dxf_bytes``) e a persistência de experimentos
    (``ExperimentRecorder`` e ``load_experiment``).
"""

from .excel import read_projeto_from_excel
from .cad_dxf import sapatas_to_dxf_bytes
from .experiments import (
    SCHEMA_VERSION,
    ExperimentManifest,
    ExperimentRecorder,
    ExperimentRun,
    compute_metrics,
    load_experiment,
    summarise_history,
)

__all__ = [
    "read_projeto_from_excel",
    "sapatas_to_dxf_bytes",
    "SCHEMA_VERSION",
    "ExperimentManifest",
    "ExperimentRecorder",
    "ExperimentRun",
    "compute_metrics",
    "load_experiment",
    "summarise_history",
]
