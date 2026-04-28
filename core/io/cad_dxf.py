"""DXF (AutoCAD) writer for the in-plane footing layout.

Replaces the legacy ``save_dxf`` from ``pages/sapatas.py`` that used
``NamedTemporaryFile(delete=False)`` and never cleaned up. The new
implementation writes the DXF straight into an in-memory text buffer,
encodes it to bytes and avoids any filesystem footprint.

Resumo em português:
    Geração do arquivo DXF com o arranjo das sapatas. A versão antiga
    deixava arquivos órfãos em ``/tmp``; aqui o conteúdo é montado
    inteiramente em memória.
"""

from __future__ import annotations

import io
from typing import Iterable

import ezdxf

from core.domain import Sapata


DXF_VERSION: str = "R2010"
TEXT_HEIGHT_M: float = 0.20


def sapatas_to_dxf_bytes(
    sapatas: Iterable[Sapata],
    *,
    dxf_version: str = DXF_VERSION,
    text_height: float = TEXT_HEIGHT_M,
) -> bytes:
    """This function builds an AutoCAD DXF for an in-plane layout of footings.

    Each footing is drawn as four edges connecting its four AABB
    vertices (delivered by ``Sapata.vertices``), plus a centre point at
    the column centroid and a label with the pillar identifier. The
    resulting drawing matches the schema of the legacy ``save_dxf``
    used by ``pages/sapatas.py``.

    The DXF document is rendered to a text buffer (``io.StringIO``) and
    encoded with the document's own charset (typically ``cp1252``);
    binary buffers are not supported by ``ezdxf.Drawing.write`` in the
    pinned version of the library.

    :param sapatas: Iterable of Sapata entities representing the layout
    :param dxf_version: AutoCAD DXF version tag, default ``"R2010"``
    :param text_height: Height of the label text in drawing units (metres), default 0.20

    :return: Binary DXF content suitable for download or disk persistence
    """
    doc = ezdxf.new(dxfversion=dxf_version)
    msp = doc.modelspace()

    for sapata in sapatas:
        v_sw, v_se, v_ne, v_nw = sapata.vertices
        msp.add_line(v_sw, v_se)
        msp.add_line(v_se, v_ne)
        msp.add_line(v_ne, v_nw)
        msp.add_line(v_nw, v_sw)

        centre = (sapata.pilar.xg, sapata.pilar.yg)
        msp.add_point(centre)
        msp.add_text(
            sapata.pilar.rotulo,
            dxfattribs={"height": text_height},
        ).set_dxf_attrib("insert", centre)

    text_buffer = io.StringIO()
    doc.write(text_buffer)
    encoding = getattr(doc, "encoding", "utf-8") or "utf-8"
    return text_buffer.getvalue().encode(encoding, errors="replace")
