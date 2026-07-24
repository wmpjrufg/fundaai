---
tags: [issue, baixo, recursos, resolvido]
file: pages/sapatas.py
severity: baixo
status: resolvido
resolvido_em: 2026-04-28
resolvido_em_branch: refactor/core-architecture
---

# Issue — `save_dxf` deixa tempfile órfão em `/tmp`

> [!success] Resolvido em 2026-04-28 (Sprint 3.4 + 3.5, branch `refactor/core-architecture`)
> A função legada `save_dxf` foi substituída por `core.io.sapatas_to_dxf_bytes`,
> que escreve o DXF inteiramente em memória via `io.StringIO` +
> `doc.write(buf)`. Não há mais `tempfile`, não há mais `delete=False`,
> não há mais arquivos órfãos em `/tmp`.
>
> O teste `test_dxf_writer_has_no_tempfile_side_effect` em
> `tests/test_io.py` é a salvaguarda regressiva: tira um snapshot do
> `TMPDIR` antes e depois de uma chamada e falha se algum `.dxf` novo
> aparecer.
>
> A migração foi finalizada na Sprint 3.5, quando `pages/sapatas.py`
> deixou de definir `save_dxf` localmente e passou a chamar
> `sapatas_to_dxf_bytes` da camada `core.io`.

## Sintoma original

Em [[04_Codigo/pages - sapatas.py]], `save_dxf(data)`:

```python
temp_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".dxf").name
doc.saveas(temp_file_path)
with open(temp_file_path, "rb") as file:
    return file.read()
```

`delete=False` impedia a remoção automática, e nada chamava `os.remove(temp_file_path)` depois. Cada execução deixava um arquivo `.dxf` órfão no `/tmp`.

## Por que era problema

- Em produção (Streamlit Cloud), espaço em `/tmp` é limitado.
- Em desenvolvimento local, era só sujeira mas é **má prática**.

## Correção aplicada

Em `core/io/cad_dxf.py`:

```python
def sapatas_to_dxf_bytes(sapatas, *, dxf_version="R2010", text_height=0.20) -> bytes:
    doc = ezdxf.new(dxfversion=dxf_version)
    msp = doc.modelspace()
    for sapata in sapatas:
        v_sw, v_se, v_ne, v_nw = sapata.vertices
        msp.add_line(v_sw, v_se); msp.add_line(v_se, v_ne)
        msp.add_line(v_ne, v_nw); msp.add_line(v_nw, v_sw)
        centre = (sapata.pilar.xg, sapata.pilar.yg)
        msp.add_point(centre)
        msp.add_text(sapata.pilar.rotulo, dxfattribs={"height": text_height}) \
           .set_dxf_attrib("insert", centre)
    text_buffer = io.StringIO()
    doc.write(text_buffer)
    encoding = getattr(doc, "encoding", "utf-8") or "utf-8"
    return text_buffer.getvalue().encode(encoding, errors="replace")
```

`pages/sapatas.py` agora apenas chama:

```python
from core.io import sapatas_to_dxf_bytes
...
dxf_bytes = sapatas_to_dxf_bytes(result.sapatas)
```

## Vínculo

- [[04_Codigo/pages - sapatas.py]]
- [[12_Auditoria/Sprint 3.4 - IO layer - 2026-04-28]]
- [[12_Auditoria/Sprint 3.5 - API layer - 2026-04-28]]
- [[07_Issues/Lista Mestre de Issues]]
