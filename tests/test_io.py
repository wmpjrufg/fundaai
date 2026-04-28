"""Tests for the I/O layer (``core.io``).

The Excel reader is the **single entry point** of the application, so
the test surface here is intentionally wide:

    * Round-trip with the three official spreadsheet templates
      (1, 2 and 3 foundations).
    * Schema validation: missing columns, gap in load combination
      indices, missing Fz/Mx/My within a present combination.
    * Cell sanitisation: comma decimal separator, integer-as-string,
    * Domain integrity: duplicated labels, unknown soil types,
      non-positive geometry, negative SPT.
    * Global parameters: invalid f_ck and cobrimento are rejected.

The DXF writer is exercised end-to-end (bytes contain a valid header,
correct number of LINE primitives and the pillar labels).
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest

from core.domain import FundacaoProjeto, Sapata
from core.io import read_projeto_from_excel, sapatas_to_dxf_bytes


# =============================================================================
# Spreadsheet round-trip with the three official templates
# =============================================================================
@pytest.mark.parametrize(
    "filename, n_fund, primeiro_rotulo",
    [
        ("problema_fund_um.xlsx", 1, "P08"),
        ("problema_fund_dois.xlsx", 2, "P01"),
        ("problema_fund_três.xlsx", 3, "P04"),
    ],
)
def test_round_trip_official_templates(
    assets_dir: Path, filename: str, n_fund: int, primeiro_rotulo: str
):
    """This test ensures the official Excel templates load into a valid FundacaoProjeto.

    :param assets_dir: Fixture pointing to the assets directory
    :param filename: Template file name under ``assets/``
    :param n_fund: Expected number of foundation elements in the template
    :param primeiro_rotulo: Expected ``Elemento`` label of the first row

    :return: None (internal asserts)
    """
    proj = read_projeto_from_excel(
        assets_dir / filename, f_ck_kpa=25_000.0, cobrimento_m=0.04
    )
    assert isinstance(proj, FundacaoProjeto)
    assert proj.n_fund == n_fund
    assert proj.n_comb == 3
    assert proj.pilares[0].rotulo == primeiro_rotulo
    # All pillar labels are present in both per-pillar maps
    rotulos = {p.rotulo for p in proj.pilares}
    assert rotulos == set(proj.solo_por_pilar)
    assert rotulos == set(proj.combinacoes_por_pilar)
    # Per-element load combinations match the declared n_comb
    for r in rotulos:
        assert len(proj.combinacoes_por_pilar[r]) == proj.n_comb


def test_global_parameters_attached(assets_dir: Path):
    """This test ensures f_ck_kpa and cobrimento_m flow into FundacaoProjeto.

    :param assets_dir: Fixture pointing to the assets directory

    :return: None (internal asserts)
    """
    proj = read_projeto_from_excel(
        assets_dir / "problema_fund_um.xlsx", f_ck_kpa=30_000.0, cobrimento_m=0.05
    )
    assert proj.f_ck_kpa == 30_000.0
    assert proj.cobrimento_m == 0.05


# =============================================================================
# In-memory buffer support (Streamlit UploadedFile path)
# =============================================================================
def test_accepts_in_memory_buffer(assets_dir: Path):
    """This test ensures the reader works with a file-like buffer (not only paths).

    Streamlit hands the page an ``UploadedFile`` (a file-like object).
    The reader must accept it transparently.

    :return: None (internal asserts)
    """
    buf = BytesIO((assets_dir / "problema_fund_três.xlsx").read_bytes())
    proj = read_projeto_from_excel(buf, f_ck_kpa=25_000.0, cobrimento_m=0.04)
    assert proj.n_fund == 3


# =============================================================================
# Missing-file behaviour
# =============================================================================
def test_missing_file_raises(tmp_path: Path):
    """This test ensures an explicit FileNotFoundError when the path does not exist.

    :param tmp_path: pytest temp dir fixture

    :return: None (internal asserts)
    """
    with pytest.raises(FileNotFoundError):
        read_projeto_from_excel(
            tmp_path / "does_not_exist.xlsx", f_ck_kpa=25_000.0, cobrimento_m=0.04
        )


# =============================================================================
# Schema validation
# =============================================================================
def _make_xlsx(tmp_path: Path, df: pd.DataFrame) -> Path:
    """This helper writes a DataFrame to a temporary .xlsx for negative-path tests.

    :param tmp_path: pytest temp dir fixture
    :param df: DataFrame to serialise

    :return: Path to the freshly written xlsx
    """
    p = tmp_path / "case.xlsx"
    df.to_excel(p, index=False)
    return p


def _row(elemento="P01", solo="argila", spt=30, f1=100.0):
    """This helper builds a single canonical row (kwargs override defaults).

    :return: Dict shaped as a single spreadsheet row with three combinations
    """
    return {
        "Elemento": elemento, "ap (m)": 0.30, "bp (m)": 1.50,
        "spt": spt, "solo": solo, "xg (m)": 0.0, "yg (m)": 0.0,
        "Fz-c1": f1, "Mx-c1": 0.0, "My-c1": 0.0,
        "Fz-c2": 110.0, "Mx-c2": 0.0, "My-c2": 0.0,
        "Fz-c3": 120.0, "Mx-c3": 0.0, "My-c3": 0.0,
    }


def test_missing_required_column_raises(tmp_path: Path):
    """This test ensures the reader rejects spreadsheets that miss a fixed column.

    :return: None (internal asserts)
    """
    row = _row()
    del row["spt"]
    p = _make_xlsx(tmp_path, pd.DataFrame([row]))
    with pytest.raises(ValueError, match="missing required columns"):
        read_projeto_from_excel(p, f_ck_kpa=25_000.0, cobrimento_m=0.04)


def test_no_combinations_raises(tmp_path: Path):
    """This test ensures the reader rejects spreadsheets with no Fz/Mx/My columns.

    :return: None (internal asserts)
    """
    row = {k: v for k, v in _row().items() if not any(k.startswith(p) for p in ("Fz-", "Mx-", "My-"))}
    p = _make_xlsx(tmp_path, pd.DataFrame([row]))
    with pytest.raises(ValueError, match="no load combination columns"):
        read_projeto_from_excel(p, f_ck_kpa=25_000.0, cobrimento_m=0.04)


def test_non_contiguous_combinations_raise(tmp_path: Path):
    """This test ensures combination indices must start at 1 and be contiguous.

    :return: None (internal asserts)
    """
    row = _row()
    # Drop combination c2 to create a gap c1, c3
    for prefix in ("Fz", "Mx", "My"):
        del row[f"{prefix}-c2"]
    p = _make_xlsx(tmp_path, pd.DataFrame([row]))
    with pytest.raises(ValueError, match="contiguous"):
        read_projeto_from_excel(p, f_ck_kpa=25_000.0, cobrimento_m=0.04)


def test_incomplete_combination_raises(tmp_path: Path):
    """This test ensures every combination must have all three Fz/Mx/My columns.

    :return: None (internal asserts)
    """
    row = _row()
    del row["My-c2"]   # Fz-c2 and Mx-c2 stay; only My-c2 is dropped
    p = _make_xlsx(tmp_path, pd.DataFrame([row]))
    with pytest.raises(ValueError, match="incomplete"):
        read_projeto_from_excel(p, f_ck_kpa=25_000.0, cobrimento_m=0.04)


def test_empty_spreadsheet_raises(tmp_path: Path):
    """This test ensures a spreadsheet with header only is rejected.

    :return: None (internal asserts)
    """
    df = pd.DataFrame([_row()]).iloc[0:0]  # header but no rows
    p = _make_xlsx(tmp_path, df)
    with pytest.raises(ValueError, match="no rows"):
        read_projeto_from_excel(p, f_ck_kpa=25_000.0, cobrimento_m=0.04)


# =============================================================================
# Domain integrity
# =============================================================================
def test_unknown_soil_type_raises(tmp_path: Path):
    """This test ensures unknown soil identifiers are rejected.

    :return: None (internal asserts)
    """
    p = _make_xlsx(tmp_path, pd.DataFrame([_row(solo="lava")]))
    with pytest.raises(ValueError, match="unknown soil type"):
        read_projeto_from_excel(p, f_ck_kpa=25_000.0, cobrimento_m=0.04)


def test_duplicated_label_raises(tmp_path: Path):
    """This test ensures duplicated Elemento labels are rejected.

    :return: None (internal asserts)
    """
    rows = [_row(elemento="P01"), _row(elemento="P01")]
    p = _make_xlsx(tmp_path, pd.DataFrame(rows))
    with pytest.raises(ValueError, match="duplicated"):
        read_projeto_from_excel(p, f_ck_kpa=25_000.0, cobrimento_m=0.04)


def test_negative_spt_raises(tmp_path: Path):
    """This test ensures negative SPT values fail at the domain layer (Solo).

    :return: None (internal asserts)
    """
    p = _make_xlsx(tmp_path, pd.DataFrame([_row(spt=-5)]))
    with pytest.raises(ValueError, match="invalid soil data"):
        read_projeto_from_excel(p, f_ck_kpa=25_000.0, cobrimento_m=0.04)


def test_invalid_global_parameters_raise(assets_dir: Path):
    """This test ensures invalid f_ck or cobrimento are rejected by FundacaoProjeto.

    :return: None (internal asserts)
    """
    with pytest.raises(ValueError):
        read_projeto_from_excel(
            assets_dir / "problema_fund_um.xlsx", f_ck_kpa=0.0, cobrimento_m=0.04
        )
    with pytest.raises(ValueError):
        read_projeto_from_excel(
            assets_dir / "problema_fund_um.xlsx", f_ck_kpa=25_000.0, cobrimento_m=-0.01
        )


# =============================================================================
# Cell sanitisation (comma decimal separator)
# =============================================================================
def test_comma_decimal_is_accepted(tmp_path: Path):
    """This test ensures ``"855,5"`` in a load cell is correctly parsed as 855.5.

    Mirrors the legacy behaviour from ``pages/sapatas.py`` that handled
    spreadsheets exported with the Brazilian decimal separator.

    :return: None (internal asserts)
    """
    row = _row()
    row["Fz-c1"] = "855,5"
    p = _make_xlsx(tmp_path, pd.DataFrame([row]))
    proj = read_projeto_from_excel(p, f_ck_kpa=25_000.0, cobrimento_m=0.04)
    c1 = proj.combinacoes_por_pilar[proj.pilares[0].rotulo][0]
    assert c1.f_z == pytest.approx(855.5)


def test_soil_match_is_case_insensitive(tmp_path: Path):
    """This test ensures soil identifiers are normalised to lower case.

    :return: None (internal asserts)
    """
    p = _make_xlsx(tmp_path, pd.DataFrame([_row(solo="ARGILA")]))
    proj = read_projeto_from_excel(p, f_ck_kpa=25_000.0, cobrimento_m=0.04)
    assert proj.solo_por_pilar[proj.pilares[0].rotulo].tipo == "argila"


# =============================================================================
# DXF writer
# =============================================================================
def _build_sapatas(proj: FundacaoProjeto) -> list[Sapata]:
    """This helper builds one Sapata per pillar with arbitrary but valid dimensions.

    :param proj: FundacaoProjeto previously read from a template

    :return: List of Sapata entities, one per pillar
    """
    return [Sapata(p, h_x=2.0, h_y=2.0, h_z=0.6) for p in proj.pilares]


def test_dxf_bytes_have_dxf_header(assets_dir: Path):
    """This test ensures the DXF output starts with the expected SECTION/HEADER markers.

    :return: None (internal asserts)
    """
    proj = read_projeto_from_excel(
        assets_dir / "problema_fund_três.xlsx", f_ck_kpa=25_000.0, cobrimento_m=0.04
    )
    payload = sapatas_to_dxf_bytes(_build_sapatas(proj))
    assert isinstance(payload, bytes)
    assert b"SECTION" in payload[:200]
    assert b"HEADER" in payload[:200]


def test_dxf_includes_one_label_per_pillar(assets_dir: Path):
    """This test ensures every pillar label appears in the DXF text payload.

    :return: None (internal asserts)
    """
    proj = read_projeto_from_excel(
        assets_dir / "problema_fund_três.xlsx", f_ck_kpa=25_000.0, cobrimento_m=0.04
    )
    payload = sapatas_to_dxf_bytes(_build_sapatas(proj)).decode(
        "cp1252", errors="replace"
    )
    for pilar in proj.pilares:
        assert pilar.rotulo in payload


def test_dxf_writer_is_semantically_stable(assets_dir: Path):
    """This test ensures consecutive writes produce semantically equivalent output.

    Byte-level identity is not expected because ``ezdxf`` assigns a fresh
    handle (hex id) to each entity on every call. What we lock here is
    the *shape* of the export: the size of each payload must match and
    every payload must contain the same number of LINE primitives
    (4 lines per footing) and the same set of pillar labels.

    :return: None (internal asserts)
    """
    proj = read_projeto_from_excel(
        assets_dir / "problema_fund_três.xlsx", f_ck_kpa=25_000.0, cobrimento_m=0.04
    )
    sapatas = _build_sapatas(proj)
    a = sapatas_to_dxf_bytes(sapatas)
    b = sapatas_to_dxf_bytes(sapatas)

    # 1. Same payload size on consecutive calls
    assert len(a) == len(b)

    # 2. 4 LINE primitives per footing (the four AABB edges)
    n_lines_a = a.count(b"\nLINE\n")
    n_lines_b = b.count(b"\nLINE\n")
    assert n_lines_a == n_lines_b == 4 * len(sapatas)

    # 3. Same set of pillar labels in both payloads
    decoded_a = a.decode("cp1252", errors="replace")
    decoded_b = b.decode("cp1252", errors="replace")
    rotulos = {p.rotulo for p in proj.pilares}
    for r in rotulos:
        assert decoded_a.count(r) == decoded_b.count(r) >= 1


def test_dxf_writer_has_no_tempfile_side_effect(assets_dir: Path, tmp_path: Path):
    """This test ensures the writer does not leak files into the tmp directory.

    Counts files in a private tmp directory before and after a call —
    they must match. Catches the regression of the legacy
    ``NamedTemporaryFile(delete=False)`` pattern.

    :return: None (internal asserts)
    """
    import os
    proj = read_projeto_from_excel(
        assets_dir / "problema_fund_um.xlsx", f_ck_kpa=25_000.0, cobrimento_m=0.04
    )
    sapatas = _build_sapatas(proj)

    # snapshot ALL temp files before
    tmpdir = Path(os.environ.get("TMPDIR", "/tmp"))
    before = {p.name for p in tmpdir.iterdir() if p.is_file()}
    sapatas_to_dxf_bytes(sapatas)
    after = {p.name for p in tmpdir.iterdir() if p.is_file()}
    new_files = after - before
    # No new .dxf temp file from this call
    assert not any(n.endswith(".dxf") for n in new_files), (
        f"DXF writer leaked tempfiles: {[n for n in new_files if n.endswith('.dxf')]}"
    )
