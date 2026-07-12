"""Excel reader for the FundaIA input spreadsheet.

This module is the **entry point** of the application. Every
foundation project that the optimiser sees enters the system through
``read_projeto_from_excel``. Schema validation, error reporting and
type sanitisation are all centralised here so that the rest of the
core layers can rely on a clean, validated ``FundacaoProjeto``.

Resumo em português:
    Leitor da planilha Excel de entrada. Esta camada é a porta única
    de entrada do projeto: garante schema, sanitiza valores e devolve
    um ``FundacaoProjeto`` validado, livre de surpresas para as camadas
    seguintes.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import IO, Iterable, Union

import pandas as pd

from core.domain import Combinacao, FundacaoProjeto, Pilar, Solo


# --- Schema constants --------------------------------------------------------
REQUIRED_FIXED_COLUMNS: tuple[str, ...] = (
    "Elemento",
    "ap (m)",
    "bp (m)",
    "spt",
    "solo",
    "xg (m)",
    "yg (m)",
)

VALID_SOIL_TYPES: frozenset[str] = frozenset(
    {"pedregulho", "areia", "silte", "argila"}
)

LOAD_PREFIXES: tuple[str, ...] = ("Fz-", "Mx-", "My-")
COMB_PATTERN = re.compile(r"^(Fz|Mx|My)-c(\d+)$")


PathOrBuffer = Union[str, Path, IO[bytes]]


# =============================================================================
# Public API
# =============================================================================
def read_projeto_from_excel(
    path_or_buffer: PathOrBuffer,
    *,
    f_ck_kpa: float,
    cobrimento_m: float,
    sheet_name: int | str = 0,
) -> FundacaoProjeto:
    """This function reads a foundation project from an Excel spreadsheet.

    Strict schema validation is enforced: missing columns, unknown soil
    types, non-positive geometry, gaps in the per-combination columns
    (e.g. ``Fz-c2`` present without ``Mx-c2``) and duplicated element
    labels are all rejected with explicit ``ValueError`` messages.

    The integer ``n_comb`` is inferred from the highest combination
    index found in the columns, and every combination from ``c1`` up to
    ``cN`` must have all three fields (``Fz``, ``Mx``, ``My``) present.
    The legacy ``Mx``/``My`` labels follow the FundaIA convention:
    ``Mx`` is the moment component that produces eccentricity along X,
    and ``My`` the component that produces eccentricity along Y. When
    importing reactions from software that reports moments about the
    global axes, convert them before filling the spreadsheet.

    :param path_or_buffer: Path to an .xlsx/.xls file or a file-like
                           object (e.g. a Streamlit ``UploadedFile``)
    :param f_ck_kpa: Characteristic concrete compressive strength [kPa]
    :param cobrimento_m: Concrete cover [m]
    :param sheet_name: Sheet index or name passed to pandas (default: first sheet)

    :return: Validated FundacaoProjeto entity ready for the engineering layer

    :raises ValueError: When the schema is violated (missing/unexpected
                        columns, gaps in the load combinations,
                        duplicated labels, unknown soil types or invalid
                        global parameters)
    :raises FileNotFoundError: When ``path_or_buffer`` is a path that does not exist
    """
    df = _read_dataframe(path_or_buffer, sheet_name=sheet_name)
    _validate_fixed_schema(df)
    n_comb = _detect_n_comb(df)
    _sanitize_loads_inplace(df, n_comb)
    return _build_projeto(df, n_comb=n_comb, f_ck_kpa=f_ck_kpa, cobrimento_m=cobrimento_m)


# =============================================================================
# Internals
# =============================================================================
def _read_dataframe(path_or_buffer: PathOrBuffer, *, sheet_name: int | str) -> pd.DataFrame:
    """This helper wraps pandas.read_excel with a friendlier missing-file error.

    :param path_or_buffer: File path or file-like buffer to read
    :param sheet_name: Sheet index or name forwarded to pandas

    :return: DataFrame with the raw spreadsheet contents

    :raises FileNotFoundError: When ``path_or_buffer`` is a path that does not exist
    """
    if isinstance(path_or_buffer, (str, Path)) and not Path(path_or_buffer).exists():
        raise FileNotFoundError(f"input spreadsheet not found: {path_or_buffer!s}")
    return pd.read_excel(path_or_buffer, sheet_name=sheet_name)


def _validate_fixed_schema(df: pd.DataFrame) -> None:
    """This helper ensures the seven mandatory columns and a non-empty body are present.

    :param df: Raw DataFrame read from the spreadsheet

    :return: None

    :raises ValueError: When required columns are missing or the body is empty
    """
    missing = [c for c in REQUIRED_FIXED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "spreadsheet is missing required columns: "
            f"{missing}. expected at least {list(REQUIRED_FIXED_COLUMNS)}."
        )
    if df.shape[0] == 0:
        raise ValueError("spreadsheet has no rows; expected at least one foundation.")


def _detect_n_comb(df: pd.DataFrame) -> int:
    """This helper infers the number of load combinations from the column names.

    Scans columns matching the ``Fz-c{i}|Mx-c{i}|My-c{i}`` pattern and
    returns the largest index found, after asserting that no
    intermediate index is missing and that every combination has the
    full ``Fz/Mx/My`` triple.

    :param df: Raw DataFrame read from the spreadsheet

    :return: Number of declared load combinations [int]

    :raises ValueError: When no combination is found, when an
                        intermediate combination index is missing, or
                        when a combination is missing one of Fz/Mx/My
    """
    matches: list[tuple[str, int]] = []
    for col in df.columns:
        m = COMB_PATTERN.match(str(col))
        if m:
            matches.append((m.group(1), int(m.group(2))))

    if not matches:
        raise ValueError(
            "no load combination columns found; expected columns matching "
            "Fz-c{i}, Mx-c{i}, My-c{i} for i = 1, 2, ..."
        )

    indices = sorted({i for _, i in matches})
    expected = list(range(1, indices[-1] + 1))
    if indices != expected:
        missing = sorted(set(expected) - set(indices))
        raise ValueError(
            "load combination indices must be contiguous starting at 1; "
            f"missing combinations: c{missing}."
        )

    n_comb = indices[-1]
    for i in range(1, n_comb + 1):
        for prefix in ("Fz", "Mx", "My"):
            col = f"{prefix}-c{i}"
            if col not in df.columns:
                raise ValueError(
                    f"combination c{i} is incomplete; column {col!r} is missing."
                )
    return n_comb


def _sanitize_loads_inplace(df: pd.DataFrame, n_comb: int) -> None:
    """This helper coerces every load column to float, accepting comma decimals.

    Mirrors the legacy behaviour of ``pages/sapatas.py``: columns
    starting with ``Fz-``, ``Mx-`` or ``My-`` are first cast to string,
    then commas are replaced by dots, then cast back to float. Cells
    that cannot be coerced raise a clear error pointing to the column.

    :param df: DataFrame to sanitise (mutated in place)
    :param n_comb: Expected number of combinations (used to scope the loop)

    :return: None

    :raises ValueError: When a load cell cannot be coerced to float
    """
    cols = [
        f"{prefix}-c{i}"
        for i in range(1, n_comb + 1)
        for prefix in ("Fz", "Mx", "My")
    ]
    for col in cols:
        try:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .astype(float)
            )
        except ValueError as exc:  # pragma: no cover  (mensagem clara pro usuario)
            raise ValueError(
                f"column {col!r} contains non-numeric values that could not "
                f"be coerced to float: {exc}"
            ) from exc


def _build_projeto(
    df: pd.DataFrame,
    *,
    n_comb: int,
    f_ck_kpa: float,
    cobrimento_m: float,
) -> FundacaoProjeto:
    """This helper assembles the validated DataFrame into a FundacaoProjeto.

    Iterates row-wise turning each line into a ``Pilar`` plus the
    matching ``Solo`` and a list of ``Combinacao``. Duplicated element
    labels and unknown soil types are rejected here.

    :param df: Sanitised DataFrame with fixed columns and the load columns
    :param n_comb: Number of declared load combinations
    :param f_ck_kpa: Characteristic concrete compressive strength [kPa]
    :param cobrimento_m: Concrete cover [m]

    :return: FundacaoProjeto root aggregator with validated entities
    """
    pilares: list[Pilar] = []
    solo_por_pilar: dict[str, Solo] = {}
    combinacoes_por_pilar: dict[str, list[Combinacao]] = {}
    seen_labels: set[str] = set()

    for idx, row in df.iterrows():
        rotulo = str(row["Elemento"]).strip()
        if not rotulo or rotulo.lower() == "nan":
            raise ValueError(f"row {int(idx)} has an empty 'Elemento' label.")
        if rotulo in seen_labels:
            raise ValueError(
                f"duplicated 'Elemento' label {rotulo!r}; each row must be unique."
            )
        seen_labels.add(rotulo)

        tipo_solo = str(row["solo"]).strip().lower()
        if tipo_solo not in VALID_SOIL_TYPES:
            raise ValueError(
                f"row {int(idx)} ({rotulo!r}): unknown soil type {tipo_solo!r}; "
                f"expected one of {sorted(VALID_SOIL_TYPES)}."
            )

        try:
            pilar = Pilar(
                rotulo=rotulo,
                a_p=float(row["ap (m)"]),
                b_p=float(row["bp (m)"]),
                xg=float(row["xg (m)"]),
                yg=float(row["yg (m)"]),
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(f"row {int(idx)} ({rotulo!r}) has invalid pillar data: {exc}") from exc

        try:
            solo = Solo(tipo=tipo_solo, spt=float(row["spt"]))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"row {int(idx)} ({rotulo!r}) has invalid soil data: {exc}") from exc

        combs = [
            Combinacao(
                rotulo=f"c{i}",
                f_z=float(row[f"Fz-c{i}"]),
                m_x=float(row[f"Mx-c{i}"]),
                m_y=float(row[f"My-c{i}"]),
            )
            for i in range(1, n_comb + 1)
        ]

        pilares.append(pilar)
        solo_por_pilar[rotulo] = solo
        combinacoes_por_pilar[rotulo] = combs

    return FundacaoProjeto(
        pilares=pilares,
        solo_por_pilar=solo_por_pilar,
        combinacoes_por_pilar=combinacoes_por_pilar,
        f_ck_kpa=f_ck_kpa,
        cobrimento_m=cobrimento_m,
    )
