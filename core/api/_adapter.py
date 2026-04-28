"""Internal bridge between domain entities and the legacy DataFrame format.

The legacy ``fundacao._avaliar_projeto`` consumes a ``pandas.DataFrame``
with very specific column names (``Elemento``, ``ap (m)``, ``bp (m)``,
``spt``, ``solo``, ``xg (m)``, ``yg (m)``, ``Fz-c{i}``, ``Mx-c{i}``,
``My-c{i}``). The new API accepts ``FundacaoProjeto`` entities; this
adapter rebuilds the DataFrame in the exact format that the legacy
function expects.

Keeping the adapter in a private module makes the dependency on the
DataFrame layout explicit and isolated. Sprint 3.8 (vectorisation) will
let us drop it altogether.

Resumo em português:
    Adaptador interno que converte ``FundacaoProjeto`` no DataFrame
    no formato esperado pela versão legada de ``_avaliar_projeto``.
    Mantém a regressão numérica intocada enquanto a camada de domínio
    se firma.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from core.domain import FundacaoProjeto, Sapata


def projeto_to_dataframe(projeto: FundacaoProjeto) -> pd.DataFrame:
    """This function rebuilds the legacy DataFrame layout from a ``FundacaoProjeto``.

    The output mirrors what ``pd.read_excel`` produces from one of the
    official templates: 7 fixed columns followed by ``3 * n_comb`` load
    columns (``Fz-c{i}``, ``Mx-c{i}``, ``My-c{i}`` for each combination).

    :param projeto: Validated FundacaoProjeto root aggregator

    :return: DataFrame ready to be passed as ``args[0]`` to
             ``fundacao._avaliar_projeto``
    """
    rows: list[dict[str, object]] = []
    for pilar in projeto.pilares:
        solo = projeto.solo_por_pilar[pilar.rotulo]
        combs = projeto.combinacoes_por_pilar[pilar.rotulo]
        row: dict[str, object] = {
            "Elemento": pilar.rotulo,
            "ap (m)": float(pilar.a_p),
            "bp (m)": float(pilar.b_p),
            "spt": float(solo.spt),
            "solo": solo.tipo,
            "xg (m)": float(pilar.xg),
            "yg (m)": float(pilar.yg),
        }
        for comb in combs:
            row[f"Fz-{comb.rotulo}"] = float(comb.f_z)
            row[f"Mx-{comb.rotulo}"] = float(comb.m_x)
            row[f"My-{comb.rotulo}"] = float(comb.m_y)
        rows.append(row)
    return pd.DataFrame(rows)


def design_vector_to_sapatas(
    x: Sequence[float],
    projeto: FundacaoProjeto,
) -> list[Sapata]:
    """This function rebuilds the list of Sapata entities from a flat design vector.

    The optimiser hands back a flat vector ``[hx_0, hy_0, hz_0, ...,
    hx_{N-1}, hy_{N-1}, hz_{N-1}]``. This function slices it in groups
    of three and pairs each group with the corresponding pillar.

    :param x: Flat design vector of length ``3 * n_fund``
    :param projeto: FundacaoProjeto whose ``pilares`` define the order

    :return: List of Sapata entities, one per pillar (project order)

    :raises ValueError: When ``len(x) != 3 * projeto.n_fund``
    """
    n = projeto.n_fund
    if len(x) != 3 * n:
        raise ValueError(
            f"design vector has length {len(x)}; expected {3 * n} "
            f"(3 variables per pillar, n_fund={n})."
        )
    sapatas: list[Sapata] = []
    for i, pilar in enumerate(projeto.pilares):
        h_x = float(x[3 * i + 0])
        h_y = float(x[3 * i + 1])
        h_z = float(x[3 * i + 2])
        sapatas.append(Sapata(pilar=pilar, h_x=h_x, h_y=h_y, h_z=h_z))
    return sapatas


def sapatas_to_design_vector(sapatas: Sequence[Sapata]) -> list[float]:
    """This function flattens a list of Sapata entities into a design vector.

    :param sapatas: List of Sapata entities

    :return: Flat design vector ``[hx_0, hy_0, hz_0, ..., hx_{N-1}, hy_{N-1}, hz_{N-1}]``
    """
    out: list[float] = []
    for s in sapatas:
        out.extend([float(s.h_x), float(s.h_y), float(s.h_z)])
    return out
