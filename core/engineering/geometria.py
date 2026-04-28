"""Pillar-footing geometric compatibility constraint.

Resumo em português:
    Restrição geométrica que impõe um balanço mínimo entre a face do
    pilar e a borda da sapata em uma direção principal. Default da
    folga: 0,10 m por lado.
"""

from __future__ import annotations


def checagem_geometria(
    dim_sapata: float,
    dim_pilar: float,
    balanco_min: float = 0.10,
) -> float:
    """This function returns the minimum-overhang constraint between pillar and footing.

    Encodes the rule h_sapata >= dim_pilar + 2 * balanco_min in the
    constraint convention g <= 0 used across the project. Equivalent
    closed form:
        g = 1 + 2*balanco_min/dim_pilar - dim_sapata/dim_pilar.

    :param dim_sapata: Footing dimension along the analysed axis [m]
    :param dim_pilar: Pillar dimension along the same axis [m]
    :param balanco_min: Minimum required overhang on each side, default 0.10 m

    :return: Design constraint value g; g <= 0 means the geometry is feasible
    """
    delta_ap = 2 * balanco_min / dim_pilar
    delta_hx = dim_sapata / dim_pilar
    return 1 + delta_ap - delta_hx
