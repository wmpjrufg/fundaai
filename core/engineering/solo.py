"""Soil bearing capacity helpers (geotechnical layer).

Resumo em português:
    Tensão admissível do solo a partir do índice SPT (NBR 6122) e do
    tipo de solo. Mantém a correlação empírica clássica do método dos
    práticos.
"""

from __future__ import annotations


def tensao_adm_solo(solo: str, spt: float) -> float:
    """This function returns the admissible soil pressure from the SPT index.

    Empirical correlation based on the practitioners' method:
        - pedregulho:        sigma_adm = SPT/30 * 1000  [kPa]
        - areia:             sigma_adm = SPT/40 * 1000  [kPa]
        - silte or argila:   sigma_adm = SPT/50 * 1000  [kPa]

    The soil identifier is matched case-insensitively. The default branch
    (silte/argila) is intentionally chosen for any unknown label, which
    mirrors the historical behaviour of the project.

    :param solo: Soil type identifier ("pedregulho", "areia", "silte" or "argila")
    :param spt: SPT index (Nspt), dimensionless

    :return: Admissible soil pressure [kPa]
    """
    s = solo.lower()
    if s == "pedregulho":
        return spt / 30 * 1e3
    if s == "areia":
        return spt / 40 * 1e3
    return spt / 50 * 1e3
