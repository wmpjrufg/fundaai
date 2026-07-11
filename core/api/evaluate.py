"""Evaluation entry point — runs the pseudo-objective without optimising.

Useful for diagnostics, regression tests and notebooks that need to
inspect the constraint table for an arbitrary design. The function
delegates to ``fundacao._avaliar_projeto`` to preserve the numerical
behaviour validated by the regression suite (Sprint 2 baseline
``of = 19.70604234767181``).

Resumo em português:
    Função pura para avaliar um conjunto fixo de sapatas sem rodar a
    otimização. Devolve `of` e a tabela de restrições por elemento.
"""

from __future__ import annotations

from typing import Sequence

from core.api._adapter import (
    projeto_to_dataframe,
    sapatas_to_design_vector,
)
from core.api.types import EvaluationResult
from core.domain import FundacaoProjeto, Sapata
from fundacao import _avaliar_projeto


# Constraint columns extracted from the annotated DataFrame produced by
# ``fundacao._avaliar_projeto``. Aggregated per element, after the
# combination-wise expansion has been collapsed via ``max``.
_PER_ELEMENT_CONSTRAINTS: tuple[str, ...] = (
    "g sobreposicao",
    "g punção secao C",
    "g punção secao Clinha",
    "g punção",
    "g tensao",
    "g geometria",
)


def evaluate(
    projeto: FundacaoProjeto,
    sapatas: Sequence[Sapata],
    *,
    penalty: float | None = None,
) -> EvaluationResult:
    """This function evaluates the pseudo-objective for a fixed design.

    The supplied ``sapatas`` must have the same length and order as
    ``projeto.pilares``. The function rebuilds the legacy design vector
    plus the legacy DataFrame and calls ``fundacao._avaliar_projeto`` to
    obtain the volume + penalised constraints. It then unpacks the
    annotated DataFrame into a per-element mapping for downstream
    reporting.

    :param projeto: Validated FundacaoProjeto root aggregator
    :param sapatas: Sapata entities (length must equal ``projeto.n_fund``)
    :param penalty: Penalty factor passed to ``_avaliar_projeto``.
                    ``None`` keeps the engineering default

    :return: EvaluationResult with ``of_total``, the supplied sapatas
             and the per-element constraint mapping

    :raises ValueError: When the number of sapatas does not match the project
    """
    if len(sapatas) != projeto.n_fund:
        raise ValueError(
            f"received {len(sapatas)} sapatas; expected {projeto.n_fund} "
            f"(one per pillar)."
        )

    df_input = projeto_to_dataframe(projeto)
    x_vec = sapatas_to_design_vector(sapatas)

    args = (
        df_input,
        projeto.n_comb,
        projeto.f_ck_kpa,
        projeto.cobrimento_m,
    )
    if penalty is not None:
        args = args + (penalty,)

    of_total, df_anotado = _avaliar_projeto(x_vec, args=args)

    constraints: dict[str, dict[str, float]] = {}
    for pilar, sapata in zip(projeto.pilares, sapatas):
        idx = df_anotado.index[df_anotado["Elemento"] == pilar.rotulo]
        if len(idx) == 0:
            continue
        row = df_anotado.loc[idx[0]]
        constraints[pilar.rotulo] = {
            name: float(row[name]) for name in _PER_ELEMENT_CONSTRAINTS
        }

    return EvaluationResult(
        of_total=float(of_total),
        sapatas=tuple(sapatas),
        constraints=constraints,
    )
