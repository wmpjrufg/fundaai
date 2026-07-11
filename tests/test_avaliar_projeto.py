"""Testes de regressao numerica e contrato de `fundacao._avaliar_projeto`.

A Sprint 0 fundiu `obj_felipe_lucas` e `obj_teste` em torno de um nucleo
compartilhado `_avaliar_projeto(x, args, *, penalty=None)`. Este arquivo
trava o comportamento dessa funcao em tres frentes:

    1. **Regressao numerica**: o caso de tres fundacoes com a semente 42
       e penalidade default deve continuar produzindo
       `of = 19,70604234767181` (valor capturado em 2026-04-27).
    2. **Parametrizacao do penalty**: `penalty=1e1` deve coincidir com o
       default, `penalty=1e6` deve violar bem mais a OF.
    3. **Wrappers concordam**: `obj_felipe_lucas` (escalar) e `obj_teste`
       (tupla) devem produzir o mesmo `of`.

Os testes pressupõem o caso `assets/data/problema_fund_três.xlsx` (3 fundacoes,
3 combinacoes) carregado pela fixture `df_problema_tres` em `conftest.py`.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from core.api.objective import avaliar_projeto_fast
from fundacao import (
    _PENALTY_DEFAULT,
    _avaliar_projeto,
    obj_felipe_lucas,
    obj_teste,
)


# =============================================================================
# Fixtures locais
# =============================================================================
@pytest.fixture
def x_seed42(df_problema_tres: pd.DataFrame) -> np.ndarray:
    """This fixture builds the canonical design vector used in the regression baseline.

    O vetor eh gerado por `np.random.uniform(0.6, 3.0, 3 * N_fun)` com
    `np.random.seed(42)`, o mesmo procedimento usado no smoke test que
    produziu o valor de referencia `of = 19,70604234767181`.

    :param df_problema_tres: Fixture com o DataFrame de 3 fundacoes

    :return: Vetor numpy com `3 * N_fun` variaveis de projeto
    """
    n_fun = df_problema_tres.shape[0]
    np.random.seed(42)
    return np.random.uniform(0.6, 3.0, size=3 * n_fun)


# =============================================================================
# Regressao numerica
# =============================================================================
@pytest.mark.regression
def test_baseline_three_foundations_returns_19_706(
    df_problema_tres: pd.DataFrame,
    cfg_calibracao: Dict[str, Any],
    x_seed42: np.ndarray,
):
    """This test locks the canonical OF value of the three-foundation baseline.

    Valor capturado em 2026-04-27 logo apos a Sprint 0 (penalty=10):
    `of = 19,70604234767181`. Se este teste falhar apos uma futura
    refatoracao, ha alteracao de comportamento numerico que precisa
    ser confirmada explicitamente.

    :param df_problema_tres: Fixture com 3 fundacoes (problema_fund_três)
    :param cfg_calibracao: Fixture com a calibracao de projeto
    :param x_seed42: Vetor de projeto canonico (seed 42)

    :return: Nada (assert interno)
    """
    of, df = _avaliar_projeto(
        x_seed42,
        args=(
            df_problema_tres,
            cfg_calibracao["n_comb"],
            cfg_calibracao["f_ck_kpa"],
            cfg_calibracao["cob_m"],
        ),
    )
    assert of == pytest.approx(19.70604234767181, rel=1e-12)
    # df_anotado deve conter as colunas de saida obrigatorias
    cols = {
        "volume (m3)", "g sobreposicao", "tensao adm. (kPa)",
        "g punção secao C", "g tensao", "g geometria", "volume final (m3)",
    }
    assert cols.issubset(df.columns)


# =============================================================================
# Parametrizacao do penalty
# =============================================================================
@pytest.mark.regression
def test_penalty_default_constante_eh_10(
    df_problema_tres: pd.DataFrame,
    cfg_calibracao: Dict[str, Any],
    x_seed42: np.ndarray,
):
    """This test verifies that `_PENALTY_DEFAULT` is exactly 10.0 (historical hardcoded value).

    A constante substitui o `1E1` espalhado pelo codigo antigo. Trocar
    seu valor seria uma mudanca de calibracao que precisa ser
    explicitamente discutida com o orientador.

    :return: Nada (assert interno)
    """
    assert _PENALTY_DEFAULT == pytest.approx(10.0)


@pytest.mark.regression
def test_penalty_explicit_10_matches_default(
    df_problema_tres: pd.DataFrame,
    cfg_calibracao: Dict[str, Any],
    x_seed42: np.ndarray,
):
    """This test verifies that passing `penalty=10` reproduces the default behaviour.

    Esta foi uma das motivacoes da Sprint 0: o quinto elemento de `args`
    era silenciosamente ignorado pela versao antiga. Hoje ele eh
    respeitado, e `penalty=_PENALTY_DEFAULT` deve devolver exatamente o
    mesmo valor que o default.

    :return: Nada (assert interno)
    """
    args_default = (
        df_problema_tres,
        cfg_calibracao["n_comb"],
        cfg_calibracao["f_ck_kpa"],
        cfg_calibracao["cob_m"],
    )
    args_explicit = args_default + (_PENALTY_DEFAULT,)

    of_default, _ = _avaliar_projeto(x_seed42, args=args_default)
    of_explicit, _ = _avaliar_projeto(x_seed42, args=args_explicit)
    assert of_default == pytest.approx(of_explicit, rel=1e-15)


@pytest.mark.regression
def test_penalty_high_increases_of_when_violations_exist(
    df_problema_tres: pd.DataFrame,
    cfg_calibracao: Dict[str, Any],
    x_seed42: np.ndarray,
):
    """This test verifies that a much larger penalty produces a much larger OF.

    No vetor canonico (seed 42) ha violacoes; aumentar penalty de 1e1
    para 1e6 deve aumentar a OF em varias ordens de grandeza. Comprova
    que penalty agora eh efetivamente parametrizavel (issue Args extras
    em obj_teste, resolvida na Sprint 0).

    :return: Nada (assert interno)
    """
    base_args = (
        df_problema_tres,
        cfg_calibracao["n_comb"],
        cfg_calibracao["f_ck_kpa"],
        cfg_calibracao["cob_m"],
    )
    of_low, _ = _avaliar_projeto(x_seed42, args=base_args + (1e1,))
    of_high, _ = _avaliar_projeto(x_seed42, args=base_args + (1e6,))
    assert of_high > of_low * 1_000.0   # crescimento amplo, nao uma diferenca marginal


# =============================================================================
# Concordancia entre wrappers
# =============================================================================
@pytest.mark.regression
def test_obj_felipe_lucas_equals_obj_teste(
    df_problema_tres: pd.DataFrame,
    cfg_calibracao: Dict[str, Any],
    x_seed42: np.ndarray,
):
    """This test verifies that `obj_felipe_lucas` and `obj_teste` agree on the scalar OF.

    Ambos sao wrappers finos sobre `_avaliar_projeto`; o escalar deve
    coincidir exatamente. Isto garante que a fusao da Sprint 0 nao
    introduziu divergencia entre as duas APIs publicas.

    :return: Nada (assert interno)
    """
    args = (
        df_problema_tres,
        cfg_calibracao["n_comb"],
        cfg_calibracao["f_ck_kpa"],
        cfg_calibracao["cob_m"],
    )
    of_a = obj_felipe_lucas(x_seed42, args=args)
    of_b, _ = obj_teste(x_seed42, args=args)
    assert of_a == pytest.approx(of_b, rel=1e-15)


# =============================================================================
# Robustez do wrapper
# =============================================================================
@pytest.mark.regression
def test_args_aceita_4_ou_5_elementos(
    df_problema_tres: pd.DataFrame,
    cfg_calibracao: Dict[str, Any],
    x_seed42: np.ndarray,
):
    """This test verifies backwards compatibility for args with 4 or 5 elements.

    A versao antiga aceitava 4 elementos. Os notebooks historicamente
    passavam 5 (com penalidade ignorada). A nova versao aceita ambos
    e respeita o quinto elemento como `penalty`.

    :return: Nada (assert interno)
    """
    args4 = (
        df_problema_tres,
        cfg_calibracao["n_comb"],
        cfg_calibracao["f_ck_kpa"],
        cfg_calibracao["cob_m"],
    )
    args5 = args4 + (_PENALTY_DEFAULT,)
    # Nao levanta excecao em nenhuma das duas formas
    obj_felipe_lucas(x_seed42, args=args4)
    obj_felipe_lucas(x_seed42, args=args5)


# =============================================================================
# Guarda de altura util (h_z <= cob) — fast e legacy falham identicamente
# =============================================================================
@pytest.mark.regression
def test_hz_below_cover_raises_on_both_implementations(
    df_problema_tres: pd.DataFrame,
    cfg_calibracao: Dict[str, Any],
):
    """This test ensures both FO variants reject a non-positive effective depth.

    Um candidato com ``h_z <= cob`` inverteria o sinal de ``tau_sd2`` e
    leria a puncao como viavel. A guarda explicita (fast: upfront;
    legacy: dentro de ``verificacao_puncao_sapata``) transforma o regime
    fisicamente sem sentido em erro imediato nas duas implementacoes.

    :return: Nada (assert interno)
    """
    n_fun = df_problema_tres.shape[0]
    x_bad = np.tile([1.0, 1.0, cfg_calibracao["cob_m"]], n_fun)  # h_z == cob
    args = (
        df_problema_tres,
        cfg_calibracao["n_comb"],
        cfg_calibracao["f_ck_kpa"],
        cfg_calibracao["cob_m"],
    )
    with pytest.raises(ValueError, match="effective depth"):
        avaliar_projeto_fast(x_bad, args)
    with pytest.raises(ValueError, match="effective depth"):
        _avaliar_projeto(x_bad, args)


# =============================================================================
# Paridade da avaliação por componentes (Frente C / CBO)
# =============================================================================
@pytest.mark.regression
def test_componentes_theta_bit_identico_ao_fast(
    df_problema_tres: pd.DataFrame,
    cfg_calibracao: Dict[str, Any],
    x_seed42: np.ndarray,
):
    """Theta devolvido por componentes é bit-idêntico ao do avaliador fast.

    ``avaliar_projeto_componentes`` compartilha o núcleo numérico e a
    expressão final de ``avaliar_projeto_fast`` — a igualdade exigida
    aqui é exata (==), não aproximada. Também valida o contrato dos
    componentes: volume bruto <= Theta e vetor g com os 4 grupos.

    :return: Nada (assert interno)
    """
    from core.api.objective import avaliar_projeto_componentes

    args = (
        df_problema_tres,
        cfg_calibracao["n_comb"],
        cfg_calibracao["f_ck_kpa"],
        cfg_calibracao["cob_m"],
    )
    theta, volume, g = avaliar_projeto_componentes(x_seed42, args)
    assert theta == avaliar_projeto_fast(x_seed42, args)
    assert theta == 19.70604234767181
    assert g.shape == (4,)
    assert volume <= theta

    rng = np.random.default_rng(2026)
    n_fun = df_problema_tres.shape[0]
    for x in rng.uniform(0.6, 3.0, size=(200, 3 * n_fun)):
        t_c, v_c, g_c = avaliar_projeto_componentes(x, args)
        assert t_c == avaliar_projeto_fast(x, args)
        assert v_c <= t_c + 1e-12
