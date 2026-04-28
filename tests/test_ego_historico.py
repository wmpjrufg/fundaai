"""Testes do contrato de historico de `metapy_toolbox.ego.ego_01_architecture`.

A Sprint 1 corrigiu duas falhas que comprometiam a interpretacao
cientifica do EGO:

    * Pontos novos eram registrados com `ITER = 0` e `ID` constante.
    * `pages/sapatas.py` reusava o mesmo `x_ini` em todas as repeticoes
      do laco de robustez.

Este arquivo trava o comportamento corrigido para evitar regressao
silenciosa em sprints futuras (refatoracao estrutural, mudanca para
modelos substitutos fisicamente informados, etc.).

Os testes usam a funcao benchmark `sphere(x) = sum(x_i^2)` para nao
acoplar o EGO ao caso de fundacoes (que ja eh exercitado em
`test_avaliar_projeto.py`).
"""

from __future__ import annotations

import numpy as np
import pytest

from mealpy import GA

from metapy_toolbox import (
    ego_01_architecture,
    initial_population_01,
    sphere,
)


# =============================================================================
# Helpers
# =============================================================================
def _build_optimizer() -> dict:
    """This helper builds a small mealpy GA dictionary for the inner optimizer.

    Mantemos `epoch` e `pop_size` baixos para que a suite rode em poucos
    segundos. O objetivo dos testes nao eh medir convergencia mas sim
    travar a estrutura do historico e a reprodutibilidade da seed.

    :return: Dicionario `{'optimizer algorithm': mealpy.GA.BaseGA(...)}`
    """
    return {"optimizer algorithm": GA.BaseGA(epoch=10, pop_size=20)}


# =============================================================================
# Estrutura do historico (issue: Historico do EGO com ITER e ID incorretos)
# =============================================================================
@pytest.mark.optimization
class TestHistoricoEGO:
    """This class verifies that the EGO history dataframe is correctly populated.

    Antes da Sprint 1, todos os pontos novos do laco eram inseridos com
    `ITER = 0` e `ID = n_pop - 1`, corrompendo a analise de convergencia.
    A correcao usa `ITER = t` e `ID = max(ID) + 1`.
    """

    def test_iter_cobre_zero_ate_n_gen(self):
        """This test ensures `df['ITER']` covers exactly `0..n_gen`.

        :return: Nada (assert interno)
        """
        x_l, x_u = [-5.0, -5.0], [5.0, 5.0]
        n_pop, n_gen = 8, 4
        x_ini = initial_population_01(n_pop, 2, x_l, x_u, seed=123, use_lhs=True)
        _, _, df = ego_01_architecture(
            obj=sphere, n_gen=n_gen, initial_population=x_ini,
            x_lower=x_l, x_upper=x_u,
            params_opt=_build_optimizer(), seed=123,
        )
        iters = sorted(df["ITER"].unique().tolist())
        assert iters == list(range(0, n_gen + 1))

    def test_total_de_linhas_eh_npop_mais_ngen(self):
        """This test ensures the dataframe contains exactly `n_pop + n_gen` rows.

        :return: Nada (assert interno)
        """
        x_l, x_u = [-5.0, -5.0], [5.0, 5.0]
        n_pop, n_gen = 8, 4
        x_ini = initial_population_01(n_pop, 2, x_l, x_u, seed=123, use_lhs=True)
        _, _, df = ego_01_architecture(
            obj=sphere, n_gen=n_gen, initial_population=x_ini,
            x_lower=x_l, x_upper=x_u,
            params_opt=_build_optimizer(), seed=123,
        )
        assert len(df) == n_pop + n_gen

    def test_ids_sao_unicos(self):
        """This test ensures every row receives a unique ID.

        Um falhe aqui sinaliza que a iteracao do EGO voltou a copiar a
        variavel `n` do loop inicial em vez de gerar `max(ID) + 1`.

        :return: Nada (assert interno)
        """
        x_l, x_u = [-5.0, -5.0], [5.0, 5.0]
        n_pop, n_gen = 8, 4
        x_ini = initial_population_01(n_pop, 2, x_l, x_u, seed=123, use_lhs=True)
        _, _, df = ego_01_architecture(
            obj=sphere, n_gen=n_gen, initial_population=x_ini,
            x_lower=x_l, x_upper=x_u,
            params_opt=_build_optimizer(), seed=123,
        )
        ids = df["ID"].tolist()
        assert len(set(ids)) == len(ids)

    def test_cada_iteracao_tem_uma_linha_nova(self):
        """This test ensures that every iteration `t > 0` adds exactly one row.

        :return: Nada (assert interno)
        """
        x_l, x_u = [-5.0, -5.0], [5.0, 5.0]
        n_pop, n_gen = 6, 5
        x_ini = initial_population_01(n_pop, 2, x_l, x_u, seed=7, use_lhs=True)
        _, _, df = ego_01_architecture(
            obj=sphere, n_gen=n_gen, initial_population=x_ini,
            x_lower=x_l, x_upper=x_u,
            params_opt=_build_optimizer(), seed=7,
        )
        for t in range(1, n_gen + 1):
            assert (df["ITER"] == t).sum() == 1


# =============================================================================
# Reprodutibilidade (issue: n_rep reusa populacao inicial / propagacao de seed)
# =============================================================================
@pytest.mark.optimization
class TestReprodutibilidadeEGO:
    """This class verifies that the new `seed` parameter actually reproduces runs."""

    def test_lhs_eh_reproducivel_com_mesma_seed(self):
        """This test ensures that two LHS calls with the same seed produce identical samples.

        :return: Nada (assert interno)
        """
        a = initial_population_01(10, 3, [-1, -1, -1], [1, 1, 1], seed=999, use_lhs=True)
        b = initial_population_01(10, 3, [-1, -1, -1], [1, 1, 1], seed=999, use_lhs=True)
        assert np.allclose(a, b)

    def test_lhs_difere_com_seeds_diferentes(self):
        """This test ensures that different seeds produce different LHS samples.

        :return: Nada (assert interno)
        """
        a = initial_population_01(10, 3, [-1, -1, -1], [1, 1, 1], seed=1, use_lhs=True)
        b = initial_population_01(10, 3, [-1, -1, -1], [1, 1, 1], seed=2, use_lhs=True)
        assert not np.allclose(a, b)

    def test_ego_com_mesma_seed_produz_mesmo_best_of(self):
        """This test ensures that two EGO runs with identical seed return identical OF.

        Eh o teste mais forte do contrato de reprodutibilidade. Se este
        teste passar consistentemente, a equipe pode reportar
        `media ± std` sobre n_rep execucoes com seeds controladas.

        :return: Nada (assert interno)
        """
        x_l, x_u = [-5.0, -5.0], [5.0, 5.0]
        n_pop, n_gen, seed = 8, 2, 999
        # Reconstruimos os GAs do zero em cada chamada porque mealpy
        # mantem estado interno entre solves.
        x_ini_a = initial_population_01(n_pop, 2, x_l, x_u, seed=seed, use_lhs=True)
        _, of_a, _ = ego_01_architecture(
            obj=sphere, n_gen=n_gen, initial_population=x_ini_a,
            x_lower=x_l, x_upper=x_u,
            params_opt={"optimizer algorithm": GA.BaseGA(epoch=10, pop_size=20)},
            seed=seed,
        )
        x_ini_b = initial_population_01(n_pop, 2, x_l, x_u, seed=seed, use_lhs=True)
        _, of_b, _ = ego_01_architecture(
            obj=sphere, n_gen=n_gen, initial_population=x_ini_b,
            x_lower=x_l, x_upper=x_u,
            params_opt={"optimizer algorithm": GA.BaseGA(epoch=10, pop_size=20)},
            seed=seed,
        )
        assert of_a == pytest.approx(of_b, rel=1e-12)

    def test_ego_aceita_kwarg_seed(self):
        """This test ensures the `seed` keyword argument exists in the signature.

        Falha aqui significa que alguem removeu o parametro `seed`
        introduzido na Sprint 1 (regressao da API publica).

        :return: Nada (assert interno)
        """
        import inspect
        params = inspect.signature(ego_01_architecture).parameters
        assert "seed" in params
        assert params["seed"].default is None
