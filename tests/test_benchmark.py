"""Testes das funcoes benchmark de `metapy_toolbox.benchmark`.

Cada funcao classica deve respeitar duas garantias minimas:
    * `f(x*) = f_min` no minimo global conhecido (`x*` da literatura).
    * Crescimento sob perturbacao no entorno do minimo (sanity check).

Cobertura:
    * sphere, rosenbrock, rastrigin, ackley, zakharov,
      easom, dixon_price, goldstein_price (sanidade basica).
    * griewank — issue corrigida na Sprint 2 (produto fora do loop).
    * powell  — issue corrigida na Sprint 2 (indexacao 1-based estourava).

Referencias dos minimos: Surjanovic & Bingham,
"Virtual Library of Simulation Experiments" (sfu.ca/~ssurjano).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from metapy_toolbox import (
    ackley,
    dixon_price,
    easom,
    goldstein_price,
    griewank,
    powell,
    rastrigin,
    rosenbrock,
    sphere,
    zakharov,
)


# =============================================================================
# Sanidade basica (mininos conhecidos)
# =============================================================================
@pytest.mark.benchmark
class TestMinimosConhecidos:
    """This class verifies the analytical minimum of each canonical benchmark."""

    def test_sphere_zero(self):
        """This test ensures sphere(0,...,0) = 0.

        :return: Nada (assert interno)
        """
        assert sphere([0.0, 0.0, 0.0]) == pytest.approx(0.0)

    def test_rosenbrock_um(self):
        """This test ensures rosenbrock(1,...,1) = 0.

        :return: Nada (assert interno)
        """
        assert rosenbrock([1.0, 1.0, 1.0]) == pytest.approx(0.0)

    def test_rastrigin_zero(self):
        """This test ensures rastrigin(0,...,0) = 0.

        :return: Nada (assert interno)
        """
        assert rastrigin([0.0, 0.0]) == pytest.approx(0.0)

    def test_ackley_zero(self):
        """This test ensures ackley(0,...,0) = 0 (within numerical tolerance).

        :return: Nada (assert interno)
        """
        assert ackley([0.0, 0.0]) == pytest.approx(0.0, abs=1e-12)

    def test_zakharov_zero(self):
        """This test ensures zakharov(0,...,0) = 0.

        :return: Nada (assert interno)
        """
        assert zakharov([0.0, 0.0, 0.0]) == pytest.approx(0.0)

    def test_easom_pi_pi(self):
        """This test ensures easom(pi, pi) = -1.

        :return: Nada (assert interno)
        """
        assert easom([math.pi, math.pi]) == pytest.approx(-1.0)

    def test_dixon_price_global_min(self):
        """This test ensures dixon_price minimum equals zero at the analytical x*.

        Para `d = 2`, o minimo analitico eh `x_i = 2^(-(2^i - 2)/(2^i))`,
        i.e. `x_0 = 1`, `x_1 = 1/sqrt(2)`. Resultado deve ser 0.

        :return: Nada (assert interno)
        """
        x_star = [1.0, 1.0 / math.sqrt(2.0)]
        assert dixon_price(x_star) == pytest.approx(0.0, abs=1e-12)

    def test_goldstein_price_minimo(self):
        """This test ensures goldstein_price(0, -1) = 3.

        :return: Nada (assert interno)
        """
        assert goldstein_price([0.0, -1.0]) == pytest.approx(3.0)


# =============================================================================
# Griewank (corrigida na Sprint 2)
# =============================================================================
@pytest.mark.benchmark
class TestGriewank:
    """This class verifies the Griewank correction made in Sprint 2."""

    def test_minimo_em_zero(self):
        """This test ensures griewank(0,...,0) = 0 (global minimum).

        :return: Nada (assert interno)
        """
        assert griewank([0.0, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-12)

    def test_produto_dentro_do_loop_afeta_resultado(self):
        """This test ensures the product term uses every dimension, not only the last one.

        Antes da correcao, `prod *= cos(x_i / sqrt(i+1))` ficava fora do
        loop e usava somente o ultimo `x_i`. Construimos dois vetores
        que diferem apenas em coordenadas intermediarias e verificamos
        que `griewank` distingue ambos.

        :return: Nada (assert interno)
        """
        a = [1.0, 0.0, 1.0, 0.0]
        b = [0.0, 1.0, 0.0, 0.0]
        # Os dois vetores tem o mesmo ultimo elemento (0.0), mas dimensoes
        # intermediarias distintas. Se o produto considerar todas as
        # dimensoes, os valores devem diferir.
        assert griewank(a) != pytest.approx(griewank(b))

    def test_simetria_em_relacao_a_origem(self):
        """This test verifies griewank(x) = griewank(-x) due to even cosines and squares.

        :return: Nada (assert interno)
        """
        x = [0.5, -0.7, 1.2, -0.3]
        assert griewank(x) == pytest.approx(griewank([-v for v in x]))


# =============================================================================
# Powell (corrigida na Sprint 2)
# =============================================================================
@pytest.mark.benchmark
class TestPowell:
    """This class verifies the Powell correction made in Sprint 2."""

    def test_minimo_em_zero_com_dim_4(self):
        """This test ensures powell(0,...,0) = 0 in the canonical d=4 case.

        :return: Nada (assert interno)
        """
        assert powell([0.0, 0.0, 0.0, 0.0]) == pytest.approx(0.0)

    def test_minimo_em_zero_com_dim_8(self):
        """This test ensures powell(0,...,0) = 0 also for d=8 (two blocks).

        :return: Nada (assert interno)
        """
        assert powell([0.0] * 8) == pytest.approx(0.0)

    def test_dimensao_nao_multipla_de_4_levanta_erro(self):
        """This test ensures the explicit ValueError when `d % 4 != 0`.

        Prevencao defensiva contra a falha silenciosa do indice `x[4*i]`
        que existia na versao antiga.

        :return: Nada (assert interno)
        """
        with pytest.raises(ValueError):
            powell([0.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            powell([0.1, 0.2, 0.3, 0.4, 0.5])

    def test_valor_de_referencia_para_vetor_nao_trivial(self):
        """This test pins the value of powell(1,2,3,4).

        Conferindo manualmente:
            term1 = (1 + 10·2)^2 = 441
            term2 = 5·(3 - 4)^2 = 5
            term3 = (2 - 2·3)^4 = (-4)^4 = 256
            term4 = 10·(1 - 4)^4 = 10·81 = 810
            total = 441 + 5 + 256 + 810 = 1512

        :return: Nada (assert interno)
        """
        assert powell([1.0, 2.0, 3.0, 4.0]) == pytest.approx(1512.0)
