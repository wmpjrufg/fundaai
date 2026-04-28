"""Unit tests for the domain layer (``core.domain``).

Cover invariants and small helpers of the immutable entities ``Solo``,
``Pilar``, ``Combinacao`` and ``FundacaoProjeto``, plus the mutable
``Sapata`` whose dimensions are the optimisation variables.

These tests are pure (no DataFrames, no pytest fixtures from
``conftest.py``) so that the domain layer can be exercised in isolation
from the rest of the project.
"""

from __future__ import annotations

import pytest

from core.domain import (
    Combinacao,
    FundacaoProjeto,
    Pilar,
    Sapata,
    Solo,
)


# =============================================================================
# Solo
# =============================================================================
class TestSolo:
    """This class verifies the ``Solo`` immutable entity.

    Confirms field acceptance, the ``sigma_adm_kpa`` derivation against
    the engineering helper and the SPT validation invariant.
    """

    def test_sigma_adm_delegates_to_engineering(self):
        """This test ensures ``sigma_adm_kpa`` returns the engineering layer value.

        :return: None (internal asserts)
        """
        assert Solo("argila", 50).sigma_adm_kpa == pytest.approx(1000.0)
        assert Solo("areia", 40).sigma_adm_kpa == pytest.approx(1000.0)
        assert Solo("pedregulho", 30).sigma_adm_kpa == pytest.approx(1000.0)

    def test_negative_spt_raises(self):
        """This test ensures the SPT non-negativity invariant.

        :return: None (internal asserts)
        """
        with pytest.raises(ValueError):
            Solo("argila", -1.0)

    def test_solo_is_frozen(self):
        """This test ensures the dataclass is immutable.

        :return: None (internal asserts)
        """
        s = Solo("argila", 30)
        with pytest.raises((AttributeError, TypeError)):
            s.spt = 99  # type: ignore[misc]


# =============================================================================
# Pilar
# =============================================================================
class TestPilar:
    """This class verifies the ``Pilar`` immutable entity."""

    def test_pilar_carries_geometry_and_centroid(self):
        """This test ensures the dataclass keeps the supplied fields verbatim.

        :return: None (internal asserts)
        """
        p = Pilar("P04", a_p=0.30, b_p=1.50, xg=32.10, yg=20.37)
        assert p.rotulo == "P04"
        assert p.a_p == 0.30
        assert p.b_p == 1.50
        assert p.xg == 32.10
        assert p.yg == 20.37

    def test_non_positive_dimensions_raise(self):
        """This test ensures positive cross-section invariants.

        :return: None (internal asserts)
        """
        with pytest.raises(ValueError):
            Pilar("P", a_p=0.0, b_p=1.0, xg=0.0, yg=0.0)
        with pytest.raises(ValueError):
            Pilar("P", a_p=1.0, b_p=-0.5, xg=0.0, yg=0.0)

    def test_pilar_is_frozen(self):
        """This test ensures the dataclass is immutable.

        :return: None (internal asserts)
        """
        p = Pilar("P04", 0.30, 1.50, 0.0, 0.0)
        with pytest.raises((AttributeError, TypeError)):
            p.xg = 1.0  # type: ignore[misc]


# =============================================================================
# Combinacao
# =============================================================================
class TestCombinacao:
    """This class verifies the ``Combinacao`` immutable entity."""

    def test_fields(self):
        """This test ensures all four fields round-trip unchanged.

        :return: None (internal asserts)
        """
        c = Combinacao("c1", f_z=855.5, m_x=-3.7, m_y=9.2)
        assert (c.rotulo, c.f_z, c.m_x, c.m_y) == ("c1", 855.5, -3.7, 9.2)


# =============================================================================
# Sapata
# =============================================================================
class TestSapata:
    """This class verifies the ``Sapata`` mutable design entity."""

    def _pilar(self) -> Pilar:
        """This helper returns a canonical pillar at the origin.

        :return: Pilar instance centred at (0, 0)
        """
        return Pilar("P", a_p=0.30, b_p=0.30, xg=0.0, yg=0.0)

    def test_volume_matches_product(self):
        """This test ensures the volume property equals h_x * h_y * h_z.

        :return: None (internal asserts)
        """
        s = Sapata(self._pilar(), h_x=2.0, h_y=3.0, h_z=0.5)
        assert s.volume == pytest.approx(3.0)

    def test_vertices_are_centred_on_pillar(self):
        """This test ensures the four vertices form an AABB centred on the pillar.

        :return: None (internal asserts)
        """
        s = Sapata(self._pilar(), h_x=2.0, h_y=2.0, h_z=0.6)
        v_sw, v_se, v_ne, v_nw = s.vertices
        assert v_sw == (-1.0, -1.0)
        assert v_se == (1.0, -1.0)
        assert v_ne == (1.0, 1.0)
        assert v_nw == (-1.0, 1.0)

    def test_dimensions_can_be_updated(self):
        """This test ensures the design variables are mutable for the optimiser.

        :return: None (internal asserts)
        """
        s = Sapata(self._pilar(), h_x=1.0, h_y=1.0, h_z=0.6)
        s.h_x = 2.5
        s.h_y = 1.7
        assert s.volume == pytest.approx(2.5 * 1.7 * 0.6)

    def test_non_positive_dimensions_raise(self):
        """This test ensures strictly positive dimension invariants.

        :return: None (internal asserts)
        """
        p = self._pilar()
        with pytest.raises(ValueError):
            Sapata(p, h_x=0.0, h_y=1.0, h_z=0.6)
        with pytest.raises(ValueError):
            Sapata(p, h_x=1.0, h_y=1.0, h_z=-0.1)


# =============================================================================
# FundacaoProjeto
# =============================================================================
class TestFundacaoProjeto:
    """This class verifies the ``FundacaoProjeto`` root aggregator."""

    def _three_element_project(self) -> FundacaoProjeto:
        """This helper builds a small but realistic 3-element project.

        Mirrors the canonical ``problema_fund_três.xlsx`` test case
        (P04, P05, P16; clay; SPT 35/45/43; three load combinations).

        :return: FundacaoProjeto instance with three pillars
        """
        pilares = [
            Pilar("P04", 0.30, 1.50, 32.10, 20.37),
            Pilar("P05", 0.25, 1.20, 35.30, 20.37),
            Pilar("P16", 0.25, 1.20, 27.75, 18.07),
        ]
        solo = {
            "P04": Solo("argila", 35),
            "P05": Solo("argila", 45),
            "P16": Solo("argila", 43),
        }
        comb = {
            "P04": [
                Combinacao("c1", 855.5, -3.7, 9.2),
                Combinacao("c2", 891.9, -60.0, 0.6),
                Combinacao("c3", 908.5, -36.9, 0.4),
            ],
            "P05": [
                Combinacao("c1", 478.6, -3.1, 5.0),
                Combinacao("c2", 496.0, -27.6, 0.7),
                Combinacao("c3", 508.0, -27.6, 0.7),
            ],
            "P16": [
                Combinacao("c1", 377.3, -1.9, 4.4),
                Combinacao("c2", 259.2, -27.8, 0.1),
                Combinacao("c3", 383.8, 27.7, 0.2),
            ],
        }
        return FundacaoProjeto(
            pilares=pilares,
            solo_por_pilar=solo,
            combinacoes_por_pilar=comb,
            f_ck_kpa=25_000.0,
            cobrimento_m=0.04,
        )

    def test_n_fund_and_n_comb(self):
        """This test ensures the convenience counts are correct.

        :return: None (internal asserts)
        """
        proj = self._three_element_project()
        assert proj.n_fund == 3
        assert proj.n_comb == 3

    def test_missing_solo_raises(self):
        """This test ensures the aggregator detects an unmatched pillar in the soil map.

        :return: None (internal asserts)
        """
        with pytest.raises(ValueError):
            FundacaoProjeto(
                pilares=[Pilar("P04", 0.30, 1.50, 0.0, 0.0)],
                solo_por_pilar={},  # missing P04
                combinacoes_por_pilar={"P04": [Combinacao("c1", 100.0, 0.0, 0.0)]},
                f_ck_kpa=25_000.0,
                cobrimento_m=0.04,
            )

    def test_missing_combinations_raises(self):
        """This test ensures the aggregator detects an unmatched pillar in the loads map.

        :return: None (internal asserts)
        """
        with pytest.raises(ValueError):
            FundacaoProjeto(
                pilares=[Pilar("P04", 0.30, 1.50, 0.0, 0.0)],
                solo_por_pilar={"P04": Solo("argila", 35)},
                combinacoes_por_pilar={},  # missing P04
                f_ck_kpa=25_000.0,
                cobrimento_m=0.04,
            )

    def test_invalid_globals_raise(self):
        """This test ensures the aggregator validates ``f_ck_kpa`` and ``cobrimento_m``.

        :return: None (internal asserts)
        """
        pilares = [Pilar("P", 0.30, 0.30, 0.0, 0.0)]
        solo = {"P": Solo("argila", 30)}
        comb = {"P": [Combinacao("c1", 100.0, 0.0, 0.0)]}
        with pytest.raises(ValueError):
            FundacaoProjeto(pilares, solo, comb, f_ck_kpa=0.0, cobrimento_m=0.04)
        with pytest.raises(ValueError):
            FundacaoProjeto(pilares, solo, comb, f_ck_kpa=25_000.0, cobrimento_m=-0.01)
