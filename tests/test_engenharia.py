"""Testes da camada de engenharia (NBR 6118 e NBR 6122).

Os testes deste arquivo travam o comportamento numerico das funcoes
analiticas usadas na verificacao de sapatas isoladas:

    * `tensao_adm_solo`         (correlacao SPT -> sigma_adm)
    * `calcular_sigma_max_min`  (flexao composta na base)
    * `checagem_tensao_max_min` (g_tensao = sigma/sigma_adm - 1)
    * `checagem_geometria`      (balanco minimo pilar-sapata)
    * `verificacao_puncao_sapata` (secao critica C, NBR 6118 item 19.5)
    * `sobreposicao_sapatas`    (interseccao AABB entre dois retangulos)

Sao escritos apos as correcoes da Sprint 0 e da Sprint 1 (branch
`refactor/code-base-v2`) para permitir que a Sprint 3 (refatoracao
estrutural com POO + UI separada + vetorizacao) avance sem regressao
numerica silenciosa.
"""

from __future__ import annotations

import math

import pytest

import numpy as np

from fundacao import (
    calcular_sigma_max_min,
    checagem_geometria,
    checagem_tensao_max_min,
    k_tabela_19_2,
    rho_minimo_flexao,
    sobreposicao_matrix,
    sobreposicao_sapatas,
    tensao_adm_solo,
    verificacao_puncao_sapata,
    verificacao_puncao_sapata_c_linha,
)


# =============================================================================
# tensao_adm_solo
# =============================================================================
@pytest.mark.engineering
class TestTensaoAdmSolo:
    """This class verifies the empirical SPT -> sigma_adm correlation.

    Tres ramos sao validados (`pedregulho`, `areia`, `silte/argila`),
    bem como o comportamento case-insensitive da string de tipo.
    """

    def test_pedregulho_divide_por_30(self):
        """This test ensures `pedregulho` uses the SPT/30 coefficient.

        :return: Nada (asserts internos)
        """
        assert tensao_adm_solo("pedregulho", 30) == pytest.approx(1000.0)
        assert tensao_adm_solo("pedregulho", 15) == pytest.approx(500.0)

    def test_areia_divide_por_40(self):
        """This test ensures `areia` uses the SPT/40 coefficient.

        :return: Nada (asserts internos)
        """
        assert tensao_adm_solo("areia", 40) == pytest.approx(1000.0)
        assert tensao_adm_solo("areia", 20) == pytest.approx(500.0)

    def test_argila_divide_por_50(self):
        """This test ensures `argila` falls in the default SPT/50 branch.

        :return: Nada (asserts internos)
        """
        assert tensao_adm_solo("argila", 50) == pytest.approx(1000.0)
        assert tensao_adm_solo("argila", 10) == pytest.approx(200.0)

    def test_silte_usa_mesmo_coef_argila(self):
        """This test ensures `silte` follows the silte-or-argila branch.

        :return: Nada (asserts internos)
        """
        assert tensao_adm_solo("silte", 50) == pytest.approx(1000.0)

    def test_case_insensitive(self):
        """This test ensures `solo` matching is case-insensitive.

        :return: Nada (asserts internos)
        """
        assert tensao_adm_solo("PEDREGULHO", 30) == pytest.approx(1000.0)
        assert tensao_adm_solo("Areia", 40) == pytest.approx(1000.0)
        assert tensao_adm_solo("Argila", 50) == pytest.approx(1000.0)


# =============================================================================
# calcular_sigma_max_min
# =============================================================================
@pytest.mark.engineering
class TestCalcularSigmaMaxMin:
    """This class verifies the eccentric soil pressure formula on the base of a footing."""

    def test_sem_momento_sigma_uniforme_com_peso_proprio(self):
        """This test verifies uniform pressure with explicit self-weight.

        Quando os momentos sao nulos, sigma_max e sigma_min coincidem
        com a parcela axial (F_z + gamma_c V)/(hx·hy), sem o antigo
        majorador 1,30.

        :return: Nada (asserts internos)
        """
        f_zk, h_x, h_y, h_z = 1000.0, 2.0, 2.0, 0.60
        sigma_max, sigma_min = calcular_sigma_max_min(
            f_zk, 0.0, 0.0, h_x, h_y, h_z
        )
        sigma_fz = (f_zk + 25.0 * h_x * h_y * h_z) / (h_x * h_y)
        assert sigma_max == pytest.approx(sigma_fz)
        assert sigma_min == pytest.approx(sigma_fz)

    def test_excentricidade_pura_em_x_usa_hx_por_convencao_do_projeto(self):
        """This test checks the FundaIA Mx convention on a non-square footing.

        No FundaIA, `M_x` significa a componente de momento que produz
        excentricidade na direcao x (`M_x = F_z e_x`). Por isso, mesmo
        em uma sapata nao quadrada, o termo relativo usa `h_x`. Se a
        entrada vier de um software que reporta momentos em torno dos
        eixos estruturais, ela deve ser convertida antes da importacao.

        :return: Nada (asserts internos)
        """
        f_zk, m_xk, h_x, h_y, h_z = 1000.0, 50.0, 2.0, 4.0, 0.60
        sigma_max, _ = calcular_sigma_max_min(f_zk, m_xk, 0.0, h_x, h_y, h_z)
        area = h_x * h_y
        sigma_fz = (f_zk + 25.0 * area * h_z) / area
        sigma_mx = 6.0 * m_xk / (area * h_x)
        esperado = sigma_fz + sigma_mx
        assert sigma_max == pytest.approx(esperado)

    def test_modulo_dos_momentos_eh_aplicado(self):
        """This test ensures that negative moments are taken in absolute value.

        Trocar o sinal de `M_x` (com `M_y = 0`) nao deve alterar
        sigma_max nem sigma_min, pois a funcao toma o modulo internamente.

        :return: Nada (asserts internos)
        """
        f_zk, h_x, h_y, h_z = 1000.0, 2.0, 2.0, 0.60
        a = calcular_sigma_max_min(f_zk, 50.0, 0.0, h_x, h_y, h_z)
        b = calcular_sigma_max_min(f_zk, -50.0, 0.0, h_x, h_y, h_z)
        assert a == pytest.approx(b)

    def test_sigma_min_negativa_preserva_tracao(self):
        """This test verifies that tensile pressures are not clipped or factored.

        Tensoes de tracao permanecem negativas para entrar diretamente
        na restricao de nao-tracao.

        :return: Nada (asserts internos)
        """
        f_zk, m_xk, h_x, h_y, h_z = 100.0, 200.0, 2.0, 2.0, 0.60
        _, sigma_min = calcular_sigma_max_min(f_zk, m_xk, 0.0, h_x, h_y, h_z)
        area = h_x * h_y
        sigma_fz = (f_zk + 25.0 * area * h_z) / area
        sigma_mx = 6.0 * m_xk / (area * h_x)
        esperado = sigma_fz - sigma_mx
        assert sigma_min == pytest.approx(esperado)
        assert sigma_min < 0.0


# =============================================================================
# checagem_tensao_max_min
# =============================================================================
@pytest.mark.engineering
class TestChecagemTensaoMaxMin:
    """This class verifies the constraint formula for soil pressure violations."""

    def test_pressao_compressiva_no_limite(self):
        """This test verifies g = 0 when sigma equals sigma_adm.

        :return: Nada (asserts internos)
        """
        assert checagem_tensao_max_min(500.0, 500.0) == pytest.approx(0.0)

    def test_pressao_compressiva_dentro_do_limite(self):
        """This test verifies g < 0 for admissible compressive pressures.

        :return: Nada (asserts internos)
        """
        assert checagem_tensao_max_min(250.0, 500.0) == pytest.approx(-0.5)

    def test_pressao_compressiva_acima_do_limite(self):
        """This test verifies g > 0 when the soil capacity is exceeded.

        :return: Nada (asserts internos)
        """
        assert checagem_tensao_max_min(750.0, 500.0) == pytest.approx(0.5)

    def test_pressao_tracao_marca_violacao(self):
        """This test verifies that tensile pressures (sigma < 0) yield g > 0.

        Para `sigma < 0`, o ramo eh `g = -sigma / sigma_adm`, devolvendo
        valor positivo independente da magnitude.

        :return: Nada (asserts internos)
        """
        g = checagem_tensao_max_min(-100.0, 500.0)
        assert g == pytest.approx(0.2)
        assert g > 0


# =============================================================================
# checagem_geometria
# =============================================================================
@pytest.mark.engineering
class TestChecagemGeometria:
    """This class verifies the minimum overhang constraint between pillar and footing."""

    def test_sapata_no_limite_geometrico(self):
        """This test verifies g = 0 when h = a_p + 2·delta exactly.

        Com `delta = 0,10` (default) e `a_p = 0,30`, a sapata no limite
        deve ter `h = 0,30 + 0,20 = 0,50` m.

        :return: Nada (asserts internos)
        """
        assert checagem_geometria(0.50, 0.30) == pytest.approx(0.0, abs=1e-9)

    def test_sapata_maior_que_limite(self):
        """This test verifies g < 0 for a footing wider than the minimum required.

        :return: Nada (asserts internos)
        """
        g = checagem_geometria(1.00, 0.30)   # h muito maior que pilar+2*0.10
        assert g < 0

    def test_sapata_menor_que_limite(self):
        """This test verifies g > 0 for a footing that does not respect the minimum overhang.

        :return: Nada (asserts internos)
        """
        g = checagem_geometria(0.30, 0.30)   # h igual ao pilar (sem balanco)
        assert g > 0

    def test_balanco_personalizado(self):
        """This test verifies the constraint when the user supplies a custom `balanco_min`.

        :return: Nada (asserts internos)
        """
        # Com balanco_min=0.20, o limite passa para a_p + 0.40
        assert checagem_geometria(0.70, 0.30, balanco_min=0.20) == pytest.approx(
            0.0, abs=1e-9
        )


# =============================================================================
# verificacao_puncao_sapata
# =============================================================================
@pytest.mark.engineering
class TestVerificacaoPuncaoSapata:
    """This class verifies the punching-shear check at the C critical section (NBR 6118 19.5)."""

    def test_perimetro_critico_eh_2_a_mais_b(self):
        """This test verifies that the critical perimeter equals 2·(a_p + b_p).

        :return: Nada (asserts internos)
        """
        _, _, u_rd2, _ = verificacao_puncao_sapata(
            h_z=0.60, f_ck=25_000, a_p=0.30, b_p=0.40, f_zk=500.0, cob=0.04
        )
        assert u_rd2 == pytest.approx(2.0 * (0.30 + 0.40))

    def test_tau_resistente_nao_depende_da_carga(self):
        """This test verifies that tau_rd2 depends only on f_ck (and the cover via d).

        Trocar `f_zk` deve mover apenas tau_sd2; tau_rd2 permanece o mesmo
        para identico `(h_z, f_ck, cob)`.

        :return: Nada (asserts internos)
        """
        _, tau_rd2_a, _, _ = verificacao_puncao_sapata(
            h_z=0.60, f_ck=25_000, a_p=0.30, b_p=0.40, f_zk=200.0, cob=0.04
        )
        _, tau_rd2_b, _, _ = verificacao_puncao_sapata(
            h_z=0.60, f_ck=25_000, a_p=0.30, b_p=0.40, f_zk=2000.0, cob=0.04
        )
        assert tau_rd2_a == pytest.approx(tau_rd2_b)

    def test_g_rd2_aumenta_com_a_carga(self):
        """This test verifies that the constraint becomes more violated with larger F_zk.

        :return: Nada (asserts internos)
        """
        _, _, _, g_low = verificacao_puncao_sapata(
            h_z=0.60, f_ck=25_000, a_p=0.30, b_p=0.40, f_zk=200.0, cob=0.04
        )
        _, _, _, g_high = verificacao_puncao_sapata(
            h_z=0.60, f_ck=25_000, a_p=0.30, b_p=0.40, f_zk=5000.0, cob=0.04
        )
        assert g_high > g_low

    def test_alpha_v2_segue_formula_da_norma(self):
        """This test verifies the explicit form alpha_v2 = 1 - f_ck/250 (with f_ck in MPa).

        Para `f_ck = 25 MPa = 25 000 kPa`: alpha_v2 = 1 - 25/250 = 0,90.
        Logo tau_rd2 = 0,27 * 0,90 * (25 000 / 1,4) ≈ 4339,2857 kPa.

        :return: Nada (asserts internos)
        """
        _, tau_rd2, _, _ = verificacao_puncao_sapata(
            h_z=0.60, f_ck=25_000, a_p=0.30, b_p=0.40, f_zk=500.0, cob=0.04
        )
        esperado = 0.27 * (1 - 25 / 250) * (25_000 / 1.4)
        assert tau_rd2 == pytest.approx(esperado)


# =============================================================================
# sobreposicao_sapatas
# =============================================================================
@pytest.mark.engineering
class TestSobreposicaoSapatas:
    """This class verifies the AABB intersection used in the overlap check.

    A funcao espera 8 coordenadas por sapata na ordem (x1,y1,...,x4,y4),
    formando um retangulo alinhado aos eixos. Os testes constroem
    retangulos atraves de uma helper para manter os asserts legiveis.
    """

    @staticmethod
    def _rect(xc: float, yc: float, hx: float, hy: float):
        """This helper builds the 8-tuple of vertex coordinates for a centred AABB.

        :param xc: Coordenada x do centroide do retangulo
        :param yc: Coordenada y do centroide do retangulo
        :param hx: Largura total na direcao x
        :param hy: Altura total na direcao y

        :return: Tupla `(x1, y1, x2, y2, x3, y3, x4, y4)` com os
                 vertices percorridos no sentido (SW, SE, NE, NW)
        """
        return (
            xc - hx / 2, yc - hy / 2,
            xc + hx / 2, yc - hy / 2,
            xc + hx / 2, yc + hy / 2,
            xc - hx / 2, yc + hy / 2,
        )

    def test_retangulos_afastados_overlap_zero(self):
        """This test ensures non-touching rectangles produce zero area.

        :return: Nada (asserts internos)
        """
        a = self._rect(0, 0, 1, 1)
        b = self._rect(10, 10, 1, 1)
        assert sobreposicao_sapatas(*a, *b) == pytest.approx(0.0)

    def test_retangulos_apenas_encostando_overlap_zero(self):
        """This test ensures edge-touching rectangles produce zero area (open AABB).

        :return: Nada (asserts internos)
        """
        a = self._rect(0, 0, 2, 2)
        b = self._rect(2, 0, 2, 2)   # encosta na lateral direita
        assert sobreposicao_sapatas(*a, *b) == pytest.approx(0.0)

    def test_retangulos_identicos_overlap_completo(self):
        """This test ensures fully overlapping rectangles produce hx · hy.

        :return: Nada (asserts internos)
        """
        a = self._rect(0, 0, 2, 3)
        b = self._rect(0, 0, 2, 3)
        assert sobreposicao_sapatas(*a, *b) == pytest.approx(2.0 * 3.0)

    def test_retangulos_com_intersecao_parcial(self):
        """This test verifies the area of a partial intersection (1×1 quadrado).

        Dois retangulos 2×2 centrados em (0,0) e (1,1) compartilham um
        quadrado de lado 1 -> area = 1,0.

        :return: Nada (asserts internos)
        """
        a = self._rect(0, 0, 2, 2)
        b = self._rect(1, 1, 2, 2)
        assert sobreposicao_sapatas(*a, *b) == pytest.approx(1.0)

    def test_simetria_i_j(self):
        """This test verifies that swapping the order of the two rectangles preserves the area.

        :return: Nada (asserts internos)
        """
        a = self._rect(0, 0, 2, 2)
        b = self._rect(0.5, 0.3, 1.5, 2.5)
        assert sobreposicao_sapatas(*a, *b) == pytest.approx(
            sobreposicao_sapatas(*b, *a)
        )


# =============================================================================
# sobreposicao_matrix (vetorizada — Sprint 3.8)
# =============================================================================
@pytest.mark.engineering
class TestSobreposicaoMatrix:
    """This class verifies the vectorised N×N overlap matrix.

    `sobreposicao_matrix` deve produzir resultados *exatamente* iguais
    aos da versao escalar `sobreposicao_sapatas` para qualquer
    configuracao, com diagonal zerada e matriz simetrica. Esta e a
    rede de seguranca da Sprint 3.8 (substituicao do laco duplo
    df.iterrows() por uma matriz numpy).
    """

    @staticmethod
    def _bounds(centros, dims):
        """This helper turns lists of (xc, yc) and (hx, hy) into AABB arrays.

        :param centros: Lista de tuplas (xc, yc) — centroides
        :param dims:    Lista de tuplas (hx, hy) — dimensoes da sapata

        :return: Tupla `(xmin, xmax, ymin, ymax)` como arrays numpy
        """
        centros = np.asarray(centros, dtype=float)
        dims = np.asarray(dims, dtype=float)
        xmin = centros[:, 0] - dims[:, 0] / 2
        xmax = centros[:, 0] + dims[:, 0] / 2
        ymin = centros[:, 1] - dims[:, 1] / 2
        ymax = centros[:, 1] + dims[:, 1] / 2
        return xmin, xmax, ymin, ymax

    def test_diagonal_zerada(self):
        """This test ensures the diagonal of the overlap matrix is exactly zero.

        :return: Nada (asserts internos)
        """
        xmin, xmax, ymin, ymax = self._bounds(
            [(0, 0), (1, 1), (5, 5)], [(2, 2), (2, 2), (1, 1)]
        )
        m = sobreposicao_matrix(xmin, xmax, ymin, ymax)
        assert np.all(np.diag(m) == 0.0)

    def test_matriz_simetrica(self):
        """This test ensures M[i, j] == M[j, i] for every pair.

        :return: Nada (asserts internos)
        """
        xmin, xmax, ymin, ymax = self._bounds(
            [(0, 0), (0.7, 0.4), (3, -1)], [(2, 2), (1.5, 2.5), (2, 2)]
        )
        m = sobreposicao_matrix(xmin, xmax, ymin, ymax)
        assert np.allclose(m, m.T, rtol=0, atol=0)

    def test_concorda_com_versao_escalar(self):
        """This test compares the matrix entry-by-entry against the legacy scalar function.

        Constroi um cenario com 4 sapatas em diferentes regimes (sem
        sobreposicao, com sobreposicao parcial, identicas) e verifica
        que cada entrada (i, j) coincide com o valor de
        `sobreposicao_sapatas` aplicado aos vertices das duas sapatas.

        :return: Nada (asserts internos)
        """
        centros = [(0, 0), (0.5, 0.5), (10, 10), (0, 0)]
        dims = [(2.0, 2.0), (1.0, 1.0), (2.0, 2.0), (2.0, 2.0)]
        xmin, xmax, ymin, ymax = self._bounds(centros, dims)
        m = sobreposicao_matrix(xmin, xmax, ymin, ymax)

        n = len(centros)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                xc_i, yc_i = centros[i]
                hx_i, hy_i = dims[i]
                xc_j, yc_j = centros[j]
                hx_j, hy_j = dims[j]
                rect_i = (
                    xc_i - hx_i / 2, yc_i - hy_i / 2,
                    xc_i + hx_i / 2, yc_i - hy_i / 2,
                    xc_i + hx_i / 2, yc_i + hy_i / 2,
                    xc_i - hx_i / 2, yc_i + hy_i / 2,
                )
                rect_j = (
                    xc_j - hx_j / 2, yc_j - hy_j / 2,
                    xc_j + hx_j / 2, yc_j - hy_j / 2,
                    xc_j + hx_j / 2, yc_j + hy_j / 2,
                    xc_j - hx_j / 2, yc_j + hy_j / 2,
                )
                assert m[i, j] == pytest.approx(
                    sobreposicao_sapatas(*rect_i, *rect_j), rel=0, abs=0
                )

    def test_caso_sem_sobreposicao_devolve_zeros(self):
        """This test ensures three far-apart rectangles produce an all-zero matrix.

        :return: Nada (asserts internos)
        """
        xmin, xmax, ymin, ymax = self._bounds(
            [(0, 0), (100, 100), (-100, 100)], [(1, 1), (1, 1), (1, 1)]
        )
        m = sobreposicao_matrix(xmin, xmax, ymin, ymax)
        assert np.all(m == 0.0)

    def test_caso_unitario_n_igual_1(self):
        """This test ensures a single rectangle yields a 1×1 zero matrix.

        :return: Nada (asserts internos)
        """
        xmin, xmax, ymin, ymax = self._bounds([(0, 0)], [(2, 2)])
        m = sobreposicao_matrix(xmin, xmax, ymin, ymax)
        assert m.shape == (1, 1)
        assert m[0, 0] == 0.0


# =============================================================================
# Guardrails — edge cases of the analytical helpers (Sprint 4.8)
# =============================================================================
@pytest.mark.engineering
class TestEngineeringEdgeCases:
    """This class documents edge-case behaviour of the engineering helpers.

    The audit raised four concerns:
      1. tensao_adm_solo with unknown soil silently falls back to spt/50.
      2. tensao_adm_solo with spt=0 returns 0; downstream divisions blow up.
      3. calcular_sigma_max_min divides by f_zk; f_zk=0 explodes.
      4. verificacao_puncao_sapata uses d = h_z - cob; h_z <= cob explodes.
    These tests pin the **current** behaviour so a future fix is an
    intentional change reviewed against the regression baseline.
    """

    def test_tensao_adm_solo_unknown_soil_falls_back_to_spt_over_50(self):
        """An unknown soil label silently picks the silte/argila branch.

        Locked here as a guardrail. The Excel reader already validates
        the soil string against the official set, so this branch is
        only reachable from direct programmatic calls; if a future
        sprint replaces the silent fallback with a ValueError, this
        test must be updated explicitly.
        """
        assert tensao_adm_solo("pedra rara", 50) == pytest.approx(1000.0)
        assert tensao_adm_solo("UNKNOWN", 25) == pytest.approx(500.0)

    def test_tensao_adm_solo_spt_zero_returns_zero(self):
        """spt=0 -> sigma_adm=0; downstream callers must handle this."""
        assert tensao_adm_solo("argila", 0) == 0.0

    def test_checagem_tensao_zero_admissible_is_undefined(self):
        """sigma_adm=0 raises ZeroDivisionError; downstream guard required."""
        with pytest.raises(ZeroDivisionError):
            checagem_tensao_max_min(sigma=10.0, sigma_adm=0.0)

    def test_sigma_max_min_zero_load_raises(self):
        """f_zk=0 is outside the supported load-combination domain."""
        with pytest.raises(ValueError, match="f_zk"):
            calcular_sigma_max_min(f_zk=0.0, m_xk=10.0, m_yk=10.0,
                                   h_x=1.0, h_y=1.0, h_z=0.6)

    def test_puncao_h_z_equal_to_cover_raises(self):
        """h_z == cob makes d = 0; the guard raises an explicit ValueError.

        Updated from the Sprint 4.8 guardrail (which pinned the old
        ``ZeroDivisionError``): the function now validates ``d > 0``
        upfront so an unbuildable footing fails loudly instead of
        depending on the division to blow up.
        """
        with pytest.raises(ValueError, match="effective depth"):
            verificacao_puncao_sapata(
                h_z=0.04, f_ck=25_000.0, a_p=0.30, b_p=0.30,
                f_zk=500.0, cob=0.04,
            )

    def test_puncao_h_z_below_cover_raises(self):
        """h_z < cob makes d < 0; the guard raises an explicit ValueError.

        Updated from the Sprint 4.8 guardrail (which pinned the old
        silent behaviour of returning a *negative* tau_sd2 that made
        the constraint read as feasible). A negative effective depth is
        physically meaningless, so it now fails loudly.
        """
        with pytest.raises(ValueError, match="effective depth"):
            verificacao_puncao_sapata(
                h_z=0.03, f_ck=25_000.0, a_p=0.30, b_p=0.30,
                f_zk=500.0, cob=0.04,
            )


# =============================================================================
# Punção — seção crítica C' (NBR 6118 item 19.5, contorno a 2d da face)
# =============================================================================
@pytest.mark.engineering
class TestPuncaoCLinha:
    """This class verifies the C' punching check against hand-computed values.

    Referencia da formulacao: NBR 6118:2014 item 19.5 (contorno a 2d,
    cantos circulares, Tabela 19.2 para transferencia de momento e
    Tabela 17.3 para rho_min), conforme sistematizado em Santos, Lima
    Neto e Ferreira (2018), IBRACON Struct. Mater. J. 11(2).
    """

    def test_rho_minimo_valores_tabelados_e_interpolacao(self):
        """Tabela 17.3: valores exatos nas classes e interpolacao entre elas."""
        assert rho_minimo_flexao(25_000.0) == pytest.approx(0.0015)
        assert rho_minimo_flexao(50_000.0) == pytest.approx(0.00208)
        assert rho_minimo_flexao(90_000.0) == pytest.approx(0.00256)
        # interpolacao linear entre C35 (0.164%) e C40 (0.179%)
        assert rho_minimo_flexao(37_500.0) == pytest.approx(0.0017150)
        # piso abaixo de C20 (fora da faixa estrutural, documentado)
        assert rho_minimo_flexao(15_000.0) == pytest.approx(0.0015)

    def test_k_tabela_19_2_valores_e_saturacao(self):
        """Tabela 19.2: exatos, interpolados e saturados nos limites."""
        assert k_tabela_19_2(0.5) == pytest.approx(0.45)
        assert k_tabela_19_2(1.0) == pytest.approx(0.60)
        assert k_tabela_19_2(1.5) == pytest.approx(0.65)   # interp 1.0-2.0
        assert k_tabela_19_2(3.0) == pytest.approx(0.80)
        assert k_tabela_19_2(0.2) == pytest.approx(0.45)   # satura em 0.5
        assert k_tabela_19_2(10.0) == pytest.approx(0.80)  # satura em 3.0

    def test_valor_de_referencia_calculado_a_mao(self):
        """Caso de referencia: pilar 0.30x1.50 m, C25, hz=0.60, cob=0.04.

        Calculo independente (d = 0.56 m):
            u1'    = 2(1.80) + 4*pi*0.56          = 10.63717...
            W_px   = 0.045 + 0.45 + 3.36 + 5.0176 + 2*pi*0.56*0.30
            W_py   = 1.125 + 0.45 + 0.672 + 5.0176 + 2*pi*0.56*1.50
            Kx     = K(0.2 -> 0.5)                = 0.45
            Ky     = K(5.0 -> 3.0)                = 0.80
            tau_sd1 = 1.4*900/(u1'*d) + 0.45*1.4*60/(W_px*d)
                      + 0.80*1.4*10/(W_py*d)
            k_e     = 1 + sqrt(20/56)             = 1.59761...
            tau_rd1 = 0.13*k_e*(100*0.0015*25)^(1/3)*1000
        """
        import math
        d = 0.56
        u1 = 2.0 * (0.30 + 1.50) + 4.0 * math.pi * d
        w_px = 0.30**2 / 2 + 0.30 * 1.50 + 4 * 1.50 * d + 16 * d**2 \
            + 2 * math.pi * d * 0.30
        w_py = 1.50**2 / 2 + 0.30 * 1.50 + 4 * 0.30 * d + 16 * d**2 \
            + 2 * math.pi * d * 1.50
        tau_sd_esp = (1.4 * 900.0) / (u1 * d) \
            + 0.45 * (1.4 * 60.0) / (w_px * d) \
            + 0.80 * (1.4 * 10.0) / (w_py * d)
        k_e = 1.0 + math.sqrt(20.0 / 56.0)
        tau_rd_esp = 0.13 * k_e * (100.0 * 0.0015 * 25.0) ** (1 / 3) * 1000.0

        tau_sd1, tau_rd1, u_rd1, g_rd1 = verificacao_puncao_sapata_c_linha(
            h_z=0.60, f_ck=25_000.0, a_p=0.30, b_p=1.50,
            f_zk=900.0, m_xk=-60.0, m_yk=10.0, cob=0.04,
        )
        assert u_rd1 == pytest.approx(u1, rel=1e-12)
        assert tau_sd1 == pytest.approx(tau_sd_esp, rel=1e-12)
        assert tau_rd1 == pytest.approx(tau_rd_esp, rel=1e-12)
        assert g_rd1 == pytest.approx(tau_sd_esp / tau_rd_esp - 1.0, rel=1e-12)
        # o momento entra em modulo: sinal de m_xk nao pode reduzir tau_sd1
        tau_sd1_pos, *_ = verificacao_puncao_sapata_c_linha(
            h_z=0.60, f_ck=25_000.0, a_p=0.30, b_p=1.50,
            f_zk=900.0, m_xk=60.0, m_yk=10.0, cob=0.04,
        )
        assert tau_sd1_pos == pytest.approx(tau_sd1, rel=1e-15)

    def test_g_diminui_com_altura(self):
        """A folga da verificacao C' cresce monotonicamente com h_z."""
        gs = [
            verificacao_puncao_sapata_c_linha(
                h_z=h, f_ck=25_000.0, a_p=0.30, b_p=1.50,
                f_zk=900.0, m_xk=60.0, m_yk=10.0, cob=0.04,
            )[3]
            for h in (0.60, 1.00, 2.00, 3.00)
        ]
        assert all(g2 < g1 for g1, g2 in zip(gs, gs[1:]))

    def test_h_z_igual_ao_cobrimento_levanta_erro(self):
        """Mesma guarda de altura util da secao C."""
        with pytest.raises(ValueError, match="effective depth"):
            verificacao_puncao_sapata_c_linha(
                h_z=0.04, f_ck=25_000.0, a_p=0.30, b_p=0.30,
                f_zk=500.0, cob=0.04,
            )
