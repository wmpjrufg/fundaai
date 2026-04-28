"""Configuracao compartilhada da suite de testes do FundaIA.

Adiciona a raiz do repositorio ao sys.path para que os testes possam
importar `fundacao` e `metapy_toolbox` sem necessidade de instalar o
projeto. Tambem expoe fixtures comuns (caminho dos assets, dataframe
canonico do problema de tres fundacoes, configuracao de calibracao
historica) reutilizadas em multiplos arquivos de teste.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

# --- Path setup --------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --- Fixtures de paths -------------------------------------------------------
@pytest.fixture(scope="session")
def repo_root() -> Path:
    """This fixture returns the absolute path to the repository root.

    :return: Caminho absoluto para a raiz do repositorio FundaIA
    """
    return ROOT


@pytest.fixture(scope="session")
def assets_dir(repo_root: Path) -> Path:
    """This fixture returns the path to the project assets directory.

    :param repo_root: Fixture com a raiz do repositorio

    :return: Caminho absoluto para a pasta `assets/`
    """
    return repo_root / "assets"


# --- Fixtures de dados canonicos --------------------------------------------
@pytest.fixture(scope="session")
def df_problema_um(assets_dir: Path) -> pd.DataFrame:
    """This fixture loads the single-foundation reference problem (`problema_fund_um`).

    :param assets_dir: Fixture com o caminho da pasta `assets/`

    :return: DataFrame com 1 fundacao e 3 combinacoes de carregamento
    """
    return pd.read_excel(assets_dir / "problema_fund_um.xlsx")


@pytest.fixture(scope="session")
def df_problema_tres(assets_dir: Path) -> pd.DataFrame:
    """This fixture loads the three-foundation reference problem (`problema_fund_três`).

    Eh o caso usado como referencia de regressao numerica nas Sprints
    0 e 1: `_avaliar_projeto(x_seed42, args=(df, 3, 25e3, 0.04))`
    deve produzir `of = 19,706042` quando `seed=42` e `n_pop=9`
    geram a mesma populacao LHS.

    :param assets_dir: Fixture com o caminho da pasta `assets/`

    :return: DataFrame com 3 fundacoes e 3 combinacoes de carregamento
    """
    return pd.read_excel(assets_dir / "problema_fund_três.xlsx")


@pytest.fixture(scope="session")
def cfg_calibracao() -> Dict[str, Any]:
    """This fixture exposes the historical calibration used by the FundaIA UI.

    Espelha os defaults da pagina de dimensionamento (`pages/sapatas.py`)
    para que os testes de regressao usem exatamente os mesmos parametros
    de projeto que produziram os resultados parciais reportados.

    :return: Dicionario com as chaves
             [n_comb] = numero de combinacoes (int)
             [f_ck_kpa] = resistencia caracteristica do concreto (kPa)
             [cob_m] = cobrimento da armadura (m)
             [h_min_m] = dimensao minima da sapata (m)
             [h_max_m] = dimensao maxima da sapata (m)
    """
    return {
        "n_comb": 3,
        "f_ck_kpa": 25_000.0,
        "cob_m": 0.04,
        "h_min_m": 0.60,
        "h_max_m": 3.00,
    }
