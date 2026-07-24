---
tags: [projeto, stack]
---

# Stack Tecnológico

## Linguagem
- **Python 3.10+** (conforme README).

## UI
- **streamlit 1.52** — multi-page (`st.navigation`), session_state para i18n.
- **matplotlib** — plot do arranjo em planta.
- **ezdxf** — exportação para AutoCAD (DXF R2010).

## Otimização
- **mealpy 3.0.3** — GA (`GA.BaseGA`), PSO e demais metaheurísticas.
- **scipy.optimize** — minimizers locais (L-BFGS-B, SLSQP, TNC, trust-constr).
- **scipy.stats.qmc.LatinHypercube** — amostragem da pop. inicial.
- **metapy_toolbox** — biblioteca interna com EGO, GA, GWO.

## Aprendizado de máquina (surrogate)
- **scikit-learn 1.7** — `GaussianProcessRegressor`, `Pipeline`, `StandardScaler`, kernels (`RBF`, `Matern`, `RationalQuadratic`, `DotProduct`, `ExpSineSquared`, `WhiteKernel`, `ConstantKernel`).
- **joblib** — persistência dos modelos (`.pkl`).
- **multiprocessing.Pool** — treino paralelo dos GPRs.

## Dados
- **pandas / openpyxl / xlsxwriter** — leitura e escrita de Excel.
- **numpy** — operações numéricas.

## Dev/Ops
- **playwright** — `ops/wake_up.py` para acordar app no Streamlit Cloud.
- **pip-chill** — geração de `requirements.txt`.

## Arquivos-chave de configuração
- `requirements.txt` — ⚠️ atualmente em UTF-16/BOM, ver [[07_Issues/Issue - requirements.txt UTF-16]].
- `.gitignore` — ignora `*.txt` (curioso — pode mascarar dados).
- `env-setup.py` — bootstrap multi-OS.
