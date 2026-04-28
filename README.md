# FundaIA

> Plataforma computacional para o **dimensionamento otimizado de sapatas isoladas** em concreto armado, integrando critérios estruturais, geotécnicos e geométricos da NBR 6118 e da NBR 6122 em um ambiente único de projeto. A solução é obtida por uma arquitetura híbrida do tipo **Efficient Global Optimization (EGO)**, combinando um modelo substituto baseado em Regressão por Processos Gaussianos (GPR) com um Algoritmo Genético (AG) como otimizador interno da função de aquisição *Expected Improvement* (EI).

---

## Sumário

- [Visão geral](#visão-geral)
- [Pipeline de execução](#pipeline-de-execução)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Setup do ambiente](#setup-do-ambiente)
- [Como rodar a aplicação](#como-rodar-a-aplicação)
- [Suite de testes](#suite-de-testes)
- [Stack técnica](#stack-técnica)
- [Status atual](#status-atual)
- [Próximos passos](#próximos-passos)

---

## Visão geral

O dimensionamento de sapatas isoladas é tradicionalmente conduzido por procedimentos iterativos de tentativa e verificação, dependentes da experiência do projetista e frequentemente conservadores quanto ao consumo de concreto. O **FundaIA** propõe uma alternativa computacional que formaliza o problema como uma **otimização mono-objetivo penalizada**, cuja função objetivo é o volume total de concreto, sujeita a restrições de:

- **Tensão admissível do solo** (correlação empírica com SPT, NBR 6122).
- **Flexão composta na base da sapata** (σ_max e σ_min com excentricidades nos dois eixos).
- **Punção na seção crítica C** (NBR 6118 item 19.5).
- **Compatibilidade geométrica pilar-sapata** (balanço mínimo configurável).
- **Não sobreposição** entre fundações vizinhas (modelagem AABB).

A solução é obtida por uma arquitetura híbrida em três níveis: o **EGO** ([Jones, Schonlau & Welch, 1998](https://doi.org/10.1023/A:1008306431147)) orquestra a busca, um **GPR** ([Williams & Rasmussen, 2006](https://gaussianprocess.org/gpml/)) atua como surrogate da função objetivo cara, e um **AG** ([Mealpy](https://mealpy.readthedocs.io/)) maximiza a função de aquisição *Expected Improvement* a cada iteração.

A interface web é construída em **Streamlit** ([fundaai.streamlit.app](https://fundaai.streamlit.app/)) e permite que o usuário forneça os dados de projeto via planilha Excel, parametrize o método, execute a otimização e exporte o resultado tanto em Excel quanto em DXF para integração direta com o fluxo tradicional de CAD.

A pesquisa associada é desenvolvida no contexto de uma Iniciação Científica em andamento.

## Pipeline de execução

```
┌────────────────┐   ┌──────────────────┐   ┌──────────────────────┐
│ Excel de       │──▶│ Sanitização e    │──▶│ Configuração da      │
│ entrada (Nspt, │   │ leitura via      │   │ otimização (f_ck,    │
│ ap, bp, F, M)  │   │ pandas/openpyxl  │   │ cob, h_min, h_max)   │
└────────────────┘   └──────────────────┘   └──────────┬───────────┘
                                                       │
                                                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ ego_01_architecture                                             │
│   1. População inicial via Latin Hypercube (seed propagada)     │
│   2. Avaliação real da função pseudo-objetivo penalizada Θ(x)   │
│   3. Treino do GPR (Pipeline StandardScaler + GPR)              │
│   4. Maximização de EI por GA interno (Mealpy)                  │
│   5. Avaliação real do candidato e atualização da base          │
│   6. Repetição por n_gen iterações                              │
│   7. Repetição externa por n_rep com seeds independentes        │
└──────────────────────────────────┬──────────────────────────────┘
                                   ▼
            ┌────────────────────────────────────────────┐
            │ Resultado: dimensões ótimas (h_x, h_y, h_z)│
            │ + verificações detalhadas + arranjo 2D     │
            │ + exportação Excel e DXF                   │
            └────────────────────────────────────────────┘
```

## Estrutura do repositório

```
fundaIA/
├── app.py                       # entry-point Streamlit (navegação multi-página)
├── fundacao.py                  # núcleo de engenharia (NBR 6118/6122) + GPR + função objetivo
├── pages/
│   ├── home.py                  # página inicial bilíngue (PT/EN) + download do template
│   └── sapatas.py               # página de dimensionamento (upload, otimização, resultados)
├── metapy_toolbox/              # biblioteca interna de otimização
│   ├── ego.py                   # arquitetura EGO híbrida
│   ├── funcs.py                 # LHS, fitness, evaluation, bounds
│   ├── genetic_algorithm.py     # GA com 8 operadores de crossover
│   ├── grey_wolf.py             # GWO (disponível, não usado em produção)
│   └── benchmark.py             # funções clássicas (sphere, rosenbrock, ackley, ...)
├── tests/                       # suite pytest (55 testes)
│   ├── conftest.py              # fixtures compartilhadas
│   ├── test_engenharia.py       # NBR 6118/6122 (26 testes)
│   ├── test_avaliar_projeto.py  # regressão numérica do núcleo (6 testes)
│   ├── test_ego_historico.py    # contrato do EGO (8 testes)
│   └── test_benchmark.py        # funções benchmark (15 testes)
├── ops/
│   └── wake_up.py               # robô Playwright para acordar app no Streamlit Cloud
├── assets/
│   ├── problema_fund_um.xlsx    # template: 1 fundação, 3 combinações
│   ├── problema_fund_dois.xlsx  # template: 2 fundações, 3 combinações
│   ├── problema_fund_três.xlsx  # template oficial: 3 fundações, 3 combinações
│   ├── data/                    # datasets de estudo (toy problem)
│   ├── tables/                  # métricas exportadas pelos notebooks
│   └── graphics/                # figuras geradas pelos experimentos GPR
├── models/                      # modelos GPR persistidos (.pkl) para análise
├── old/                         # material legado preservado para referência
├── *.ipynb                      # notebooks de teste e exploração da FO/GPR
├── requirements.txt             # dependências (UTF-8, organizadas por categoria)
├── pytest.ini                   # configuração da suite de testes
└── env-setup.py                 # bootstrap multi-OS do venv
```

## Setup do ambiente

### Requisitos
- Python 3.10 ou superior.
- Acesso a `pip` e a um shell padrão (bash, zsh, PowerShell).

### Instalação manual

```bash
# 1. Clone do repositório
git clone https://github.com/wmpjrufg/fundaIA.git
cd fundaIA

# 2. Criação do ambiente virtual
python3 -m venv .venv

# 3. Ativação (Linux / macOS)
source .venv/bin/activate
# ou (Windows PowerShell)
.venv\Scripts\Activate.ps1

# 4. Instalação das dependências
pip install -r requirements.txt
```

### Bootstrap automático

Alternativa multi-OS: `python env-setup.py` cria o venv e instala as dependências automaticamente.

## Como rodar a aplicação

```bash
streamlit run app.py
```

A interface abre no navegador (em geral `http://localhost:8501`). Fluxo de uso:

1. Definir os parâmetros gerais (`fck`, cobrimento, dimensões mínima/máxima da sapata, número de gerações e tamanho da população).
2. Fazer upload da planilha Excel de entrada (template disponível para download na página inicial).
3. Clicar em **Dimensionar** — o método EGO+GPR+AG é executado em `n_rep = 5` repetições com seeds independentes.
4. Visualizar o arranjo otimizado em planta e exportar os resultados em Excel ou DXF.

### Schema da planilha de entrada

| Coluna | Unidade | Descrição |
|---|---|---|
| `Elemento` | — | rótulo do pilar (ex.: `P04`) |
| `ap (m)`, `bp (m)` | m | dimensões do pilar |
| `spt` | — | índice de sondagem SPT |
| `solo` | — | `pedregulho`, `areia`, `silte` ou `argila` |
| `xg (m)`, `yg (m)` | m | coordenadas do centróide do pilar |
| `Fz-c{i}` | kN | carga axial característica na combinação `i` |
| `Mx-c{i}`, `My-c{i}` | kN·m | momentos característicos na combinação `i` |

## Suite de testes

A suite de testes funciona como **rede de segurança regressiva** para o comportamento numérico atual, viabilizando refatorações futuras sem risco de regressão silenciosa.

```bash
# Toda a suite (55 testes em ~3 segundos)
pytest

# Apenas os testes da camada de engenharia
pytest -m engineering

# Apenas os testes de regressão numérica (trava o OF de referência)
pytest -m regression

# Apenas os testes do contrato do EGO (histórico, IDs, reprodutibilidade)
pytest -m optimization

# Apenas os testes das funções benchmark
pytest -m benchmark
```

Cobertura atual: **55 testes** distribuídos entre engenharia (26), regressão do `_avaliar_projeto` (6), contrato do `ego_01_architecture` (8) e benchmarks (15). Trava o valor de referência `of = 19,70604234767181` para o caso `problema_fund_três.xlsx` com seed canônica.

## Stack técnica

| Camada | Tecnologia |
|---|---|
| Linguagem | Python 3.10+ |
| Interface web | [Streamlit](https://streamlit.io/) |
| Surrogate (GPR) | [scikit-learn](https://scikit-learn.org/) — `GaussianProcessRegressor` + kernels (`RBF`, `Matern`, `RationalQuadratic`, `DotProduct`, `ExpSineSquared`, `WhiteKernel`) |
| Metaheurísticas | [Mealpy](https://mealpy.readthedocs.io/) — `GA.BaseGA`, PSO, GWO, etc. |
| Otimização local | `scipy.optimize` — L-BFGS-B, SLSQP, TNC, trust-constr |
| Amostragem | `scipy.stats.qmc.LatinHypercube` |
| Persistência de modelos | [joblib](https://joblib.readthedocs.io/) |
| Planilhas | [openpyxl](https://openpyxl.readthedocs.io/) + [xlsxwriter](https://xlsxwriter.readthedocs.io/) |
| Exportação CAD | [ezdxf](https://ezdxf.readthedocs.io/) |
| Testes | [pytest](https://docs.pytest.org/) |
| Operação (wake-up) | [Playwright](https://playwright.dev/) |

## Status atual

O projeto está em **fase de saneamento de código e validação experimental**, conforme roadmap interno. Marcos recentes:

- ✅ **Saneamento de base** — codificação de `requirements.txt`, eliminação de duplicação em `pages/sapatas.py`, remoção de módulo morto, fusão de funções-objetivo redundantes, parametrização do fator de penalidade.
- ✅ **Correções científicas** — histórico do EGO com `ITER` e `ID` corretos, repetições de robustez agora independentes, parâmetro `seed` propagado para reprodutibilidade.
- ✅ **Rede de segurança** — suite `pytest` com 55 testes cobrindo engenharia, regressão numérica, contrato do EGO e benchmarks.
- 🔄 **Em curso** — refatoração estrutural (separação de UI e domínio, modelagem orientada a objetos, vetorização da função objetivo).
- 🔮 **Próxima frente** — incorporação explícita do problema de empacotamento (posicionamento conjunto das fundações como variável de projeto).

## Próximos passos

A frente seguinte da pesquisa contempla a **incorporação do problema de empacotamento (packing 2D)** ao processo de otimização, tratando as posições `(xg, yg)` como variáveis de projeto adicionais sob restrições rígidas de não sobreposição e fronteira do lote. O objetivo é convergir para uma versão acoplada do problema (sizing + layout) coerente com o escopo da pesquisa em andamento.

Para questões técnicas ou colaborações, abrir uma *issue* neste repositório.
