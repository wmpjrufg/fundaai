---
tags: [melhorias, refactor, arquitetura, sugestao]
---

# Refactor — Separar UI de Domínio

> [!note] Sugestão
> Hoje [[04_Codigo/pages - sapatas.py]] mistura: leitura Excel, parametrização, instanciação do GA, laço de repetições, pós-processamento, plot, DXF. Quebrar em camadas isola responsabilidades.

## Antes (procedural)

```python
# pages/sapatas.py — 530 linhas
df = pd.read_excel(uploaded)
for rep in range(n_rep):
    x_new, best_of, _ = ego_01_architecture(obj_felipe_lucas, ...)
    if best_of < best_of_aux: ...
plot_data(payload); save_dxf(payload)
```

## Depois (camadas)

```python
# ui/streamlit/pages/sapatas.py
from core.io.excel import carregar_projeto
from core.api.optimize import otimizar
from core.io.cad_dxf import exportar
projeto = carregar_projeto(uploaded)
config  = ConfigOtimizacao(...)
result  = otimizar(projeto, config)
st.dataframe(result.to_dataframe())
st.pyplot(result.plot_layout())
st.download_button(..., data=exportar(result.layout))
```

## Função pura `otimizar`

```python
def otimizar(projeto: FundacaoProjeto, config: ConfigOtimizacao) -> ResultadoOtimizacao:
    """Sem Streamlit, sem I/O. Pode ser chamada por CLI, API, notebook ou UI."""
```

Vantagens imediatas:

- Testar sem subir Streamlit.
- Reusar de notebooks / CLI / FastAPI.
- Trocar UI sem tocar lógica.

## Padrões úteis

- **Hexagonal architecture (ports & adapters)**: domínio no centro, adapters de I/O nas bordas.
- **DTO**: `ConfigOtimizacao` é um DTO validado por [[10_Melhorias/Refactor - Configuração com Pydantic]].

## Vínculos

- [[10_Melhorias/Refactor - Plano Geral]]
- [[10_Melhorias/Refactor - POO Domain Model]]
