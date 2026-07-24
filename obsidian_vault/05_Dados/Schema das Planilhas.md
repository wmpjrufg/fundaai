---
tags: [dados, schema]
---

# Schema das Planilhas de Entrada

Todas as planilhas em `assets/` (e `assets/data/`) seguem o mesmo schema.

## Colunas obrigatórias

| Coluna | Tipo | Unidade | Descrição |
|---|---|---|---|
| `Elemento` | str | — | rótulo (ex.: `P04`) |
| `ap (m)` | float | m | dimensão do pilar em X |
| `bp (m)` | float | m | dimensão do pilar em Y |
| `spt` | float | — | índice de [[02_Engenharia/SPT - Sondagem]] |
| `solo` | str | — | `pedregulho` / `areia` / `silte` / `argila` |
| `xg (m)` | float | m | coordenada X do centróide |
| `yg (m)` | float | m | coordenada Y do centróide |

## Colunas dinâmicas (uma para cada combinação `c1..cN`)

| Coluna | Tipo | Unidade | Descrição |
|---|---|---|---|
| `Fz-c{i}` | float | kN | carga axial característica |
| `Mx-c{i}` | float | kN·m | componente que gera variação de pressão ao longo de X (`h_x`) |
| `My-c{i}` | float | kN·m | componente que gera variação de pressão ao longo de Y (`h_y`) |

> [!warning] Convenção de momentos
> `Mx` e `My` são nomes internos do FundaIA para a formulação de tensão. Se a planilha for montada a partir de software estrutural, conferir se o momento em torno do eixo global X não deve entrar como variação na direção Y, e vice-versa. A importação deve entregar ao FundaIA as componentes já convertidas para a convenção acima.

## Sanitização aplicada em [[04_Codigo/pages - sapatas.py]]

```python
for col in df.columns:
    if col.startswith(("Fz-", "Mx-", "My-")):
        df[col] = df[col].astype(str).str.replace(",", ".", regex=False).astype(float)
```

⇒ aceita decimais escritos com vírgula.

## Templates disponíveis

| Arquivo | N fundações | N combinações |
|---|---|---|
| `assets/problema_fund_um.xlsx` | 1 | 3 |
| `assets/problema_fund_dois.xlsx` | 2 | 3 |
| `assets/problema_fund_três.xlsx` | 3 | 3 (default no botão de download) |
| `assets/data/toy_problem.xlsx` | 3 | 3 (estudos GPR) |
| `assets/data/toy_problem_copy{,_2,_3}.xlsx` | 3 | 3 (variações para experimentos) |

## Vínculos

- [[04_Codigo/pages - home.py]] (download do template)
- [[04_Codigo/pages - sapatas.py]] (upload e leitura)
- [[01_Projeto/Pipeline de Execução]]
