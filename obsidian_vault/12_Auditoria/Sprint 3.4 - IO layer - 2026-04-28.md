---
tags: [refactor, sprint, log, arquitetura, io]
data: 2026-04-28
branch: refactor/core-architecture
escopo: Sprint 3.4 — camada de IO (Excel reader + DXF writer)
---

# Sprint 3.4 — IO layer

> Log da quarta sub-sprint da Sprint 3 (refactor estrutural). Introduz
> a camada de IO em `core/io/`, isolando leitura de planilhas e
> exportação de DXF da página Streamlit. Esta é a **porta única de
> entrada** do projeto — daí a cobertura de testes ampla (21 casos).

## Escopo executado

| # | Item | Arquivo | Status |
|---|---|---|---|
| 1 | Leitor Excel `read_projeto_from_excel` | `core/io/excel.py` | ✅ |
| 2 | Escritor DXF `sapatas_to_dxf_bytes` | `core/io/cad_dxf.py` | ✅ |
| 3 | Re-exports `core.io` | `core/io/__init__.py` | ✅ |
| 4 | Testes (21 casos) | `tests/test_io.py` | ✅ |

## Decisões de design

### Excel reader (`read_projeto_from_excel`)

Recebe **`path`** (str/Path) ou **buffer file-like** (`UploadedFile` do Streamlit). A assinatura é:

```python
read_projeto_from_excel(
    path_or_buffer, *, f_ck_kpa, cobrimento_m, sheet_name=0
) -> FundacaoProjeto
```

Pipeline interno:
1. **`_read_dataframe`** — `pd.read_excel`, com erro claro se o path não existe.
2. **`_validate_fixed_schema`** — checa as 7 colunas obrigatórias e que o body não está vazio.
3. **`_detect_n_comb`** — descobre N a partir das colunas `Fz-c{i}|Mx-c{i}|My-c{i}`, exigindo:
   - índices contíguos a partir de `c1` (sem gaps);
   - cada combinação tem o trio Fz/Mx/My completo.
4. **`_sanitize_loads_inplace`** — `str → replace("," → ".") → float` para todas as 3·N colunas (preserva o comportamento histórico de aceitar vírgula decimal).
5. **`_build_projeto`** — itera linha-a-linha:
   - rejeita `Elemento` vazio ou duplicado;
   - normaliza `solo` (lowercase) e rejeita tipos desconhecidos;
   - constrói `Pilar`, `Solo`, `Combinacao` (cada construtor faz suas próprias validações);
   - monta `FundacaoProjeto`.

Mensagens de erro são **explícitas e numeradas por linha** quando falham por linha. Exemplos:

```
spreadsheet is missing required columns: ['spt']. expected at least [...].
load combination indices must be contiguous starting at 1; missing combinations: c[2].
combination c2 is incomplete; column 'My-c2' is missing.
row 1 ('P01'): unknown soil type 'lava'; expected one of ['areia', 'argila', 'pedregulho', 'silte'].
duplicated 'Elemento' label 'P01'; each row must be unique.
```

### DXF writer (`sapatas_to_dxf_bytes`)

```python
sapatas_to_dxf_bytes(sapatas: Iterable[Sapata]) -> bytes
```

- Recebe diretamente entidades `Sapata` (em vez do dicionário ad-hoc do legado).
- Gera o DXF inteiramente em memória via `io.StringIO` + `doc.write(buf)`.
- Encoding: usa `doc.encoding` (tipicamente `cp1252`).
- **Sem `tempfile`** — resolve a issue [[07_Issues/Issue - DXF tempfile não removido]].
- Cada sapata: 4 LINE + 1 POINT + 1 TEXT (label do pilar).

## Testes (21 casos em `tests/test_io.py`)

### Round-trip com templates oficiais (4 testes)
- Template 1 fundação (`problema_fund_um.xlsx`).
- Template 2 fundações (`problema_fund_dois.xlsx`).
- Template 3 fundações (`problema_fund_três.xlsx`).
- Parâmetros globais `f_ck_kpa` / `cobrimento_m` flow correto.

### Compatibilidade com Streamlit (1 teste)
- Aceita `BytesIO` (simulando `UploadedFile`).

### Missing file (1 teste)
- `FileNotFoundError` explícito quando o path não existe.

### Schema validation (5 testes)
- Coluna obrigatória ausente.
- Sem nenhuma combinação (Fz/Mx/My).
- Combinações com gap (c1 + c3, sem c2).
- Combinação incompleta (Fz-c2 + Mx-c2 sem My-c2).
- Spreadsheet vazia (header sem rows).

### Domínio (4 testes)
- Tipo de solo desconhecido.
- Elemento duplicado.
- SPT negativo.
- `f_ck_kpa = 0` ou `cobrimento_m < 0`.

### Sanitização (2 testes)
- Vírgula decimal (`"855,5"` → 855.5).
- Tipo de solo case-insensitive.

### DXF writer (4 testes)
- Header válido (`SECTION`, `HEADER` nos primeiros bytes).
- Cada label de pilar aparece no payload.
- Estabilidade semântica entre chamadas (mesmo tamanho, mesmo número de LINEs, mesmas labels — **byte-a-byte difere** porque ezdxf gera handle único por entidade).
- Não gera tempfile órfão.

## Validação

```text
=== AST ===
  ✓ 38 arquivos Python OK

=== Imports core completos ===
  ✓ core.engineering, core.domain, core.io all import cleanly

=== End-to-end smoke ===
  read: n_fund=3, n_comb=3, fck=25000.0 kPa
  dxf: 17216 bytes; LINE count = 18  (= 6 unique LINE token + 12 inside payload)

=== pytest ===
  91 passed in 3.x s
    test_avaliar_projeto.py    6
    test_benchmark.py         15
    test_domain.py            15
    test_ego_historico.py      8
    test_engenharia.py        26
    test_io.py                21
```

A trava de regressão `of = 19,70604234767181` permanece intocada.
A camada de IO ainda **não** é consumida por `pages/sapatas.py` — a integração acontece na Sprint 3.5 (API layer).

## Próxima sub-sprint (Sprint 3.5)

API layer: `core/api/optimize.py` com a função pura
`optimize(projeto, config) -> OptimisationResult` que orquestra IO,
domínio, engenharia e otimização. `pages/sapatas.py` passará a ser
um shell fino de Streamlit que chama `read_projeto_from_excel`,
`optimize` e `sapatas_to_dxf_bytes`.

## Vínculos

- [[10_Melhorias/Roadmap Sugerido]]
- [[10_Melhorias/MOC - Melhorias]]
- [[12_Auditoria/Sprint 3 - Refactor estrutural - kickoff - 2026-04-27]]
- [[12_Auditoria/Sprint 3.2 - Engineering migration - 2026-04-28]]
- [[12_Auditoria/Sprint 3.3 - Domain entities - 2026-04-28]]
- [[07_Issues/Issue - DXF tempfile não removido]] — pode ser marcada como resolvida na Sprint 3.5 quando `pages/sapatas.py` migrar para a nova API
- [[01_Projeto/Convenções do Projeto]]
