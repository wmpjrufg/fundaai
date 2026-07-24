---
tags: [issue, medio, resolvido]
severity: medio
status: resolvido
resolvido_em: 2026-04-27
resolvido_em_branch: fix/code-sanitization-and-tests
---

# Issue — Notebooks com paths quebrados (`assets/el08.xlsx`)

> [!success] Resolvido em 2026-04-27 (Sprint 2, branch `fix/code-sanitization-and-tests`)
> Os notebooks `testes_fo_filipe.ipynb` e `testes_otm.ipynb` agora apontam
> para `assets/problema_fund_três.xlsx` (3 fundações, mesmo schema, alinhado
> com a UI). Substituição feita por script Python preservando o JSON dos
> notebooks; ambos continuam válidos (`json.load` sem erro).

## Sintoma original

Os notebooks abaixo carregavam um arquivo que **não existia** no estado atual do repositório:

- [[06_Notebooks/testes_fo_filipe]] — `pd.read_excel(r"assets\el08.xlsx")`
- [[06_Notebooks/testes_otm]] — `pd.read_excel(r"assets\el08.xlsx")`

`assets/el08.xlsx` foi removido (ou nunca commitado) durante a reorganização da pasta de assets — agora os datasets vivem em `assets/` (problemas de fundação) e em `assets/data/` (toy problem).

## Por que era problema

- Quem clonasse o repositório não conseguia executar esses notebooks.
- Se fossem usados para gerar tabelas/figuras do relatório, qualquer rebuild quebrava.

## Correção aplicada

Substituição do path em ambos os notebooks:

```diff
- pd.read_excel(r"assets\el08.xlsx")
+ pd.read_excel(r"assets/problema_fund_três.xlsx")
```

Escolhido `problema_fund_três.xlsx` por ser o template oficial usado pela
UI (`pages/sapatas.py` o oferece como download na página inicial), tendo
mesmo schema e 3 fundações como o `el08.xlsx` original aparentava ter.

## Vínculo

- [[06_Notebooks/testes_fo_filipe]]
- [[06_Notebooks/testes_otm]]
- [[05_Dados/Assets - Templates Excel]]
- [[07_Issues/Lista Mestre de Issues]]
