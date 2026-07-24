# Sprint 5.6 - Correcoes pos-avaliacao e piloto Fase B - 2026-07-12

## Contexto

Lucas trouxe uma avaliacao externa (Consensus/Claude) sobre o artigo. A avaliacao foi considerada majoritariamente coerente: o artigo 1 esta metodologicamente mais forte e honesto, mas a auditoria de decomposicao mostrou que os casos atuais sao quase separaveis. A decisao foi fazer tres correcoes imediatas e iniciar a Fase B sem misturar ainda seus resultados ao protocolo congelado do artigo 1.

## Correcoes feitas no artigo/vault

### 1. Resumo e abstract

`docs/artigo_ic_lucas/main.tex` foi atualizado para mencionar explicitamente a auditoria por Differential Evolution decomposta por sapata:

- volumes factiveis que empatam ou melhoram o melhor protocolo global em `<0,01%`, `0,77%` e `2,08%`;
- interpretacao: os casos atuais sao quase separaveis.

Motivo: o corpo do artigo ja tratava a decomposicao, mas o resumo ainda nao avisava o leitor desse limite metodologico.

### 2. Nota CBO antiga

`obsidian_vault/11_Frentes_de_Pesquisa/Bayesian Optimization Constrained.md` foi corrigida. A versao antiga ainda registrava numeros pre-correcao e afirmava que o ganho "cresce com a dimensao". A nota agora registra:

- CBO melhora media de `Theta` frente ao EGO em `1,5% / 9,3% / 21,7%`;
- melhora melhor volume factivel em `0,8% / 3,5% / 15,3%`;
- Wilcoxon-Holm pareado EGO x CBO: `p=0,014`, `<0,001`, `<0,001`;
- factibilidade CBO: `63% / 37% / 83%`, contra `83% / 83% / 83%` do EGO;
- ressalva: nao atribuir causalmente o ganho a dimensionalidade porque os casos atuais sao quase separaveis.

### 3. NBR 6118

`obsidian_vault/02_Engenharia/NBR 6118.md` foi atualizado com a checagem externa:

- Catálogo ABNT lista `ABNT NBR 6118:2026` como norma ativa;
- DIN Media registra `ABNT NBR 6118:2026-03-11`;
- ABECE informa que a Emenda 1:2026, em conjunto com a ABNT NBR 6118:2023, equivale a ABNT NBR 6118:2026;
- pendencia antes da submissao: conferir a forma bibliografica exata no acesso institucional/ABNT Colecao.

## Fase B iniciada

Criado `scripts/run_packing_phase_b_pilot.py`.

Objetivo do piloto:

- criar caso acoplado minimo com packing ativo;
- testar variaveis `(h_x, h_y, h_z, dx, dy)` por sapata;
- manter o avaliador de engenharia atual (`avaliar_projeto_componentes`) como nucleo;
- adicionar restricoes de contencao do pilar na sapata deslocada e fronteira retangular do lote;
- ajustar momentos efetivos por deslocamento:
  - `Mx_eff = Mx_input - Fz * dx`;
  - `My_eff = My_input - Fz * dy`.

Artefatos:

- `experiments/phase_b_packing_pilot/summary.csv`;
- `experiments/phase_b_packing_pilot/designs.csv`;
- `experiments/phase_b_packing_pilot/config.json`.

## Resultado do piloto

Caso minimo derivado de `assets/data/problema_fund_dois.xlsx`, com dois pilares reposicionados sinteticamente a `2,00 m`.

| Modo | Volume | `g_sob` | Factivel | Leitura |
|---|---:|---:|---|---|
| `individual_centered` | 4,750747 m3 | 0,2307 | Nao | Otimos individuais violam sobreposicao quando montados juntos. |
| `fixed_centers` | 4,929703 m3 | 0,0000 | Sim | Resolver packing so por dimensoes exige maior volume. |
| `packing_offsets` | 4,525122 m3 | 0,0000 | Sim | Permitir posicionamento reduz volume factivel no caso acoplado minimo. |

Interpretacao:

- A Fase B ja tem um caso minimo em que decomposicao por sapata falha.
- O posicionamento como variavel tem efeito real, reduzindo volume factivel em relacao ao redimensionamento com centros fixos.
- Ainda nao e evidencia estatistica; o proximo passo e transformar o piloto em benchmark pareado com casos acoplados congelados.

## Proximos passos tecnicos

1. Promover o piloto para uma funcao de objetivo 5N formal em `core/api` ou modulo experimental dedicado.
2. Definir sinais e convencoes finais dos momentos induzidos por deslocamento com revisao estrutural.
3. Criar 3--5 casos acoplados congelados:
   - dois-pilares minimo;
   - tres-pilares denso;
   - seis a dez pilares sintetico ou real;
   - caso com fronteira de lote ativa.
4. Comparar CBO, EGO penalizado, GWO/DE e aleatoria sob orcamento pareado.
5. Decidir depois dos resultados se o artigo 1 ganha uma secao curta de "estudo acoplado preliminar" ou se a Fase B vira artigo 2.

## Validacoes executadas

- `latexmk -pdf -g -interaction=nonstopmode main.tex` em `docs/artigo_ic_lucas`.
  - PDF gerado com 22 paginas.
  - Sem erro de citacao/referencia.
  - Avisos restantes: layout (`overfull/underfull`, resumo mais longo e `balance`).
- `.venv/bin/python -m compileall scripts/run_decomposition_baseline.py scripts/run_packing_phase_b_pilot.py scripts/make_paper_artifacts.py`.
- `.venv/bin/python -m pytest -q`.
  - Resultado: suite completa verde.
