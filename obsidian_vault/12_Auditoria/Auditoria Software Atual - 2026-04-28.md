---
tags: [auditoria, software, refactor, validação, fundaia, vault, roadmap]
data: 2026-04-28
status: revisado
escopo: Auditoria estática + testes do software atual após refatoração
---

# Auditoria Software Atual - 2026-04-28

## Resumo executivo

O estado atual do FundaIA está **bem melhor estruturado** do que a versão anterior: a separação `core/`, `frontend/`, `scripts/`, `notebooks/`, `archive/`, `docs/` e `obsidian_vault/` está coerente com o roadmap; a suíte de testes está forte; o caminho principal do aplicativo está funcional; e a trilha recomendada no vault continua correta: **validar e fechar a etapa atual FundaIA + EGO-GPR antes de iniciar bin packing/layout completo**.

Resultado da suíte:

```text
.venv/bin/python -m pytest
211 passed in 21.85s
```

Não encontrei uma quebra evidente no fluxo principal do app. O caminho de produção atual é:

```text
Excel -> core.io.read_projeto_from_excel
      -> FundacaoProjeto
      -> core.api.optimize
      -> EGO + GPR + mealpy.GA.BaseGA
      -> OptimisationResult
      -> evaluate / exportacao / 2D / 3D / historico EGO
```

Mas existem pontos que precisam ser tratados antes de usar o software como base final de artigo:

1. O **EGO-GPR de produção está coerente**, mas a justificativa científica deve continuar honesta: hoje a função objetivo direta não parece cara; o custo caro está mais no treinamento do GPR e na preparação para extensões futuras.
2. O **GA próprio em `core/optimization/genetic_algorithm.py` não está pronto para ser apresentado como algoritmo validado**. Há bugs nos operadores e no loop principal. A UI não usa esse GA próprio; ela usa `mealpy.GA.BaseGA`.
3. O **GWO próprio em `core/optimization/grey_wolf.py` ainda deve ser considerado experimental/legado**, não validado.
4. A verificação de **punção ainda é parcial**: seção C implementada, seção C' pendente.
5. A arquitetura prometida no `ARCHITECTURE.md` ainda tem pequenas divergências com o código, principalmente `core.domain.Solo` importando `core.engineering.solo`.
6. O vault está majoritariamente alinhado, mas algumas notas estão desatualizadas em relação ao estado pós-Sprint 4.7.

## Arquivos e áreas verificados

### Código principal

- `app.py`
- `fundacao.py`
- `core/domain/*.py`
- `core/engineering/*.py`
- `core/optimization/*.py`
- `core/api/*.py`
- `core/io/*.py`
- `core/observability/*.py`
- `frontend/pages/*.py`
- `frontend/components/*.py`
- `frontend/theme/*.py`
- `scripts/*.py`
- `tests/*.py`

### Documentação e vault

- `README.md`
- `ARCHITECTURE.md`
- `docs/README.md`
- `docs/articles/README.md`
- `docs/contexto_academico/README.md`
- `obsidian_vault/10_Melhorias/Roadmap Sugerido.md`
- `obsidian_vault/10_Melhorias/Guia - Validação antes do Bin Packing.md`
- `obsidian_vault/07_Issues/Lista Mestre de Issues.md`
- `obsidian_vault/12_Auditoria/Sprint 3.6 ... Sprint 4.7`
- `obsidian_vault/01_Projeto/Contexto Acadêmico - IC Lucas e TCC Filipe Amaral.md`
- `obsidian_vault/08_Artigos/Index de Artigos.md`

## Veredito por camada

| Camada | Estado | Veredito |
| --- | --- | --- |
| `core.domain` | Bom, com uma quebra arquitetural pequena | Entidades claras, mas `Solo` depende de `core.engineering` |
| `core.engineering` | Bom para etapa atual, com limites técnicos explícitos | Tensão, geometria, punção C e overlap funcionam; faltam defesas de edge case e C' |
| `core.optimization` | Misto | EGO coerente; cache bom; benchmarks saneados; GA/GWO próprios não validados |
| `core.api` | Bom | `optimize` e `evaluate` isolam bem o uso do legado |
| `core.io` | Bom | Excel, DXF e experimentos estão bem encaixados |
| `frontend` | Bom e funcional | UI atual está mais rica; há um input morto (`n_comb`) e i18n ainda não centralizado |
| `scripts` | Funcional, mas com docs divergentes | `env_setup.py` cria `venv`, README diz `.venv` |
| `tests` | Muito bom | 211 testes passando; ainda faltam testes para GA/GWO próprios e edge cases de engenharia |
| vault | Coerente no roadmap | Precisa atualizar notas antigas sobre `metapy_toolbox`, componentes planejados e estado real pós-Sprint 4.7 |

## Achados críticos e importantes

### 1. `core.domain` viola a regra arquitetural de não depender de `core.engineering`

Em `core/domain/solo.py`, a entidade `Solo` importa `tensao_adm_solo`:

```python
from core.engineering.solo import tensao_adm_solo
```

Isso contradiz o `ARCHITECTURE.md`, que diz que `core.domain` depende de nada dentro do projeto. Não quebra o app, e os testes passam, mas enfraquece a pureza arquitetural.

Recomendação futura:

- Ou mover `sigma_adm_kpa` para um serviço/função de engenharia;
- Ou aceitar formalmente no `ARCHITECTURE.md` que `Solo.sigma_adm_kpa` é uma conveniência que cruza a camada.

Minha preferência: manter `core.domain` puro e calcular `sigma_adm_kpa` na engenharia/API.

### 2. O caminho de produção usa `mealpy.GA.BaseGA`, não o GA próprio

Em `core/api/optimize.py`, o otimizador interno da aquisição EI é:

```python
GA.BaseGA(epoch=config.ga_epoch, pop_size=config.ga_pop_size)
```

Isso é bom: a UI não depende do GA artesanal em `core/optimization/genetic_algorithm.py`. Para o artigo, deve ficar claro:

- "Algoritmo genético interno" = implementação da biblioteca `mealpy`;
- O GA próprio do antigo `metapy_toolbox` existe no repositório, mas não é o motor de produção.

### 3. O GA próprio contém bugs e não deve ser usado como baseline ainda

Arquivo: `core/optimization/genetic_algorithm.py`.

Problemas encontrados:

- `simulated_binary_crossover` gera dois filhos idênticos, porque `neighbor_b` repete a mesma expressão de `neighbor_a`.
- `multi_point_crossover` itera sobre os valores da máscara (`0` ou `1`) e usa isso como índice; com vetores maiores, os filhos ficam compostos só por genes das posições 0 e 1.
- Se `selection_type` não for roleta, `i_selected = None`; se houver crossover, a consulta por pai pode falhar.
- Se `mutation_type` não for random walk, `report_mutation` pode ser usado sem ter sido definido.
- Não há testes cobrindo esses operadores.

Validações rápidas feitas durante a auditoria:

```text
simulated_binary_crossover([1,2], [3,4]) -> filhos idênticos
multi_point_crossover([10,20,30,40], [1,2,3,4]) -> filho [2,10,2,2]
```

Conclusão: **não usar esse GA próprio em comparação científica antes de corrigir e testar**. O app continua seguro porque não depende dele.

### 4. GWO ainda é experimental e metodologicamente suspeito

Arquivo: `core/optimization/grey_wolf.py`.

Problemas:

- A distância `D` é calculada por norma escalar acumulada, enquanto o GWO clássico usa distância vetorial por dimensão.
- Falta `abs` na distância `D = |C * X_p - X|`.
- Ainda há placeholder literal:

```python
df['DIVERSITY'] = 'aqui implementa função lucas'
```

O vault já registrava o placeholder em `Issue - Placeholder Diversidade GWO`, mas a auditoria encontrou também a divergência da fórmula de movimento.

Conclusão: **GWO não deve ser citado como algoritmo implementado/validado**. Pode ser citado como código legado/experimental, se necessário.

### 5. `best_avg_worst` depende de índice resetado

Arquivo: `core/optimization/funcs.py`.

O código usa:

```python
best_idx = int(df['OF'].idxmin())
df['X_0'].values[best_idx]
```

Isso só é seguro quando o índice do DataFrame é `0..n-1`. O GA/GWO resetam o índice antes de chamar, então o uso atual tende a passar. Mas a função isolada quebra com DataFrame filtrado sem `reset_index`.

Validação rápida:

```text
DataFrame com índice [5, 6] -> IndexError: index 5 is out of bounds
```

Recomendação futura: usar `.loc[best_idx, ...]` ou converter o `idxmin` para posição com `df.index.get_loc(best_idx)`.

### 6. Engenharia está coerente, mas faltam guardrails para casos-limite

Pontos fortes:

- `tensao_adm_solo` está simples e testada.
- `calcular_sigma_max_min` preserva o comportamento histórico.
- `checagem_geometria` está clara.
- `sobreposicao_matrix` reproduz a versão escalar e está bem testada.
- `_avaliar_projeto` manteve a regressão `of = 19.70604234767181`.

Pontos que podem quebrar ou gerar resultado inválido:

- `tensao_adm_solo`: solo desconhecido cai no ramo `spt/50`, por compatibilidade histórica. Entrada via Excel já valida tipo de solo, mas chamadas diretas ainda aceitam erro silencioso.
- `spt = 0` gera `sigma_adm = 0`; depois `checagem_tensao_max_min` divide por zero.
- `calcular_sigma_max_min` divide por `f_zk`, `h_x` e `h_y`; se `f_zk = 0` ou dimensões inválidas forem passadas por fora do fluxo normal, quebra.
- `verificacao_puncao_sapata` usa `d = h_z - cob`; se `h_z <= cob`, a tensão fica inválida.
- `verificacao_puncao_sapata` implementa só a seção C, não C'.

Recomendação para pesquisa séria:

- Manter os testes atuais;
- Adicionar testes de borda;
- Deixar no artigo que a punção é parcial enquanto C' não for implementada;
- Não escrever que a formulação está "plenamente conforme NBR 6118" enquanto C' e outros detalhes normativos não forem cobertos.

### 7. A correlação SPT -> tensão admissível deve ser descrita como empírica

O código e o vault tratam `SPT/30`, `SPT/40`, `SPT/50` como prática adotada. Isso é aceitável para uma ferramenta preliminar, mas para artigo precisa ser blindado:

> correlação empírica adotada para estimativa preliminar de tensão admissível, compatível com o escopo de pré-dimensionamento.

Evitar:

> fórmula diretamente prescrita pela NBR 6122.

Isso é coerente com as observações anteriores feitas no artigo da IC.

### 8. Input `n_comb` aparece na UI, mas não é usado

Em `frontend/pages/sapatas.py`, existe:

```python
n_comb_ui = st.number_input(...)
```

Mas `read_projeto_from_excel` infere `n_comb` pelas colunas do Excel, e `n_comb_ui` não entra no `OptimisationConfig`.

Isso não quebra a conta, mas confunde o usuário: alterar o campo "Número de combinações" não muda nada. O caminho mais coerente é:

- remover esse input; ou
- transformá-lo em informação detectada automaticamente depois do upload.

### 9. `metapy_toolbox` está em estado removido, mas algumas docs dizem que há shim

Estado atual local:

```text
import metapy_toolbox -> namespace vazio, sem símbolos públicos
attrs -> []
```

O `notebooks/README.md` e a Sprint 4.3 dizem que o shim foi removido, e isso bate com o filesystem. Porém:

- `core/optimization/__init__.py` ainda diz que o pacote antigo continua como shim;
- algumas notas antigas do vault em `04_Codigo/metapy_toolbox ...` ainda apontam para o caminho antigo;
- `tests/conftest.py` e docstrings de alguns testes ainda falam em `metapy_toolbox`.

Isso é documentação desatualizada, não bug de runtime. Mas, para contexto de IA e para evitar confusão futura, vale atualizar.

### 10. README/ARCHITECTURE têm pequenas partes pós-4.7 desatualizadas

Exemplos:

- `README.md` e `ARCHITECTURE.md` ainda descrevem `frontend/components/` como "planned/scaffold", mas hoje já existem `footings_3d.py`, `ego_chart.py`, `result_export.py`.
- `ARCHITECTURE.md` ainda tem uma seção dizendo que `frontend/components/` está vazio e planeja mover o 3D para lá; isso já foi feito.
- `core/__init__.py` ainda parece texto da Sprint 3.1: diz que nada de `fundacao.py`/`metapy_toolbox` foi movido.

Conclusão: a implementação avançou mais do que alguns textos estruturais.

### 11. Script de setup tem divergência entre README e código

`README.md` e `scripts/README.md` dizem que o setup automático cria `.venv/`.

Mas `scripts/env_setup.py` cria `venv/`:

```python
python -m venv venv
./venv/bin/pip
source venv/bin/activate
```

Além disso, `requirements.txt` ainda comenta que Playwright fica em `ops/requirements.txt`, mas a pasta atual é `scripts/requirements.txt`.

Não afeta o app nem os testes, mas é atrito para instalação nova.

## Coerência com o vault

### O que está alinhado

O vault está correto nos pontos principais:

- A trilha ativa é validar o FundaIA/EGO-GPR antes do bin packing.
- O bin packing/layout completo ainda não está implementado.
- A etapa atual deve ser descrita como dimensionamento de sapatas isoladas com posições fornecidas.
- A sobreposição atual é uma restrição geométrica AABB, não layout optimization completo.
- A punção C' continua pendente.
- `docs/articles` está bem organizada e com mapa PDF -> ficha.
- O contexto acadêmico da IC, relatório parcial e TCC do Filipe está bem encaixado em `obsidian_vault/01_Projeto/Contexto Acadêmico - IC Lucas e TCC Filipe Amaral.md`.
- O roadmap "validar antes do bin packing" continua sendo a decisão certa.

### O que precisa ser atualizado no vault

Sugestões de atualização:

- Criar/atualizar issue: "GA próprio contém operadores incorretos".
- Atualizar `Issue - Placeholder Diversidade GWO` para incluir também a fórmula de movimento.
- Atualizar notas `04_Codigo/metapy_toolbox ...` para apontar para `core/optimization/...` ou marcar como histórico.
- Atualizar `MOC - Melhorias` e `ARCHITECTURE.md` com Sprint 4.5-4.7 já concretizadas em componentes.
- Atualizar `core/__init__.py` no futuro, porque a docstring está muito desatualizada.
- Registrar input morto `n_comb_ui` como issue de UI.
- Registrar divergência `.venv` vs `venv` no setup.

## Situação dos algoritmos

### EGO + GPR + EI

Estado: **coerente para produção atual**.

Pontos positivos:

- Seed propagada para LHS, GPR e `mealpy`.
- Histórico do EGO corrigido (`ITER` e `ID` coerentes).
- Cache do surrogate implementado de forma conservadora.
- Progress callback bem isolado.
- Testes cobrem histórico, seed, cache e caminho API.

Pontos de cautela:

- A função objetivo direta atual é barata; a defesa do EGO deve ser científica/metodológica e voltada a extensões futuras.
- `kernel_index=-1` pega o último kernel de `constroi_kernel()`. O vault/artigo precisa decidir se o kernel de produção é "k20" ou outra convenção.
- O EGO usa penalização exterior linear. Isso é aceitável para a etapa atual, mas ainda precisa comparação com baseline de orçamento equivalente.

### GA da `mealpy`

Estado: **ok como otimizador interno da aquisição EI**.

É o que a UI usa. Como ele otimiza a acquisition function, não é o mesmo que dizer "o problema estrutural foi resolvido por GA puro".

### GA próprio

Estado: **não validado**.

Não usar como baseline de artigo ainda. Corrigir operadores e adicionar testes antes.

### GWO próprio

Estado: **não validado**.

Não usar como evidência científica ainda. Corrigir fórmula e placeholder antes.

### Benchmarks

Estado: **bom**.

`griewank` e `powell` foram corrigidos e testados. A camada de benchmark parece saneada para testes isolados de otimizadores.

## Estado do software para artigo

### O que já é defendável

- Ferramenta computacional em Streamlit para dimensionamento otimizado de sapatas isoladas.
- Entrada por planilha validada.
- Formulação por volume penalizado.
- Restrições de tensão no solo, geometria mínima, sobreposição preliminar e punção C. Atualização 2026-07-10: punção C′ também foi implementada na Sprint 5.2.
- Pipeline EGO-GPR com Expected Improvement e GA interno via `mealpy`.
- Reprodutibilidade por seed.
- Persistência de experimentos com manifest/config/env/project/history/summary/metrics.
- Visualizações 2D/3D e exportações.
- Suite de testes com 211 testes passando.

### O que deve ser declarado como limite

- Histórico superado: punção C' não estava implementada em 2026-04-28, mas foi incorporada na Sprint 5.2.
- Posição/layout não otimizado; `xg` e `yg` vêm da planilha.
- Sobreposição é AABB preliminar, não bin packing formal.
- Não há bulbo de tensão nem interação geotécnica entre sapatas próximas.
- A correlação SPT-tensão admissível é empírica/preliminar.
- Resultados finais ainda precisam ser gerados com orçamento equivalente, múltiplas seeds, factibilidade e comparação adequada.
- GA/GWO próprios não são métodos validados no estado atual.

## Roadmap recomendado a partir daqui

### Antes de qualquer bin packing

1. Atualizar docs/vault para refletir o estado real pós-Sprint 4.7.
2. Fechar a linguagem técnica da função objetivo e da punção parcial.
3. Decidir oficialmente:
   - kernel de produção;
   - 20 vs 21 kernels;
   - sobreposição contada 1x ou 2x;
   - se C' entra agora ou fica como limite.
4. Criar casos congelados de validação:
   - 1 sapata;
   - 2 sapatas distantes;
   - 2 sapatas próximas;
   - 3 sapatas do problema atual.
5. Rodar experimentos finais com:
   - seeds registradas;
   - mesmo orçamento de avaliações reais;
   - EGO-GPR;
   - random search/Monte Carlo;
   - opcional: GA puro externo validado, não o GA próprio atual.
6. Reportar:
   - melhor volume factível;
   - média/desvio;
   - taxa de factibilidade;
   - violação máxima;
   - tempo;
   - número de avaliações reais;
   - histórico de convergência.

### Depois disso

Iniciar a frente bin packing/layout, já como outro problema:

- posições como variáveis;
- fronteira do lote;
- margens construtivas;
- decisão sapata isolada vs associada;
- bulbo de tensão;
- recalque/interação;
- modelos formais de packing.

## Checklist de validação desta auditoria

- [x] `git status --short` limpo antes e depois da auditoria.
- [x] Estrutura do repositório lida.
- [x] `README.md` e `ARCHITECTURE.md` conferidos.
- [x] `core/domain` conferido.
- [x] `core/engineering` conferido.
- [x] `core/optimization` conferido.
- [x] `core/api` conferido.
- [x] `core/io` conferido.
- [x] `frontend/pages` e `frontend/components` conferidos.
- [x] `scripts` conferido.
- [x] Vault conferido contra Roadmap, Guia antes do Bin Packing, Issues e Sprints 3.6-4.7.
- [x] `docs/articles` e contexto acadêmico conferidos em nível estrutural.
- [x] Suíte completa executada: `211 passed in 21.85s`.
- [x] Testes rápidos adicionais rodados para confirmar problemas em GA próprio e `best_avg_worst`.

## Conclusão

A refatoração ficou **boa e útil**: o software agora tem arquitetura, testes, persistência experimental, cache, UI melhor e documentação de pesquisa. Isso é base real para fechar a etapa atual e escrever o artigo.

O ponto mais importante é não superestimar o que está pronto: o FundaIA atual valida uma etapa de **dimensionamento otimizado com posições fixas**, não um sistema completo de layout/bin packing. Para artigo, essa honestidade é força, não fraqueza.

Minha recomendação é:

> Fechar validação de engenharia + experimentos reprodutíveis + limpeza de documentação antes de implementar bin packing.

Com isso, o próximo passo fica academicamente defensável e tecnicamente mais seguro.
