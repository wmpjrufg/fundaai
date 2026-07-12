# Plano da Frente 2 - Empacotamento de sapatas e interação por bulbo de tensões

**Projeto:** FundaIA  
**Branch de trabalho:** `codex/frente-2-binpacking`  
**Data:** 2026-07-12  
**Ponto de partida preservado:** branch `codex/lucas-frente-1-artigo`, commit `fdba0ed90`, com o artigo 1 e os artefatos congelados.

## 1. Decisão estratégica

A Frente 2 deve ser separada da Frente 1. A Frente 1 fica como artigo de pré-dimensionamento geométrico experimental, com centros fixos ou casos quase separáveis. A Frente 2 passa a estudar **posicionamento conjunto de sapatas em planta**, com restrições de empacotamento e, em uma etapa posterior, interação geotécnica por bulbo de tensões.

Essa separação evita três problemas:

1. misturar resultados quase finais do artigo 1 com resultados exploratórios;
2. confundir verificações estruturais locais com interação geotécnica entre sapatas;
3. criar uma narrativa metodológica instável, em que o artigo 1 muda de escopo no meio da submissão.

## 2. Conclusão sobre o projeto externo `bin_packing_3d`

Foi analisado o projeto em:

`/Users/lucasteixeira/Documents/bin_packing_3d`

O projeto externo tem:

- domínio Pydantic para instâncias de 3D-BPP;
- validação geométrica por AABB;
- restrições de distância mínima entre itens;
- decoder por **Extreme Points**;
- Simulated Annealing;
- core Python e C++;
- API FastAPI;
- persistência SQLite;
- GPR/EGO para configuração;
- frontend React/Three.js;
- testes automatizados.

Validação local executada:

```bash
.venv/bin/python -m pytest
```

Resultado: `65 passed, 1 warning`.

### 2.1. O que é reaproveitável

O que deve ser reaproveitado como padrão técnico/conceitual:

- validação AABB e funções de distância entre caixas/retângulos;
- ideia de restrição de distância mínima como contrato explícito;
- representação indireta quando houver muitas variáveis de posição;
- registro de convergência e artefatos;
- separação entre domínio, validador, solver e experimento;
- protocolo experimental com seeds pareadas e orçamento controlado;
- referências de packing, especialmente Extreme Points e Simulated Annealing.

### 2.2. O que não deve ser consumido diretamente

Não recomendo usar a API do `bin_packing_3d` como motor principal da Frente 2.

Motivo: o problema externo é **3D Bin Packing clássico**: itens retangulares dentro de caixas, minimizando número de caixas. O problema do FundaIA é diferente: sapatas em planta, com cargas, momentos, tensão admissível, punção, geometria de pilar, limite de lote e, possivelmente, interação geotécnica.

Usar a API diretamente exigiria transformar sapatas em "itens" e o terreno em "caixa". Isso ajudaria apenas em não sobreposição geométrica, mas perderia a parte de engenharia. Portanto, a integração correta é:

```text
aproveitar conceitos e padrões de validação
não substituir a função objetivo do FundaIA por um solver 3D-BPP
```

## 3. Formulação inicial da Frente 2

### 3.1. Variáveis de projeto

Para cada sapata `i`, a Frente 2 deve começar com cinco variáveis:

```text
h_x_i, h_y_i, h_z_i, dx_i, dy_i
```

onde:

- `h_x_i`: dimensão da sapata na direção x;
- `h_y_i`: dimensão da sapata na direção y;
- `h_z_i`: altura;
- `dx_i`: deslocamento do centro da sapata em relação ao pilar no eixo x;
- `dy_i`: deslocamento do centro da sapata em relação ao pilar no eixo y.

Rotação em planta pode ficar fora da primeira versão. Sapatas rotacionadas exigiriam OBB/SAT em vez de AABB e aumentariam o risco sem necessidade imediata.

### 3.2. Objetivo

Manter o objetivo principal da Frente 1:

```text
minimizar volume total de concreto
```

com penalizações ou tratamento explícito de restrições.

### 3.3. Restrições mínimas da Frente 2

As restrições iniciais devem ser:

1. **não sobreposição em planta** entre sapatas;
2. **pilar contido na sapata**, com balanço mínimo;
3. **sapata dentro dos limites do lote**;
4. **tensão admissível no solo**;
5. **punção C e C'**;
6. **geometria mínima**;
7. **interação por bulbo de tensões**, inicialmente opcional e estudada por ablação.

## 4. Bulbo de tensões: como tratar sem fragilizar o artigo

### 4.1. Resposta curta

Sim, faz sentido considerar bulbo de tensões. Mas a restrição não deve nascer como uma distância fixa arbitrária entre sapatas.

O caminho mais defensável é modelar a interação como **acréscimo de tensão vertical no solo causado por sapatas vizinhas**, usando superposição:

```text
Δσ_z,total(P, z) = Σ Δσ_z,j(P, z)
```

Depois, define-se uma métrica de interação, por exemplo:

```text
I_i = tensão induzida por vizinhas / tensão induzida pela própria sapata i
```

ou:

```text
I_i = acréscimo de tensão das vizinhas / σ_adm,i
```

Essa métrica vira restrição ou indicador experimental.

### 4.2. Por que não começar com "distância mínima entre sapatas"

Uma distância fixa, como `S >= 2B` ou `S >= 3B`, é fácil de implementar, mas é fraca cientificamente porque:

- não depende da carga;
- não depende da dimensão real da sapata;
- não depende da profundidade analisada;
- não distingue solos;
- não mede o quanto uma sapata realmente interfere na outra.

Ela pode ser usada como baseline geométrico simples, mas não como principal contribuição geotécnica.

### 4.3. Modelo recomendado para a primeira implementação

Implementar dois níveis:

#### Nível A - aproximação 2V:1H

Usar a aproximação 2V:1H como modo rápido e didático.

Em profundidade `z`, a carga de uma sapata retangular é distribuída em uma área aumentada:

```text
A_z = (h_x + z) * (h_y + z)
```

se for adotada a propagação 2 vertical para 1 horizontal em cada lado.

Essa aproximação serve para triagem, testes e comparação.

#### Nível B - Boussinesq/Fadum para área retangular

Implementar cálculo de acréscimo de tensão vertical sob área retangular uniformemente carregada, usando superposição de cantos. Para sapatas retangulares alinhadas aos eixos:

```text
Δσ_z(P, z) = q * I_z
```

em que:

- `q` é a pressão de contato;
- `I_z` é o fator de influência;
- `P` é o ponto de controle;
- `z` é a profundidade avaliada.

Esse nível deve ser o padrão metodológico após validação.

### 4.4. Pontos de controle

Para cada sapata `i`, avaliar a influência das demais sapatas em:

- centro da sapata;
- quatro cantos;
- opcionalmente uma malha 3 x 3.

Profundidades iniciais:

```text
z = 0.5B_i, 1.0B_i, 2.0B_i, 4.0B_i
```

onde `B_i = min(h_x_i, h_y_i)`.

Essas profundidades devem ser parametrizadas, não fixadas no código.

### 4.5. Métricas candidatas

Métrica 1 - razão vizinha/própria:

```text
R_i = max_{P,z} (Σ_{j != i} Δσ_z,j(P,z)) / max(Δσ_z,i(P,z), ε)
```

Restrição:

```text
g_bulbo = R_i / R_lim - 1 <= 0
```

Métrica 2 - razão vizinha/tensão admissível:

```text
R_i = max_{P,z} (Σ_{j != i} Δσ_z,j(P,z)) / σ_adm,i
```

Restrição:

```text
g_bulbo = R_i / R_lim - 1 <= 0
```

Métrica 3 - indicador apenas descritivo:

Usar `R_i` para comparar layouts, mas não penalizar na primeira campanha. Isso permite entender se a restrição muda os resultados antes de incorporá-la como critério duro.

### 4.6. Valor de corte

Não recomendo fixar `R_lim` como verdade de projeto agora.

Para pesquisa, usar estudo paramétrico:

```text
R_lim ∈ {0.10, 0.20, 0.30}
```

e declarar que é uma regra experimental de interação, não uma prescrição normativa.

## 5. Arquitetura proposta no FundaIA

### 5.1. Novos módulos

Criar:

```text
core/engineering/layout.py
core/engineering/stress_influence.py
core/api/layout_objective.py
```

Responsabilidades:

- `layout.py`: bounds AABB, contenção do pilar, limite de lote, distância entre retângulos;
- `stress_influence.py`: 2V:1H, Boussinesq/Fadum, superposição e índice de interação;
- `layout_objective.py`: avaliador da Frente 2 com variáveis `h_x, h_y, h_z, dx, dy`.

### 5.2. Não quebrar a Frente 1

Não alterar a assinatura de `avaliar_projeto_fast` para incluir offsets.

Criar um avaliador novo para a Frente 2. Isso reduz risco de quebrar o artigo 1.

### 5.3. Contrato de restrições

Manter o contrato atual:

```text
g <= 0  -> viável
g > 0   -> violado
```

Grupos da Frente 2:

```text
sob, pun, ten, geo, contain, boundary, bulbo
```

## 6. Plano de implementação

### Sprint B1 - Base geométrica 2D

Objetivo: transformar o piloto em código estável.

Tarefas:

1. criar `core/engineering/layout.py`;
2. mover para ele a lógica de:
   - AABB das sapatas;
   - não sobreposição;
   - contenção do pilar;
   - limite de lote;
   - distância mínima entre retângulos;
3. escrever testes unitários para:
   - toque por borda sem sobreposição;
   - sobreposição positiva;
   - pilar dentro/fora da sapata;
   - sapata dentro/fora do lote;
   - simetria da distância entre retângulos.

Critério de pronto:

```text
pytest passa e o piloto da Fase B usa o módulo novo
```

### Sprint B2 - Avaliador oficial da Frente 2

Objetivo: separar o avaliador de layout do avaliador do artigo 1.

Tarefas:

1. criar `core/api/layout_objective.py`;
2. formalizar o vetor `[h_x, h_y, h_z, dx, dy]`;
3. ajustar momentos efetivos:

```text
Mx_eff = Mx_input - Fz * dx
My_eff = My_input - Fz * dy
```

4. retornar componentes:

```text
theta, volume, g_sob, g_pun, g_ten, g_geo, g_contain, g_boundary
```

5. criar testes de regressão contra `run_packing_phase_b_pilot.py`.

Critério de pronto:

```text
o piloto vira uma chamada limpa ao avaliador novo
```

### Sprint B3 - Bulbo de tensões nível A

Objetivo: implementar a aproximação 2V:1H como baseline de interação.

Tarefas:

1. criar `core/engineering/stress_influence.py`;
2. implementar `stress_2v1h_rect(q, hx, hy, dx, dy, z)`;
3. implementar superposição entre sapatas;
4. calcular `interaction_ratio`;
5. criar `g_bulbo` opcional;
6. testar:
   - tensão reduz com profundidade;
   - tensão reduz com afastamento horizontal;
   - superposição é soma das contribuições;
   - sapata sem vizinha tem `g_bulbo <= 0`.

Critério de pronto:

```text
existe restrição geotécnica opcional, simples e testada
```

### Sprint B4 - Bulbo de tensões nível B

Objetivo: implementar Boussinesq/Fadum para área retangular.

Tarefas:

1. implementar fator de influência para retângulo carregado;
2. usar superposição de cantos para ponto arbitrário;
3. comparar com 2V:1H em casos simples;
4. adicionar testes de simetria, monotonicidade e limites;
5. documentar hipóteses:
   - meio elástico semi-infinito;
   - carga uniformemente distribuída;
   - sapata flexível/rígida simplificada;
   - solo homogêneo no primeiro modelo.

Critério de pronto:

```text
Boussinesq/Fadum vira o método padrão, 2V:1H fica como baseline simples
```

### Sprint B5 - Casos acoplados congelados

Objetivo: criar benchmark real da Frente 2.

Casos mínimos:

1. duas sapatas próximas, como no piloto atual;
2. três sapatas em corredor estreito;
3. quatro sapatas em lote com divisa;
4. caso com cargas assimétricas e momentos relevantes;
5. caso com restrição de bulbo ativa.

Cada caso deve ter:

- entrada congelada;
- limites de lote;
- cargas;
- `σ_adm`;
- `R_lim` do bulbo;
- melhor solução conhecida até o momento;
- figura em planta.

### Sprint B6 - Algoritmos e baselines

Comparar:

1. solução centralizada;
2. decomposição por sapata;
3. DE com dimensões apenas;
4. DE com `dx, dy`;
5. EGO penalizado;
6. CBO;
7. busca aleatória;
8. opcionalmente, um construtor geométrico inspirado em Extreme Points 2D.

Não usar o solver 3D-BPP como baseline principal, porque ele não resolve as verificações estruturais.

### Sprint B7 - Protocolo experimental

Usar:

- 30 repetições pareadas;
- mesmo orçamento de avaliações reais;
- comparação por tempo de parede como análise secundária;
- Wilcoxon pareado;
- correção de Holm;
- tamanhos de efeito;
- intervalos de confiança bootstrap;
- factibilidade estrita.

Métricas:

- volume factível;
- melhor `theta`;
- taxa de factibilidade;
- `g_sob`;
- `g_bulbo`;
- índice de interação por bulbo;
- tempo;
- avaliações;
- distância mínima entre sapatas;
- deslocamentos `dx, dy`.

### Sprint B8 - Integração visual

Adicionar ao frontend:

- planta com sapatas, pilares e lote;
- linhas de afastamento;
- mapa simples de interação por bulbo;
- tabela de restrições;
- exportação do layout.

Isso deve vir depois do avaliador e dos testes, não antes.

## 7. Como citar essa frente no artigo 1

No artigo 1, manter como trabalho futuro:

```text
O próximo estágio é o posicionamento conjunto de sapatas, no qual as coordenadas
dos centros das fundações passam a ser variáveis de projeto e restrições de
empacotamento, divisa e interação geotécnica por bulbo de tensões podem tornar
o problema efetivamente acoplado.
```

Não inserir resultados da Frente 2 no artigo 1 até que o protocolo esteja completo.

## 8. Fontes técnicas

### Fundações e tensão no solo

- ABNT NBR 6122:2019 - Projeto e execução de fundações. Referência normativa brasileira para fundações. Catálogo/consulta: <https://www.abntcatalogo.com.br/> e página comercial resumida: <https://www.normas.com.br/visualizar/abnt-nbr-nm/5248/abnt-nbr6122-projeto-e-execucao-de-fundacoes>.
- FHWA. *Geotechnical Engineering Circular No. 6: Shallow Foundations*. Documento técnico de referência para fundações superficiais e análises de recalque/tensões: <https://www.fhwa.dot.gov/engineering/geotech/pubs/010943.pdf>.
- Fadum, R. E. *Influence Values for Estimating Stresses in Elastic Foundations*. Fonte clássica para valores de influência de tensões em fundações elásticas: <https://www.issmge.org/uploads/publications/1/43/1948_03_0020.pdf>.
- Material didático sobre Boussinesq/Fadum para carregamento retangular, útil para conferência independente de fórmula: <https://vulcanhammer.net/2020/01/29/analytical-boussinesq-solutions-for-strip-square-and-rectangular-loads/>.
- Aula/nota técnica sobre métodos Boussinesq, Westergaard, Fadum, Newmark e 2:1: <https://ce.arizval.com/civil/geotechnical-engineering/07-vertical-stresses/content/>.

### Empacotamento e posicionamento

- Crainic, T. G.; Perboli, G.; Tadei, R. *Extreme Point-Based Heuristics for Three-Dimensional Bin Packing*. INFORMS Journal on Computing, 2008. DOI: `10.1287/ijoc.1070.0250`. Página: <https://pubsonline.informs.org/doi/10.1287/ijoc.1070.0250>.
- Martello, S.; Pisinger, D.; Vigo, D. *The Three-Dimensional Bin Packing Problem*. Operations Research, 2000. Base clássica para 3D-BPP e lower bounds.

### Otimização com restrições

- Gardner, J. R. et al. *Bayesian Optimization with Inequality Constraints*. ICML, 2014. PDF: <https://proceedings.mlr.press/v32/gardner14.pdf>.
- BoTorch documentation, constraints em Bayesian Optimization, atualizada em 2026 e citando Gardner et al. e Letham et al.: <https://botorch.org/docs/constraints>.

## 9. Risco metodológico e mitigação

| Risco | Mitigação |
| --- | --- |
| Bulbo virar regra arbitrária | Implementar como índice paramétrico e estudar `R_lim`, não como valor fixo definitivo. |
| Frente 2 contaminar artigo 1 | Manter branches separadas e citar apenas como trabalho futuro. |
| API do 3D-BPP distorcer o problema | Reaproveitar padrões, não o solver como função objetivo principal. |
| Aumento de dimensão prejudicar EGO/CBO | Começar com poucos casos e comparar também DE/CMA-ES/Random Search. |
| Boussinesq exigir hipóteses fortes | Documentar solo homogêneo, meio elástico e carga uniforme; deixar recalque por camadas como trabalho futuro. |

## 10. Próximo passo imediato

O próximo passo técnico é a Sprint B1:

```text
criar core/engineering/layout.py
extrair validações geométricas do piloto
escrever testes unitários
manter o artigo 1 intocado
```

