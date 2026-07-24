# Plano da Frente 2 - posicionamento de sapatas, empacotamento e bulbo de tensões

- **Projeto:** FundaIA
- **Data:** 12/07/2026
- **Ponto de partida:** artigo 1 preservado como estudo de pré-dimensionamento geométrico com posições fixas.

---

## 1. Ideia central

A Frente 2 deve ser separada do artigo atual.

O artigo 1 ficou centrado no pré-dimensionamento geométrico de sapatas isoladas, com as posições dos pilares já conhecidas. Essa escolha está coerente com o código e com os resultados, principalmente porque os três casos usados no artigo são quase separáveis: as sapatas não competem de fato por espaço.

A Frente 2 começa quando as posições das sapatas também passam a ser variáveis de projeto. Nesse caso, o problema muda de natureza:

- as sapatas podem se aproximar;
- a sobreposição deixa de ser uma restrição apenas de segurança e passa a governar soluções;
- limites de lote e divisas passam a importar;
- deslocar uma sapata altera os momentos efetivos;
- pode haver interação geotécnica entre fundações próximas.

Por isso, a Frente 2 deve ser tratada como uma continuação metodológica, não como um enxerto no artigo 1.

---

## 2. Por que não misturar com o artigo 1

Misturar os resultados exploratórios de packing no artigo atual criaria três problemas:

1. o artigo deixaria de ter um escopo claro;
2. os resultados do piloto ainda não teriam o mesmo nível de validação estatística;
3. seria necessário reabrir toda a metodologia, porque o vetor de projeto mudaria de `[h_x, h_y, h_z]` para `[h_x, h_y, h_z, dx, dy]`.

O mais seguro é manter o artigo 1 como uma entrega fechada e usar a Frente 2 como próxima etapa. No artigo atual, basta citar o posicionamento conjunto como trabalho futuro.

---

## 3. Relação com o projeto externo de bin packing

Foi analisado o projeto externo:

```text
/Users/lucasteixeira/Documents/bin_packing_3d
```

Ele tem várias ideias úteis:

- validação geométrica por AABB;
- cálculo de distância mínima entre caixas;
- representação indireta de posições;
- heurísticas de posicionamento;
- registro de resultados;
- separação entre domínio, validador, solver e experimento;
- testes automatizados.

Os testes locais desse projeto passaram:

```text
65 passed, 1 warning
```

Mesmo assim, ele não deve ser usado diretamente como motor da Frente 2. O motivo é simples: o problema dele é 3D Bin Packing clássico, com itens dentro de caixas. O problema do FundaIA é outro: sapatas em planta, com cargas, momentos, tensão admissível, punção, pilar contido, limites de lote e possivelmente interação geotécnica.

O que faz sentido reaproveitar é o padrão conceitual:

```text
usar as ideias geométricas e a organização
sem transformar o problema de fundações em um 3D-BPP genérico
```

---

## 4. Formulação inicial

Para cada sapata `i`, a Frente 2 deve começar com cinco variáveis:

```text
h_x_i, h_y_i, h_z_i, dx_i, dy_i
```

onde:

- `h_x_i` e `h_y_i` são as dimensões em planta;
- `h_z_i` é a altura;
- `dx_i` é o deslocamento do centro da sapata em relação ao pilar no eixo x;
- `dy_i` é o deslocamento do centro da sapata em relação ao pilar no eixo y.

Rotação em planta deve ficar fora da primeira versão. Se as sapatas puderem rotacionar, a geometria deixa de ser AABB e passa a exigir OBB/SAT, aumentando o risco da implementação sem ganho imediato para a pesquisa.

O objetivo continua sendo:

```text
minimizar volume total de concreto
```

mas agora com restrições adicionais de posicionamento.

---

## 5. Restrições mínimas

A primeira versão da Frente 2 deve verificar:

1. não sobreposição entre sapatas;
2. pilar contido na sapata;
3. balanço mínimo em torno do pilar;
4. sapata dentro dos limites do lote;
5. tensão admissível no solo;
6. punção nos contornos `C` e `C'`;
7. geometria mínima;
8. índice de interação por bulbo de tensões, inicialmente opcional.

O contrato de restrições deve continuar o mesmo do artigo:

```text
g <= 0  -> viável
g > 0   -> violado
```

Grupos esperados:

```text
sob, pun, ten, geo, contain, boundary, bulbo
```

---

## 6. Momentos efetivos com deslocamento

Quando a sapata deixa de ficar centrada no pilar, o braço de alavanca muda. Por isso, o piloto já adotou:

```text
Mx_eff = Mx_input - Fz * dx
My_eff = My_input - Fz * dy
```

Essa relação precisa ficar documentada e testada, porque é ela que conecta a variável de layout à verificação de tensão no solo.

Também é importante manter a convenção de sinais clara. O código atual já documenta a convenção interna dos momentos, mas a Frente 2 deve reforçar isso para evitar erro de interpretação com planilhas vindas de softwares estruturais.

---

## 7. Bulbo de tensões

Faz sentido estudar interação por bulbo de tensões, mas ela não deve começar como uma regra fixa do tipo:

```text
distância mínima >= 2B
```

Uma distância fixa é fácil de implementar, mas é fraca como argumento técnico, porque não depende da carga, da dimensão real da sapata, da profundidade avaliada nem da tensão admissível do solo.

O caminho mais defensável é tratar a interação como acréscimo de tensão vertical provocado por sapatas vizinhas:

```text
Delta sigma_z,total(P, z) = soma das contribuições das sapatas vizinhas
```

Depois disso, pode-se definir um índice de interação. Por exemplo:

```text
R_i = tensão induzida pelas vizinhas / tensão induzida pela própria sapata i
```

ou:

```text
R_i = tensão induzida pelas vizinhas / sigma_adm,i
```

Esse índice pode ser usado de duas formas:

- como indicador descritivo, sem penalizar a solução no primeiro momento;
- como restrição experimental, com limite paramétrico.

Uma forma de restrição seria:

```text
g_bulbo = R_i / R_lim - 1 <= 0
```

Por enquanto, `R_lim` não deve ser apresentado como valor normativo. O mais correto é estudar valores como:

```text
R_lim = 0,10; 0,20; 0,30
```

e declarar que se trata de um critério experimental de interação.

---

## 8. Dois níveis para o bulbo

### Nível A: aproximação 2V:1H

O primeiro nível pode usar a aproximação 2V:1H, porque ela é simples, didática e serve como baseline.

Em uma profundidade `z`, a área carregada pode ser aproximada por:

```text
A_z = (h_x + z) * (h_y + z)
```

Essa aproximação não deve ser o argumento final da pesquisa, mas é útil para testar a arquitetura, os pontos de controle e a superposição.

### Nível B: Boussinesq/Fadum para área retangular

Depois da versão simples, o padrão metodológico deve ser o cálculo do acréscimo de tensão vertical sob área retangular carregada, usando fatores de influência e superposição.

Para sapatas retangulares alinhadas aos eixos:

```text
Delta sigma_z(P, z) = q * I_z
```

onde:

- `q` é a pressão de contato;
- `I_z` é o fator de influência;
- `P` é o ponto de controle;
- `z` é a profundidade analisada.

Esse nível é mais defensável, desde que as hipóteses fiquem explícitas: solo homogêneo, meio elástico semi-infinito, carga uniformemente distribuída e fundação simplificada.

---

## 9. Pontos de controle

Para cada sapata, a influência das vizinhas pode ser avaliada em:

- centro da sapata;
- quatro cantos;
- opcionalmente uma malha `3 x 3`.

Profundidades iniciais:

```text
z = 0,5B; 1,0B; 2,0B; 4,0B
```

com:

```text
B = min(h_x, h_y)
```

Essas profundidades devem ser parâmetros de configuração, não números fixos escondidos no código.

---

## 10. Arquitetura sugerida

Não alterar a assinatura de `avaliar_projeto_fast`. Esse avaliador deve continuar preservado para o artigo 1.

Para a Frente 2, criar módulos próprios:

```text
core/engineering/layout.py
core/engineering/stress_influence.py
core/api/layout_objective.py
```

Responsabilidades:

- `layout.py`: AABB, não sobreposição, contenção do pilar, limites de lote e distância entre retângulos;
- `stress_influence.py`: 2V:1H, Boussinesq/Fadum, superposição e índice de interação;
- `layout_objective.py`: avaliador da Frente 2 com variáveis `[h_x, h_y, h_z, dx, dy]`.

Essa separação evita quebrar o artigo 1 e deixa claro que a Frente 2 é outro problema.

---

## 11. Plano de implementação

### Sprint B1 - geometria 2D

Criar `core/engineering/layout.py` e mover para lá a lógica de:

- AABB das sapatas;
- sobreposição;
- toque por borda sem sobreposição;
- pilar dentro/fora da sapata;
- sapata dentro/fora do lote;
- distância mínima entre retângulos.

Critério de pronto:

```text
pytest passa e o piloto usa o módulo novo
```

### Sprint B2 - avaliador oficial da Frente 2

Criar `core/api/layout_objective.py`.

O avaliador deve:

- receber `[h_x, h_y, h_z, dx, dy]` por sapata;
- calcular momentos efetivos;
- chamar as verificações já existentes de tensão e punção;
- retornar volume, `theta` e restrições por grupo;
- reproduzir o piloto atual como teste de regressão.

Critério de pronto:

```text
o piloto deixa de ter lógica própria e passa a usar o avaliador novo
```

### Sprint B3 - bulbo nível A

Implementar a aproximação 2V:1H em `stress_influence.py`.

Testes mínimos:

- tensão diminui com a profundidade;
- tensão diminui com afastamento horizontal;
- superposição soma as contribuições;
- sapata sem vizinha tem índice de interação nulo ou aceitável.

### Sprint B4 - bulbo nível B

Implementar Boussinesq/Fadum para área retangular.

Testes mínimos:

- simetria;
- monotonicidade com profundidade;
- comparação qualitativa com 2V:1H;
- limites sem divisão por zero;
- documentação das hipóteses geotécnicas.

### Sprint B5 - casos acoplados congelados

Criar uma bancada própria da Frente 2, com pelo menos:

1. duas sapatas próximas, como no piloto;
2. três sapatas em corredor estreito;
3. quatro sapatas próximas a divisas do lote;
4. caso com momentos relevantes;
5. caso em que o índice de bulbo fique ativo.

Cada caso deve ter entrada congelada, limites de lote, cargas, tensão admissível, figura em planta e melhor solução conhecida até o momento.

### Sprint B6 - algoritmos e baselines

Comparar:

- solução centralizada;
- decomposição por sapata;
- DE apenas com dimensões;
- DE com `dx, dy`;
- EGO penalizado;
- CBO;
- busca aleatória;
- opcionalmente um construtor geométrico inspirado em Extreme Points 2D.

O solver 3D-BPP externo não deve ser baseline principal, porque ele não resolve as verificações de engenharia.

### Sprint B7 - protocolo experimental

Usar a mesma disciplina do artigo 1:

- 30 repetições;
- sementes pareadas;
- orçamento de avaliações controlado;
- teste de Wilcoxon pareado;
- correção de Holm;
- taxa de factibilidade;
- volume factível;
- violação máxima por grupo;
- tempo de parede;
- índice de interação por bulbo.

### Sprint B8 - visualização

Depois do avaliador e dos testes, adicionar à interface:

- planta com sapatas, pilares e lote;
- linhas de afastamento;
- indicação de sobreposição ou folgas;
- mapa simples de interação por bulbo;
- tabela de restrições;
- exportação do layout.

---

## 12. Como citar no artigo 1

A frase sugerida para o artigo atual é:

```text
O próximo estágio da pesquisa é o posicionamento conjunto de sapatas,
em que as coordenadas dos centros das fundações passam a ser variáveis
de projeto e restrições de empacotamento, divisa e interação geotécnica
podem tornar o problema efetivamente acoplado.
```

Não vale inserir os resultados do piloto no artigo 1 enquanto não houver protocolo próprio.

---

## 13. Riscos e cuidados

| Risco | Como controlar |
| --- | --- |
| Bulbo virar regra arbitrária | Usar índice paramétrico e estudar sensibilidade de `R_lim`. |
| Frente 2 contaminar o artigo 1 | Manter avaliador, branch e protocolo separados. |
| Usar 3D-BPP de forma forçada | Reaproveitar ideias geométricas, não o solver como motor. |
| Aumento de dimensão prejudicar EGO/CBO | Comparar com DE, busca aleatória e baselines geométricos. |
| Hipóteses fortes de Boussinesq/Fadum | Declarar solo homogêneo, meio elástico e carga uniforme. |
| Sobreposição contada duas vezes | Resolver antes dos benchmarks da Frente 2. |

---

## 14. Fontes técnicas

### Fundações e tensões no solo

- ABNT NBR 6122: referência normativa brasileira para projeto e execução de fundações. Conferir a versão e forma bibliográfica final no Catálogo/ABNT Coleção antes de submissão: <https://www.abntcatalogo.com.br/>.
- FHWA. *Geotechnical Engineering Circular No. 6: Shallow Foundations*. Documento técnico de referência para fundações superficiais e análise de tensões/recalques em projetos de pontes e infraestrutura: <https://www.fhwa.dot.gov/engineering/geotech/pubs/010943.pdf>.
- Fadum, R. E. (1948). *Influence Values for Estimating Stresses in Elastic Foundations*. Fonte clássica para fatores de influência de tensões verticais em fundações elásticas: <https://www.issmge.org/publications/publication/influence-values-for-estimating-stresses-in-elastic-foundations>.
- Newmark, N. M. (1942). *Influence charts for computation of stresses in elastic foundations*. Referência clássica complementar para cálculo gráfico/por influência de tensões: <https://hdl.handle.net/2142/4170>.

### Empacotamento e layout

- Martello, S.; Pisinger, D.; Vigo, D. (2000). *The Three-Dimensional Bin Packing Problem*. Operations Research, 48(2), 256-267. DOI: `10.1287/opre.48.2.256`.
- Crainic, T. G.; Perboli, G.; Tadei, R. (2008). *Extreme Point-Based Heuristics for Three-Dimensional Bin Packing*. INFORMS Journal on Computing, 20(3), 368-384. DOI: `10.1287/ijoc.1070.0250`. Página: <https://pubsonline.informs.org/doi/10.1287/ijoc.1070.0250>.
- Iori, M.; de Lima, V. L.; Martello, S.; Miyazawa, F. K.; Monaci, M. (2020). *Exact Solution Techniques for Two-dimensional Cutting and Packing*. Referência útil para diferenciar packing ortogonal 2D de packing 3D clássico: <https://arxiv.org/abs/2004.12619>.

### Otimização com restrições

- Gardner, J. R. et al. (2014). *Bayesian Optimization with Inequality Constraints*. Base do CBO usado no projeto: <https://proceedings.mlr.press/v32/gardner14.html>.
- Eriksson, D.; Poloczek, M. (2021). *Scalable Constrained Bayesian Optimization*. Referência para versões escaláveis com região de confiança.
- BoTorch documentation, seção de constraints. Útil como referência prática atual para distinguir restrições de parâmetro e restrições de saída em Bayesian Optimization: <https://botorch.org/docs/constraints>.

---

## 15. Próximo passo

O próximo passo imediato é a Sprint B1:

```text
criar core/engineering/layout.py
extrair validações geométricas do piloto
escrever testes unitários
manter o artigo 1 intocado
```

Só depois disso faz sentido implementar o bulbo de tensões e abrir uma bancada experimental completa da Frente 2.
