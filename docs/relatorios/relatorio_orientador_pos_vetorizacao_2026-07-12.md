# Relatório de evolução do FundaIA após a vetorização da sobreposição

- **Projeto:** FundaIA
- **Autor:** Lucas Teixeira Correia
- **Data:** 12/07/2026
- **Ponto de partida:** versão em que a principal mudança era a vetorização da verificação de sobreposição entre sapatas.

---

## 1. Visão geral

Depois da última versão apresentada, o projeto mudou bastante. A vetorização da sobreposição continuou sendo importante, porque deixou a função objetivo muito mais rápida, mas ela acabou virando só uma parte de uma reorganização maior.

O trabalho mais recente consolidou três coisas:

1. uma base de código mais organizada e testável;
2. uma formulação de engenharia mais clara para tensão no solo e punção;
3. um protocolo experimental mais honesto para o artigo.

O ponto principal é que o artigo atual não está mais tentando parecer um projeto executivo completo de fundações. Ele foi reposicionado como um estudo de pré-dimensionamento geométrico otimizado de sapatas isoladas, com hipóteses declaradas, resultados regeneráveis e limitações explícitas. Isso deixa o trabalho mais forte, porque separa o que já está implementado do que ainda precisa entrar em uma etapa posterior.

---

## 2. O que mudou no código

### 2.1. Organização do projeto

O código foi reorganizado em camadas. A lógica de engenharia, otimização, entrada e saída, API e interface deixou de ficar misturada em poucos arquivos grandes.

Hoje a estrutura principal está assim:

- `core/domain`: entidades e validações de domínio;
- `core/engineering`: fórmulas de engenharia, como tensão, punção, solo e sobreposição;
- `core/api`: funções de avaliação e otimização usadas pela interface e pelos scripts;
- `core/optimization`: EGO, GPR, CBO, GA e rotinas de otimização;
- `core/io`: leitura de planilhas, exportações e persistência de experimentos;
- `frontend`: interface Streamlit;
- `scripts`: execução dos experimentos e geração dos artefatos do artigo.

Essa mudança não altera a ideia do método, mas deixa o projeto mais fácil de auditar. Também facilita separar o artigo atual da próxima etapa, que envolve posicionamento conjunto das sapatas.

### 2.2. Vetorização da sobreposição

A verificação de sobreposição entre sapatas, que antes era feita por laços par a par, foi reescrita usando uma matriz `N x N` em NumPy.

Cada célula da matriz representa a área de interseção entre duas sapatas em planta. A diagonal é zerada, porque uma sapata não deve ser comparada com ela mesma.

Na prática, o cálculo passou a ser:

```text
interseção em x * interseção em y = área de sobreposição
```

Um exemplo simples ajuda. Imagine três sapatas retangulares vistas em planta:

```text
S1: x = [0,0 ; 2,0]   y = [0,0 ; 2,0]
S2: x = [1,5 ; 3,5]   y = [0,5 ; 2,5]
S3: x = [4,0 ; 5,0]   y = [0,0 ; 1,0]
```

Para comparar `S1` com `S2`, o código calcula primeiro quanto elas se cruzam em cada direção:

```text
interseção em x = min(2,0, 3,5) - max(0,0, 1,5) = 2,0 - 1,5 = 0,5 m
interseção em y = min(2,0, 2,5) - max(0,0, 0,5) = 2,0 - 0,5 = 1,5 m

área sobreposta = 0,5 * 1,5 = 0,75 m2
```

Já `S1` com `S3` não se cruza em `x`, porque `S1` termina em `x = 2,0` e `S3` só começa em `x = 4,0`. Então a interseção em `x` vira zero e a área sobreposta também é zero.

Com três sapatas, a matriz `N x N` fica assim:

```text
           S1     S2     S3
S1       0,00   0,75   0,00
S2       0,75   0,00   0,00
S3       0,00   0,00   0,00
```

A diagonal é zero porque `S1` contra `S1`, `S2` contra `S2` e `S3` contra `S3` não interessam. A matriz é simétrica porque a área de sobreposição de `S1` com `S2` é a mesma de `S2` com `S1`.

Essa simetria explica a pendência da dupla contagem: se a penalidade global simplesmente somar todas as células da matriz, o par `S1`-`S2` aparece duas vezes. Para uma leitura global da sobreposição, seria preciso somar só metade da matriz, por exemplo apenas os pares acima da diagonal, ou dividir a soma por dois. No código atual, a matriz é usada para montar uma restrição por sapata; por isso a decisão precisa ser fechada com cuidado antes da Frente 2, em que a sobreposição vai ficar ativa.

O resultado numérico foi preservado em relação à versão escalar, mas o tempo caiu muito nos casos com mais sapatas. Essa parte está em `core/engineering/packing.py`.

Ainda existe uma decisão pendente: como a matriz é simétrica, a penalidade global de sobreposição pode estar contando o mesmo par duas vezes. Nos casos do artigo isso não altera os resultados, porque a sobreposição fica inativa por construção. Mas para a Frente 2, em que o packing passa a ser ativo, essa decisão precisa ser fechada.

### 2.3. Função objetivo consolidada

A função objetivo atual trabalha com as dimensões de cada sapata:

```text
[h_x, h_y, h_z]
```

O objetivo principal é minimizar o volume de concreto:

```text
V = h_x * h_y * h_z
```

Como uma solução de volume mínimo pode violar critérios de projeto, o valor otimizado é uma pseudo-função objetivo penalizada:

```text
Theta = volume + penalidades
```

As restrições seguem a convenção:

```text
g <= 0  -> restrição atendida
g > 0   -> restrição violada
```

Os grupos atuais são:

- sobreposição em planta;
- punção;
- tensão no solo;
- geometria mínima em torno do pilar.

A função principal está em `core/api/objective.py`. Também foi criada uma versão por componentes, usada pelo CBO, que separa volume e restrições.

---

## 3. Ajustes de engenharia

### 3.1. Tensão no solo

A formulação da tensão no solo foi corrigida para ficar mais coerente fisicamente.

Antes havia coeficientes legados, como `1,05` e `1,30`, e o peso próprio da sapata não aparecia de forma limpa. Agora o peso próprio é calculado diretamente a partir do volume:

```text
peso_proprio = gamma_c * h_x * h_y * h_z
```

com:

```text
gamma_c = 25 kN/m3
```

A tensão máxima e mínima na base usam flexão composta:

```text
area = h_x * h_y
sigma_axial = (F_z + peso_proprio) / area
sigma_mx = 6 * |M_x| / (area * h_x)
sigma_my = 6 * |M_y| / (area * h_y)

sigma_max = sigma_axial + sigma_mx + sigma_my
sigma_min = sigma_axial - sigma_mx - sigma_my
```

Com isso, uma sapata maior ou mais alta passa a transmitir mais peso ao solo, como deveria acontecer. Também foi documentada a convenção interna dos momentos:

```text
Mx = Fz * ex
My = Fz * ey
```

Essa observação é importante porque alguns softwares estruturais podem fornecer momentos em torno dos eixos globais, e nesses casos pode ser necessário converter a convenção antes de importar os dados.

### 3.2. Tensão admissível por SPT

O código ainda usa uma correlação simples para estimar a tensão admissível:

```text
pedregulho: Nspt / 30 * 1000
areia:      Nspt / 40 * 1000
silte/argila: Nspt / 50 * 1000
```

Essa correlação ficou tratada como hipótese preliminar de pré-dimensionamento, não como prescrição direta da NBR 6122. Essa distinção foi colocada no código e no artigo.

Esse é um dos principais pontos a melhorar antes de uma submissão mais forte: ou a correlação precisa ser melhor justificada por bibliografia geotécnica, ou deve ser substituída por um modelo mais bem calibrado.

### 3.3. Punção nos contornos C e C'

Antes a verificação de punção considerava apenas o contorno `C`, na face do pilar. A versão atual também verifica o contorno `C'`, a `2d` da face do pilar, seguindo a leitura da NBR 6118 para punção.

O avaliador considera o pior caso:

```text
g_puncao = max(g_C, g_C')
```

No contorno `C'`, foram adotadas duas hipóteses conservadoras:

- como o projeto ainda não dimensiona armadura de flexão, usa-se taxa mínima de armadura;
- a reação do solo dentro do perímetro crítico não é abatida da solicitação.

Nos três casos do artigo, a punção ficou folgada. Ou seja, a verificação ficou mais completa, mas não governou os resultados obtidos.

### 3.4. Validações de fronteira

Também foram adicionadas proteções para evitar entradas fisicamente inválidas:

- `Fz` precisa ser positivo, porque o modelo atual considera contato comprimido solo-sapata;
- `h_z` precisa ser maior que o cobrimento, para garantir altura útil positiva;
- `f_ck` é esperado em kPa, por exemplo `25 MPa = 25000 kPa`;
- dimensões e parâmetros básicos são checados antes da otimização.

Essas validações ajudam a evitar soluções aparentemente boas, mas geradas por erro de unidade ou entrada fora do domínio do modelo.

---

## 4. Interface e rastreabilidade

A interface Streamlit foi ampliada. Ela agora permite acompanhar melhor uma execução e entender visualmente o resultado.

Foram adicionados:

- visualização 2D/3D das sapatas;
- gráfico de convergência do EGO;
- progresso ao vivo;
- cancelamento cooperativo;
- exportação de artefatos;
- página de experimentos para comparação entre algoritmos.

Também foram adicionados logs estruturados e persistência de experimentos. Cada rodada pode salvar configurações, histórico, métricas, ambiente e artefatos. Isso é importante porque os números do artigo não ficam soltos: eles podem ser regenerados a partir de scripts e arquivos persistidos.

---

## 5. Protocolo experimental

O protocolo do artigo foi fechado para evitar comparações frágeis.

Foram usados três casos:

- Caso 1: uma sapata, 3 variáveis;
- Caso 2: duas sapatas, 6 variáveis;
- Caso 3: três sapatas, 9 variáveis.

Parâmetros comuns:

```text
h_min = 0,60 m
h_max = 3,00 m
f_ck = 25 MPa
cobrimento = 0,04 m
30 repetições por célula experimental
sementes pareadas entre algoritmos
```

Foram definidos dois cenários:

- **S1:** todos os algoritmos com 150 avaliações reais;
- **S2:** buscas diretas com 3000 avaliações, para testar o que acontece quando a função objetivo é barata.

A comparação estatística usa teste pareado de Wilcoxon com correção de Holm. Isso é mais adequado do que tratar as repetições como independentes, porque as sementes foram pareadas.

O resultado principal é equilibrado: sob orçamento igual de avaliações, os métodos assistidos por modelo substituto ficam competitivos. Mas, quando a função objetivo é muito barata e as buscas diretas recebem orçamento maior, PSO/GWO conseguem soluções melhores em menos tempo de parede. Essa conclusão é importante, porque evita uma justificativa exagerada do EGO no problema atual.

---

## 6. CBO e tratamento explícito de restrições

Além do EGO penalizado, foi implementada uma versão de otimização bayesiana com restrições, baseada em Gardner et al. (2014).

A diferença é esta:

```text
EGO penalizado:
1 GP aprende volume + penalidades

CBO:
1 GP aprende o volume
1 GP aprende cada grupo de restrição
```

A aquisição usada segue a ideia:

```text
ECI(x) = EI(volume) * P(g_1 <= 0) * ... * P(g_k <= 0)
```

Se ainda não há ponto factível, o método prioriza encontrar a região factível.

Nos resultados do artigo, o CBO melhorou a média de `Theta` em relação ao EGO e encontrou bons volumes factíveis, mas ainda perdeu factibilidade estrita em alguns casos. A leitura correta é que o CBO é promissor, principalmente para a próxima fase com restrições ativas de posicionamento, mas ainda precisa de regra final de seleção factível.

---

## 7. Auditoria por decomposição

Uma crítica importante era que os casos do artigo talvez fossem quase separáveis. Em outras palavras: otimizar cada sapata isoladamente e juntar os resultados poderia dar quase o mesmo resultado que otimizar tudo junto.

Essa hipótese foi testada com um baseline de Differential Evolution por sapata:

1. otimiza-se cada sapata separadamente;
2. junta-se a solução;
3. reavalia-se tudo no avaliador global.

Resultado:

| Caso | Volume decomposto | Melhor protocolo global | Diferença |
| --- | ---: | ---: | ---: |
| Caso 1 | 3,108824 m3 | 3,108826 m3 | menor que 0,01% |
| Caso 2 | 4,750747 m3 | 4,787486 m3 | 0,77% |
| Caso 3 | 2,122252 m3 | 2,167259 m3 | 2,08% |

Isso mostra que os casos atuais são bons para testar a formulação, a penalidade, o GPR, o CBO e o protocolo. Mas eles ainda não demonstram plenamente o ganho do posicionamento conjunto, porque a sobreposição está inativa.

Essa auditoria foi incorporada ao artigo para deixar a limitação clara.

---

## 8. Fase B: posicionamento e packing

Depois da auditoria de decomposição, foi iniciado um piloto de posicionamento conjunto.

Nessa nova formulação, cada sapata pode ter:

```text
h_x, h_y, h_z, dx, dy
```

Os deslocamentos `dx` e `dy` representam o deslocamento do centro da sapata em relação ao pilar. Com isso, as sapatas podem se mover em planta, desde que:

- o pilar continue dentro da sapata;
- a sapata continue dentro dos limites do lote;
- não haja sobreposição;
- tensão, punção e geometria continuem atendidas.

No piloto, os momentos efetivos foram ajustados assim:

```text
Mx_eff = Mx_input - Fz * dx
My_eff = My_input - Fz * dy
```

Resultado do caso mínimo:

- Com os ótimos individuais centralizados, o volume ficou em `4,750747 m3`, mas a solução não foi factível, porque as sapatas se sobrepõem (`g_sob = 0,2307`).
- Mantendo centros fixos e redimensionando as sapatas, a solução ficou factível, sem sobreposição, mas o volume subiu para `4,929703 m3`.
- Permitindo deslocamentos `dx, dy`, a solução também ficou factível, sem sobreposição, e o volume caiu para `4,525122 m3`.

Esse resultado ainda é piloto. Ele não deve entrar como resultado principal do artigo atual. O papel dele é mostrar que a próxima fase é relevante e que, quando o posicionamento entra no problema, a decomposição por sapata deixa de ser suficiente.

---

## 9. Situação do artigo

O artigo em `docs/artigo_ic_lucas` foi bastante revisado.

As principais mudanças foram:

- mudança para formato de duas colunas;
- reposicionamento como pré-dimensionamento geométrico experimental;
- inclusão da punção nos contornos `C` e `C'`;
- correção da tensão no solo;
- inclusão do CBO;
- protocolo S1/S2;
- estudo de penalidade e kernels;
- auditoria por decomposição;
- discussão mais clara das limitações;
- dados, figuras e tabelas regeneráveis por scripts.

O PDF atual compila com 22 páginas. Na última verificação, não havia citações indefinidas nem erro de compilação.

O texto está coerente com o código. O principal cuidado é manter o escopo bem delimitado: ainda não há dimensionamento de armadura, flexão, cisalhamento unidirecional, ancoragem, custo total, recalque ou modelo geotécnico calibrado.

---

## 10. Pendências reais

Esses pontos ainda precisam ser tratados com cuidado:

1. **Correlação `Nspt`-tensão admissível:** continua sendo a hipótese geotécnica mais fraca. Precisa ser validada, substituída ou muito bem declarada como aproximação preliminar.
2. **Combinações de ações:** ainda falta separar formalmente verificações de serviço e estado limite último.
3. **Projeto estrutural completo:** o artigo não dimensiona armadura nem faz verificação completa de flexão, cisalhamento unidirecional, ancoragem e custo.
4. **Sobreposição contada duas vezes:** não afeta os casos atuais, mas precisa ser resolvida antes da Frente 2.
5. **Frente B ainda é piloto:** precisa virar protocolo experimental com casos acoplados congelados, seeds pareadas e estatística.
6. **Normas:** é necessário conferir a forma bibliográfica final da NBR 6118 e da NBR 6122 antes da submissão.
7. **Nome do software:** o repositório usa majoritariamente FundaIA, enquanto o artigo aparece como FundaAI. É melhor padronizar antes de enviar.

---

## 11. Validações feitas

Foram conferidos:

- implementação da função objetivo contra `core/api/objective.py`;
- tensão no solo contra `core/engineering/tensao.py`;
- punção contra `core/engineering/puncao.py`;
- CBO contra `core/optimization/cbo.py`;
- piloto de packing contra `scripts/run_packing_phase_b_pilot.py`;
- resultados do artigo contra os artefatos e notas de auditoria.

Também foi executada a suíte de testes:

```text
264 testes passaram
```

E o artigo compilou em LaTeX:

```text
PDF com 22 páginas, sem erro de compilação
```

---

## 12. Referências principais que sustentam as mudanças

**Engenharia e fundações**

- ABNT NBR 6118: usada para as verificações de punção nos contornos `C` e `C'`. A forma bibliográfica final deve ser conferida antes da submissão, porque há edição 2023, versão corrigida e emenda de 2026. Consulta pública: Catálogo ABNT e DIN Media.
- ABNT NBR 6122: referência geral para projeto e execução de fundações. A versão atual precisa ser conferida no Catálogo/ABNT Coleção antes da submissão.
- Santos, D. F. A.; Lima Neto, A. F.; Ferreira, M. P. (2018). *Punching shear resistance of reinforced concrete footings: evaluation of design codes*. Revista IBRACON de Estruturas e Materiais. DOI: `10.1590/S1983-41952018000200011`.
- Wang, Y.; Kulhawy, F. H. (2008). *Economic Design Optimization of Foundations*. DOI: `10.1061/(ASCE)1090-0241(2008)134:8(1097)`.
- Nigdeli, S. M.; Bekdas, G.; Yang, X.-S. (2018). *Metaheuristic Optimization of Reinforced Concrete Footings*. DOI: `10.1007/s12205-018-2010-6`.
- Waheed et al. (2022, 2025), sobre ferramentas e otimização econômica de sapatas de concreto armado.

**Otimização e modelos substitutos**

- Jones, Schonlau e Welch (1998), base do EGO. DOI: `10.1023/A:1008306431147`.
- Williams e Rasmussen (1995), base de processos gaussianos para regressão.
- Shahriari et al. (2016) e Schulz et al. (2018), revisão/tutorial de Bayesian Optimization e GPR.
- Gardner et al. (2014), base do CBO com restrições de desigualdade. Fonte: PMLR.
- Eriksson e Poloczek (2021), SCBO como caminho escalável futuro.
- Mathern et al. (2021), uso de CBO em projeto estrutural. DOI: `10.1007/s00158-020-02720-2`.

**Geotecnia baseada em dados**

- Ahmad et al. (2021), GPR para capacidade de carga de fundações superficiais. DOI: `10.3390/app112110317`.
- Khajehzadeh, Keawsawasvong e Nehdi (2022), soft computing em capacidade de carga e otimização de fundações. DOI: `10.3390/su14031847`.
- Fattahi, Ghaedi e Armaghani (2025), previsão de recalques com técnicas inteligentes. DOI: `10.32604/cmes.2025.062390`.

---

## 13. Resumo curto

Desde a vetorização da sobreposição, o FundaIA deixou de ser apenas uma implementação rápida da função objetivo e passou a ter uma base experimental mais completa. A engenharia foi corrigida em pontos importantes, principalmente tensão no solo e punção `C'`. O protocolo experimental foi fechado com seeds, orçamento, estatística e artefatos regeneráveis. O artigo ficou mais honesto ao assumir o escopo de pré-dimensionamento geométrico e ao mostrar que os casos atuais são quase separáveis.

O projeto está em bom estado para apresentação e discussão. O que ainda precisa de atenção antes de uma submissão mais forte é a parte geotécnica da tensão admissível por SPT, a separação formal das combinações de ações, a padronização normativa e a transformação da Fase B de packing em experimento completo.
