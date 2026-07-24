---
tags: [engenharia, sapata, fundacao-rasa, guia, didatico, formulas]
aliases: [Guia Sapatas, Guia Dimensionamento, Guia Waheed 2025]
data: 2026-04-29
fontes:
  - Waheed et al. (2025) — DOI 10.1007/s41062-024-01823-9
  - Waheed et al. (2022) — DOI 10.3390/buildings12040471
  - Wang & Kulhawy (2008) — Economic Design Optimization of Foundations
  - Gandomi & Kashani (2018) — Cost Minimization of Shallow Foundation
  - NBR 6118 (2026) — Projeto de estruturas de concreto
  - NBR 6122 (2022) — Projeto e execução de fundações, Emenda 1
  - Bezerra et al. (2024) — Elementos de fundação
  - Ahmad et al. (2021) — GPR Bearing Capacity Shallow Foundations
  - Kashani et al. (2020) — Optimum Design of Shallow Foundation
  - ACI 318‑14 (2014) — Building code requirements for structural concrete
---

# Guia Didático — Dimensionamento de Sapatas Isoladas

> Este guia foi escrito para alguém que **não é da engenharia** (você, Lucas), tomando como núcleo o paper recente **Waheed et al. 2025** ([[08_Artigos/Waheed et al. 2025 - Economical Design RC Isolated Footings]]) e cruzando com os outros artigos da pasta `docs/articles/` e com as notas técnicas já existentes no vault.
>
> Para **cada fórmula** do paper, o guia mostra:
> 1. a equação,
> 2. o que cada símbolo significa,
> 3. a intuição física (em palavras simples),
> 4. como ela se relaciona com o que o **FundaIA** já faz (ou não faz) hoje.

---

## 0 · Como ler este guia

- As referências usam o estilo **`[Sobrenome AAAA, Eq./Sec. X]`** e ao final tem uma seção `Referências` com PDF e ficha do vault.
- As fórmulas estão em **LaTeX** (Obsidian renderiza nativamente).
- Quando tem `→ FundaIA`, é um ponteiro pro arquivo do código (`core/...`) ou pra nota técnica do vault.
- Quando tem `⚠`, é uma diferença importante entre o paper e o FundaIA, ou um cuidado.

---

## 1 · O que é uma sapata isolada e por que ela existe?

### 1.1 — A função da sapata

Imagine um prédio. Os **pilares** carregam o peso de toda a estrutura: vigas, lajes, paredes, móveis, pessoas. Esse peso desce pelos pilares como se fossem pernas. Mas pernas finas concentram peso numa área pequena — se você apoiar um piano só com 4 pernas finas num assoalho de madeira, ele afunda. O pilar de concreto faz exatamente isso com o solo.

A **sapata** é uma "almofada" de concreto armado, mais larga que o pilar, que fica enterrada no solo. Ela **espalha** a carga do pilar numa área maior, fazendo a pressão (força/área) cair pra um valor que o solo aguenta sem afundar nem romper.

> 📚 **Referência didática introdutória**: [[08_Artigos/Bezerra et al. 2024 - Elementos de Fundacao]] (visão geral de tipos de fundação e quando usar cada um). Ver também [[02_Engenharia/Sapatas Isoladas]] no vault.

### 1.2 — Por que se chama "isolada"?

Porque cada sapata atende **um único pilar**. Existem outros tipos:

| Tipo | Quando se usa |
| --- | --- |
| **Isolada** | Um pilar por sapata. É o caso do FundaIA. |
| **Associada / corrida** | Pilares próximos compartilham uma sapata mais comprida. |
| **Radier** | Uma laje gigante embaixo de toda a obra. Solo muito ruim ou cargas muito altas. |
| **Estaca / tubulão** | Solo só aguenta carga em camadas profundas (fundação **profunda**, não rasa). |

A sapata isolada é o caso mais comum em obras pequenas e médias [Bezerra 2024; Waheed 2022, Sec. Introduction].

### 1.3 — Por que **otimizar** a sapata?

Sapata é cara. Cada metro cúbico de concreto custa dinheiro, gera CO₂ e consome aço. Em projetos tradicionais o engenheiro usa **regras de bolso** que tendem a sobredimensionar — sai do lado seguro, mas joga material fora.

Os papers que sustentam o FundaIA atacam exatamente isso:

- **[Wang & Kulhawy 2008]** — formulam fundação como problema de **otimização com restrições**: dimensões são variáveis, custo (ou volume) é a função objetivo, normas são restrições. Esse é o paradigma.
- **[Gandomi & Kashani 2018]** — comparam várias **metaheurísticas** (algoritmos evolucionários, enxame) na minimização de custo de fundação rasa, e discutem o efeito da posição da coluna por meio de variáveis adicionais. O artigo mostra que a escolha do algoritmo importa: TLBO se sai bem nos casos estudados, WOA mais fraco.
- **[Waheed 2022]** — apresenta a **planilha-ferramenta** que os mesmos autores depois evoluem em 2025. Já reportava economia de até **44%** em casos numéricos.
- **[Waheed 2025]** ⭐ — leva a ideia adiante com **sapatas escalonadas (stepped footings)** + objetivos eco-eficientes (carbono, energia incorporada).

A motivação econômica e ambiental, então, está bem firmada na literatura.

> 🔗 No FundaIA: [[03_Otimizacao/Formulação do Problema]] e [[01_Projeto/Visão Geral do Projeto]].

---

## 2 · As três famílias de falha que a sapata tem que evitar

Antes das fórmulas, é fundamental entender **o que** estamos verificando. O paper organiza as falhas em três grupos [Waheed 2025, Sec. "Design of single-stepped isolated footings"]:

### 2.1 — Falhas geotécnicas (problema do solo)
- **Capacidade de carga**: o solo cede sob a sapata.
- **Recalque**: o solo afunda demais (pode até estar dentro da capacidade, mas afunda mais que o aceitável).
- **Deslizamento**: a sapata escorrega lateralmente.

### 2.2 — Falhas estruturais (problema da própria sapata como peça de concreto)
- **Flexão (bending)**: a sapata "envergaria" como uma viga em balanço sob a carga do solo.
- **Cisalhamento de uma direção (one-way shear)**: ruptura por corte ao longo de um plano.
- **Cisalhamento de duas direções / punção (two-way / punching shear)**: o pilar atravessa a sapata como um soco — falha em formato de tronco de cone.
- **Aderência (bond)**: as barras de aço deslizam dentro do concreto e perdem ancoragem.

### 2.3 — Falhas de equilíbrio
- **Tombamento**: cargas excêntricas levantam um lado da sapata. A regra de bolso é manter a **excentricidade** menor que **1/6 do lado** — assim a base toda fica comprimida (é o "núcleo central de inércia").

> ⚠ O paper Waheed 2025 segue **ACI 318-14** (norma americana). O FundaIA segue **NBR 6118** (norma brasileira). As verificações são **as mesmas em conceito**, mas as constantes mudam. Esse é o ponto número 1 a ter em mente ao comparar.

---

## 2.5 · Vocabulário fundamental (leia esta seção antes das fórmulas!)

Esta seção é **a chave** para entender as fórmulas. Ela explica os conceitos que vão aparecer o tempo todo, usando analogias do dia a dia. Se você dominar esses 12 conceitos, o resto fica natural.

### 2.5.1 — Força, carga e peso

**Força** é qualquer "empurrão" ou "puxão" sobre um corpo. A unidade é o **Newton (N)**. Em engenharia civil quase sempre falamos em **kilonewton (kN)**: 1 kN ≈ peso de 100 kg na superfície da Terra.

**Carga** é só um sinônimo culto de **força aplicada numa estrutura**. Quando você lê "carga de 1500 kN no pilar", é o mesmo que "uma força de 1500 kN está empurrando esse pilar pra baixo".

**Peso** é a força que a gravidade faz sobre uma massa. Massa em kg → peso em N (multiplica por 9,81). Mas em engenharia raramente importa essa distinção: no contexto de sapatas, "peso" e "carga axial" andam juntos.

> 💡 **Por que isso importa**: a sapata existe para **espalhar** uma força (a carga do pilar) numa área grande. Quanto maior a área, menor a pressão (força ÷ área) sobre o solo.

### 2.5.2 — Carga axial, lateral e cortante

Uma força aplicada num pilar pode vir em três direções:

- **Carga axial** (ou **vertical**): paralela ao eixo do pilar, ou seja, "pra baixo" (compressão) ou "pra cima" (tração — raro). É **a principal carga** que a sapata transmite ao solo. Símbolo no paper: `P` (Eq. 1) ou `P_D + P_L`. No FundaIA: `F_z` ou `F_zk`.

- **Carga lateral** (ou **horizontal**): perpendicular ao eixo do pilar. Vem do **vento**, de **terremoto**, de **empuxo de terra** atrás de muros. Não vamos detalhar aqui porque o Waheed 2025 e o FundaIA não tratam carga lateral diretamente — só momento.

- **Cortante** (ou **shear**): força que tenta **cortar** uma seção, como uma tesoura cortando papel. Não é uma carga aplicada de fora — é a **força interna** que aparece dentro do material por causa de outras cargas. Mais detalhes em §2.5.7.

### 2.5.3 — Momento (fletor)

**Momento** é o que faz **torcer** ou **dobrar** alguma coisa. Pense em apertar um parafuso com uma chave de boca: você aplica uma força na ponta do cabo, e isso vira um **momento** (também chamado de **torque**) no parafuso. Quanto maior o cabo, maior o momento, mesmo aplicando a mesma força.

Matematicamente: **momento = força × distância perpendicular**. Unidade: **kN·m** (kilonewton vezes metro).

#### De onde vem o momento numa sapata?

Três fontes principais:

1. **Vento** empurrando o prédio lateralmente. O prédio quer tombar; isso vira momento na base do pilar.
2. **Carga descentrada**: se a carga do andar não cai exatamente no eixo do pilar (porque o pilar está num canto, ou tem balanço), aparece momento no pé do pilar.
3. **Pilar inclinado / desaprumo**: pequenas imperfeições viram momentos.

> 💡 **Quando o paper ou eu falo "se houver momento"**, é literalmente: "se existir alguma dessas três situações". Numa edificação simétrica e sem vento, momento ≈ 0 e a fórmula da pressão do solo simplifica para apenas `P/A`. Mas quase nenhum projeto real é tão limpo, então **o caso geral inclui momentos**.

No paper, os momentos são `M_B` (momento em torno do eixo paralelo a `B`) e `M_L` (em torno do paralelo a `L`). No FundaIA, são `M_x` e `M_y`. **Mesma coisa, nomes diferentes**.

### 2.5.4 — Excentricidade

**Excentricidade** é uma forma elegante de expressar momento em **unidades de comprimento**. Ela responde: "**onde** a carga axial efetivamente cai?".

$$ e = \frac{M}{P} $$

Se `e = 0`: a carga cai exatamente no centro da sapata.
Se `e = 0,3 m`: é como se a carga axial estivesse aplicada 30 cm fora do centro.

> 💡 **Analogia**: imagine empurrar uma bandeja com prato de comida para baixo. Se você empurra no centro, ela desce reto. Se você empurra na borda, ela tomba. **Onde** você empurra é a excentricidade. **Quanto** você empurra é a carga axial.

#### A regra do "1/6 do lado" (núcleo central)

Pra uma base retangular, existe um teorema clássico:

> Se a carga axial cai dentro de um losango central (chamado **núcleo central de inércia**), cuja semidiagonal é `lado/6`, então a base **inteira** fica comprimida.
>
> Se cai fora desse losango, **parte da base levanta** (tração).

É por isso que a regra `e_L ≤ L/6` e `e_B ≤ B/6` aparece no paper [Eq. 1-5] e em qualquer livro de fundações.

### 2.5.5 — Tensão (pressão)

**Tensão** é **força distribuída por unidade de área**. Em geral usamos o símbolo `σ` (sigma).

$$ \sigma = \frac{F}{A} $$

Unidades: Pa (Pascal = 1 N/m²), kPa (1 000 Pa), MPa (1 000 000 Pa). Para você ter referência:
- Pressão atmosférica: ≈ 100 kPa
- Pneu de carro: ≈ 200 kPa
- Solo argiloso típico aguenta ≈ 200–400 kPa
- Concreto C25: aguenta 25 MPa = 25 000 kPa em compressão

> 💡 **Tensão é o "quanto está apertado"**. A força total `F` numa estrutura grande pode ser enorme, mas se a área `A` for grande também, a tensão (o quanto cada cm² está sofrendo) pode ser perfeitamente segura. O **trabalho da sapata** é justamente aumentar `A` pra reduzir `σ` no solo.

### 2.5.6 — Concreto armado, armadura e cobrimento

**Concreto** é forte em **compressão** (esmagamento) e **fraco em tração** (puxão). Imagine pedra: dá pra empilhar pedras umas sobre as outras (compressão), mas se você tentar pendurar uma pedra puxando-a por baixo (tração), ela quebra.

**Aço** é forte tanto em compressão quanto em tração — mas é caro e enferruja.

**Concreto armado** combina os dois: o concreto cuida da compressão, o aço (em forma de barras = **armadura** ou **rebar** em inglês) cuida da tração. As barras ficam **enterradas dentro do concreto**.

#### Cobrimento

**Cobrimento** (`Cov_f` no paper, `cob` no FundaIA) é a distância entre a **face externa** do concreto e a **primeira barra de aço**. Serve pra:
1. Proteger as barras de **corrosão** (água, gás carbônico atravessando o concreto).
2. Proteger contra **fogo** (concreto isola termicamente).
3. Garantir **aderência** (o concreto precisa "abraçar" a barra com força).

Valores típicos: 2,5 a 5 cm. No FundaIA o default é 4 cm; no Waheed 2025, 63,5 mm (≈ 6,4 cm).

#### Altura útil `d` ou `d_e`

Como as barras estão enterradas, a **altura efetiva da peça pra cálculo de flexão** não é a altura total `h`. É:

$$ d_e \approx h - \text{cobrimento} - \frac{d_b}{2} $$

onde `d_b` é o diâmetro da barra. **`d_e` é a distância do topo do concreto até o centro de gravidade da armadura inferior** — o "braço útil" da peça.

### 2.5.7 — Cisalhamento e punção

**Cisalhamento** é o tipo de esforço interno que aparece quando uma parte do material **escorrega** em relação à parte vizinha. Tesoura cortando papel é o exemplo perfeito: as duas lâminas aplicam forças paralelas mas de sentidos opostos em planos próximos, e o papel cede no plano entre elas.

Numa sapata, há dois tipos:

- **Cisalhamento de uma direção** (one-way shear): falha **ao longo de um plano vertical reto** que atravessa a sapata. Exemplo: imagine cortar a sapata com uma faca gigante a uma distância `d_e` da face do pilar.

- **Cisalhamento de duas direções / punção** (two-way / punching shear): o pilar **fura a sapata como um soco**. A falha tem **forma de tronco de cone** (uma "rampa" inclinada de 45° em torno do pilar). É o **modo de falha mais perigoso** em sapatas porque é súbito e frágil — sem aviso prévio.

#### Por que existe um "perímetro crítico"?

Pra calcular punção, a engenharia mediu/modelou que a fissura cônica corta o concreto numa superfície inclinada. A **projeção** dessa superfície em planta é um **retângulo (ou círculo) ao redor do pilar**, a uma distância de meia altura útil (`d_e/2`) da face. Esse contorno é o **perímetro crítico** `b_o` (paper) ou `u_rd1` (NBR).

> 💡 **Intuição**: pra calcular se o pilar fura a sapata, em vez de modelar a falha 3D em forma de cone (caro), você "achata" o cone numa superfície vertical e calcula a tensão de cisalhamento atuando nele. É uma simplificação genial — mas exige medir esse perímetro com cuidado.

### 2.5.8 — Flexão e viga em balanço

**Flexão** é o que faz uma régua **dobrar** quando você apoia uma ponta na mesa e empurra a outra. A régua fica curvada: o **lado de cima** estica (tração), o **lado de baixo** comprime.

**Viga em balanço (cantilever)** é uma viga engastada de um lado e livre do outro, como um trampolim de piscina. A carga no extremo livre faz a viga dobrar pra baixo.

> 💡 **Por que isso importa**: a sapata se comporta **exatamente como uma viga em balanço** que sai do pilar pros lados. A pressão do solo empurra a parte saliente da sapata pra cima, e ela tende a dobrar como o trampolim. Resultado: tração na **face inferior** e compressão na superior. A armadura tem que ir na **base inferior** pra resistir essa tração.

### 2.5.9 — Cargas mortas (D), vivas (L), serviço e cálculo

Em estruturas existem dois tipos básicos de carga:

- **Carga morta / permanente** (`D` de Dead, ou `G` em normas europeias): peso da própria estrutura — concreto, paredes, cobertura. Não muda com o uso.

- **Carga viva / sobrecarga** (`L` de Live, ou `Q` em normas europeias): peso de **uso** — pessoas, móveis, carros, neve. Varia ao longo do tempo.

#### Serviço × cálculo

- **Carga de serviço** (`P` no paper, `F_zk` no FundaIA — "k" de característica): a carga **real**, sem majoração. Usada para verificar **conforto / dimensionamento geotécnico** (a pressão admissível do solo é checada com carga de serviço, porque é a carga **real** que vai sobre o solo).

- **Carga de cálculo / fatorada** (`P_u` no paper, com `u` de **ultimate**): a carga real **multiplicada por um fator de segurança** (ex.: 1,4 pra carga viva, 1,2 pra carga morta). Usada para verificar o **concreto armado** (cisalhamento, flexão), porque a estrutura tem que suportar com folga o pior caso.

> 💡 **Por que dois valores**: as **falhas geotécnicas** (solo cedendo) e as **falhas estruturais** (concreto rompendo) têm naturezas diferentes. Solo é probabilístico/empírico → check com carga real + margem embutida na `q_adm`. Concreto é mecânico/calculável → check com carga majorada pra garantir folga.

### 2.5.10 — Resistência característica e de cálculo (`f_ck`, `f_cd`)

Toda resistência de material tem dois números:

- **`f_ck`** (concreto) ou **`f_yk`** (aço): valor **característico** = aquele que 95% dos corpos-de-prova rompem **acima** dele (5% rompem abaixo). Pra concreto C25, `f_ck = 25 MPa`.

- **`f_cd`** ou **`f_yd`**: valor **de cálculo** = característico **dividido** por um coeficiente de segurança parcial.
  - NBR 6118: `f_cd = f_ck / 1,4` (concreto), `f_yd = f_yk / 1,15` (aço).
  - ACI 318: usa filosofia diferente (fator `φ` aplicado na resistência, em vez de dividir).

> 💡 **Por que dois números**: por causa da **variabilidade**. O concreto que sai da betoneira não é exatamente igual de uma virada pra outra. A norma diz: "use o valor que 95% das amostras superam, e ainda divide por 1,4 pra ficar bem dentro do seguro".

### 2.5.11 — Coeficientes de segurança e fatores

Cada formulazinha tem alguns números misteriosos. Os principais são:

| Símbolo | Significado | Valor típico |
|---|---|---|
| `γ_c` | Coef. parcial do concreto (NBR) | 1,4 |
| `γ_s` | Coef. parcial do aço (NBR) | 1,15 |
| `γ_f` | Coef. de majoração das ações (NBR) | 1,4 |
| `φ` | Fator de redução da resistência (ACI) | 0,75 (cisalhamento), 0,90 (flexão) |
| `α_v2` (NBR) | Fator de fragilidade do concreto na punção | `1 − f_ck/250` |
| `λ` (ACI) | Fator do tipo de concreto | 1,0 (normal), 0,75 (leve) |
| `γ_c` (FundaIA) | Peso específico do concreto usado para peso próprio | 25 kN/m³ |

> 💡 **Por que isso existe**: o mundo é incerto. As normas embutem 30 a 50% de margem em vários pontos pra cobrir cargas surpresa, falhas de execução, variação de material, etc. A engenharia estrutural moderna se construiu sobre **calibrar essas margens** com dados estatísticos, em vez de "chutar pra cima" como antigamente.
>
> No FundaIA atual, os fatores antigos `1,05` e `1,30` foram removidos da tensão solo-sapata. O peso próprio entra por `γ_c h_x h_y h_z`; combinações de ações e coeficientes normativos devem ser tratados explicitamente nas cargas ou em uma etapa própria de combinações.

### 2.5.12 — Volume, área e peso (saída do projeto)

Ao final, um projeto entrega quantidades:

- **`V_c`**: volume de concreto a comprar (m³).
- **`W_st`**: peso de aço a comprar (kg).
- **`A_f`**: área de fôrma de madeira a alugar/comprar (m²) — é o que dá forma ao concreto enquanto ele endurece.

A função objetivo do paper [Eq. 13] é simplesmente: **custo = soma do que você gasta com cada um desses três insumos**. O FundaIA usa só `V_c` (volume) por simplificação — porque na prática `V_c` é quem domina o custo total em sapatas comuns.

---

## 3 · As variáveis que a otimização escolhe

[Waheed 2025, Tabela 1] define **8 variáveis de projeto** para a sapata escalonada:

| Variável | Unidade | Significado | Limite inferior | Limite superior |
|---|---|---|---|---|
| `L` | m | Comprimento da sapata | 1 | 10 |
| `B` | m | Largura da sapata | 1 | 10 |
| `x` | m | Comprimento até o degrau | 0 | 10 |
| `y` | m | Largura até o degrau | 0 | 10 |
| `d₁` | mm | Altura da sapata na face do pilar (parte alta) | 150 | 1000 |
| `d₂` | mm | Altura da sapata na face do degrau (parte baixa) | 150 | 500 |
| `A_sL` | mm²/m | Área de aço ao longo de L | A_s,min | A_s,max |
| `A_sB` | mm²/m | Área de aço ao longo de B | A_s,min | A_s,max |

> 🔗 **Comparação com FundaIA** ([[02_Engenharia/Sapatas Isoladas]], [[04_Codigo/fundacao.py]]):
>
> O FundaIA hoje só otimiza **3 variáveis por sapata**: `h_x`, `h_y`, `h_z` (sapata simples, sem degrau, sem armadura como variável). Detalhes:
> - **Não tem `x` e `y`** (degrau): o FundaIA assume sapata em bloco retangular único — equivale a `x = 0` e `y = 0` no Waheed 2025, ou `d₁ = d₂`.
> - **Não otimiza armadura**: `A_sL` e `A_sB` não são variáveis. A NBR 6118 e o cálculo de quantidade de aço **não são feitos hoje** no FundaIA. Isso é uma diferença importante.
> - **Otimiza várias sapatas ao mesmo tempo** (uma por pilar) e cuida de **não-sobreposição** entre elas (`packing`), o que o Waheed 2025 não faz (ele otimiza uma sapata por vez).

> ⭐ **O que isso significa pro seu projeto?** O paper Waheed 2025 mostra um **espaço de variáveis maior e mais rico** (degrau + armadura). Pra você, isso é uma janela natural pra evolução do FundaIA: adicionar armadura como variável é um passo lógico se quiser sair de "minimizar volume" pra "minimizar custo de verdade".

---

## 4 · As fórmulas do paper, uma a uma

A partir daqui o guia segue **a numeração exata do paper Waheed 2025**.

### 4.1 — Eq. (1)–(5): Pressão no solo e excentricidade

[Waheed 2025, Eq. 1-5]:

$$
\begin{cases}
\sigma_{\max/\min} = \dfrac{P}{A} \pm \dfrac{M_B \, B/2}{I_B} \pm \dfrac{M_L \, L/2}{I_L} \\[6pt]
q_{\max} \le q_{net} \\[2pt]
q_{\min} \ge 0 \\[2pt]
e_L \le L/6 \\[2pt]
e_B \le B/6
\end{cases}
$$

**Tradução em palavras**:

#### O que essa fórmula está calculando?

Está calculando **a pressão que o solo recebe da sapata** em cada canto da base. Por quê? Porque se em **algum canto** a pressão for maior que o solo aguenta (`q_admissível`), o solo cede ali — recalque diferencial — e a obra trinca.

Se a carga axial caísse exatamente no centro **e não houvesse momento**, a pressão seria **uniforme** em toda a base: bastava calcular `σ = P/A` e pronto. Mas como **quase sempre há momento** (vento, carga assimétrica, pilar de canto), a pressão fica **inclinada** — mais alta de um lado, mais baixa do outro. A fórmula da flexão composta é a forma fechada dessa inclinação.

#### Termo a termo

- `P/A` (parte uniforme): a **pressão média** que existiria se a carga fosse perfeitamente centrada. `P` = carga axial total no pilar (em kN). `A = L × B` = área da base da sapata. `P/A` sai em kPa.

- `M_B · (B/2) / I_B`: a **correção de pressão na borda da largura** por causa do momento `M_B`. Vamos destrinchar:
  - `M_B`: momento aplicado em torno do eixo paralelo a B (ou seja, o momento que **inclina a sapata na direção L**).
  - `B/2`: distância do centro até a borda — é onde a correção é máxima.
  - `I_B = L · B³ / 12`: o momento de inércia que mede **quanto a base resiste a rodar nessa direção**. Quanto mais larga a base, maior `I_B`, menor a correção. **Inércia grande = base "rígida" → distribui melhor a carga**.
  - O sinal `±` significa: numa borda você **soma** (pressão maior), na borda oposta você **subtrai** (pressão menor).

- `M_L · (L/2) / I_L`: análogo, mas pra inclinação na direção B. `I_L = B · L³ / 12`.

> 💡 **Por que `B/2 / I_B` e não `B/2 / I_L`?** Porque o momento `M_B` faz a base rodar no plano que **estica** ao longo da largura. A "distância" relevante na fórmula `(M·y/I)` é a **distância do eixo neutro até a fibra mais distante**, e o eixo neutro nesse caso passa pelo centro paralelo a `B`. A maior distância é `B/2`. E o momento de inércia que entra é o **em relação ao eixo paralelo a B** = `I_B = L · B³ / 12`. (Vai ficar mais claro quando você ler [[02_Engenharia/Flexão Composta - Sigma Max e Min]]).

Para uma base retangular:

$$ I_L = \frac{B \, L^3}{12}, \qquad I_B = \frac{L \, B^3}{12} $$

(Note: a notação do paper usa `I_B` pro momento de inércia "associado a M_B", o que pode confundir. Em livros brasileiros é mais comum escrever `I_x` e `I_y`.)

#### A forma usada no FundaIA

Substituindo `I = (lado_perpendicular · lado³)/12`, o termo de flexão vira `6M / (lado_perp · lado²)`, que é equivalente a `6M/(A·lado)` quando `A = L·B`. O FundaIA atual usa essa forma dimensional diretamente, somando o peso próprio explícito à parcela axial:

$$ \sigma_{\max/\min} =
\frac{F_z + W_c}{A}
\pm \frac{6M_x}{A\,h_x}
\pm \frac{6M_y}{A\,h_y}, \qquad
W_c = \gamma_c h_x h_y h_z $$

Essa forma evita tratar o peso próprio por fator fixo e deixa a convenção de momentos explícita: `Mx` varia a pressão ao longo de `h_x`; `My` varia ao longo de `h_y`.

#### Por que `q_max ≤ q_net` (segunda restrição)

A `q_net` (em kPa) é **a pressão líquida que aquele solo aguenta**. Vem ou de ensaio (placa), ou de fórmulas teóricas (Terzaghi), ou de correlação empírica com SPT (no FundaIA: `q_adm = N_SPT/k · 1000`). Já tem **margem de segurança embutida** (por isso "admissível").

A palavra **líquida** é importante: você está enterrando a sapata a uma profundidade `h_e`. Antes de a obra existir, naquela cota o solo já estava sustentando o peso da terra acima. Quando você escava, "tira" esse peso. Quando coloca a sapata, "devolve" parte. Só a **diferença** (a pressão líquida que a sapata adiciona ao que já estava lá) é que conta. Por isso `q_net = q_adm − γ_solo · h_e`.

> ⚠ **Cuidado de OCR**: no PDF original do paper, esta linha aparece com texto bagunçado — `q3net qmax / q3min 0` — que deve ser lido como `q_max ≤ q_net` e `q_min ≥ 0`. Foi um problema de extração do PDF, não do paper de verdade.

#### Por que `q_min ≥ 0` (terceira restrição)

Solo **não puxa**: ele só comprime. Se a fórmula te der `q_min < 0` (tração), na realidade aquele canto da sapata simplesmente **levanta** e perde contato com o solo. A análise toda fica inválida.

> 💡 **Visualize**: imagine uma régua apoiada numa mesa. Se você empurrar a régua reto pra baixo, ela cola na mesa toda. Se empurrar de lado (criando momento), uma ponta se levanta. A fórmula `σ = P/A ± Mc/I` "não sabe" disso e te dá um valor negativo na ponta levantada. Aí entra a restrição `q_min ≥ 0`: "só vou aceitar projetos onde **toda** a base fique em contato".

#### Por que `e ≤ lado/6` (quarta e quinta restrições)

Como vimos em §2.5.4, isso é uma **forma equivalente** de exigir `q_min ≥ 0`. Quando a excentricidade `e = M/P` ultrapassa `lado/6`, a base começa a tracionar — ou seja, o `q_min` da fórmula vira negativo. As normas mantêm as duas formas (`q_min ≥ 0` **e** `e ≤ lado/6`) por redundância pedagógica e por convivência com fórmulas alternativas que partem da excentricidade direto.

> 🔗 **No FundaIA** ([[02_Engenharia/Flexão Composta - Sigma Max e Min]], `core/engineering/tensao.py`):
>
> ```python
> area = h_x * h_y
> peso_proprio = 25.0 * area * h_z
> sigma_axial = (f_zk + peso_proprio) / area
> sigma_mx = 6.0 * abs(m_xk) / (area * h_x)
> sigma_my = 6.0 * abs(m_yk) / (area * h_y)
> sigma_max = sigma_axial + sigma_mx + sigma_my
> sigma_min = sigma_axial - sigma_mx - sigma_my
> ```
>
> É a forma `P/A ± M c/I`, com o peso próprio calculado explicitamente por volume:
> - `peso_proprio = gamma_c · h_x · h_y · h_z`, com `gamma_c = 25 kN/m3`.
> - `sigma_mx` usa `h_x` porque, na convenção interna do FundaIA, `Mx` é a componente que gera variação de pressão ao longo de X.
> - `sigma_my` usa `h_y` pela mesma lógica na direção Y.
> - Os fatores antigos `1,05` e `1,30` foram removidos. Coeficientes normativos e combinações de ações devem entrar explicitamente nas cargas ou em uma etapa futura de combinações.
>
> ⚠ A restrição `e ≤ L/6` **não está checada explicitamente** como `g_excentricidade` no FundaIA — ela é implicitamente respeitada quando `σ_min ≥ 0` (porque ultrapassar L/6 faz a tração aparecer). É essa a forma que aparece no `checagem_tensao_max_min`.

#### Subfórmula didática: tensão admissível do solo (FundaIA)

O FundaIA usa um **método empírico baseado em SPT** ([[02_Engenharia/Tensão Admissível do Solo]], [[02_Engenharia/SPT - Sondagem]]):

$$
\sigma_{adm} = \frac{N_{SPT}}{k} \cdot 1000 \;\; [\text{kPa}]
$$

| Solo | k |
|---|---|
| Pedregulho | 30 |
| Areia | 40 |
| Silte / argila | 50 |

Esse é o **"método dos práticos"** simplificado, originário das correlações de Terzaghi & Peck. Existem métodos mais modernos (Décourt-Quaresma, Aoki-Velloso) que ainda **não estão no FundaIA**.

> 📚 [Ahmad et al. 2021] propõe substituir esse método empírico por um **regressor por Gaussian Process (GPR)** ajustado em ensaios de laboratório/campo. Esse paper é uma referência **direta** pra você porque o FundaIA já usa GPR no surrogate de otimização — usar GPR pra estimar capacidade de carga do solo é uma evolução natural.

---

### 4.2 — Eq. (6)–(10): Cisalhamento (one-way + punção/two-way + excêntrica)

[Waheed 2025, Eq. 6-10] — segundo o ACI 318-14:

$$
\begin{cases}
\phi V_n \ge V_u \\[6pt]
V_n = \phi \cdot 0{,}17 \, \lambda \sqrt{f_c} \, b \, d_e & \text{(one-way shear)}\\[6pt]
V_n = \phi \cdot 0{,}33 \sqrt{f_c} \, \lambda \, b_o \, \dfrac{d_e}{2} & \text{(two-way / punching, base)}\\[6pt]
V_n = \phi \cdot 0{,}17\,\bigl(1 + \dfrac{2}{\beta}\bigr)\, \lambda \sqrt{f_c} \, b_o \, \dfrac{d_e}{2} & \text{(punching, fator forma)}\\[6pt]
V_n = \phi \cdot 0{,}083 \,\bigl( \dfrac{\alpha_s d_e}{b_o} + 2 \bigr)\, \lambda \sqrt{f_c} \, b_o \, \dfrac{d_e}{2} & \text{(punching, fator posição)}
\end{cases}
$$

**Tradução em palavras**:

#### O que essa fórmula está calculando?

Está calculando se **o concreto da sapata aguenta** dois tipos de cisalhamento:

1. **One-way shear** (uma direção): o "corte de tesoura" ao longo de um plano vertical reto. Uma analogia: imagine cortar a sapata como você corta uma fatia de bolo — uma **lâmina** vertical descendo numa linha reta a uma distância `d_e` da face do pilar. A força que tenta fazer essa fatia escorregar é `V_u`. O concreto aguenta `V_n`.

2. **Two-way / punching shear** (duas direções): o pilar empurrando como um soco. A falha é um **tronco de cone** de concreto se desprendendo ao redor do pilar. Em vez de calcular o cone em 3D (caro), a engenharia "achata" a falha numa superfície vertical envolvendo o pilar a uma distância `d_e/2` da face, e calcula a tensão de cisalhamento ali.

#### Símbolos da fórmula

- **`V_u`** (kN): cisalhamento **atuante** = a força de corte que a carga produz, **com fatores de majoração** (o `u` é de **ultimate**, ou seja, "valor de cálculo / carga fatorada", veja §2.5.9). É calculado integrando a pressão do solo `q_u` sobre a área **fora** da seção crítica.

- **`V_n`** (kN): cisalhamento **resistente nominal** = a capacidade do concreto resistir, segundo a fórmula. "**Nominal**" significa "valor antes de aplicar fator de redução de segurança"; depois multiplicamos por `φ` (que é menor que 1) pra ter o valor de projeto.

- **`φ`** (adimensional): fator de redução de capacidade no ACI. Veja §2.5.11. ≈ 0,75 pra cisalhamento. **`φV_n` é o que você pode realmente "contar" como capacidade**.

- **`λ`** (adimensional): fator pro tipo de concreto. Concreto comum: `λ = 1`. Concreto leve (com argila expandida, por exemplo): `λ < 1` porque é mais frágil. O paper adota `λ = 1`.

- **`f_c`** (MPa): a resistência característica de compressão do concreto (= o `f_ck` da NBR). C25 → `f_c = 25 MPa`.

- **`√f_c`**: por que **raiz quadrada**? Porque a resistência ao cisalhamento do concreto **não cresce linearmente** com a resistência à compressão. Empiricamente, ela cresce **com a raiz**. Concreto duas vezes mais forte na compressão é só √2 ≈ 1,41 vezes mais forte no cisalhamento. Essa observação vem de centenas de ensaios e é base do ACI desde os anos 60.

- **`b`** (m): largura da faixa que está sendo cortada (one-way). Em geral toma-se `b = 1 m` ou a largura total da sapata.

- **`d_e`** (mm ou m): **altura útil** = altura da sapata menos cobrimento menos meio diâmetro de barra. **É o "braço útil" do concreto resistir**. Veja §2.5.6.

- **`b_o`** (m): **perímetro crítico** da punção (na NBR é chamado `u`, no FundaIA é `u_rd2`). É o **contorno fechado ao redor do pilar** ao longo do qual a tensão de cisalhamento é avaliada. No ACI, fica a `d_e/2` da face do pilar; perto do pilar `b_o ≈ 2(c1 + c2 + 2·d_e/2)`.

- **`β`** (adimensional): razão lado maior / lado menor do pilar. Pilar quadrado → `β = 1`. Pilar muito alongado (1m × 0,3m) → `β = 3,33`.

- **`α_s`** (adimensional): constante de **posição do pilar**:
  - 40 → pilar **interno** (centro da edificação): a punção pode "fechar" 360° em torno.
  - 30 → pilar de **borda**: só ≈ 270° de concreto disponível.
  - 20 → pilar de **canto**: só ≈ 180°.

A regra final é: **`φV_n ≥ V_u`** em **todas** as seções críticas. Para a sapata escalonada do paper, isso significa quatro pontos: face do pilar (com `d_{e1}`) e face do degrau (com `d_{e2}`), em ambas as direções (L e B). Daí saem as 6 restrições g1-g6.

#### Por que três fórmulas para punção?

O ACI dá o **menor entre três** valores de `V_n`. Por quê? Porque o cisalhamento de punção depende de **três fatores** que podem dominar separadamente, e a norma quer o caso **mais crítico**:

1. **`V_n = φ · 0,33 √f_c · λ · b_o · d_e/2`** — caso base, calibrado pra colunas razoavelmente quadradas e bem dentro da sapata. Considera apenas o concreto puro.

2. **`V_n = φ · 0,17 (1 + 2/β) λ √f_c · b_o · d_e/2`** — fator de **forma do pilar**. Quando `β` é grande (pilar alongado), o termo `(1 + 2/β) → 1`, reduzindo `V_n`. Por que? Porque em pilares muito retangulares, o lado curto recebe muito mais cisalhamento concentrado por metro de perímetro do que o lado longo, e a fórmula reta superestima a capacidade.

3. **`V_n = φ · 0,083 (α_s d_e / b_o + 2) λ √f_c · b_o · d_e/2`** — fator de **posição do pilar**. Pilar de canto (`α_s = 20`) reduz mais ainda `V_n`. Por que? O cone de punção não consegue se desenvolver de todos os lados — falta concreto pra "abraçar" o soco do pilar.

> 💡 **A norma não escolhe pra você qual usar**: você calcula os três e usa o menor. O ACI deixa explícito porque é cada um, e a fórmula menor é o **gargalo**.

> 🔗 **No FundaIA** ([[02_Engenharia/Verificação à Punção]], `core/engineering/puncao.py`):
>
> ```python
> alpha_v2 = 1 - (f_ck / 1000) / 250
> f_cd = f_ck / 1.4
> tau_rd2 = 0.27 * alpha_v2 * f_cd
> u_rd2 = 2 * (a_p + b_p)
> tau_sd2 = (1.4 * f_zk) / (u_rd2 * d)
> g_rd2 = tau_sd2 / tau_rd2 - 1
> ```
>
> A NBR 6118 trabalha com **tensões** (kPa) e **dois perímetros críticos**:
> - **Perímetro C**: na face do pilar — `u = 2(a_p + b_p)` — verifica esmagamento da biela.
> - **Perímetro C'**: a `2d` da face — verifica diagonal-tração/punção propriamente dita.
>
> O FundaIA atual implementa os dois contornos: C em `verificacao_puncao_sapata` e C′ em `verificacao_puncao_sapata_c_linha`. A issue antiga [[07_Issues/Issue - Punção seção C linha comentada]] fica apenas como registro histórico da correção.
>
> ⭐ A constante `0,27` da NBR vem de uma versão simplificada da Eq. CEB-FIP. As constantes `0,33` / `0,17` / `0,083` do ACI vêm da derivação semi-empírica do ACI Committee 318. Conceitualmente **fazem a mesma coisa**: limitam a tensão de cisalhamento ao redor do pilar.

---

### 4.3 — Eq. (11)–(12): Flexão

[Waheed 2025, Eq. 11-12]:

$$
\begin{cases}
\phi M_n \ge M_u \\[6pt]
\phi M_n = \phi \cdot A_s \cdot f_y \cdot \bigl( d_e - \dfrac{a}{2} \bigr)
\end{cases}
$$

**Tradução em palavras**:

#### O que essa fórmula está calculando?

Está calculando se **a armadura inferior da sapata aguenta** o momento fletor que a pressão do solo gera. A sapata, sob a pressão do solo, **enverga como uma viga em balanço** (veja §2.5.8) que "sai" do pilar pra cada lado: o solo empurra a parte saliente da sapata pra cima, ela quer dobrar pra baixo, a face inferior estica (tração), a face superior comprime. Quem segura a tração é o aço, quem segura a compressão é o concreto.

#### Símbolos

- **`M_u`** (kN·m): momento fletor **atuante** na seção crítica (com cargas fatoradas — `u` de ultimate). Calcula-se integrando a pressão do solo sobre a área de sapata fora dessa seção, multiplicada pela distância ao "engastamento". A seção crítica fica na **face do pilar** (na pad simples) ou na **face do pilar e na face do degrau** (na escalonada).

- **`M_n`** (kN·m): momento **resistente nominal** = quanto a seção de concreto+aço aguenta sem romper, antes de aplicar fator de segurança.

- **`φ`** (adimensional): fator de redução; ≈ 0,90 pra flexão no ACI (a flexão é mais "gentil" que cisalhamento, daí φ maior).

- **`A_s`** (mm²/m): **área de aço por metro de largura** da sapata. Por que **por metro**? Porque a sapata não é uma "viga" pontual: ela é uma laje (placa) e a armadura é distribuída. Você expressa "quantos mm² de aço passam por cada metro perpendicular".

- **`f_y`** (MPa): **tensão de escoamento** do aço — a tensão a partir da qual o aço **deixa de voltar pra forma original** (entra em deformação plástica). Aço CA-50 brasileiro: `f_y = 500 MPa`. ASTM Grade 60 do paper: `f_y = 415 MPa`.

- **`a`** (mm): **profundidade do bloco retangular equivalente** (modelo de Whitney). Em palavras simples: do topo do concreto pra baixo, há uma faixa de altura `a` onde supõe-se que **todo o concreto** está comprimido a `0,85·f_c`. Abaixo dessa faixa o concreto está sem tensão (a tração fica por conta do aço).

- **`(d_e − a/2)`**: o **braço de alavanca interno** entre o aço (puxando, na altura `d_e` desde o topo) e o concreto (empurrando, no centro do bloco de Whitney, na altura `a/2` desde o topo). A distância entre os dois é `d_e − a/2`.

#### Como essa fórmula sai (sem matemática chata)

A peça está em equilíbrio. Significa que a soma das forças = 0 e a soma dos momentos = 0.

1. **Soma de forças horizontais = 0**: força puxando do aço = força empurrando do concreto.
   - Força no aço: `T = A_s · f_y` (área × tensão).
   - Força no concreto: `C = 0,85 · f_c · b · a` (tensão equivalente × área de bloco).
   - Igualando: `a = A_s · f_y / (0,85 · f_c · b)`. Daí sai a fórmula do `a`.

2. **Soma de momentos**: o momento que a peça aguenta é a força do par (T = C) vezes o braço entre eles:
   - `M_n = T · (d_e − a/2) = A_s · f_y · (d_e − a/2)`.

Aplicando o fator `φ`:

$$ \phi M_n = \phi \cdot A_s \cdot f_y \cdot \bigl(d_e - a/2\bigr) $$

> 💡 **Por que `0,85·f_c` e não `f_c`?** Porque a distribuição real de tensões no concreto comprimido **não é retangular** — ela tem forma de "parábola amassada". O modelo de Whitney aproxima essa forma por um retângulo equivalente que **dá o mesmo momento resultante**. Para conseguir essa equivalência, foi calibrado o fator `0,85` em centenas de ensaios.

#### Restrição final

A regra é simples: **`φM_n ≥ M_u`**. Se a armadura escolhida produz um `M_n` insuficiente, ou aumenta `A_s` (mais aço), ou aumenta `d_e` (sapata mais alta), ou aumenta `f_y` (aço mais resistente).

A intuição da régua: pense na sapata como uma régua engastada num lado. Sob carga, ela quer dobrar. A armadura na **face inferior** (lado tracionado) impede a régua de partir. `A_s · f_y` é a "força que o aço aguenta puxando", e `(d_e − a/2)` é o braço dessa força até o centro de compressão.

> 🔗 **No FundaIA**: **não implementado**.
>
> O FundaIA não calcula armadura nem verifica flexão. Hoje ele assume que o usuário escolherá a armadura num passo posterior, e foca só em **dimensionar o bloco de concreto**. Essa é uma das **lacunas** apontadas pela auditoria do vault ([[12_Auditoria/Auditoria 2026-04-27 - Vault vs Projeto]]).
>
> ⭐ Adicionar a checagem de flexão e a armadura como variável de otimização é o caminho **natural** para o FundaIA evoluir de "minimizar volume" para "minimizar custo" no espírito do Waheed 2025.

---

### 4.4 — Espaçamento e comprimento de ancoragem (constraints g22-g25, sem fórmula numerada)

#### Por que isso importa? (intuição antes da fórmula)

Imagine que você tem barras de aço **espalhadas** dentro do concreto, formando uma malha. Três coisas precisam ser garantidas pra que a malha funcione:

1. **As barras não podem estar muito juntas** — se ficarem coladas, o pedrisco do concreto **não consegue passar entre elas** durante a concretagem, e fica um buraco. Daí a regra `S_min` (espaçamento mínimo): tipicamente 1 a 1,5 × o tamanho do pedrisco máximo, ou ≈ 5–10 cm.

2. **As barras não podem estar muito separadas** — se ficarem com "buracos" de 50 cm entre elas, regiões inteiras da sapata ficam sem aço, e a tração não tem quem segure. Daí a regra `S_max` (espaçamento máximo): tipicamente 25 cm pra sapatas, ou 2× a espessura da peça.

3. **As barras precisam ficar bem "agarradas" ao concreto** — se a barra é puxada pela força de tração da flexão, ela tem que conseguir transmitir essa força pro concreto **gradualmente** ao longo do seu comprimento, em vez de simplesmente sair deslizando como espaguete. O comprimento mínimo pra isso acontecer é o **comprimento de ancoragem `l_db`**.

#### Comprimento de ancoragem

A fórmula do ACI 318-14 §25.4 (simplificada, com `ψ_t = ψ_e = 1`) é:

$$ l_d = \frac{f_y \, \psi_t \, \psi_e}{1{,}1 \, \lambda \sqrt{f_c}} \cdot \frac{d_b}{c_b/d_b + K_{tr}/d_b} $$

Termo a termo:

- `f_y` (MPa): aço mais resistente exige ancorar mais comprimento (porque a barra precisa "transferir" mais força ao concreto).
- `√f_c`: concreto mais forte agarra melhor → ancoragem mais curta.
- `ψ_t`: fator pra barras horizontais altas (concreto fica menos compactado embaixo). Em sapatas: `ψ_t = 1`.
- `ψ_e`: fator pra barras epóxi-revestidas (menos aderência). Em sapatas comuns: `ψ_e = 1`.
- `λ`: fator pro tipo de concreto (1 pra normal).
- `d_b`: diâmetro da barra. Barras mais grossas exigem mais comprimento de ancoragem.
- `c_b/d_b + K_{tr}/d_b`: termo de "confinamento" (estribos ajudam a ancorar). Em sapatas geralmente é 2,5.

#### As restrições g22 e g23 do paper

`l_db ≤ L/2 − Cov_f − d_b` e `l_db ≤ B/2 − Cov_f − d_b`.

Em palavras: **a metade da sapata, descontado o cobrimento e o diâmetro da barra, precisa ter pelo menos o comprimento de ancoragem `l_db`**. Por que metade? Porque o ponto crítico de tração é a face do pilar, e a barra "ancora" indo dela pra borda da sapata. Essa metade precisa ser comprida o bastante pra a barra "agarrar" antes de chegar na borda.

> 🔗 **No FundaIA**: **não implementado** (segue a mesma lógica da flexão acima — como o FundaIA não calcula armadura, não checa ancoragem).

---

### 4.5 — Eq. (13): Função objetivo de **custo**

[Waheed 2025, Eq. 13]:

$$ f(\text{cost}) = C_s \cdot W_{st} + C_c \cdot V_c + C_f \cdot A_f $$

#### O que é "função objetivo"?

**Função objetivo** (FO) é o número que o algoritmo de otimização tenta **minimizar** (ou maximizar). É o "placar" que decide se um projeto é melhor que outro.

No FundaIA, a FO é o **volume total de concreto** somado em todas as sapatas, **mais penalidades** pra projetos que violam alguma restrição. Quanto menor o número, melhor o projeto. No Waheed 2025, a FO é o **custo total**.

#### Tradução em palavras

O custo total da sapata é a soma do custo das três quantidades que você compra na obra:

- **`W_st`** (kg) — peso de aço (steel weight). `C_s` (Rs/kg) é o custo unitário do aço.
- **`V_c`** (m³) — volume de concreto. `C_c` (Rs/m³) é o custo unitário.
- **`A_f`** (m²) — área de **fôrmas** (formwork): as "caixas" de madeira ou metal que dão forma ao concreto enquanto ele cura (em geral 7 a 28 dias). Depois que o concreto endurece, as fôrmas são removidas. `C_f` (Rs/m²) é o custo unitário.

> 💡 **Por que fôrma é cara**: ela tem que ser **estanque** (não vazar concreto), **resistente** (suporta a pressão da massa de concreto fresco) e **lisa** (a face do concreto que sai dela é a face visível). Em sapatas, é menos crítica que em vigas/pilares, mas ainda assim entra no custo.

Os exemplos do paper [Waheed 2025, Tabela 3] usam:
- `C_c = 7460–7856 Rs/m³` (Rs = rúpia paquistanesa; ≈ 25 USD/m³),
- `C_s = 120 Rs/kg` (≈ 0,40 USD/kg),
- `C_f` não detalhado para o stepped (assumido com base no perímetro).

#### Por que essa fórmula é uma simplificação?

Numa obra real, o custo de uma sapata inclui também:
- **Mão de obra** (escavação, ferragem, concretagem),
- **Equipamento** (escavadeira, bomba de concreto),
- **Transporte** (concreto de usina até a obra),
- **BDI** (impostos, lucro, riscos).

A fórmula do Waheed 2025 captura só os **três insumos diretos** (concreto, aço, fôrma). Isso é razoável pra **comparar projetos entre si** (porque os outros custos são proporcionais ao tamanho do projeto), mas **subestima o valor absoluto**.

> 🔗 **No FundaIA**: **objetivo é volume**, não custo:
>
> ```python
> # núcleo da função objetivo
> obj = sum( h_x * h_y * h_z   for cada sapata )  +  penalidades
> ```
>
> Isso é uma simplificação: como `V_c` domina em sapatas comuns (`C_c · V_c` ≫ `C_s · W_st` na maioria dos casos), minimizar volume **se aproxima** de minimizar custo, mas **não é a mesma coisa**.
>
> Curiosidade do Waheed 2025: ele mostra que **minimizar volume e minimizar energia incorporada andam juntos** (ver §4.6 abaixo), o que é um argumento a favor do volume como proxy razoável.

---

### 4.6 — Eq. (14): Função objetivo de custo "truncado" (sem fôrma)

[Waheed 2025, Eq. 14]:

$$ f_{cost} = C_s \cdot W_{st} + C_c \cdot V_c $$

Igual à Eq. (13) mas sem o termo `C_f · A_f`. O paper passa a usar essa versão pra comparar de forma justa com as fórmulas de carbono e energia (que também só têm dois termos). É uma escolha metodológica pra que as três funções objetivo sejam **estruturalmente comparáveis** [Waheed 2025, Sec. "Formulation of optimization problem for eco-efficient design"].

---

### 4.7 — Eq. (15): Função objetivo de **emissões de carbono**

[Waheed 2025, Eq. 15]:

$$ f_{emissions} = e_s \cdot W_{st} + e_c \cdot V_c $$

#### O que é "emissão de carbono incorporada"?

Antes mesmo de a obra existir, o **cimento** que vira concreto já foi fabricado numa fábrica de cimento que emitiu CO₂ pra atmosfera. As barras de aço já saíram de um alto-forno que queimou carvão. Toda essa "história ambiental" do material é chamada de **carbono incorporado** (embodied carbon).

A unidade típica é **kg CO₂ por kg de material** (pra aço) ou **kg CO₂ por m³ de material** (pra concreto). Os números do paper vêm da base **ICE V2.0** (Inventory of Carbon and Energy), uma referência reconhecida internacionalmente [Hammond et al. 2011].

- **`e_s` = 1,99 kg CO₂/kg de aço**: cada kg de barra de aço CA-50/Grade 60 "carrega" 1,99 kg de CO₂ emitido durante sua fabricação (alto-forno, laminação, transporte).
- **`e_c`** (kg CO₂ / m³): emissão por m³ de concreto. **Depende da resistência** porque concreto mais forte tem mais cimento, e o cimento é o vilão (a fabricação de cimento Portland produz ≈ 0,9 kg de CO₂ por kg de cimento). Tabela do paper:

| f_c (MPa) | Emissão (kg CO₂/m³) | Energia incorporada (MJ/m³) |
| --- | --- | --- |
| 20 | 240 | 1680 |
| 25 | 256,8 | 1776 |
| 30 | 271,2 | 1872 |
| 35 | 288 | 1968 |
| 40 | 316,8 | 2112 |
| 50 | 362,4 | 2400 |

A intuição: cada metro cúbico de concreto tem "história ambiental" — dá pra rastrear a tonelada de CO₂ que foi emitida pra fabricar o cimento, transportar, etc. O aço também (alto-forno emite muito CO₂).

---

### 4.8 — Eq. (16): Função objetivo de **energia incorporada**

[Waheed 2025, Eq. 16]:

$$ f_{energy} = E_s \cdot W_{st} + E_c \cdot V_c $$

#### O que é "energia incorporada"?

É **toda a energia gasta** pra produzir um material — desde extrair a matéria-prima do solo, processar, transportar, até deixar o material pronto na obra. Inclui energia elétrica, queima de combustíveis fósseis, etc.

A unidade é **MJ (megajoule) por kg ou m³**. Pra ter referência: 1 kWh = 3,6 MJ. Então 35,4 MJ/kg de aço significa que cada kg de aço "consumiu" ≈ 9,8 kWh de energia na fabricação.

- **`E_s` = 35,4 MJ/kg de aço**: alto-forno, laminação a quente, transporte.
- **`E_c`** (MJ/m³): energia incorporada do concreto, **depende da resistência** (concreto mais forte = mais cimento = mais energia gasta na clinquerização). Mesma tabela do §4.7.

**A descoberta interessante** do Waheed 2025 [§Results and discussion, Sec. eco-efficient]:

> Otimizar pelo objetivo **energia incorporada** dá resultados **tão bons quanto custo** e **melhores que carbono** — porque no concreto a energia e a massa estão muito correlacionadas, e isso evita que o algoritmo "trapaceie" usando muito concreto barato em CO₂ pra economizar aço.

Concretamente:
- Otimizar carbono → mais aço e mais concreto → mais caro e mais energia.
- Otimizar custo → equilíbrio bom em todos os 3 indicadores.
- Otimizar energia → equilíbrio bom em todos os 3 indicadores (até melhor que custo em alguns casos).

> ⭐ Pro FundaIA: isso é um insight **importante**. Se um dia você quiser fazer multi-objetivo ou trocar a função objetivo, **energia incorporada é uma escolha defensável** — combina ganho ambiental e econômico simultaneamente. Hoje o volume mínimo do FundaIA está mais próximo de "energia incorporada" do que de "carbono", já que volume mínimo → menos concreto → menos energia.

---

### 4.9 — Tabela 2: as 25 restrições `g₁(x)` a `g₂₅(x)`

A tabela completa do paper consolida tudo o que vimos. Resumindo em grupos:

| Grupo | Restrições | O que verifica |
|---|---|---|
| Punção + cisalhamento excêntrico | g1, g2 | `V'_u + V''_u ≤ φV_c` em `d_{e1}/2` (face do pilar) e `d_{e2}/2` (face do degrau) |
| One-way shear | g3, g4, g5, g6 | `V_u ≤ φV_c` em `d_{e1}` e `d_{e2}` ao longo de L e B |
| Flexão | g7, g8, g9, g10 | `M_u ≤ φM_n` na face do pilar e face do degrau, em ambas as direções |
| Geometria do degrau | g11, g12, g13, g14 | A altura útil cabe dentro do degrau (não invade a fronteira) |
| Tensão no solo | g16 | `q_max ≤ q_net` |
| Área da sapata | g17 | `A_req ≤ A_pro` |
| Excentricidade | g18, g19 | `e_L ≤ L/6`, `e_B ≤ B/6` |
| Limites de armadura | g20, g21 | `A_{s,min} ≤ A_s ≤ A_{s,max}` |
| Ancoragem | g22, g23 | `l_db ≤ L/2 − cob − d_b` e idem em B |
| Espaçamento | g24, g25 | `S_min ≤ S ≤ S_max` |

Note que **não existe `g15`** — é um lapso do paper, segundo a numeração da Tabela 2. Isso é normal em papers; a numeração não é "auditada" pelo Springer.

> 🔗 **No FundaIA**, em [[03_Otimizacao/Penalização de Restrições]] e `core/api/evaluate.py`:
>
> | Restrição FundaIA | Equivalente Waheed 2025 |
> | --- | --- |
> | `g_tensao` (σ_max ≤ σ_adm) | g16 |
> | `g_min_inferior` (σ_min ≥ 0) | parte de g18/g19 implícito |
> | `g_punção` (perímetro C) | g1 (parcial — só esmagamento, não tração diagonal) |
> | `g_geometria_x`, `g_geometria_y` (balanço mínimo) | sem equivalente direto (regra prática brasileira) |
> | `g_packing` (não-sobreposição) | sem equivalente (Waheed otimiza 1 sapata por vez) |
>
> ⚠ **Lacunas**: o FundaIA **não checa** flexão (g7-g10), one-way shear (g3-g6), ancoragem (g22-g23), espaçamento (g24-g25), nem limites de armadura (g20-g21) — porque o objetivo é só volume, não dimensiona armadura.

---

## 5 · O algoritmo do paper vs. o do FundaIA

### 5.1 — O que o Waheed 2025 usa

[Waheed 2025, Methodology]: **Algoritmo Genético (GA)** implementado pelo add-in **Evolver** do Excel, com:
- Tamanho da população: 50,
- Taxa de mutação: 0,075 (0,1 na fase eco-eficiente),
- Taxa de crossover: 0,5,
- Critério de parada: 20.000 trials.

Roda 2 vezes por exemplo (2 runs) e fica com o melhor.

### 5.2 — O que o FundaIA usa

[[03_Otimizacao/EGO - Efficient Global Optimization]] + [[03_Otimizacao/Gaussian Process Regressor]] + [[03_Otimizacao/Algoritmo Genético]]:

```
para cada repetição (n_rep, default 5):
    1. LHS gera n_pop pontos iniciais e avalia a FO real
    2. para iter = 1..n_gen:
        a. Treina GPR (surrogate) na história
        b. Maximiza Expected Improvement com GA interno
        c. Avalia o novo ponto na FO real
        d. Adiciona à história
    3. retorna o melhor da rep
retorna o melhor entre todas as reps
```

> ⭐ **Diferença metodológica fundamental**: o Waheed 2025 usa **GA puro** (Evolver), enquanto o FundaIA usa **EGO + GPR + GA interno** — isso é **mais sofisticado** do ponto de vista de eficiência amostral (gasta menos avaliações reais da FO pra chegar perto do ótimo), o que justamente é a contribuição do paper Jones, Schonlau & Welch (1998) que está em [[08_Artigos/Jones et al. 1998 - Efficient Global Optimization]].

> 📚 **Comparação justa**: pra publicar artigo a partir do FundaIA, um experimento natural é rodar **o mesmo problema** com:
> 1. GA puro (estilo Waheed 2025),
> 2. GWO puro (estilo Gandomi & Kashani 2018),
> 3. EGO+GPR+GA (FundaIA),
>
> e comparar **número de avaliações da FO até atingir 99% do ótimo**. Esse seria um experimento computacional limpo, defensável academicamente.

---

## 6 · O que esse paper traz pro FundaIA?

Síntese das contribuições do Waheed 2025 mapeadas pro estado atual do projeto:

### 6.1 — Validação metodológica
✅ Confirma que **GA + Excel** funciona como ferramenta prática — mas **não é o estado da arte**. O FundaIA, ao usar EGO+GPR+GA, está um passo à frente em termos de eficiência amostral.

### 6.2 — Argumento ambiental forte
⭐ **Energia incorporada > emissões de carbono** como objetivo eco-eficiente. Isso é diretamente útil pro discurso do FundaIA: minimizar volume é uma boa proxy de minimizar energia incorporada (porque concreto é o termo dominante). Você tem **base bibliográfica recente** (2025) pra justificar a escolha do volume como FO no artigo.

### 6.3 — Caminhos de evolução do FundaIA

| Sugestão (do paper) | Status no FundaIA hoje | Custo de implementação |
|---|---|---|
| Sapatas escalonadas (variáveis x, y, d₂) | ❌ não tem | Médio: precisa repensar a função objetivo e adicionar 3 variáveis por sapata |
| Armadura como variável (A_sL, A_sB) | ❌ não tem | Alto: precisa implementar verificações de flexão, ancoragem, espaçamento |
| Função objetivo de custo (Eq. 13/14) | ❌ usa volume | Baixo: trivial se já tiver armadura |
| Função objetivo de carbono / energia | ❌ não tem | Baixo: também trivial após armadura |
| Múltiplos pilares / packing | ✅ FundaIA tem (Waheed 2025 não) | já implementado |
| Surrogate (GPR) e EGO | ✅ FundaIA tem (Waheed 2025 não) | já implementado |
| Punção C' (perímetro a 2d) | ✅ implementado na Sprint 5.2 | Manter testes e documentação alinhados |

### 6.4 — Limitações importantes do paper (não esconda)

- O Waheed 2025 usa **ACI 318-14**, não NBR 6118 — então constantes e fatores são diferentes.
- O paper trata **uma sapata** por vez, sem layout de obra → não trata a sobreposição entre sapatas.
- A "Evolver" do Excel é uma caixa-preta comercial — repetibilidade científica mais limitada que código aberto (mealpy/scipy do FundaIA).
- O paper usa **moeda PKR** (rúpias paquistanesas) e custos locais; números absolutos não comparam diretamente com Brasil. As **proporções** (% de redução) é o que vale.

---

## 7 · O que ainda **não foi visto** e seria saudável ler depois

### 7.1 — Pra entender melhor a **estrutura** do problema

- [[08_Artigos/Wang e Kulhawy 2008 - Economic Design Optimization of Foundations]] — paper-paradigma da formulação "fundação como problema de otimização".
- [[08_Artigos/Gandomi e Kashani 2018 - Cost Minimization Shallow Foundation]] — comparação entre 7 metaheurísticas pra minimizar custo. Bom pra ver "o quanto o algoritmo importa".
- [[08_Artigos/Kashani et al. 2020 - Optimum Design of Shallow Foundation]] — variação do anterior, mais recente.

### 7.2 — Pra entender o algoritmo (EGO + GPR)

- [[08_Artigos/Jones et al. 1998 - Efficient Global Optimization]] — paper clássico do EGO.
- [[08_Artigos/Schulz et al. 2018 - Tutorial Gaussian Process Regression]] — tutorial didático de GPR (vale ouro pra você).
- [[08_Artigos/Williams e Rasmussen - Gaussian Processes for Regression]] — capítulo do livro-referência.
- [[08_Artigos/Shahriari et al. 2016 - Review Bayesian Optimization]] — review extensa.
- [[08_Artigos/Snoek et al. 2012 - Practical Bayesian Optimization]] — aspectos práticos.

### 7.3 — Pra entender o lado **geotécnico** (capacidade de carga do solo)

- [[08_Artigos/NBR 6122 1996 - Projeto e Execucao de Fundacoes]] — norma brasileira de fundações (ainda não substituída por edição mais recente desse vault — ler com cuidado de data).
- [[08_Artigos/Ahmad et al. 2021 - GPR Bearing Capacity Shallow Foundations]] — usa GPR pra estimar capacidade de carga (linha similar à do FundaIA mas em outra etapa do pipeline).
- [[08_Artigos/Bezerra et al. 2024 - Elementos de Fundacao]] — visão geral didática em português.

### 7.4 — Pra futuras frentes (não-prioritário agora)

- [[08_Artigos/Mbock et al. 2019 - Optimal Forms Shallow Foundations]] — formas alternativas (T, escalonadas) — relevante se quiser implementar stepped no FundaIA.
- [[08_Artigos/Juang e Wang 2013 - Reliability Robust Spread Foundations]] — projeto **com incerteza** (probabilístico). É uma frente de evolução possível.
- [[11_Frentes_de_Pesquisa/Physics-Informed Surrogates]] — frente declarada de interesse no vault.

---

## 8 · Glossário rápido (símbolos do paper)

| Símbolo | Significado | Unidade |
|---|---|---|
| `L`, `B` | Comprimento e largura da sapata | m |
| `x`, `y` | Posição do degrau (até onde a sapata é alta) | m |
| `c1`, `c2` | Lados do pilar | mm |
| `d₁`, `d₂` | Altura total da sapata na face do pilar e do degrau | mm |
| `d_{e1}`, `d_{e2}` | Altura útil (descontado cobrimento + meia barra) | mm |
| `A_sL`, `A_sB` | Área de aço por metro ao longo de L e B | mm²/m |
| `f_c` (`f_ck` na NBR) | Resistência do concreto | MPa |
| `f_y` | Tensão de escoamento do aço | MPa |
| `P_D`, `P_L` | Cargas de dead load (peso próprio) e live load (uso) | kN |
| `M_B`, `M_L` | Momentos em torno dos eixos perpendiculares à largura e ao comprimento | kN·m |
| `q_max`, `q_min` | Pressão máx. e mín. sob a base | kPa |
| `q_net` | Pressão admissível líquida | kPa |
| `e_L`, `e_B` | Excentricidades | m |
| `V_u` | Cisalhamento atuante | kN |
| `V_n` | Cisalhamento resistente nominal | kN |
| `b_o` | Perímetro crítico de punção | m |
| `β` | Razão dos lados do pilar | — |
| `α_s` | Constante de posição do pilar | — |
| `M_u`, `M_n` | Momento atuante e resistente nominal | kN·m |
| `a` | Profundidade do bloco equivalente de Whitney | mm |
| `φ` | Fator de redução de capacidade (segurança) | — |
| `λ` | Fator do tipo de concreto (1 pra normal) | — |
| `Cov_f` | Cobrimento de concreto | mm |
| `d_b` | Diâmetro da barra | mm |
| `l_db` | Comprimento de ancoragem | mm |
| `S_min`, `S_max` | Espaçamento mínimo e máximo | mm |
| `V_c`, `W_st`, `A_f` | Volume de concreto, peso de aço, área de fôrma | m³, kg, m² |
| `C_c`, `C_s`, `C_f` | Custos unitários | Rs/m³, Rs/kg, Rs/m² |
| `e_c`, `e_s` | Emissões de CO₂ unitárias | kg CO₂/m³, kg CO₂/kg |
| `E_c`, `E_s` | Energia incorporada unitária | MJ/m³, MJ/kg |

---

## 9 · Próximos passos sugeridos (caminho de aprendizado)

Se você quiser **aprender de fato** o que o seu projeto faz, sugiro esta ordem:

1. ✅ **Este guia** (você está aqui).
2. Leia [[02_Engenharia/Sapatas Isoladas]] e [[02_Engenharia/Tensão Admissível do Solo]] (curtas, ancoram o vocabulário).
3. Leia [[02_Engenharia/Flexão Composta - Sigma Max e Min]] e [[02_Engenharia/Verificação à Punção]] — são as duas verificações que **o FundaIA faz hoje**.
4. Abra `core/engineering/tensao.py` e `core/engineering/puncao.py` e leia o código com este guia ao lado.
5. Leia o **abstract + introdução + conclusão** do Waheed 2025 (15 minutos).
6. Leia o paper [Schulz et al. 2018] sobre GPR — é tutorial e didático.
7. Leia o paper [Jones et al. 1998] sobre EGO — é a base do algoritmo.
8. Leia, agora, o **Waheed 2025 inteiro** — vai ser muito mais fácil com tudo isso na cabeça.

---

## 10 · Referências completas

### Fonte primária (foco deste guia)

- **Waheed, J., Azam, R., Riaz, M.R., Shakeel, M.** (2025). *Optimization-based innovative approach for economical design of reinforced concrete isolated footings*. Innovative Infrastructure Solutions 10:56. DOI: 10.1007/s41062-024-01823-9.
  - PDF: `docs/articles/01_artigo_1_ego_gpr/2025_waheed_et_al_economical_design_rc_isolated_footings.pdf`
  - Ficha: [[08_Artigos/Waheed et al. 2025 - Economical Design RC Isolated Footings]]

### Fontes complementares utilizadas neste guia

- **Waheed et al. (2022)**. *Metaheuristic-Based Practical Tool for Optimal Design of Reinforced Concrete Isolated Footings*. Buildings 12:471. DOI: 10.3390/buildings12040471.
  - Ficha: [[08_Artigos/Waheed et al. 2022 - Practical Tool RC Isolated Footings]]
- **Wang, Y. & Kulhawy, F.H. (2008)**. *Economic Design Optimization of Foundations*. J. Geotech. Geoenviron. Eng.
  - Ficha: [[08_Artigos/Wang e Kulhawy 2008 - Economic Design Optimization of Foundations]]
- **Gandomi, A.H. & Kashani, A.R. (2018)**. *Construction Cost Minimization of Shallow Foundation Using Recent Swarm Intelligence Techniques*. IEEE Trans. Industrial Informatics.
  - Ficha: [[08_Artigos/Gandomi e Kashani 2018 - Cost Minimization Shallow Foundation]]
- **Kashani, A.R. et al. (2020)**. *Optimum Design of Shallow Foundation*.
  - Ficha: [[08_Artigos/Kashani et al. 2020 - Optimum Design of Shallow Foundation]]
- **Bezerra, B.O., Santos Neto, E.F., Souza, D.S. (2024)**. *Elementos de fundação: indicação de aplicação partindo do pré-projeto*. Ciências Exatas e Tecnológicas.
  - Ficha: [[08_Artigos/Bezerra et al. 2024 - Elementos de Fundacao]]
- **Ahmad, M. et al. (2021)**. *GPR Bearing Capacity Shallow Foundations*.
  - Ficha: [[08_Artigos/Ahmad et al. 2021 - GPR Bearing Capacity Shallow Foundations]]
- **NBR 6118 (ABNT, 2026)**. *Projeto de estruturas de concreto — Procedimento*. (Item 19.5: Punção)
  - Ficha de apoio: [[02_Engenharia/NBR 6118]]
- **NBR 6122 (ABNT, 2022)**. *Projeto e execução de fundações — Emenda 1*.
  - Ficha: [[08_Artigos/NBR 6122 1996 - Projeto e Execucao de Fundacoes]]
- **ACI 318-14 (2014)**. *Building code requirements for structural concrete*. American Concrete Institute, Farmington Hills (citado pelo Waheed 2025; norma de referência do paper).

### Fundamentação metodológica (algoritmo do FundaIA, não usado pelo paper)

- **Jones, D.R., Schonlau, M., Welch, W.J. (1998)**. *Efficient Global Optimization of Expensive Black-Box Functions*.
  - Ficha: [[08_Artigos/Jones et al. 1998 - Efficient Global Optimization]]
- **Williams, C.K.I. & Rasmussen, C.E.** *Gaussian Processes for Regression*.
  - Ficha: [[08_Artigos/Williams e Rasmussen - Gaussian Processes for Regression]]
- **Schulz, E., Speekenbrink, M., Krause, A. (2018)**. *A tutorial on Gaussian process regression*.
  - Ficha: [[08_Artigos/Schulz et al. 2018 - Tutorial Gaussian Process Regression]]
- **Shahriari, B. et al. (2016)**. *Taking the Human Out of the Loop: A Review of Bayesian Optimization*.
  - Ficha: [[08_Artigos/Shahriari et al. 2016 - Review Bayesian Optimization]]

### Notas técnicas internas do vault

- [[02_Engenharia/Sapatas Isoladas]]
- [[02_Engenharia/NBR 6118]]
- [[02_Engenharia/Tensão Admissível do Solo]]
- [[02_Engenharia/Flexão Composta - Sigma Max e Min]]
- [[02_Engenharia/Verificação à Punção]]
- [[02_Engenharia/Restrição de Geometria]]
- [[02_Engenharia/SPT - Sondagem]]
- [[03_Otimizacao/Formulação do Problema]]
- [[03_Otimizacao/EGO - Efficient Global Optimization]]
- [[03_Otimizacao/Gaussian Process Regressor]]
- [[03_Otimizacao/Penalização de Restrições]]

---

> [!tip] Dica de estudo
> Você não precisa ler **tudo** de uma vez. O caminho mais útil é:
> 1. ler o **vocabulário fundamental** (§2.5) primeiro — sem isso, as fórmulas não fazem sentido,
> 2. ler **uma seção de fórmula** (§4.1 hoje, por exemplo),
> 3. abrir o **código correspondente** no FundaIA (`core/engineering/tensao.py`),
> 4. comparar **fórmula × implementação**,
> 5. anotar dúvidas no vault.
>
> Faça isso 4-5 vezes e você vai sair entendendo de verdade o domínio do seu próprio projeto. O guia foi escrito pra ser usado **junto com o código**, não no lugar dele.
