# Diagnóstico da formulação: sapata concêntrica e penalização dentro do GP

Documento de trabalho — análise dos dois problemas acoplados identificados na
formulação atual do FundaAI, com base no manuscrito em `paper/` (seções 2, 4 e 5).

---

## Resumo

Dois problemas independentes na aparência, mas que são o mesmo problema:

1. **A formulação concêntrica** (sapata sempre centrada no pilar) cria um domínio
   viável não-convexo e, em casos de proximidade, desconexo ou vazio.
2. **A penalização quadrática dentro do GP** impede o modelo substituto de
   enxergar esse domínio — e não existe valor de `α` que resolva.

Corrigir apenas um dos dois não resolve o caso de sobreposição.

---

## 1. O que "concêntrica" significa na formulação atual

As equações de vértice (`section2_theory.tex`, instanciadas em
`section5_application.tex:347`) definem:

```
x₁ = x_g − h_x/2      x₂ = x_g + h_x/2
y₁ = y_g − h_y/2      y₄ = y_g + h_y/2
```

O centro da sapata **é** o centro do pilar `(x_g, y_g)`. Sempre.

> **Consequência: a posição da sapata não é variável de projeto. Só a forma é.**

São 3 variáveis por elemento (`h_x`, `h_y`, `h_z`) e nenhuma delas move a sapata.
Para separar duas sapatas que colidem, o otimizador só pode **mudar o formato** —
nunca **afastá-las**.

```
CONCÊNTRICA (hoje)          COM EXCENTRICIDADE
   ┌─────┐                     ┌────┐
   │  ●  │ ← sapata presa      │ ●  │ ← pilar fora do centro
   └──┬──┘   no eixo do pilar  └────┘
   ┌──┴──┐                          ┌────┐
   │  ●  │   colidem                │  ● │
   └─────┘                          └────┘
   só resta deformar          dá para afastar
```

---

## 2. Por que isso quebra o caso P01–P02

Dados das Tabelas `dois_elemento` e `carga_2_elementos`:

| Elemento | a_p (m) | b_p (m) | SPT | x_g (m) | y_g (m) | F_z máx (kN) |
|---|---|---|---|---|---|---|
| P01 | 0,25 | 1,20 | 10 | 6,900 | 26,255 | 511,6 |
| P02 | 0,30 | 1,50 | 12 | 7,985 | 25,105 | 915,9 |

Distância entre eixos dos pilares:

- **Δx = 7,985 − 6,900 = 1,085 m**
- **Δy = 26,255 − 25,105 = 1,150 m**

### 2.1 A condição de não-sobreposição é uma disjunção

Dois retângulos concêntricos nos pilares não se sobrepõem se estiverem separados
em x **ou** em y:

```
Ramo A:  h_x1 + h_x2  ≤  2·Δx = 2,17 m
Ramo B:  h_y1 + h_y2  ≤  2·Δy = 2,30 m
```

Como é um **OU**, o domínio viável é a *união* de dois conjuntos — portanto
**não-convexo**. Isso já é ruim para um GP, que assume superfície suave e
estacionária. Mas piora.

### 2.2 O ramo B é vazio

Mínimo geométrico imposto pela restrição pilar–sapata (`h_y ≥ b_p + 2δ`, δ = 0,05):

- P01: `h_y ≥ 1,20 + 0,10 = 1,30 m`
- P02: `h_y ≥ 1,50 + 0,10 = 1,60 m`
- **Soma mínima possível = 2,90 m > 2,30 m**

**Não existe solução separada em y**, nem espremendo as duas sapatas até o limite
construtivo. O ramo B é infactível por construção.

### 2.3 O ramo A é uma fatia fina

Sobra o ramo A: `h_x1 + h_x2 ≤ 2,17 m`, ou seja, sapatas com ~1,0 m de largura
cada. Mas a capacidade do solo exige área.

Com `σ_adm = N/50 · 1000`:

- P01: N = 10 → σ_adm = 200 kPa → `A ≥ 1,30 · 1,05 · 511,6 / 200 ≈ 3,5 m²`
- P02: N = 12 → σ_adm = 240 kPa → `A ≥ 1,30 · 1,05 · 915,9 / 240 ≈ 5,2 m²`

(sem contar os momentos, que agravam)

Com `h_x ≈ 1,0 m`, isso força **h_y ≈ 3,5 a 4,5 m**.

> A única região viável é um par de sapatas **muito alongadas em y e estreitas em x**
> — uma fatia fina, anisotrópica, e distante de qualquer coisa que um DoE por
> Latin Hypercube vá amostrar.

### 2.4 Verificação: o modelo bate com o resultado publicado

Com a solução publicada (`tab:dimensoes_2_fundacoes`: 2,99 × 1,40 e 3,00 × 1,90):

```
overlap_x = (2,99 + 3,00)/2 − 1,085 = 1,910 m
overlap_y = (1,40 + 1,90)/2 − 1,150 = 0,500 m
A_overlap = 1,910 × 0,500        = 0,955 m²

g_P01 = 0,955 / (2,99 × 1,40) = 0,955 / 4,186 = 0,228
g_P02 = 0,955 / (3,00 × 1,90) = 0,955 / 5,700 = 0,168
```

A tabela `resultado_g_2_fundacoes` reporta **0,23** e **0,18**. Bate exatamente —
o que confirma que o diagnóstico acima descreve a formulação real, não uma
hipótese.

---

## 3. A correção da formulação: excentricidade

Introduzir `e_x`, `e_y` como variáveis de projeto (sapata deslocada em relação ao
pilar). O ramo A passa a ser:

```
h_x1/2 + h_x2/2 − (e_x2 − e_x1)  ≤  Δx
```

Você compra separação **sem sacrificar área**. O domínio viável deixa de ser uma
fatia e vira um volume conexo.

O preço físico é honesto e as equações já existem no manuscrito: a excentricidade
gera momento adicional `M = F_z · e`, que entra direto nas Eq. de `σ_max` / `σ_min`
(`section2_theory.tex`, Eq. `eq:sigma_max` e `eq:sigma_min`).

É sapata excêntrica / de divisa — prática corrente de escritório, não invenção.

**Efeito colateral positivo:** o problema deixa de ser "3n variáveis box-constrained"
e vira **5n variáveis com acoplamento espacial** — um problema de layout/packing
genuíno.

---

## 4. A armadilha da penalidade baixa

Reduzir `α` faz o **ajuste do GP** funcionar. Não faz a **otimização** funcionar.

Conta com `α = 10` na própria solução publicada
(`section4_method.tex:12`, `P = α · Σ max(0, g)²`):

| Violação | g | Custo `10·g²` | Volume do elemento | Custo relativo |
|---|---|---|---|---|
| Sobreposição P01 | 0,23 | 0,529 | 2,55 m³ | 21 % |
| Sobreposição P02 | 0,18 | 0,324 | 3,48 m³ | 9 % |
| **Tensão no solo P02 (c2)** | **0,04** | **0,016** | 3,48 m³ | **0,5 %** |

> **Estourar a capacidade de carga do solo em 4% custa 0,016** numa função
> objetivo da ordem de 6 m³.

O otimizador vai violar a tensão admissível toda vez que isso lhe render mais de
~0,3 % de volume — e vai fazer isso **corretamente**, porque é literalmente o que
a função objetivo manda fazer.

**O algoritmo não errou. A função objetivo é que diz que segurança geotécnica
vale 0,016 m³.**

---

## 5. O diagnóstico real: não existe α bom

Este é o ponto central, e é a razão de o assunto ser publicável.

| α | Efeito no GP | Efeito na otimização |
|---|---|---|
| **Alto** (10⁶) | Escala explode (~10⁶ m³), *kink* em g=0, comportamento não-estacionário — chapado dentro da região viável, quadrático fora. GP assume suavidade e escala de comprimento única → **R² = 0,69** | Restrições respeitadas |
| **Baixo** (10¹) | Superfície suave, GP ajusta bem → **R² = 0,99** | **Região viável não é respeitada** — violações da Tabela `resultado_g_2_fundacoes` |

**Não existe α que resolva os dois.** É um trade-off estrutural entre
*"surrogate preciso"* e *"ótimo viável"*.

Reduzir `α` não foi achar um hiperparâmetro bom — foi andar até a outra ponta do
mesmo trade-off. É exatamente por isso que a literatura de otimização bayesiana
com restrições existe.

Note também que a interpretação atual no manuscrito
(`section5_application.tex:20`) apresenta a escolha de `α = 10¹` como uma
*descoberta sobre hiperparâmetros*. Na verdade é o diagnóstico de um defeito da
formulação. Precisa ser reescrita.

---

## 6. A saída: Constrained EI

Parar de colocar a penalidade **dentro** do GP:

1. Um GP para o **volume** `V(X)` — suave, é literalmente um produto de variáveis.
2. Um GP para **cada restrição** `g_j(X)` separadamente — cada uma suave.
3. Trocar EI por **Constrained EI (EIC)**:

$$
\mathrm{EIC}(\mathbf{x}) = \mathrm{EI}(\mathbf{x}) \cdot
\prod_j \Phi\!\left(\frac{-\mu_{g_j}(\mathbf{x})}{\sigma_{g_j}(\mathbf{x})}\right)
$$

O segundo termo é a **probabilidade de viabilidade**. Se o modelo acha que a
restrição vai ser violada, a aquisição vai a zero — não importa quão bom seja o
volume.

> **E o `α` desaparece: não há mais hiperparâmetro para calibrar.**

### Custo de implementação

Treinar `n_restrições + 1` GPs em vez de 1, e multiplicar o EI por um produto de
CDFs normais. Mudança pequena e localizada no código.

### Referências

- **Schonlau, M., Welch, W. J., Jones, D. R. (1998).** *Global versus local search
  in constrained optimization of computer models.* — EI com restrições, canônico.
- **Gardner, J. R. et al. (2014).** *Bayesian optimization with inequality
  constraints.* ICML — versão moderna do EIC.
- **Gelbart, M. A., Snoek, J., Adams, R. P. (2014).** *Bayesian optimization with
  unknown constraints.* UAI.
- **Gramacy, R. B. et al. (2016).** *Modeling an augmented Lagrangian for blackbox
  constrained optimization.* Technometrics — ALBO, muito citado em SMO.
- **Parr, J. M. et al. (2012).** *Infill sampling criteria for surrogate-based
  optimization with constraint handling.* Engineering Optimization — comparação de
  estratégias.

---

## 7. Plano de verificação sugerido

Rodar o caso P01–P02 nas quatro combinações, para separar a contribuição de cada
correção:

| # | Formulação | Tratamento de restrição | Hipótese |
|---|---|---|---|
| 1 | Concêntrica | Penalidade α = 10¹ | Estado atual — converge inviável |
| 2 | Concêntrica | Penalidade α = 10⁶ | GP degrada, mas viável (ou não converge) |
| 3 | Concêntrica | **EIC** | Deve achar a fatia alongada, ou provar infactibilidade |
| 4 | **Com e_x, e_y** | **EIC** | Deve achar solução viável e econômica |

Isso mostra, com evidência própria, que os dois problemas são independentes e
ambos necessários — que é exatamente o argumento de um artigo.
