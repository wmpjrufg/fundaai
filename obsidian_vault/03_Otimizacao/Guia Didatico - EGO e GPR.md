---
tags: [otimizacao, ego, gpr, bayesian-optimization, guia, didatico]
aliases: [Guia EGO, Guia GPR, Guia Otimização Bayesiana]
data: 2026-04-29
fontes:
  - SMT v2.9.3 — EGO documentation (smt.readthedocs.io)
  - Jones, Schonlau & Welch (1998) — Efficient Global Optimization
  - Williams & Rasmussen — Gaussian Processes for Regression
  - Schulz, Speekenbrink & Krause (2018) — Tutorial on GPR
  - Shahriari et al. (2016) — Review of Bayesian Optimization
  - Snoek, Larochelle & Adams (2012) — Practical Bayesian Optimization
  - Mockus (1975) — On Bayesian methods for seeking the extremum
  - Ginsbourger, Le Riche & Carraro (2010) — Parallel Kriging
---

# Guia Didático — EGO e GPR (Otimização Bayesiana)

> Este guia foi escrito para alguém que **não é estatístico** (você, Lucas). Ele explica de forma didática o **Gaussian Process Regression (GPR)** e o algoritmo **Efficient Global Optimization (EGO)** — os dois pilares do FundaIA — usando como núcleo a documentação oficial da biblioteca [SMT v2.9.3](https://smt.readthedocs.io/en/v2.9.3/_src_docs/applications/ego.html), cruzada com os papers fundadores (Jones 1998, Mockus 1975) e tutoriais didáticos (Schulz 2018, Shahriari 2016).
>
> Para cada conceito, o guia mostra:
> 1. **A intuição** (em palavras simples, com analogias),
> 2. **A fórmula** (LaTeX),
> 3. **Cada termo explicado**,
> 4. **Como o FundaIA usa isso na prática**.

---

## 0 · O que esse guia entrega

Ao final, você vai entender:

- **O que é uma "função objetivo cara"** e por que ela precisa de tratamento especial.
- **O que é um Gaussian Process** sem precisar saber o que é uma matriz de covariância (mas vamos chegar lá no §3).
- **Por que ele dá uma incerteza junto com a previsão**, e por que isso é mágico.
- **O que é Expected Improvement** e por que essa fórmula em particular (e não outra) — exploit × explore.
- **Como o EGO usa tudo isso pra minimizar** sem precisar avaliar a função muitas vezes.
- **As 3 funções de aquisição** (EI, SBO, LCB) e quando usar cada uma.
- **A versão paralela (qEI)** e os 4 truques de "mentira" pra sugerir vários pontos ao mesmo tempo.
- **Onde tudo isso aparece no código do FundaIA**.

---

## 1 · O problema que motiva tudo: "função objetivo cara"

### 1.1 — Função objetivo cara, em palavras

Imagine que você tem uma função `f(x)` que recebe alguns parâmetros (dimensões de uma sapata, hiperparâmetros de uma rede neural, configuração de um experimento de engenharia) e devolve um número (custo, erro, perda).

Você quer **achar o `x` que minimiza `f(x)`**. Em otimização clássica (gradiente descendente, etc.) você precisa avaliar `f` muitas vezes — milhares ou milhões. Mas existem cenários onde **avaliar `f` uma única vez já é caro**:

- **Engenharia computacional**: cada avaliação roda uma simulação CFD/FEM que dura horas ou dias.
- **Machine learning**: cada avaliação treina um modelo grande (horas/dias na GPU).
- **Experimento físico**: cada avaliação é um ensaio de laboratório (custa material e tempo de bancada).
- **FundaIA**: cada avaliação roda dezenas de verificações de NBR 6118 sobre múltiplas combinações de carga em múltiplas sapatas.

Quando avaliar `f(x)` é caro, você precisa de um **algoritmo que escolha sabiamente onde avaliar** — não pode "tentar tudo".

> 💡 **Definição clássica** [Mockus 1975]: "*Bayesian optimization is defined as an optimization technique based upon the minimization of the expected deviation from the extremum of the studied function.*" — Otimização bayesiana é uma técnica que minimiza o **desvio esperado** do extremo. A palavra-chave é **esperado**: já que não sabemos onde está o extremo, escolhemos onde olhar baseado em **expectativa probabilística**.

### 1.2 — A ideia central da otimização bayesiana

Em vez de avaliar `f` em pontos aleatórios ou seguir um gradiente local (que pode não existir), a otimização bayesiana faz o seguinte:

1. **Aproxima `f` por um modelo probabilístico** (o "surrogate" — geralmente um GPR). Esse modelo, dadas as observações que você já tem, gera uma **distribuição de probabilidade** sobre como `f` se comporta em cada ponto: "no ponto `x`, eu acho que `f` vale `μ(x)` ± `σ(x)`".

2. **Define uma "função de aquisição"** (EI, LCB, SBO etc.) que, baseada nessa distribuição, mede **o quanto vale a pena avaliar cada ponto**. A função de aquisição combina dois objetivos:
   - **Exploit** (explorar onde o modelo prediz bem) — ir onde `μ(x)` é baixa.
   - **Explore** (reduzir incerteza) — ir onde `σ(x)` é alta.

3. **Avalia o ponto que maximiza a função de aquisição**, adiciona o resultado à base, atualiza o modelo e repete.

> 💡 **Por que isso é eficiente?** Porque a função de aquisição é **barata** (uma fórmula fechada), enquanto `f` é **cara**. A cada iteração você gasta muita CPU otimizando a função de aquisição (que é grátis) pra escolher **um único ponto** onde gastar uma avaliação cara de `f`.

---

## 2 · Vocabulário básico (leia antes do GPR)

Antes de mergulhar no GPR, vamos fixar 6 conceitos que vão aparecer o tempo todo. Se você é da computação mas não da estatística, alguns serão familiares; outros não.

### 2.1 — Variável aleatória

Uma **variável aleatória** `Y` é uma quantidade que pode assumir vários valores, cada um com uma probabilidade. Pense num dado: a "face que sai" é uma variável aleatória que assume valores de 1 a 6 com probabilidade 1/6 cada.

Em problemas contínuos (como regressão), `Y` pode assumir qualquer número real, e a "probabilidade" vira uma **densidade**: `P(Y ≈ 3) ≈ p(3)·dy`.

### 2.2 — Distribuição normal (gaussiana)

A distribuição **normal** ou **gaussiana** é a famosa "curva de sino". Tem dois parâmetros:

- **Média (`μ`)**: o centro da curva. O valor mais provável.
- **Desvio padrão (`σ`)**: o quão "espalhada" a curva está. Pequeno σ = pico estreito (alta certeza). Grande σ = pico largo (alta incerteza).

Notação: `Y ~ N(μ, σ²)` significa "Y é uma normal com média μ e variância σ²".

> 💡 **Por que normal?** Porque pelo **Teorema Central do Limite**, soma de muitos efeitos aleatórios independentes tende a ser normal, então é uma escolha "inocente" e amplamente aplicável quando você não tem motivo pra escolher outra distribuição.

### 2.3 — PDF e CDF

Duas funções importantes da distribuição normal padrão (média 0, desvio 1):

- **`φ(z)` — PDF** (Probability Density Function, função densidade de probabilidade): a altura da curva em `z`. Ela responde "qual é a densidade de probabilidade neste ponto?".
  $$ \varphi(z) = \frac{1}{\sqrt{2\pi}} \, e^{-z^2/2} $$

- **`Φ(z)` — CDF** (Cumulative Distribution Function, função de distribuição acumulada): a área sob a curva da PDF até `z`. Ela responde "qual é a probabilidade de Y ser **menor ou igual** a `z`?".
  $$ \Phi(z) = P(Y \le z) = \int_{-\infty}^{z} \varphi(t)\, dt $$

> 💡 **Você não precisa decorar essas fórmulas**: na prática `φ` e `Φ` são funções prontas (em Python: `scipy.stats.norm.pdf` e `scipy.stats.norm.cdf`). O importante é saber **o que cada uma significa**.

### 2.4 — Esperança (valor esperado)

A **esperança** `E[Y]` de uma variável aleatória é a **média ponderada** de todos os valores possíveis, com peso igual à probabilidade:

$$ E[Y] = \int y \, p(y) \, dy $$

Para uma normal, `E[Y] = μ` (a esperança é a média, óbvio).

Quando você lê `E[max(f_min - Y, 0)]` no Expected Improvement, é "a média ponderada do **ganho** sobre todos os cenários possíveis de Y".

### 2.5 — Prior, posterior e bayesiano

Esse é **o coração** da otimização bayesiana, e é onde as pessoas tropeçam.

- **Prior** (antes de ver dados): sua crença sobre `f` antes de medir nada. Ex.: "eu acho que `f` é uma função suave, e em média vale 0".
- **Likelihood** (probabilidade dos dados): dado um modelo de `f`, qual é a chance de ver as observações que de fato foram vistas?
- **Posterior** (depois de ver dados): sua crença atualizada sobre `f`, combinando prior + likelihood pelo Teorema de Bayes.

> 💡 **Em palavras simples**: você começa com uma crença vaga ("`f` pode ser qualquer função suave"). Mede `f` em alguns pontos. Atualiza sua crença ("`f` é uma função suave **que passa por estes pontos**"). É exatamente o que um cientista faz: tem uma teoria, vê dados, ajusta a teoria.

### 2.6 — Kernel (função de covariância)

Um **kernel** `k(x, x')` é uma função que mede a **similaridade** entre dois pontos `x` e `x'`. No GPR, é o **modelo de suavidade** que você assume sobre `f`. Tipos comuns:

- **RBF / Gaussian / Squared Exponential**: `k(x, x') = exp(-||x - x'||² / (2 ℓ²))`. Resulta em funções **muito suaves** (infinitamente diferenciáveis). Mais detalhes em §3.4.
- **Matérn ν=2.5**: similar ao RBF mas permite funções **menos suaves** (2 vezes diferenciáveis). É o **default do FundaIA**.
- **Rational Quadratic, Periodic, etc.**: outras escolhas pra capturar comportamentos específicos.

> 💡 **Intuição**: o kernel diz "quanto a função muda entre dois pontos próximos?". Pontos próximos no espaço de entrada → kernel alto → funções similares. Pontos distantes → kernel baixo → funções podem variar muito.

---

## 3 · Gaussian Process Regression (GPR)

### 3.1 — A ideia em uma frase

> Um GPR é um **modelo probabilístico** que, dado um conjunto de observações `{(x_i, y_i)}`, gera para cada ponto `x` novo **uma média `μ(x)` e um desvio padrão `σ(x)`** que descrevem onde `f(x)` provavelmente está e o quanto isso é incerto.

A diferença pra uma regressão clássica (linear, polinomial, rede neural) é que a **regressão clássica te dá apenas o `μ(x)`** (a previsão pontual). O GPR te dá `μ(x)` **e** `σ(x)` — a previsão **e** a incerteza.

Por que essa incerteza importa? Porque o EGO **precisa dela** pra decidir onde explorar. Sem incerteza, você só sabe "minha melhor aposta é aqui"; com incerteza, você sabe "minha melhor aposta é aqui, **mas eu não sei muito sobre o que acontece ali**". É o "mas" que faz toda a diferença.

### 3.2 — Definição formal

Um **Gaussian Process** é uma **distribuição sobre funções**: em vez de cada amostra ser um número (variável aleatória clássica), cada amostra é uma função inteira. A definição matemática é:

$$ f(x) \sim \mathcal{GP}\bigl(m(x), k(x, x')\bigr) $$

onde:
- `m(x)`: função média (geralmente assumida 0 após centrar os dados — **prior** sobre o valor médio).
- `k(x, x')`: função kernel (covariância) que define a estrutura de suavidade.

A propriedade-chave é:

> **Para qualquer conjunto finito de pontos `x_1, ..., x_n`, o vetor `[f(x_1), ..., f(x_n)]` segue uma distribuição normal multivariada** com média `[m(x_1), ..., m(x_n)]` e matriz de covariância `K_{ij} = k(x_i, x_j)`.

Em outras palavras: uma "função aleatória" é representada pelo conjunto de seus valores em pontos amostrados, e esses valores juntos têm distribuição normal multivariada.

> 💡 **Mantenha a intuição**: você não precisa visualizar uma "distribuição sobre funções inteiras" — basta pensar que **em cada ponto `x`, o GPR te dá uma normal `N(μ(x), σ²(x))`**, e que essas normais estão **correlacionadas** entre si (se `x` e `x'` estão próximos, suas previsões estão correlacionadas).

### 3.3 — A fórmula da posterior (com observações)

Dado um conjunto de observações `(X, y)` com `y_i = f(x_i)`, o GPR atualiza a distribuição da função em qualquer ponto novo `x_*` para:

$$
\begin{aligned}
\mu(x_*) &= k_*^T \, K^{-1} \, y \\
\sigma^2(x_*) &= k(x_*, x_*) - k_*^T \, K^{-1} \, k_*
\end{aligned}
$$

onde:
- `K` (n×n): matriz de covariância entre os pontos observados, `K_{ij} = k(x_i, x_j)`.
- `k_*` (n×1): vetor de covariância entre o ponto novo `x_*` e os observados, `k_{*i} = k(x_*, x_i)`.
- `k(x_*, x_*)`: variância do kernel em `x_*` (auto-similaridade).

Não precisa decorar. O importante é entender **três propriedades**:

1. **`μ(x_*)` é uma combinação linear das observações**: `μ(x_*) = Σ α_i y_i`, onde os pesos `α_i` vêm de `K^{-1} k_*`. Ou seja, a previsão é uma **média ponderada** dos `y_i` observados, com peso maior pros pontos mais próximos de `x_*`.

2. **`σ²(x_*)` é zero nos pontos observados**: faz sentido — se você já mediu `f(x_i)`, não há incerteza ali. A incerteza cresce conforme você se afasta dos pontos medidos.

3. **`σ²(x_*)` depende **apenas das posições** dos pontos, não dos valores `y`**: a incerteza é geométrica, não estatística sobre os dados. Isso é crucial pro EGO: você sabe **onde tem pouca informação** mesmo antes de medir.

### 3.4 — Kernels e hiperparâmetros (sem matemática chata)

Os kernels mais comuns têm **hiperparâmetros** — números que ajustam a suavidade e amplitude da função. Vamos ver dois exemplos.

#### RBF (Radial Basis Function)

$$ k_{\text{RBF}}(x, x') = \sigma_f^2 \, \exp\!\left(-\frac{||x - x'||^2}{2 \ell^2}\right) $$

- **`σ_f²` (variance)**: a amplitude vertical da função. σ_f grande = função pode oscilar muito; pequeno = função fica perto de zero.
- **`ℓ` (length-scale)**: o "comprimento característico". `ℓ` grande = função muito suave (variações ao longo de distâncias grandes); `ℓ` pequeno = função enrugada (variações em distâncias pequenas).

#### Matérn ν=2.5

$$ k_{\text{Matérn}}(x, x') = \sigma_f^2 \left(1 + \frac{\sqrt{5}\,||x - x'||}{\ell} + \frac{5\,||x - x'||^2}{3\,\ell^2}\right) \exp\!\left(-\frac{\sqrt{5}\,||x - x'||}{\ell}\right) $$

Parece pavoroso, mas é só uma generalização do RBF que permite funções **menos suaves** (apenas 2 vezes diferenciáveis, em vez de infinitamente). É o **default do FundaIA** porque a função objetivo de sapatas tem "quinas" (mudanças bruscas quando uma restrição vira ativa).

#### Como os hiperparâmetros são escolhidos

Por **maximização de verossimilhança marginal** (em inglês, *marginal likelihood maximization*). Em essência, o algoritmo encontra os valores de `σ_f²` e `ℓ` que tornam **as observações mais prováveis** sob o modelo. É um problema de otimização interno que o `GaussianProcessRegressor` do scikit-learn resolve sozinho ao chamar `.fit()`.

> 💡 **Por que `n_restarts_optimizer=5`** (no FundaIA): porque a verossimilhança marginal é multimodal (vários picos). O scikit-learn reinicia a otimização interna 5 vezes em pontos diferentes pra **escapar de mínimos locais ruins**. É um detalhe importante — sem isso, pode dar um GPR ruim.

### 3.5 — Limitação prática: O(n³)

Olhando a fórmula da posterior, aparece `K^{-1}` — inversão de uma matriz `n × n`. Isso custa **`O(n³)`** operações.

- 100 pontos → trivial (1 ms).
- 1.000 pontos → centenas de ms.
- 10.000 pontos → minutos.
- 100.000 pontos → impraticável.

Por isso GPR é o método de escolha para **problemas com poucas avaliações** — exatamente o cenário do EGO. Se você tem milhões de exemplos, GPR não é a escolha — use uma rede neural.

> 🔗 **No FundaIA** ([[03_Otimizacao/Gaussian Process Regressor]], `core/optimization/ego.py`): cada repetição do EGO acumula `n_pop + n_gen` pontos. Com defaults `n_pop=250, n_gen=20`, ficam ~270 pontos por rep — bem dentro do regime confortável do GPR. O `SurrogateCache` do FundaIA evita re-treinar do zero quando os dados não mudam.

---

## 4 · Expected Improvement (EI) — a função de aquisição

### 4.1 — A ideia em palavras

Você já tem um conjunto de avaliações reais e sabe qual foi **o melhor valor encontrado até agora**: `f_min = min{y_1, ..., y_n}`.

Para cada ponto candidato `x` que você poderia avaliar a seguir, faça duas perguntas:

1. "Se eu avaliar `f(x)`, **quanto eu posso melhorar** sobre `f_min`?"
2. "Quão **provável** é essa melhoria?"

A resposta combinada é o **Expected Improvement** — a esperança da melhoria.

### 4.2 — Definindo "melhoria"

A "melhoria" trazida por avaliar `f(x)` é definida como:

$$ I(x) = \max(f_{\min} - Y(x),\, 0) $$

onde `Y(x)` é a variável aleatória que descreve o valor de `f(x)` segundo o GPR, ou seja, `Y(x) ~ N(μ(x), σ²(x))`.

A função `max(·, 0)` significa: **se o valor avaliado for maior que `f_min`, não há melhoria** (não piora seu melhor encontrado). Só conta a parte que de fato melhora.

### 4.3 — A fórmula do EI

A esperança de `I(x)` tem uma **forma fechada** (sai integrando a definição usando a normal do GPR):

$$
\boxed{\,\text{EI}(x) = (f_{\min} - \mu(x))\, \Phi(z) + \sigma(x)\, \varphi(z)\,, \quad z = \frac{f_{\min} - \mu(x)}{\sigma(x)}\,}
$$

Vamos destrinchar **termo a termo** porque essa fórmula tem muito em pouco espaço.

#### O termo de **exploit**: `(f_min - μ(x)) · Φ(z)`

- `(f_min - μ(x))` é a **melhoria média esperada**: se a média predita μ(x) é bem menor que f_min, esse termo é grande (positivo).
- `Φ(z)` é a **probabilidade de melhorar**: a chance (entre 0 e 1) de o valor real `f(x)` cair abaixo de f_min, segundo a distribuição normal do GPR.
- **Produto**: "ganho médio × probabilidade de ganhar".

> 💡 **Quando esse termo domina**: quando a média `μ(x)` é claramente menor que `f_min` e a incerteza `σ(x)` é pequena. Você está "explorando o que já parece bom". Isso é **exploit**.

#### O termo de **explore**: `σ(x) · φ(z)`

- `σ(x)` é a incerteza no ponto.
- `φ(z)` é a densidade da normal padrão em `z` — alta perto do `f_min` (z ≈ 0).
- **Produto**: "incerteza × densidade da chance de melhorar bem na fronteira".

> 💡 **Quando esse termo domina**: quando `σ(x)` é grande, mesmo que `μ(x)` esteja próximo de `f_min`. Você está "olhando onde tem incerteza, porque ali pode ter algo melhor que ainda não vi". Isso é **explore**.

### 4.4 — Como o EI equilibra exploit × explore automaticamente

A genialidade do EI é que esse equilíbrio é **automático**:

| Cenário | μ(x) | σ(x) | Termo dominante | EI age como |
|---|---|---|---|---|
| Vale prometedor já visitado | baixa | baixa | exploit | mira no melhor candidato |
| Região nunca explorada | desconhecida | alta | explore | reduz incerteza |
| Pico ruim já bem mapeado | alta | baixa | nem um nem outro | EI ≈ 0, ignora |
| Vale prometedor com pouca info | baixa | alta | ambos | combina explore + exploit |

> 💡 **Por que essa fórmula em particular?** Porque ela é **a esperança matemática** da função `max(f_min - Y, 0)` integrando sobre a normal do GPR. Não é uma escolha arbitrária — é a resposta exata pra "qual o ganho médio que vou ter avaliando `x`?". Outras funções de aquisição (LCB, SBO) tomam atalhos diferentes, com prós e contras (§5).

### 4.5 — O ponto que o EGO escolhe

A cada iteração, o EGO escolhe o ponto que **maximiza o EI**:

$$ x_{n+1} = \arg\max_{x} \, \text{EI}(x) $$

Essa maximização é em si **um problema de otimização**, mas é **barato**: EI é uma fórmula fechada, não custa nada avaliar. O FundaIA usa um **Algoritmo Genético interno** (mealpy GA) ou um optimizador SciPy pra encontrar o `x_{n+1}` global — porque EI é multimodal e gradiente local pode dar errado.

> 🔗 **No FundaIA** ([[03_Otimizacao/Expected Improvement]], `core/optimization/ego.py`):
>
> ```python
> def obj_ego(x, coef):
>     model, fmin = coef
>     mu, sig = model.predict(x_df, return_std=True)
>     sigma = max(sig[0], 1e-10)         # evita divisão por 0
>     z = (fmin - mu[0]) / sigma
>     of = (fmin - mu[0]) * norm.cdf(z) + sigma * norm.pdf(z)
>     return -of                          # negar pra usar minimizador
> ```
>
> Note o `max(sig[0], 1e-10)`: quando `σ → 0` (ponto já avaliado), `z → ±∞` e a divisão explode. Esse "jitter" mantém numericamente estável.

---

## 5 · As outras 2 funções de aquisição (SBO e LCB)

A documentação do SMT lista **três** critérios de aquisição. EI é o default, mas os outros têm seus usos.

### 5.1 — SBO (Surrogate-Based Optimization)

A escolha mais ingênua: **simplesmente use `μ(x)` como objetivo**.

$$ x_{n+1} = \arg\min_{x} \, \mu(x) $$

- ✅ **Simples**: não precisa de `σ`.
- ❌ **Greedy demais**: ignora completamente a incerteza. Pode ficar preso num mínimo local do surrogate.

**Quando usar**: quando o surrogate está **muito bem treinado** (muitos pontos), e você quer só refinar localmente.

### 5.2 — LCB (Lower Confidence Bound)

Em vez de média, use a **fronteira inferior do intervalo de confiança**:

$$ x_{n+1} = \arg\min_{x} \, \bigl(\mu(x) - \kappa \cdot \sigma(x)\bigr) $$

A documentação do SMT usa `κ = 3` (intervalo de confiança ≈ 99%, "μ - 3σ").

- ✅ **Honesto sobre incerteza**: o `−κσ` premia regiões incertas (porque ali o "pior caso otimista" pode ser muito bom).
- ✅ **Parâmetro intuitivo**: `κ` é o slider explore/exploit. `κ = 0` → SBO. `κ → ∞` → exploração pura.
- ❌ **Sem garantia teórica de calibração**: o "3σ" é um chute prático.

**Quando usar**: quando você quer **controle direto** sobre o quanto explorar. Útil quando o EI fica preso em "platôs" de exploit.

### 5.3 — Comparação rápida

| Critério | Fórmula | Comportamento | Custo computacional |
|---|---|---|---|
| **EI** | `(f_min − μ)Φ(z) + σ φ(z)` | Equilíbrio explore/exploit automático | Baixo |
| **SBO** | `μ(x)` | Puro exploit; pode ficar preso | Mínimo |
| **LCB** | `μ − κσ` | Explore/exploit ajustável por κ | Mínimo |

> 🔗 **No FundaIA** ([[04_Codigo/metapy_toolbox - ego.py]]): usa **EI** como default (única opção implementada hoje). Adicionar LCB seria uma **extensão fácil** se quiser experimentar — basta substituir a função `obj_ego`.

---

## 6 · O algoritmo EGO completo

Juntando GPR + EI, o pseudocódigo do EGO é:

```
EGO(F, n_iter)              # Minimiza F em n_iter iterações
  X, Y = LHS_inicial(F)     # gera pontos iniciais com Latin Hypercube
                            # (ver §7 para por que LHS)

  Para i = 1 até n_iter:
    1. mod = treinar_GPR(X, Y)             # surrogate
    2. f_min = min(Y)
    3. x_novo = arg_max EI(mod, f_min)     # otimiza EI com GA interno
    4. y_novo = F(x_novo)                  # avaliação cara
    5. X = X ∪ {x_novo}
    6. Y = Y ∪ {y_novo}

  Retorna: x* = X[arg_min Y], y* = min(Y)
```

> 💡 **Os 6 passos são o EGO inteiro.** Tudo o que você lê em papers e documentações é variação ou refinamento desse esqueleto. Entender esses 6 passos é entender o algoritmo.

### 6.1 — O passo da inicialização (LHS)

O `LHS_inicial` (Latin Hypercube Sampling) gera os primeiros pontos. **Por que LHS e não aleatório uniforme?** Porque LHS garante **cobertura uniforme em cada dimensão** isoladamente — mesmo com poucos pontos, cada projeção 1D fica bem distribuída. Random uniforme pode deixar buracos grandes em alguns eixos.

> 🔗 No FundaIA: `n_pop` define quantos pontos LHS. Default = 250, todos avaliados na função objetivo real (essa é a "iteração 0" do gráfico de histórico).

### 6.2 — Re-treino do GPR a cada iteração

Isso é importante: o GPR **re-otimiza os hiperparâmetros do kernel** a cada iteração com a base aumentada. Isso pode dar uma "virada" no modelo (kernel muito diferente) — o que é bom (modelo se ajusta) e ruim (instabilidade numérica).

> 🔗 **Cache do FundaIA** (`SurrogateCache` em `core/optimization/cache.py`): se duas iterações chegam no mesmo `(X, y)` (raro mas pode acontecer com avaliações duplicadas), o cache devolve o GPR já treinado em vez de re-fitar. Economia importante em runs longos.

---

## 7 · Versão paralela: qEI

### 7.1 — O problema

EGO clássico é **sequencial**: avalia 1 ponto, atualiza modelo, avalia o próximo. Mas e se você tem **8 cores de CPU** ociosos, ou **um cluster** com 100 nós? Faria sentido sugerir **q pontos** ao mesmo tempo e avaliá-los em paralelo.

Mas o EI sequencial não te dá isso de graça: o melhor `x_{n+1}` é único; o segundo melhor está provavelmente colado nele. Você precisaria de uma fórmula que escolha `q` pontos **diversos** simultaneamente.

### 7.2 — A solução: qEI

**qEI** [Ginsbourger 2010] generaliza EI pra `q` pontos:

$$ \text{qEI}(x_1, ..., x_q) = E\left[\max\left(f_{\min} - \min_{j=1..q} Y(x_j),\, 0\right)\right] $$

Calcular qEI **exato** é caro: envolve integrais multidimensionais sobre normais multivariadas.

### 7.3 — As 4 estratégias de "mentira" (Virtual Values)

Pra evitar o cálculo exato do qEI, a documentação do SMT implementa **estratégias de mentira**: depois de escolher o primeiro `x_1` (com EI normal), você "finge" um valor `ŷ_1` para ele, atualiza o GPR como se já tivesse medido, e usa o GPR mentiroso pra escolher `x_2`, e assim sucessivamente.

| Estratégia | Valor "mentido" `ŷ_q` | Comportamento |
|---|---|---|
| **CLmin** (Constant Liar Min) | `min(Y)` | "Diga que é o melhor já visto" — incentiva diversidade (próximos pontos não querem repetir) |
| **KB** (Kriging Believer) | `μ(x_q)` | "Diga o que o GPR prediz" — neutro |
| **KBLB** (KB Lower Bound) | `μ(x_q) − 3σ` | "Otimista" — assume melhor que o GPR prediz, força próximos pontos a explorar **outras** regiões |
| **KBUB** (KB Upper Bound) | `μ(x_q) + 3σ` | "Pessimista" — assume pior, próximos pontos podem se concentrar perto |

A documentação do SMT usa **KBLB** como default, e recomenda manter `q < 8` pra eficiência.

> 🔗 **No FundaIA**: paralelismo qEI **não está implementado**. O FundaIA tem `n_rep` repetições independentes (cada uma roda EGO sequencial completo), e isso é o que paraleliza hoje. Se em algum momento você quiser ganho real com cluster, qEI é uma frente de evolução interessante.

---

## 8 · Mapping completo: SMT EGO ↔ FundaIA

Como o FundaIA não usa SMT diretamente (usa scikit-learn + mealpy), aqui o mapping conceitual:

| Conceito SMT EGO | Equivalente no FundaIA | Onde no código |
|---|---|---|
| `surrogate=KRG` (kriging) | `GaussianProcessRegressor` do scikit-learn | `core/optimization/ego.py:gpr_pipelines` |
| `criterion=EI` | `obj_ego` (calcula EI direto) | `core/optimization/ego.py:obj_ego` |
| `n_iter` | `n_gen` (default 20) | `core/api/types.py:OptimisationConfig` |
| `n_doe` (LHS inicial) | `n_pop` (default 250) | idem |
| `n_start` (multi-start do EI) | `ga_pop_size`, `ga_epoch` (do GA interno) | `OptimisationConfig.ga_*` |
| `random_state` | `base_seed=42` | `OptimisationConfig.base_seed` |
| `n_parallel` (qEI) | ❌ não tem | — |
| `enable_tunneling` | ❌ não tem | — |
| `surrogate=GPX` (GP eXtra, mixed-int) | ❌ não tem | (mas FundaIA tem variáveis contínuas, não precisa) |

---

## 9 · O que o FundaIA poderia ganhar com SMT

Ler a documentação do SMT é útil porque ela **lista funcionalidades** que o FundaIA não tem hoje:

### 9.1 — Tunneling (penalizar re-avaliação)

> "Optional tunneling to penalize re-evaluation of known points"

Em runs longos, o GPR pode ficar querendo avaliar de novo pontos próximos a já-avaliados (fenômeno de "ghosting"). O tunneling adiciona uma penalidade para reduzir esse comportamento. **Custo**: é caro em alta dimensão. **Pro FundaIA**: como `n_gen` é pequeno (20), provavelmente não compensa.

### 9.2 — qEI paralelo

Já discutido em §7. Útil se você quiser rodar EGO num cluster.

### 9.3 — Critérios alternativos (LCB)

Adicionar LCB ao FundaIA é uma extensão de **uma função** — pode ser útil pra benchmark.

### 9.4 — Mixed-integer/categorical (GPX)

Se um dia você quiser otimizar **classes de sapata** (pad, stepped, sloped) junto com dimensões, vai precisar de surrogate que aceite variáveis categóricas. **GPX** do SMT faz isso. Hoje o FundaIA é todo contínuo.

---

## 10 · Glossário rápido

| Termo | Significado |
|---|---|
| **Black-box function** | função que você só sabe avaliar — não conhece a forma analítica nem o gradiente |
| **Surrogate** | modelo aproximado e barato que substitui a função cara |
| **Acquisition function** | função que mede "o quanto vale a pena avaliar este ponto" — EI, LCB, SBO |
| **Bayesian optimization (BO)** | termo guarda-chuva pra otimização baseada em surrogate probabilístico |
| **Kriging** | sinônimo de GPR (vem da geoestatística — Krige, 1951) |
| **Posterior** | distribuição atualizada após observar dados |
| **Prior** | distribuição inicial antes de observar dados |
| **CDF (Φ)** | função de distribuição acumulada da normal padrão |
| **PDF (φ)** | função densidade de probabilidade da normal padrão |
| **LHS** | Latin Hypercube Sampling — método de amostragem com cobertura uniforme |
| **MLE** | Maximum Likelihood Estimation — método pra ajustar hiperparâmetros |
| **Multimodal** | função com vários mínimos locais (EI tipicamente é) |
| **Length-scale (ℓ)** | hiperparâmetro do kernel: quanto a função "varia rápido" |
| **DOE** | Design of Experiments — conjunto inicial de pontos avaliados |
| **Trade-off explore/exploit** | dilema de gastar avaliações em regiões novas vs. refinar regiões conhecidas |
| **Confidence interval** | faixa onde o valor "real" cai com certa probabilidade (ex.: μ ± 3σ ≈ 99%) |

---

## 11 · Caminho de aprendizado sugerido

1. ✅ Leia este guia até aqui.
2. Abra `core/optimization/ego.py` no FundaIA e leia a função `obj_ego` linha por linha — é o EI puro em código.
3. Leia [[03_Otimizacao/Latin Hypercube Sampling]] (1 página) — entende o "iter 0".
4. Leia [Schulz, Speekenbrink & Krause 2018] — *A tutorial on Gaussian process regression*. É o melhor tutorial pra **completar a intuição**.
5. Leia [Jones, Schonlau & Welch 1998] — paper original do EGO. Tem só 38 páginas e é didático.
6. Leia [Shahriari et al. 2016] — *Taking the Human Out of the Loop: A Review of Bayesian Optimization*. Review extensa, ótima visão geral.
7. (Opcional, avançado) Capítulo 2 do **Williams & Rasmussen** — *Gaussian Processes for Machine Learning*. Livro-referência completo.

---

## 12 · Referências completas

### Fonte primária deste guia

- **SMT v2.9.3 — Documentação EGO**. <https://smt.readthedocs.io/en/v2.9.3/_src_docs/applications/ego.html>. Acesso em 2026-04-29.

### Fontes complementares utilizadas

- **Jones, D.R., Schonlau, M., Welch, W.J.** (1998). *Efficient Global Optimization of Expensive Black-Box Functions*. Journal of Global Optimization 13(4): 455–492.
  - Ficha: [[08_Artigos/Jones et al. 1998 - Efficient Global Optimization]]
- **Mockus, J.** (1975). *On Bayesian methods for seeking the extremum*. Optimization Techniques IFIP Technical Conference.
  - Ficha (OCR pendente): [[08_Artigos/The application of Bayesian methods - OCR pendente]]
- **Williams, C.K.I., Rasmussen, C.E.** (2006). *Gaussian Processes for Machine Learning*. MIT Press.
  - Ficha: [[08_Artigos/Williams e Rasmussen - Gaussian Processes for Regression]]
- **Schulz, E., Speekenbrink, M., Krause, A.** (2018). *A tutorial on Gaussian process regression*. Journal of Mathematical Psychology 85: 1–16.
  - Ficha: [[08_Artigos/Schulz et al. 2018 - Tutorial Gaussian Process Regression]]
- **Shahriari, B., Swersky, K., Wang, Z., Adams, R.P., de Freitas, N.** (2016). *Taking the Human Out of the Loop: A Review of Bayesian Optimization*. Proc. IEEE 104(1): 148–175.
  - Ficha: [[08_Artigos/Shahriari et al. 2016 - Review Bayesian Optimization]]
- **Snoek, J., Larochelle, H., Adams, R.P.** (2012). *Practical Bayesian Optimization of Machine Learning Algorithms*. NeurIPS 2012.
  - Ficha: [[08_Artigos/Snoek et al. 2012 - Practical Bayesian Optimization]]
- **Ginsbourger, D., Le Riche, R., Carraro, L.** (2010). *Kriging is well-suited to parallelize optimization*. In: Computational Intelligence in Expensive Optimization Problems.
- **Saves, P. et al.** *A general square exponential kernel to handle mixed-categorical variables for Gaussian process*. AIAA Aviation 2022 Forum.

### Notas técnicas internas do vault

- [[03_Otimizacao/EGO - Efficient Global Optimization]]
- [[03_Otimizacao/Gaussian Process Regressor]]
- [[03_Otimizacao/Expected Improvement]]
- [[03_Otimizacao/Latin Hypercube Sampling]]
- [[03_Otimizacao/Algoritmo Genético]]
- [[03_Otimizacao/Kernels GPR]]
- [[03_Otimizacao/Formulação do Problema]]
- [[03_Otimizacao/Penalização de Restrições]]

### Guia irmão (engenharia)

- [[02_Engenharia/Guia Didatico - Dimensionamento de Sapatas Isoladas]] — explica as fórmulas do **lado do problema** (sapatas, NBR 6118, ACI 318), enquanto este guia explica o **lado do algoritmo**.

---

> [!tip] Como estudar este guia
> 1. Leia §1, §2 (vocabulário) e §6 (algoritmo) numa primeira passada — você já entende o **esqueleto** do EGO.
> 2. Volte e leia §3 (GPR) com calma — é a parte mais matemática, mas a intuição de §3.1 sozinha já vale muito.
> 3. Releia §4 (EI) com a fórmula aberta no papel — destrinche `(f_min − μ)Φ(z) + σφ(z)` termo a termo.
> 4. Abra `core/optimization/ego.py` e mapeie cada linha de código no esqueleto de §6.
>
> Não tente entender tudo de uma vez. O EGO é simples no esqueleto, mas tem profundidade matemática nos detalhes — é normal precisar de várias passadas.
