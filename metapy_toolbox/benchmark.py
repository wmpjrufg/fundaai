"""Benchmark functions for optimization."""
import numpy as np


def sphere(x: list) -> float:
    """The Sphere function has d local minima except for the global one. It is continuous, convex and unimodal.

    :param x: Design variables

    :return: Objective function value
    """

    d = len(x)
    of = 0
    for i in range(d):
        x_i = x[i]
        of += x_i ** 2

    return of


def rosenbrock(x: list) -> float:
    """The Rosenbrock function is unimodal, and the global minimum lies in a narrow, parabolic valley.

    :param x: Design variables

    :return: Objective function value
    """

    d = len(x)
    sum = 0
    for i in range(d - 1):
        x_i = x[i]
        x_next = x[i + 1]
        new = 100 * (x_next - x_i ** 2) ** 2 + (x_i - 1) ** 2
        sum += new
    of = sum

    return of


def rastrigin(x: list) -> float:
    """The Rastrigin function has several local minima. It is highly multimodal, but locations of the minima are regularly distributed.

    :param x: Design variables

    :return: Objective function value
    """

    d = len(x)
    sum = 0
    for i in range(d):
        x_i = x[i]
        sum += (x_i ** 2 - 10 * np.cos(2 * np.pi * x_i))
    of = 10 * d + sum

    return of


def ackley(x: list) -> float:
    """The Ackley function in its two-dimensional form, it is characterized by a nearly flat outer region, and a large hole at the centre.

    :param x: Design variables

    :return: Objective function value
    """

    n_dimensions = len(x)
    sum1 = 0
    sum2 = 0
    a = 20
    b = 0.2
    c = 2 * np.pi
    for i in range(n_dimensions):
        x_i = x[i]
        sum1 += x_i ** 2
        sum2 += np.cos(c * x_i)
    term_1 = -a * np.exp(-b * np.sqrt(sum1 / n_dimensions))
    term_2 = -np.exp(sum2 / n_dimensions)
    of = term_1 + term_2 + a + np.exp(1)

    return of


def griewank(x: list) -> float:
    """This function evaluates the Griewank benchmark on a d-dimensional vector.

    The Griewank function has many widespread local minima regularly
    distributed. Reference: Surjanovic & Bingham
    (https://www.sfu.ca/~ssurjano/griewank.html). Global minimum at
    x = 0, f(0) = 0.

    Notas de correcao (Sprint 2):
        Versao anterior tinha o produto fora do loop (`prod *= cos(...)`
        com indentacao errada), o que usava apenas o ultimo `x_i` e
        produzia valores numericamente errados para todo `d > 1`. A
        correcao move o produto para dentro do loop, partindo de
        `prod = 1` e multiplicando por `cos(x_i / sqrt(i+1))` em cada
        passo.

    :param x: Vetor das variaveis de projeto, com `d >= 1`

    :return: Valor da funcao objetivo `sum(x_i^2)/4000 - prod(cos(x_i/sqrt(i+1))) + 1`
    """
    n_dimensions = len(x)
    soma = 0.0
    produto = 1.0
    for i in range(n_dimensions):
        x_i = x[i]
        soma += (x_i ** 2) / 4000.0
        produto *= np.cos(x_i / np.sqrt(i + 1))
    of = soma - produto + 1.0
    return of


def zakharov(x: list) -> float:
    """The Zakharov function has no local minima except the global one.

    :param x: Design variables

    :return: Objective function value
    """

    n_dimensions = len(x)
    sum_1 = 0
    sum_2 = 0
    for i in range(n_dimensions):
        x_i = x[i]
        sum_1 += x_i ** 2
        sum_2 += (0.5 * i * x_i)
    of = sum_1 + sum_2**2 + sum_2**4

    return of


def easom(x: list) -> float:
    """The Easom function has several local minima. It is unimodal, and the global minimum has a small area relative to the search space.

    :param x: Design variables

    :return: Objective function value
    """

    x_1 = x[0]
    x_2 = x[1]
    fact_1 = - np.cos(x_1) * np.cos(x_2)
    fact_2 = np.exp(- (x_1 - np.pi) ** 2 - (x_2 - np.pi) ** 2)
    of = fact_1*fact_2

    return of


def michalewicz(x: list) -> float:
    """The Michalewicz function has d! local minima, and it is multimodal. The parameter m defines the steepness of they valleys and ridges. A larger m leads to a more difficult search.

    :param x: Design variables

    :return: Objective function value
    """

    n_dimensions = len(x)
    sum = 0
    m = 10
    for i in range(n_dimensions):
        x_i = x[i]
        sum += np.sin(x_i) * (np.sin((i * x_i ** 2) / np.pi)**(2 * m))
    of = -sum

    return of


def dixon_price(x: list) -> float:
    """The Dixon-Price function is unimodal, and the global minimum lies in a narrow, parabolic valley.

    :param x: Design variables

    :return: Objective function value
    """

    x1 = x[0]
    n_dimensions = len(x)
    term1 = (x1-1)**2
    sum = 0
    for i in range(1, n_dimensions):
        x_i = x[i]
        xold = x[i-1]
        new = i * (2*x_i**2 - xold)**2
        sum = sum + new
    of = term1 + sum

    return of


def goldstein_price(x: list) -> float:
    """The Goldstein-Price function has several local minima. Dimensions: 2.

    :param x: Design variables

    :return: Objective function value
    """

    x1 = x[0]
    x2 = x[1]
    fact1A = (x1 + x2 + 1)**2
    fact1B = 19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2
    fact1 = 1 + fact1A * fact1B
    fact2A = (2*x1 - 3*x2)**2
    fact2B = 18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2
    fact2 = 30 + fact2A * fact2B
    of = fact1*fact2

    return of


def powell(x: list) -> float:
    """This function evaluates the Powell benchmark on a d-dimensional vector.

    Reference: Surjanovic & Bingham
    (https://www.sfu.ca/~ssurjano/powell.html). The function is defined
    in blocks of four variables; therefore `d` must be a multiple of 4
    (`d in {4, 8, 12, ...}`). Global minimum at x = 0, f(0) = 0.

    Notas de correcao (Sprint 2):
        Versao anterior usava indexacao 1-based (`x[4i-3], x[4i-2],
        x[4i-1], x[4i]`) sem ajuste para Python 0-based, o que
        estourava o ultimo indice `x[4i]` quando o vetor tinha tamanho
        exatamente multiplo de 4 (caso canonico do Powell). A nova
        implementacao usa o equivalente 0-based: para
        `i = 0..n_blocks - 1`, le `x[4i], x[4i+1], x[4i+2], x[4i+3]`.

    :param x: Vetor das variaveis de projeto, com `d` multiplo de 4

    :return: Valor da funcao objetivo
             `sum_{i=0..n/4-1} (x_{4i} + 10·x_{4i+1})^2 + 5·(x_{4i+2} - x_{4i+3})^2
              + (x_{4i+1} - 2·x_{4i+2})^4 + 10·(x_{4i} - x_{4i+3})^4`

    :raises ValueError: Se `len(x) % 4 != 0`
    """
    n_dimensions = len(x)
    if n_dimensions % 4 != 0:
        raise ValueError(
            f"powell exige d multiplo de 4; recebeu d={n_dimensions}."
        )
    soma = 0.0
    n_blocks = n_dimensions // 4
    for i in range(n_blocks):
        a = x[4 * i + 0]
        b = x[4 * i + 1]
        c = x[4 * i + 2]
        d = x[4 * i + 3]
        term1 = (a + 10 * b) ** 2
        term2 = 5 * (c - d) ** 2
        term3 = (b - 2 * c) ** 4
        term4 = 10 * (a - d) ** 4
        soma += term1 + term2 + term3 + term4
    return soma


def active_learning_example(x: list) -> float:
    """Active learning function.

    :param x: Design variables

    :return: Objective function value
    """

    of = (x[0] - 3.5) * np.sin((x[0] - 3.5) / (np.pi))

    return of
