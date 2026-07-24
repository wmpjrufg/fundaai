---
tags: [projeto, convencoes, padrao, qualidade]
aliases: [Convenções, Project Conventions, Style Guide]
data_de_criacao: 2026-04-27
---

# Convenções do Projeto FundaIA

> Padrões oficiais de escrita adotados no repositório FundaIA. Seguir
> estas convenções é **requisito** para qualquer contribuição de código,
> documentação inline ou commits — independentemente de quem esteja
> escrevendo (humano ou ferramenta de apoio).

## 1. Idioma de cada artefato

| Artefato | Idioma | Observação |
|---|---|---|
| **Mensagens de commit** | **inglês** | Padronização internacional; facilita revisão por colaboradores externos. |
| **Docstrings (`:param:`, `:return:`, primeira linha)** | **inglês** | Espelha a convenção já adotada em `metapy_toolbox/` (ex.: `gray_wolf_hunting`). |
| **Comentários de bloco / linhas** | **inglês ou português** | Português apenas como **resumo breve de localização** (ex.: `# Tensão admissível do solo`); explicações técnicas em inglês. |
| **Identificadores de código** (variáveis, funções, classes) | **português** quando refletem domínio normativo (NBR), **inglês** caso contrário | Já há precedente: `tensao_adm_solo`, `verificacao_puncao_sapata`. Manter por compatibilidade. |
| **README.md, notas do vault, relatórios** | **português** | Documentação interna do projeto. |
| **Logs, mensagens de erro técnicas** | **inglês** | Facilita busca em Stack Overflow / GitHub Issues. |
| **UI (Streamlit, mensagens ao usuário final)** | **PT/EN bilíngue** | Já implementado via `obter_textos()`. |

## 2. Padrão de mensagem de commit

Seguir [Conventional Commits](https://www.conventionalcommits.org/) com **assunto em inglês** e **corpo opcional em inglês**:

```
<type>(<scope>): <short subject in english>

<optional body in english explaining WHAT and WHY>
<can include numeric results, references to issues, etc.>
```

**Tipos aceitos:**

| `<type>` | Quando usar |
|---|---|
| `feat` | Nova funcionalidade |
| `fix` | Correção de bug |
| `refactor` | Refatoração sem mudança de comportamento |
| `test` | Adição/alteração de testes |
| `docs` | Documentação (README, docstrings em massa) |
| `chore` | Configuração, dependências, gitignore, infraestrutura |
| `perf` | Melhorias de performance |
| `style` | Formatação (sem mudança lógica) |

**Exemplos válidos** (extraídos do histórico do projeto):

```
test: Sprint 2 — pytest suite with 55 tests (regression safety net)
fix: Sprint 2 — fix griewank (product outside loop) and powell (indexing)
refactor: Sprint 1 — EGO history (ITER and ID) and independent n_rep
```

> Resumo em português pode aparecer no **corpo** do commit como bloco
> auxiliar, mas a primeira linha (assunto) deve ser sempre em inglês.

## 3. Padrão de docstring

Seguir o estilo já adotado no `metapy_toolbox` (ex.: `gray_wolf_hunting`):

```python
def my_function(x: list, y: float) -> float:
    """This function performs <one-line description in english>.

    Optional longer explanation in english. Pode incluir um pequeno
    resumo em português como linha de localização rápida quando
    ajudar a equipe brasileira a navegar o codigo, mas o conteudo
    tecnico principal permanece em ingles.

    :param x: First design variables vector
    :param y: Numeric coefficient that scales the result

    :return: [0] = Description of first returned value
             [1] = Description of second returned value
    """
```

**Regras:**

- Primeira linha: `This function ...` ou `This class ...`, em inglês, terminando com ponto.
- `:param <nome>:` em inglês. Português permitido após o em-dash quando ajudar a equipe a localizar (raro).
- `:return:` em inglês. Para múltiplos retornos, usar a convenção `[0] = ...`, `[1] = ...` — **não** usar `Tuple[...]` redundante.
- `:raises:` quando aplicável, em inglês.
- Unidades em colchetes (`[m]`, `[kPa]`, `[kN·m]`) — espelha a convenção de `fundacao.py`.

**Exemplo bom (do projeto):**

```python
def gray_wolf_hunting(parent_0, x_alpha, x_beta, x_delta, a, x_lower, x_upper):
    """This function performs the Grey Wolf Hunting movement.

    :param parent_0: First parent. Current solution
    :param x_alpha: Position of the best wolf at the previous iteration
    :param x_beta: Position of the second best wolf at the previous iteration
    :param x_delta: Position of the third best wolf at the previous iteration
    :param a: Parameter that decreases linearly from 2 to 0 over the iterations
    :param x_lower: Lower limit of the design variables
    :param x_upper: Upper limit of the design variables

    :return: [0] = First offspring position
             [1] = Second offspring position
             [2] = Third offspring position
             [3] = Report about the linear crossover process
    """
```

## 4. Padrão de testes

- Nome do arquivo: `tests/test_<modulo>.py`.
- Nome da classe: `class TestXxxYyy` (CamelCase, em inglês).
- Nome da função: `def test_<comportamento_esperado>(self):` (snake_case, em inglês ou português permitido por proximidade ao domínio NBR).
- Docstring: `This test ensures/verifies <invariante>` em inglês, com explicação curta em português permitida quando útil.
- Marker pytest declarado: `@pytest.mark.engineering`, `@pytest.mark.regression`, etc.

## 5. Branch naming

Seguir `<tipo>/<descricao-em-ingles-com-hifens>`:

```
feat/packing-as-decision-variable
fix/code-sanitization-and-tests
refactor/core-architecture
docs/article-ic-lucas
```

## 6. Convenções de engenharia implementadas

### Momentos `Mx` e `My`

No FundaIA, os nomes `Mx` e `My` seguem a convenção interna da função de tensão:

| Campo | Interpretação no FundaIA | Termo de tensão |
|---|---|---|
| `Mx-c{i}` | momento/componente que produz excentricidade e variação de pressão ao longo de `h_x` | `6 |Mx| / (h_x h_y h_x)` |
| `My-c{i}` | momento/componente que produz excentricidade e variação de pressão ao longo de `h_y` | `6 |My| / (h_x h_y h_y)` |

Se as cargas vierem de outro software com a convenção estrutural usual de "momento em torno do eixo X" e "momento em torno do eixo Y", a importação deve converter para esta convenção antes de preencher a planilha. Em particular, um momento em torno de X costuma gerar variação de pressão na direção Y; portanto, não assumir que o rótulo externo pode ser copiado sem conferência.

### Tensão solo-sapata

A verificação atual usa peso próprio explícito da sapata:

$$
W_c = \gamma_c h_x h_y h_z
$$

com `gamma_c = 25 kN/m3` por padrão. A tensão na base é calculada como:

$$
\sigma_{\max} =
\frac{F_z + W_c}{h_x h_y}
+ \frac{6 |M_x|}{h_x h_y h_x}
+ \frac{6 |M_y|}{h_x h_y h_y}
$$

$$
\sigma_{\min} =
\frac{F_z + W_c}{h_x h_y}
- \frac{6 |M_x|}{h_x h_y h_x}
- \frac{6 |M_y|}{h_x h_y h_y}
$$

Os fatores antigos `1,05` para peso próprio aproximado e `1,30` para tensão compressiva não fazem mais parte do contrato atual. Qualquer majoração normativa ou combinação de ações deve entrar explicitamente nas cargas de entrada ou em uma etapa futura de combinações, não dentro de `calcular_sigma_max_min`.

## 7. Histórico de adoção

| Data | Commit/Sprint | Aplicação |
|---|---|---|
| 2026-04-27 | Sprints 0/1/2 (`fix/code-sanitization-and-tests`) | Convenção parcialmente seguida; alguns commits e docstrings ainda em português. **Não corrigir retroativamente**: novos artefatos seguem 100% o padrão; legado fica até a próxima refatoração natural. |
| 2026-04-27 (criação desta nota) | — | A partir daqui, **todos** os novos commits, docstrings, docstrings de testes e mensagens devem seguir o padrão. |
| 2026-07-12 | Sprint 5.4 | Convenção `Mx/My` explicitada e tensão solo-sapata corrigida para peso próprio por volume, sem fatores fixos `1,05`/`1,30`. |

## Vínculos

- [[01_Projeto/Stack Tecnológico]]
- [[01_Projeto/Pipeline de Execução]]
- [[10_Melhorias/MOC - Melhorias]]
- [[12_Auditoria/Sprint 2 - Testes e Saneamento Experimental - 2026-04-27]]
