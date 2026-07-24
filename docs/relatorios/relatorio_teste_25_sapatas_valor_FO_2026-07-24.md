# Teste de escalabilidade com 25 sapatas — avaliação do valor da função objetivo

**Autor:** Lucas Teixeira Correia (IC — Eng. Civil, PUC Goiás)
**Data:** 2026-07-24
**Orientação:** Profa. Dra. Maria José Pereira Dantas
**Plataforma:** FundaIA (`core/api/benchmark.py`, `core/api/objective.py`)

> Relatório de diagnóstico. Não é um protocolo de publicação (3 repetições,
> orçamento modesto), mas um *probe* de escala para responder objetivamente:
> **como a formulação e o algoritmo se comportam com 20–30 sapatas, e o que
> isso diz sobre o valor da função objetivo (FO)?**

---

## 1. Objetivo do teste

Avaliar o comportamento da FO penalizada e da arquitetura de otimização em um
caso de **25 sapatas** (dimensão do vetor de projeto = 75), muito acima dos
casos congelados do artigo 1 (1, 2 e 3 sapatas; dim 3, 6 e 9). O foco é
**avaliar o valor da FO** sob três aspectos:

1. **Custo computacional** da avaliação (a FO é cara no sentido de Jones et al., 1998?);
2. **Adequação da formulação** penalizada `Θ = volume + α·Σ max(g_k, 0)` (α=10, p=1);
3. **Estrutura do problema** (separabilidade) e sua consequência para a escolha do otimizador.

## 2. Montagem do caso

- **25 pilares** numa grade 5×5 espaçada 6 m → a restrição de **não sobreposição
  é inativa por construção** (mesmo regime dos casos do artigo, escalado).
- **Cargas correlacionadas ao solo** (SPT e tipo), de modo que exista uma
  **região factível não-vazia** (verificado: `h = 2,0 m` uniforme já é factível,
  volume 200 m³; `h = 3,0 m` também).
- Pipeline **idêntico ao de produção**: `read_projeto_from_excel` →
  `run_benchmark` (mesma FO vetorizada `avaliar_projeto_fast` /
  `avaliar_projeto_componentes`). Nenhum código do projeto foi alterado.
- Comparados **EGO+GPR, CBO (ECI), GA, PSO, GWO e busca aleatória** sob:
  - **S1** — orçamento igual de **250** avaliações reais;
  - **S2** — orçamento estendido de **5000** avaliações (só para as metaheurísticas baratas).
- **Referência de decomposição:** Differential Evolution resolvendo cada sapata
  isoladamente (25 subproblemas de dim 3), remontada e reavaliada globalmente.
  Como as posições são fixas e espaçadas, essa referência aproxima o **ótimo verdadeiro**.
- 3 repetições semeadas (`base_seed = 42`).

## 3. Resultados

| Método | melhor Θ | factibilidade | melhor volume factível | tempo/rep |
|---|---:|---:|---:|---:|
| **Decomposição (DE por sapata)** | — | **100 %** | **33,2 m³** | 37 s (total, 25 sapatas) |
| EGO + GPR (250) | 82,2 | **0 %** | — | **159,9 s** |
| CBO — ECI (250) | 115,4 | **33 %** | 115,4 m³ | **155,8 s** |
| GA puro (250) | 102,0 | 0 % | — | 0,04 s |
| GWO puro (250) | 111,2 | 0 % | — | 0,04 s |
| PSO puro (250) | 124,6 | 0 % | — | 0,04 s |
| Busca aleatória (250) | 137,3 | 0 % | — | 0,04 s |
| GA puro (5000) | 44,0 | 0 % | — | 0,9 s |
| GWO puro (5000) | 39,6 | 0 % | — | 0,8 s |
| PSO puro (5000) | 69,5 | 0 % | — | 0,8 s |
| Busca aleatória (5000) | 121,6 | 0 % | — | 0,7 s |

Notas: "melhor Θ" é o pseudo-objetivo penalizado (menor = melhor); pode ser
**menor que o volume factível** porque a penalidade linear α=10 admite designs
levemente infactíveis com Θ baixo. "Factibilidade" = fração das repetições cujo
melhor projeto satisfaz **todas** as restrições com tolerância estrita
(`g ≤ 10⁻⁹`). Em **todas** as repetições infactíveis a restrição governante é a
**tensão admissível do solo** (`viol_ten` domina), coerente com o artigo 1.

## 4. Avaliação do valor da FO

### 4.1 A FO é barata — o gargalo é o surrogate, não a avaliação

A FO vetorizada custa **~0,1 ms por avaliação**, e isso se confirma até dim=75:
5000 avaliações de GA/GWO rodam em **< 1 s**. Consequentemente:

- O EGO/CBO gastam **~160 s por repetição**, ou seja **~4000× mais tempo** que
  GA/aleatória para o **mesmo** número de avaliações reais. Como a FO é trivial,
  esses ~160 s são quase inteiramente **ajuste do GPR (O(n³)) e maximização da
  aquisição em alta dimensão** — não a FO.
- **A justificativa clássica do EGO (FO cara, no sentido de Jones et al., 1998)
  não se aplica a esta formulação**, e o teste torna isso ainda mais nítido em
  escala: quanto maior o problema, mais o custo migra para o maquinário do
  surrogate, e não para a FO.

### 4.2 A FO penalizada com α=10 é fraca em dimensão alta

O teste expõe uma **limitação da formulação da FO**, não do otimizador:

- Sob orçamento igual, o EGO alcança o **menor Θ (82,2)** — mas esse projeto é
  **infactível**: volume 67 m³ com a tensão do solo **~37 % acima do admissível**.
- GA e GWO com 5000 avaliações chegam a Θ = 40–44, **todos infactíveis**: empurram
  o volume para baixo sem satisfazer a tensão.
- **"Melhor Θ" ≠ projeto viável.** Com α=10 e p=1, o ótimo penalizado não coincide
  com o ótimo restrito, e a folga cresce com a dimensão. Isto reforça, agora
  empiricamente em dim alta, a limitação já discutida na Seção 7 do artigo: a
  avaliação por Θ **precisa** ser acompanhada de métricas de factibilidade estrita.

### 4.3 O tratamento explícito de restrições (CBO) é o único que "enxerga" a factibilidade

O **CBO foi o único método conjunto a encontrar um projeto estritamente factível**
(1 de 3 repetições), porque modela as restrições em surrogates independentes e usa
aquisição por probabilidade de factibilidade. Isso **valida** a Frente C (CBO) do
artigo: em regime de restrição ativa e dimensão alta, tratar a restrição
explicitamente importa. Mas o custo é alto (156 s/rep) e o volume factível obtido
(115 m³) ainda é **3,5× o ótimo** da decomposição (33 m³).

### 4.4 Com posições fixas, a FO conjunta é (quase) separável — e não agrega valor

A referência de decomposição encontrou **33,2 m³, 100 % factível, em 37 s** —
**3,5× melhor** em volume que o melhor método conjunto (CBO) e **~6× melhor** que o
baseline trivial (todas as sapatas a `h=2,0` = 200 m³). Ou seja:

- Com posições fixas e espaçadas, **25 sapatas são 25 problemas 3D independentes**.
  A FO conjunta em dim=75 **não carrega informação de acoplamento** que a
  decomposição não capture — pelo contrário, a busca monolítica em dim=75 **fracassa**
  (custo alto + 0 % de factibilidade) onde 25 buscas de dim 3 resolvem trivialmente.
- Isto **confirma em escala** a auditoria de quase separabilidade do artigo:
  **aumentar N com posições fixas não é evidência a favor do EGO/CBO** — é
  justamente o cenário que expõe sua fragilidade.

## 5. Conclusões e implicações para a pesquisa

1. **A FO é barata e permanece barata em escala.** O valor da arquitetura
   assistida por surrogate **não** está em economizar avaliações de uma FO cara
   (ela não é cara), e sim como investimento metodológico para regimes futuros de
   avaliação genuinamente custosa. O teste fortalece esse posicionamento honesto
   já adotado no manuscrito.

2. **A formulação penalizada precisa de seleção final factível.** Em dim alta,
   α=10 é insuficiente; recomenda-se penalidade adaptativa, regras de dominância
   de factibilidade (Deb) ou seleção explícita por factibilidade estrita — não
   confiar em Θ isolado.

3. **Não escalar N com posições fixas como demonstração de mérito do otimizador.**
   Seria enganoso: a decomposição bate todos os métodos conjuntos. O recorte
   honesto do artigo 1 (pré-dimensionamento geométrico, posições fixas) deve ser
   mantido, com a separabilidade declarada.

4. **O que justifica o maquinário surrogate é o acoplamento real (Fase B —
   empacotamento).** Quando as posições viram variáveis e a não-sobreposição fica
   ativa, o problema deixa de ser separável — mas a dimensão sobe para 5N e um GP
   **global monolítico** piora ainda mais. O caminho é o já declarado na Seção 8:
   **decomposição-aware / regiões de confiança / surrogate local (SCBO)**, não EGO global.

## 6. Reprodução

Artefatos versionados em `experiments/teste_25_sapatas_2026-07-24/`:

- `run_case25_full.py` — script completo (caso, S1, S2, decomposição);
- `gen_case25.py` — smoke test de tempos;
- `caso_25_v2.xlsx` — planilha do caso sintético (entrada);
- `s1_summary.csv`, `s1_per_rep.csv`, `s1_pvalues.csv`, `s2_summary.csv` — resultados;
- `decomp_ref.json` — referência de decomposição;
- `full.log` — log completo da execução.

```bash
.venv/bin/python experiments/teste_25_sapatas_2026-07-24/run_case25_full.py
```

Ambiente: Python 3.12, numpy/scipy/scikit-learn/mealpy conforme `requirements.txt`.
Determinístico dado o mesmo `base_seed` e as mesmas versões de biblioteca.
