---
tags: [artigo, indice]
---

# 📚 Index de Artigos

Use este arquivo como índice das suas leituras. Para cada artigo, crie uma nota nesta pasta seguindo [[99_Templates/Template - Artigo]].

> [!check] Validação
> As fichas criadas a partir da biblioteca enviada foram auditadas em [[08_Artigos/Validação das Fichas - 2026-04-27]].

> [!info] PDFs locais
> A biblioteca fisica foi reorganizada em `docs/articles`. O mapa completo PDF -> ficha esta em `docs/articles/README.md`, e cada ficha abaixo possui o campo `arquivo_pdf` no frontmatter.

## Sugestões de tópicos a cobrir

### 🏗️ Engenharia / Geotecnia
- Métodos de capacidade de carga: Terzaghi, Meyerhof, Vesić, Hansen.
- SPT e correlações: Décourt-Quaresma, Aoki-Velloso, Teixeira (cf. [[02_Engenharia/SPT - Sondagem]]).
- Dimensionamento de sapatas pela [[02_Engenharia/NBR 6118]] (item 22.4) e NBR 6122 (fundações).
- Métodos de verificação à punção (cf. [[02_Engenharia/Verificação à Punção]]).

### 🧮 Otimização
- Jones, Schonlau, Welch (1998) "Efficient Global Optimization of Expensive Black-Box Functions" — base do [[03_Otimizacao/EGO - Efficient Global Optimization]].
- Forrester, Sobester, Keane "Engineering Design via Surrogate Modelling" (livro de referência).
- Mirjalili et al. (2014) "Grey Wolf Optimizer".
- Tizhoosh (2005) "Opposition-based learning" — cf. [[03_Otimizacao/Opposite e Quasi-Opposite Population]].
- Deb, K. — métodos para tratar restrições em GA (Deb's rules).

### 📦 Packing
- Survey de packing 2D: Lodi, Martello, Vigo.
- No-Fit Polygon (NFP).
- Bortfeldt & Wäscher (2013) "Constraints in container loading – A state-of-the-art review".
- Bulbo de tensão / distribuição de tensões: Boussinesq, Newmark, método 2:1.
- Sapatas associadas/combinadas: critérios de proximidade, recalque e decisão topológica.

## Prioridade atual de leitura

> [!note] Antes do packing
> Primeiro registrar e usar as leituras que sustentam o artigo 1: otimização de sapatas isoladas, EGO, GPR, Bayesian Optimization e comparação de metaheurísticas. A busca por artigos de packing fica como próxima frente.

### Para o artigo 1 — validar FundaIA/EGO-GPR

- [ ] Jones, Schonlau, Welch (1998) — EGO / Expected Improvement.
- [ ] Snoek et al. (2012) — Bayesian Optimization.
- [ ] Shahriari et al. (2016) — revisão de Bayesian Optimization.
- [ ] Schulz et al. (2018) — GPR.
- [ ] Wang e Kulhawy (2008) — otimização econômica de fundações.
- [ ] Gandomi e Kashani (2018) — metaheurísticas em fundações rasas.
- [ ] Nigdeli et al. (2018) — metaheurísticas em sapatas de concreto armado.
- [ ] Waheed et al. (2022) — ferramenta para sapatas isoladas.
- [ ] Waheed et al. (2025) — otimização econômica de sapatas isoladas.
- [ ] Gomes et al. (2018) — comparação probabilística de metaheurísticas.
- [ ] Mathern et al. (2021) — BO com restrições em projeto estrutural.

### Para a próxima frente — bin packing/layout

- [ ] Lodi, Martello, Vigo — survey de packing 2D.
- [ ] Bortfeldt & Wäscher — container loading / restrições de packing.
- [ ] Burke et al. — No-Fit Polygon.
- [ ] Stoyan / phi-functions.
- [ ] Referências de sapata associada/combinada.
- [ ] Referências de bulbo de tensão e interação entre fundações próximas.

## Lista registrada no vault
### Essenciais para o artigo 1
- [[Jones et al. 1998 - Efficient Global Optimization]]
- [[Snoek et al. 2012 - Practical Bayesian Optimization]]
- [[Shahriari et al. 2016 - Review Bayesian Optimization]]
- [[Schulz et al. 2018 - Tutorial Gaussian Process Regression]]
- [[Williams e Rasmussen - Gaussian Processes for Regression]]
- [[Wang e Kulhawy 2008 - Economic Design Optimization of Foundations]]
- [[Gandomi e Kashani 2018 - Cost Minimization Shallow Foundation]]
- [[Kashani et al. 2020 - Optimum Design of Shallow Foundation]]
- [[Nigdeli et al. 2018 - Metaheuristic Optimization RC Footings]]
- [[Waheed et al. 2022 - Practical Tool RC Isolated Footings]]
- [[Waheed et al. 2025 - Economical Design RC Isolated Footings]]
- [[Gomes et al. 2018 - Probabilistic Metric Metaheuristics]]

### Apoio técnico / contexto
- [[Ahmad et al. 2021 - GPR Bearing Capacity Shallow Foundations]]
- [[Santos et al. 2018 - Punching Shear RC Footings]]
- [[Khajehzadeh et al. 2022 - Hybrid Soft Computing Shallow Foundations]]
- [[Fattahi et al. 2025 - Settlement Prediction Intelligent Optimization]]
- [[Bezerra et al. 2024 - Elementos de Fundacao]]
- [[Khan et al. 2023 - Python Automation AEC]]
- [[NBR 6122 1996 - Projeto e Execucao de Fundacoes]]
- [[Rasheed et al. 2017 - Optimization Shallow Foundation GSA]]
- [[Morales-Castaneda et al. 2020 - Balance in Metaheuristics]]
- [[Abualigah et al. 2021 - Arithmetic Optimization Algorithm]]

### Frente C — CBO (implementada em 2026-07-11)
- [[Gardner et al. 2014 - Bayesian Optimization with Inequality Constraints]]
- [[Eriksson e Poloczek 2021 - Scalable Constrained BO]]
- [[Mathern et al. 2021 - Multiobjective Constrained BO Structural Design]]
- [[Yu et al. 2025 - PFN Constrained Engineering BO]]

### Triados e fora do escopo do artigo 1
- [[Chandra et al. 2021 - Bored Pile Cost Optimization]]
- [[Jakubczyk-Galczynska et al. 2024 - Construction Management Bayesian Networks]]
- [[Duplicatas detectadas - 2026-07-12]]

### Próximas frentes / uso futuro
- [[Juang e Wang 2013 - Reliability Robust Spread Foundations]]
- [[G09-002 - Geotechnical Engineering Shallow Foundations]]
- [[Mbock et al. 2019 - Optimal Forms Shallow Foundations]]
- [[Deng et al. 2026 - Metamaterial Autoencoder Active Learning]]
- [[The application of Bayesian methods - OCR pendente]]

---

> [!tip] Como adicionar
> 1. Criar `08_Artigos/<Nome do Artigo>.md` baseado em [[99_Templates/Template - Artigo]].
> 2. Adicionar `- [[Nome do Artigo]]` à lista acima.
> 3. Linkar ao tópico do projeto que o artigo aborda (ex.: `relevante para [[02_Engenharia/Verificação à Punção]]`).
