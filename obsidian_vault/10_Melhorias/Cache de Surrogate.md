---
tags: [melhorias, performance, sugestao]
---

# Cache de Surrogate

> [!note] Sugestão
> No EGO atual, `pipe.fit(x_train, y_train)` é chamado a **cada iteração** com a base inteira (incluindo o último ponto). Como `GPR` é `O(n³)`, isso fica caro com a base crescendo.

## Estratégias

### 1. Recomputar só o necessário
- `GPR` permite atualizações incrementais? Não no sklearn nativo, mas **GPyTorch** e **scikit-bench** sim.
- Alternativa simples: re-treinar a cada `K` iterações (e usar o GPR antigo como ponto inicial dos hiperparâmetros nas demais).

### 2. Persistir entre runs
- Hoje cada execução começa do zero. Em produção (Streamlit Cloud), poderia haver cache:
   ```python
   @st.cache_data
   def avaliar_e_treinar(projeto_hash, x): ...
   ```
- Ou guardar pares `(x, of)` num parquet em `experiments/cache/<projeto_hash>.parquet` e usar como bootstrap quando o usuário re-roda o mesmo projeto.

### 3. Pré-treinar offline
- Já existem 118 `.pkl` em [[05_Dados/Modelos GPR Treinados]] — mas a UI nunca os usa.
- Investigar se algum desses GPRs serve como **prior** para um problema novo (transfer learning).

## Vínculos

- [[03_Otimizacao/EGO - Efficient Global Optimization]]
- [[03_Otimizacao/Gaussian Process Regressor]]
- [[10_Melhorias/Refactor - Vetorização da FO]]
- [[11_Frentes_de_Pesquisa/Surrogate Multifidelidade]]
