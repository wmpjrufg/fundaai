---
tags: [melhorias, ciencia, validacao, sugestao]
---

# Validação contra problema-benchmark

> [!note] Sugestão
> Para defender resultados na IC, é essencial confrontar a saída do FundaIA com:
>
> 1. **Cálculo manual** ou **planilha clássica** de um exemplo de bibliografia.
> 2. **Software comercial** (TQS, Eberick, SAP2000) sobre o mesmo input.
> 3. **Função-benchmark de otimização** (ver [[04_Codigo/metapy_toolbox - benchmark.py]]) para validar o algoritmo isoladamente.

## Bateria proposta

### A. Engenharia
- Exemplo do livro de Carvalho & Pinheiro (vol. 2, sapatas) ⇒ verificar σ_max, punção, geometria.
- Exemplo do livro de Bastos (USP/EESC) sobre fundações.

### B. Otimização
- `sphere`, `rosenbrock`, `rastrigin`, `ackley` em D=10, D=30 ⇒ EGO+GA deve achar mínimo conhecido.
- Funções com restrições: `g05`, `g06`, `g08` (Liang et al., CEC 2006 benchmark).

### C. Acoplado (mais difícil — caso a construir internamente)
- Sapata isolada com solução **analítica conhecida** (caso simétrico, sem momentos): h_min teórico depende só de F/σ_adm.

## Métricas

- Erro relativo vs solução de referência.
- Taxa de violação de restrições.
- Número de avaliações até convergência.

## Vínculos

- [[10_Melhorias/Testes Automatizados]]
- [[10_Melhorias/Persistência de Experimentos]]
- [[04_Codigo/metapy_toolbox - benchmark.py]]
