---
tags: [guia, validacao, artigo, roadmap, packing, ego, gpr]
status: ativo
---

# Guia - Validação antes do Bin Packing

> [!summary] Decisão atual
> Antes de entrar na frente de **bin packing/layout**, fechar e validar a etapa atual: FundaIA + EGO-GPR para dimensionamento otimizado de sapatas isoladas, com posições fixas e restrição preliminar de sobreposição.

Esta nota é o roteiro curto para execução. O relatório completo está em [[09_Relatorios/Analise - Roadmap Artigo IC - 2026-04-27]].

## Por que validar primeiro

O projeto já tem uma entrega com potencial de artigo: uma arquitetura EGO-GPR integrada ao FundaIA. Mas, para publicar ou defender resultados, ela precisa estar reprodutível e tecnicamente fechada.

O bin packing completo deve vir depois porque muda a natureza do problema: além de dimensionar sapatas, passa a decidir posição, excentricidade, proximidade, fronteira do lote, possíveis sapatas associadas e interação geotécnica.

## Gate 0 - Escopo honesto da etapa atual

Usar esta formulação:

> Dimensionamento otimizado de sapatas isoladas com restrições geométricas, geotécnicas e estruturais, usando EGO-GPR, considerando posições de pilares/sapatas fornecidas como entrada e uma restrição preliminar de sobreposição em planta.

Evitar, por enquanto:

- "bin packing implementado";
- "layout otimizado";
- "posicionamento completo";
- "bulbo de tensão modelado";
- "sapatas associadas automáticas".

## Gate 1 - Saneamento mínimo

Antes de rodar experimentos finais:

- [ ] Corrigir [[07_Issues/Issue - requirements.txt UTF-16]].
- [ ] Remover [[07_Issues/Issue - Duplicação em sapatas.py]].
- [ ] Decidir [[07_Issues/Issue - obj_felipe_lucas vs obj_teste]].
- [ ] Resolver [[07_Issues/Issue - Args extras em obj_teste]].
- [ ] Corrigir [[07_Issues/Issue - Histórico do EGO com ITER e ID incorretos]].
- [ ] Corrigir [[07_Issues/Issue - n_rep reusa população inicial]].
- [ ] Decidir/documentar [[07_Issues/Issue - Sobreposição contada duas vezes]].
- [ ] Sanear [[07_Issues/Issue - Benchmarks suspeitos]] se for usar benchmarks.
- [ ] Decidir oficialmente a convenção de [[03_Otimizacao/Kernels GPR]]: 20 experimentais + 1 produção, ou 21 kernels.

Saída esperada: app instalável, código sem duplicações críticas e logs de EGO confiáveis.

## Gate 2 - Validação de engenharia

Antes de comparar algoritmos, validar que a função objetivo calcula corretamente:

- [ ] Testar `tensao_adm_solo`.
- [ ] Testar `calcular_sigma_max_min`.
- [ ] Testar `checagem_geometria`.
- [x] Testar `verificacao_puncao_sapata` na seção C.
- [x] Implementar e testar a seção C' a `2d` da face.
- [ ] Testar `sobreposicao_sapatas`.
- [ ] Criar um exemplo manual pequeno, calculado fora do otimizador, para comparar com o código.

Saída esperada: confiança de que o otimizador está otimizando uma função objetivo tecnicamente coerente.

## Gate 3 - Validação experimental ✅ (2026-07-10)

Concluído na [[12_Auditoria/Sprint 5.1 - Protocolo experimental final e casos-limite - 2026-07-10]]:

- [x] Congelar casos de 1, 2 e 3 sapatas. *(problema_fund_um/dois/três; h ∈ [0,60; 3,00] m)*
- [x] Rodar EGO-GPR com seeds registradas. *(30 seeds, 42–71)*
- [x] Rodar busca aleatória com mesmo orçamento de avaliações reais. *(algoritmo `random` na bancada)*
- [x] Baselines de metaheurística: GA, PSO e GWO (além do planejado).
- [x] 30 seeds por caso e cenário.
- [x] Reportar melhor volume factível, média±DP, mediana, taxa de factibilidade, violação máxima, tempo e avaliações. *(summary + per_rep da bancada)*
- [x] Recriar figuras/tabelas por script determinístico. *(`scripts/make_paper_artifacts.py`)*

Saída obtida: Tabelas 5–6 e Figuras 3–5 do artigo, com p-valores de Mann–Whitney.

## Gate 4 - Artigo 1

Tema recomendado:

> FundaIA: otimização assistida por processos gaussianos para dimensionamento de sapatas isoladas de concreto armado.

Contribuições defendáveis:

- formulação penalizada do dimensionamento;
- integração de restrições de engenharia;
- EGO-GPR com Expected Improvement e GA interno;
- ferramenta interativa em Streamlit;
- estudo de kernels/penalização;
- comparação com Monte Carlo/random search e, se possível, GA puro.

## Próxima etapa - Bin Packing

Só iniciar como frente principal depois que os gates acima estiverem fechados ou formalmente assumidos como limites.

Quando chegar lá, buscar artigos sobre:

- strip packing 2D;
- bin packing 2D;
- no-fit polygon;
- phi-functions para packing;
- layout optimization;
- sapatas associadas/combinadas;
- bulbo de tensão;
- Boussinesq, Newmark ou método 2:1 de distribuição de tensões;
- recalque e interação entre fundações próximas.

Nota de referência atual: [[03_Otimizacao/Problema de Empacotamento]].

## Leituras-base para esta etapa

- [[08_Artigos/Index de Artigos]].
- Jones, Schonlau e Welch (1998) - EGO.
- Snoek et al. (2012) - Bayesian Optimization.
- Shahriari et al. (2016) - revisão de Bayesian Optimization.
- Schulz et al. (2018) - GPR.
- Wang e Kulhawy (2008) - otimização econômica de fundações.
- Gandomi e Kashani (2018) - metaheurísticas em fundações rasas.
- Waheed et al. (2022, 2025) - ferramenta/otimização de sapatas isoladas.
- Gomes et al. (2018) - comparação probabilística de metaheurísticas.

## Vínculos

- [[09_Relatorios/Analise - Roadmap Artigo IC - 2026-04-27]]
- [[10_Melhorias/Roadmap Sugerido]]
- [[12_Auditoria/Auditoria 2026-04-27 - Vault vs Projeto]]
- [[03_Otimizacao/Problema de Empacotamento]]
- [[11_Frentes_de_Pesquisa/Posicionamento Conjunto - Layout + Sizing]]
