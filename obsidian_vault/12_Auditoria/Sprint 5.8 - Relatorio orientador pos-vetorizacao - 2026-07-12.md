# Sprint 5.8 - Relatório para orientação pós-vetorização

**Data:** 2026-07-12  
**Objetivo:** documentar, em linguagem didática, tudo que mudou depois da última versão vista pela orientação, cujo ponto de corte foi a vetorização da verificação de sobreposição entre sapatas na Sprint 3.8.

## Artefatos gerados

- `docs/relatorios/relatorio_orientador_pos_vetorizacao_2026-07-12.md`
- `docs/relatorios/relatorio_orientador_pos_vetorizacao_2026-07-12.docx`
- `docs/relatorios/relatorio_orientador_pos_vetorizacao_2026-07-12.pdf`

Observação: a pasta `docs/` está ignorada pelo Git pela regra atual do `.gitignore`, então esses arquivos existem localmente, mas não aparecem automaticamente em `git status`.

## Conteúdo coberto

- Linha do tempo desde a Sprint 3.8 até a Sprint 5.7.
- Explicação didática da função objetivo, restrições e convenção `g <= 0`.
- Vetorização da sobreposição e ganho de desempenho.
- Correção da tensão no solo, com peso próprio real e convenção dos momentos.
- Implementação da punção no contorno `C'` a `2d`.
- Validações de fronteira: `Fz > 0`, `h_z > cobrimento` e unidade de `f_ck`.
- Cache do GPR, persistência de experimentos e logs estruturados.
- Melhorias de interface: 3D, curva do EGO, progresso ao vivo, cancelamento e página de experimentos.
- Protocolo experimental do artigo: casos, orçamento, Wilcoxon pareado e correção de Holm.
- Estudo de penalidade, kernels do GPR e motivação do CBO.
- CBO com modelos separados para volume e restrições.
- Auditoria de decomposição por sapata e interpretação de quase separabilidade.
- Piloto da Fase B com deslocamentos `dx, dy` e restrições de packing.
- Estado do artigo em `docs/artigo_ic_lucas`.
- Pendências honestas antes de submissão: `N_spt`-tensão admissível, combinações de ações, escopo de projeto executivo e protocolo completo da Fase B.

## Revisão realizada

- O documento foi revisado para distinguir implementação consolidada, evidência experimental do artigo 1 e piloto futuro.
- A versão Markdown foi acentuada e revisada para envio acadêmico.
- A versão `.docx` foi gerada com estilo de relatório técnico e renderizada.
- O PDF final foi gerado a partir do `.docx`.
- Renderização visual conferida em páginas representativas: primeira página, página técnica intermediária e página final.

