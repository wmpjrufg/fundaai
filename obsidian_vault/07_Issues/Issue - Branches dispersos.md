---
tags: [issue, baixo, devops]
severity: baixo
---

# Issue — Branches dispersos no remote

## Sintoma

`git branch -a` mostra 16+ branches remotos:

```
IC_Filiipe (typo)
Teste_puncao
Teste_punção (com cedilha — incompatibilidade Unicode)
combinaoes_separadas (typo)
dev, dev-lucas, dev-planilhas-wander, dev-wander-final,
dev-wander-otm, dev_filipe_final, dev_filipe_finaliza,
dev_wander_relatorio, feature/dev-lucas, teste_geral
```

## Por que é problema

- Difícil saber qual é a fonte da verdade.
- Branches com cedilha podem causar problemas em alguns sistemas.
- Trabalho potencialmente perdido / esquecido.

## Correção sugerida

Auditoria conduzida pela equipe do projeto sob orientação do Prof. Wanderley:
1. Identificar branches já mergeadas (`git branch -a --merged main`).
2. Deletar as redundantes do remote.
3. Manter apenas: `main`, `refactor/code-base` (atual), e talvez 1 ou 2 features ativas.

⚠️ Não executar sem confirmação prévia da equipe — pode haver código não commitado em uma branch antiga.

## Vínculo

- [[01_Projeto/Atores e Histórico]]
