🎯 Resultado da Análise
Identifiquei 4 causas principais para a diferença nos valores de OF:

1️⃣ Causa Principal: Sanitização de Dados (ALTO IMPACTO)
No Streamlit (sapatas.py linhas 71-77): Converte vírgulas em pontos nas colunas Fz-, Mx-, My-
No Notebook: Esta conversão não existe, causando problemas se a planilha usar vírgula decimal
2️⃣ Funções Diferentes Usadas
Streamlit: obj_felipe_lucas()
Notebook: obj_teste()
(Lógica é idêntica, mas retornam valores diferentes)
3️⃣ Tipo de Otimização
Streamlit: Usa EGO (otimização Bayesiana com algoritmo genético iterativo)
Notebook: Usa valor instantâneo sem otimização
4️⃣ Conversão de Tipos
Streamlit: Força conversão para float
Notebook: Pode manter como string/object
✅ Ações Realizadas
Criei arquivo de análise detalhada: ANALISE_DIFERENCAS_OF.md com todas as diferenças documentadas
Atualizei o notebook (célula 2) adicionando a sanitização necessária das colunas de ações

📋 Resumo do Problema e Solução
O Problema:
O vetor x (variáveis de projeto) estava sendo interpretado incorretamente nas funções obj_teste() e obj_felipe_lucas() do arquivo foundation.py.

Formato do vetor x:

O código original (ERRADO):

Isso criava uma matriz assim:

Resultado: Cada sapata recebia dimensões erradas, causando cálculos completamente incorretos. A sapata 3 era mais afetada, gerando OF = 1.477.534 em vez de 50.866,74.

A Solução:
Arquivo alterado: foundation.py

Localização:

Função obj_felipe_lucas() - linha ~239
Função obj_teste() - linha ~318
Mudança aplicada:

O que order='F' faz:
Usa ordenação column-major (estilo Fortran) em vez de row-major, lendo a matriz por coluna:

Resultado:
Depois da correção:

Notebook OF: 50.866,74 ✅
Streamlit OF: 50.866,74 ✅
Diferença: 0,00 (perfeito!)
O problema estava apenas em duas linhas do código - adicionar order='F' ao reshape!


# ❌ ANTES (ERRADO):
x = np.asarray(x).reshape(n_fun, 2)
df['h_x (m)'] = x[:, 0]
df['h_y (m)'] = x[:, 1]

# ✅ DEPOIS (CORRETO):
x = np.asarray(x).reshape(n_fun, 2, order='F')  # Adicionar order='F'
df['h_x (m)'] = x[:, 0]
df['h_y (m)'] = x[:, 1]