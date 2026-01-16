"""Aplicativo Streamlit para dimensionamento de sapatas isoladas."""
import streamlit as st
from pathlib import Path

# Título do aplicativo
st.title('Dimensionamento de Sapatas')
st.write(r"""
            <p style="text-align: justify;">
            Este aplicativo tem como objetivo auxiliar no dimensionamento de sapatas isoladas,
            considerando a resistência do solo e as cargas aplicadas pelos pilares. Para isso, é necessário
            que o usuário forneça um arquivo Excel com os dados de entrada, conforme o exemplo
            disponível para download.
            </p>

            <h2>Observações:</h2>
            <ul>
            <li>O arquivo de entrada deve conter as seguintes colunas:
                <ul>
                <li>Elemento: Nome do elemento</li>
                <li>ap (m): dimensão x do pilar (m)</li>
                <li>bp (m): dimensão y do pilar (m)</li>
                <li>spt: spt</li>
                <li>solo: Tipo de solo</li>
                <li>xg (m): Coordenada x do pilar (m)</li>
                <li>yg (m): Coordenada y do pilar (m)</li>
                <li>Fz-ci (kN): Força vertical aplicada no pilar da combinação i (kN)</li>
                <li>Mx-ci (kN.m): Momento fletor em torno do eixo x (k.m) da combinação i</li>
                <li>My-ci (kN.m): Momento fletor em torno do eixo y (k.m) da combinação i</li>
                </ul>
            </li>

            <li>Não modifique o cabeçalho da planilha, pois o aplicativo faz referência a ele.</li>
            <li>Qualquer número de combinações pode ser informado na planilha, seguindo o padrão de nomenclatura.</li>

            <li>
                Os índices finais <strong>ci</strong>, <strong>ci</strong> e <strong>ci</strong> associados às ações
                (<em>Fz</em>, <em>Mx</em> e <em>My</em>) indicam a <strong>combinação de carregamento</strong> à qual cada
                valor pertence. Dessa forma, por exemplo, <em>Fz-c1</em>, <em>Mx-c1</em> e <em>My-c1</em> correspondem
                às ações da combinação 1, enquanto <em>Fz-c2</em>, <em>Mx-c2</em> e <em>My-c2</em> referem-se à combinação 2,
                e assim sucessivamente.
            </li>

            <li>A aplicação atualmente é capaz de analisar, para cada uma das sapatas as seguintes restrições: tensão no solo, tensão de punção e geometria mínima considerando a referência do pilar.</li>
            </ul>

            <p>Você pode baixar um arquivo de exemplo clicando no botão abaixo.</p>
""", unsafe_allow_html=True)

# Planilha padrão
download_template(path="assets/sapata_16.xlsx", label="📥 Baixar planilha modelo (Excel)", filename="modelo_entrada_sapatas.xlsx")
