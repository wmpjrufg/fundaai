import streamlit as st 
import pandas as pd
from my_example import obj_ic_fundacoes, tensao_adm_solo

st.title('Dimensionamento de Sapatas')
st.write(r'''
        <p style="text-align: justify;">
        Este aplicativo tem como objetivo auxiliar no dimensionamento de sapatas isoladas,
        considerando a resistência do solo e as cargas aplicadas. Para isso, é necessário
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
                                        <li>spt: spt </li>
                                        <li>solo: Tipo de solo</li>                                                                                    
                                        <li>xg (m): Coordenada x do pilar (m)</li>
                                        <li>yg (m): Coordenada y do pilar (m)</li>
                                        <li>Fz-ci (kN): Força vertical aplicada no pilar da combinação 1(kN)</li> 
                                        <li>Mx-ci (kNm): Momento fletor em torno do eixo x (kNm) da combinação i</li>
                                        <li>My-c1 (kNm): Momento fletor em torno do eixo y (kNm) da combinação i</li>
                                </ul>  
                        <li>Não modifique o cabeçalho da planilha, pois o aplicativo faz referência a ele.</li>
                        <li>Qualquer numero de combinações podem ser informados na planilha, seguindo o padrão de nomenclatura.</li>
                        </li>
                </ul>   
         
         Você pode baixar um arquivo de exemplo clicando no botão abaixo.
        ''', unsafe_allow_html=True)

#Botão para baixar tabela exemplo
with open("planilha_padrao.xlsx", "rb") as file:
    st.download_button(
        label="Download example data_não padronizado",
        data=file,
        file_name="example_data.xlsx",
        mime="text/csv"
    )

uploaded_file = st.file_uploader("Uploaded file", type=['xlsx'])

if uploaded_file is not None:
    data = pd.read_excel(uploaded_file,)
    
    fck = st.number_input('Concreto fck (KPa)', value=2500)
    if fck < 20000:   
        st.warning('Concreto fck deve ser maior ou igual a 20000')

    cob = st.number_input('Recobrimnento (m)', value=0.025)

    n_comb = st.number_input('Número de combinações estruturais', value=1)
    if n_comb < 1:
        st.warning('Número de combinações estruturais deve ser maior ou igual a 1')

    h_x = st.number_input('Dimensão x da sapata (m)', value=0.6)
    if h_x < 0.6:
        st.warning('Dimensão x da sapata deve ser maior ou igual a 0.6')
    h_y = st.number_input('Dimensão y da sapata (m)', value=0.6)
    if h_y < 0.6:
        st.warning('Dimensão y da sapata deve ser maior ou igual a 0.6')
    h_z = st.number_input('Dimensão z da sapata (m)', value=0.6)
    if h_z < 0.6:
        st.warning('Dimensão z da sapata deve ser maior ou igual a 0.6')

    x = [h_x, h_y]

    data[('fck (kPa)')] = fck
    data[('cob (m)')] = cob
    data[('n_comb')] = n_comb
    data[('h_z (m)')] = h_z

    data = tensao_adm_solo(data)

    none_variable = {'cob (m)': cob, 'fck (kPa)': fck, 'número de combinações estruturais': n_comb, 'dados estrutura': data, 'h_z (m)': h_z}

    resultado, df_resultado = obj_ic_fundacoes(x, none_variable)
    st.success(f'Volume otimizado (com penalização por restrições): {resultado:.4f} m³')
    st.dataframe(df_resultado)

    
