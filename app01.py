import streamlit as st 
import pandas as pd
from my_example import obj_ic_fundacoes, tensao_adm_solo
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_sapatas(df_resultado, h_x, h_y):
    """
    Gera uma figura matplotlib com a planta das sapatas.

    Parâmetros:
        df_resultado (DataFrame): Resultado contendo colunas 'Elemento', 'xg (m)', 'yg (m)'
        h_x (float): Dimensão da sapata em x (largura)
        h_y (float): Dimensão da sapata em y (comprimento)

    Retorna:
        fig (matplotlib.figure.Figure): Figura com a plotagem
    """
    labels = df_resultado['Elemento'].tolist()
    x = df_resultado['xg (m)'].tolist()
    y = df_resultado['yg (m)'].tolist()

    fig, ax = plt.subplots(figsize=(10, 10))

    for i in range(len(x)):
        rect = patches.Rectangle((x[i] - h_x/2, y[i] - h_y/2), h_x, h_y,
                                 linewidth=1, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)
        ax.scatter(x[i], y[i], color='red', marker='+', s=100)
        ax.annotate(labels[i], (x[i], y[i]), textcoords="offset points", xytext=(0, 10), ha='center')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Planta das Sapatas')
    ax.grid(True)
    ax.set_aspect('equal')
    return fig


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
                                        <li>Fz-ci (kN): Força vertical aplicada no pilar da combinação i(kN)</li> 
                                        <li>Mx-ci (kNm): Momento fletor em torno do eixo x (kNm) da combinação i</li>
                                        <li>My-c1 (kNm): Momento fletor em torno do eixo y (kNm) da combinação i</li>
                                </ul>  
                        <li>Não modifique o cabeçalho da planilha, pois o aplicativo faz referência a ele.</li>
                        <li>Qualquer numero de combinações podem ser informados na planilha, seguindo o padrão de nomenclatura.</li>
                        <li>Aplicação em construção, atualmete ela é capaz de analisar para uma dada dimensão de sapata se passa ou não em varias verificações.</li>
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
    
    fck = st.number_input('Concreto fck (KPa)', value=25000)
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
    if st.button("calacular"):
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

        fig = plot_sapatas(df_resultado, h_x, h_y)
        st.pyplot(fig)