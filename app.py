import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import ezdxf
import tempfile

from io import BytesIO
from my_example import obj_ic_fundacoes, tensao_adm_solo, data_comb
from metapy_toolbox import metaheuristic_optimizer

# Função para plotar o gráfico com os quadrados
def plot_data(data):
    """
    Plota um gráfico com quadrados azuis em torno dos pontos e anotações a partir dos dados fornecidos.
    Retorna a figura gerada.

    Parâmetros:
    data (dict): Um dicionário com chaves 'label', 'x', 'y', 'L x', e 'L y'.
                 'label' deve ser uma lista de rótulos, 'x' e 'y' devem ser listas de coordenadas.
                 'L x' e 'L y' definem as dimensões dos quadrados ao redor dos pontos.
                 
    Retorna:
    plt.Figure: A figura gerada.
    """
    labels = data['label']
    x = data['x']
    y = data['y']
    L_x = data['L x']
    L_y = data['L y']

    # Criando a figura e os eixos
    fig, ax = plt.subplots(figsize=(10, 10))

    # Desenhando os quadrados e pontos
    for i in range(len(x)):
        # Adiciona um quadrado azul ao redor do ponto
        square = patches.Rectangle((x[i] - L_x[i] / 2, y[i] - L_y[i] / 2),
                                   L_x[i], L_y[i], linewidth=1, edgecolor='blue', facecolor='none')
        ax.add_patch(square)

        # Adiciona o ponto
        ax.scatter(x[i], y[i], color='red', marker='+', s=100)

        # Adiciona a anotação
        ax.annotate(labels[i], (x[i], y[i]), textcoords="offset points", xytext=(0, 10), ha='center')

    # Configurando o gráfico
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Posicionamento das sapatas')
    ax.grid(True)

    # Retorna a figura
    return fig

def save_dxf(data):
    """
    Função para salvar os dados como um arquivo DXF para AutoCAD.

    Parâmetros:
    data (dict): Dados contendo as coordenadas e dimensões dos quadrados e pontos.
    filename (str): Nome do arquivo DXF a ser salvo.
    """
    labels = data['label']
    x = data['x']
    y = data['y']
    L_x = data['L x']
    L_y = data['L y']

    # Criar um novo documento DXF
    doc = ezdxf.new(dxfversion='R2010')
    msp = doc.modelspace()

    # Adicionar os quadrados e pontos ao arquivo DXF
    for i in range(len(x)):
        # Coordenadas do quadrado (quatro cantos)
        p1 = (x[i] - L_x[i] / 2, y[i] - L_y[i] / 2)
        p2 = (x[i] + L_x[i] / 2, y[i] - L_y[i] / 2)
        p3 = (x[i] + L_x[i] / 2, y[i] + L_y[i] / 2)
        p4 = (x[i] - L_x[i] / 2, y[i] + L_y[i] / 2)

        # Desenhar o quadrado no DXF usando linhas
        msp.add_line(p1, p2)
        msp.add_line(p2, p3)
        msp.add_line(p3, p4)
        msp.add_line(p4, p1)

        # Adicionar o ponto central
        msp.add_point((x[i], y[i]))

        # Adicionar o rótulo próximo ao ponto
        msp.add_text(labels[i], dxfattribs={'height': 0.2}).set_dxf_attrib('insert', (x[i], y[i]))

    # Salvar o arquivo DXF em um arquivo temporário
    temp_file_path = tempfile.NamedTemporaryFile(delete=False, suffix='.dxf').name
    doc.saveas(temp_file_path)
    print(f"Arquivo DXF temporário '{temp_file_path}' salvo com sucesso.")
    
    # Ler o arquivo DXF e retornar os dados binários
    with open(temp_file_path, "rb") as file:
        return file.read()
    

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
                                        <li>spt: spt </li>
                                        <li>solo: Tipo de solo</li>                                              
                                        <li>xg (m): Coordenada x do pilar (m)</li>
                                        <li>yg (m): Coordenada y do pilar (m)</li>
                                        <li>fz (kN): Força vertical aplicada no pilar (kN)</li> 
                                        <li>mx (kNm): Momento fletor em torno do eixo x (kNm)</li>
                                        <li>my (kNm): Momento fletor em torno do eixo y (kNm)</li>
                                        <li>combinação n: com as subcolunas Fz, Mx e My</li>
                                </ul>  
                        <li>Não modifique o cabeçalho da planilha, pois o aplicativo faz referência a ele.</li>
                        <li>Qualquer numero de combinações podem ser informados na planilha, seguindo o padrão de nomenclatura.</li>
                        </li>
                </ul>   
         
         Você pode baixar um arquivo de exemplo clicando no botão abaixo.
        ''', unsafe_allow_html=True)

with open("planilha_padrao.xlsx", "rb") as file:
    st.download_button(
        label="Download example data",
        data=file,
        file_name="example_data.xlsx",
        mime="text/csv"
    )

uploaded_file = st.file_uploader("Uploaded file", type=['xlsx'])
if uploaded_file is not None:
    data = pd.read_excel(uploaded_file, header=[0,1])

    dim_min = st.number_input('Dimensão minima da sapata (m)', value=0.60)
    if dim_min < 0.60:
        st.warning('Dimensão mínima da sapata deve ser maior ou igual a 0.60')
    dim_max = st.number_input('Dimensão máxima da sapata (m)', value=2.25)
    
    if st.button('Calculate'):
        df = tensao_adm_solo(data)
        lista_comb = data_comb(df)

        dim_a = []
        dim_b = []
        for i, row in df.iterrows():
                # Recolhendo os dados do pilar
                dados_fundacao = { 
                                        'combinações': lista_comb[i],
                                        'sigma_adm (kPa)': row[('sigma_adm (kPa)', '')],
                                }
                # Otimização
                algorithm_setup = {   
                        'number of iterations': 100,
                        'number of population': 5,
                        'number of dimensions': 2,
                        'x pop lower limit': [dim_min, dim_min], # 0.60
                        'x pop upper limit': [dim_max, dim_max], # 2.25
                        'none variable': dados_fundacao,
                        'objective function': obj_ic_fundacoes,
                        'algorithm parameters': {
                                                'selection': {'type': 'roulette'},
                                                'crossover': {'crossover rate (%)': 82, 'type':'linear'},
                                                'mutation': {'mutation rate (%)': 12, 'type': 'hill climbing', 'cov (%)': 15, 'pdf': 'gaussian'},
                                                }
                        }
                n_rep = 5
                general_setup = {   
                        'number of repetitions': n_rep,
                        'type code': 'real code',
                        'initial pop. seed': [None] * n_rep,
                        'algorithm': 'genetic_algorithm_01',
                        }
                df_all_reps, df_resume_all_reps, reports, status = metaheuristic_optimizer(algorithm_setup, general_setup)
                df_novo = df_resume_all_reps[status]
                dimensoes_sapata_a = list(df_novo['X_0_BEST'])[-1]
                dimensoes_sapata_b = list(df_novo['X_1_BEST'])[-1]
                dim_a.append(dimensoes_sapata_a)
                dim_b.append(dimensoes_sapata_b)

        # Atribuição da dimensão otimizada
        df['dimensão hx (m)'] = dim_a
        df['dimensão hy (m)'] = dim_b
        
        st.title('Resultados:')
        df = df[[col for col in df.columns if not col[0].startswith("combinação")]]
        df.columns = df.columns.get_level_values(0)
        st.table(df)

        excel_file = BytesIO()
        df.to_excel(excel_file, index=True)

        # Colocar o ponteiro no início do arquivo
        excel_file.seek(0)

        # Adicionar o botão de download
        st.download_button(
        label="Download data",
        data=excel_file,
        file_name="result_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # Plotar o gráfico
        data_plot = {
            'label': df.index,
            'x': df['xg (m)'],
            'y': df['yg (m)'],
            'L x': df['dimensão hx (m)'],
            'L y': df['dimensão hy (m)']
        }
        fig = plot_data(data_plot)
        st.pyplot(fig)

        # Salvar o arquivo DXF 
        dxf_data = save_dxf(data_plot)
        st.download_button(
            label="Download DXF",
            data=dxf_data,
            file_name="sapatas.dxf",
            mime="application/octet-stream"
        )
else:
    st.warning('Please, upload a file')

