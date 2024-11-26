import streamlit as st 
import pandas as pd

from io import BytesIO
from my_example import obj_ic_fundacoes, tensao_adm_solo, data_comb
from metapy_toolbox import metaheuristic_optimizer


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
        df['dimensão a (m)'] = dim_a
        df['dimensão b (m)'] = dim_b
        
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
else:
    st.warning('Please, upload a file')

with open("planilha_padrao.xlsx", "rb") as file:
    st.download_button(
        label="Download example data",
        data=file,
        file_name="example_data.xlsx",
        mime="text/csv"
    )