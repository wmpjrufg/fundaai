import streamlit as st 
import pandas as pd

from my_example import obj_ic_fundacoes, tensao_adm_solo
from metapy_toolbox import metaheuristic_optimizer


uploaded_file = st.file_uploader("Uploaded file", type=['xlsx'])
if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)

    if st.button('Calculate'):
        df = tensao_adm_solo(data)
        dim_a = []
        dim_b = []

        for i, row in df.iterrows():
                # Recolhendo os dados do pilar
                f_zmax = row['Fz,max (kN)']
                f_zmin = row['Fz,min (kN)']
                m_xmax = row['Mx,max (kN.m)']
                m_xmin = row['Mx,min (kN.m)']
                m_ymax = row['My,max (kN.m)']
                m_ymin = row['My,min (kN.m)']
                sigma_lim_solo = row['sigma_adm (kPa)']
                dados_fundacao = { 
                                        'Fz,max (kN)': f_zmax,
                                        'Fz,min (kN)': f_zmin,
                                        'Mx,max (kN.m)': m_xmax,
                                        'Mx,min (kN.m)': m_xmin,
                                        'My,max (kN.m)': m_ymax,
                                        'My,min (kN.m)': m_ymin,
                                        'sigma_adm (kPa)': sigma_lim_solo,
                                }
                
                # Otimização
                algorithm_setup = {   
                        'number of iterations': 100,
                        'number of population': 10,
                        'number of dimensions': 2,
                        'x pop lower limit': [0.30, 0.30], # 0.60
                        'x pop upper limit': [2.25, 2.25],
                        'none variable': dados_fundacao,
                        'objective function': obj_ic_fundacoes,
                        'algorithm parameters': {
                                                'selection': {'type': 'roulette'},
                                                'crossover': {'crossover rate (%)': 82, 'type':'linear'},
                                                'mutation': {'mutation rate (%)': 12, 'type': 'hill climbing', 'cov (%)': 15, 'pdf': 'gaussian'},
                                                }
                        }
                n_rep = 10
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

        st.title('Results')
        st.table(df)
        
else:
    st.warning('Please, upload a file')

with open("input.xlsx", "rb") as file:
    st.download_button(
        label="Download example data",
        data=file,
        file_name="example_data.xlsx",
        mime="text/csv"
    )