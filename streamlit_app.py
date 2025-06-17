# streamlit_app.py

import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os
import logging

logging.getLogger('streamlit.runtime.scriptrunner').setLevel(logging.ERROR)

# === Carregar dados ===
base_dir = os.path.dirname(os.path.abspath(__file__))
arquivo = os.path.join(base_dir, 'data', 'base_vendas_24.xlsx')

if not os.path.exists(arquivo):
    st.error(f"Arquivo não encontrado: {arquivo}")
    st.stop()

df = pd.read_excel(arquivo, sheet_name='Base vendas')
df.columns = df.columns.str.strip()
df['Emissao'] = pd.to_datetime(df['Emissao'], errors='coerce')
df = df.dropna(subset=['Emissao', 'Cliente', 'Produto', 'Quantidade'])
df['AnoMes'] = df['Emissao'].dt.to_period('M').dt.to_timestamp()
df['Cliente'] = df['Cliente'].astype(str)
df['Produto'] = df['Produto'].astype(str)
df = df[df['AnoMes'] >= '2024-01-01']

agrupado = df.groupby(['Cliente', 'Produto', 'AnoMes'])['Quantidade'].sum().reset_index()

# === Função de previsão ===
def prever(cliente, produto):
    dados = agrupado[(agrupado['Cliente'] == cliente) & (agrupado['Produto'] == produto)].copy()
    dados = dados.sort_values('AnoMes')
    dados['Previsao'] = 'Histórico'

    if dados.empty:
        return dados, "Sem dados suficientes para previsão."

    serie = dados.set_index('AnoMes')['Quantidade']
    serie.index = pd.date_range(start=serie.index.min(), periods=len(serie), freq='MS')

    try:
        modelo = ExponentialSmoothing(
            serie,
            trend='add',
            damped_trend=True,  # TORNAR A PREVISÃO MAIS REALISTA
            seasonal=None,
            initialization_method='estimated'
        )
        ajuste = modelo.fit()

        # Previsão para os próximos 6 meses, aplicando redutor de 10%
        previsao = (ajuste.forecast(6) * 0.9).round().astype(int)

        previsao = previsao.reset_index()
        previsao.columns = ['AnoMes', 'Quantidade']
        previsao['Cliente'] = cliente
        previsao['Produto'] = produto
        previsao['Previsao'] = 'Previsão'

        df_plot = pd.concat([dados, previsao], ignore_index=True)
        return df_plot, None

    except Exception:
        return dados, "Não foi possível gerar previsão (modelo não convergiu)."

# === Interface Streamlit ===
st.set_page_config(page_title="Painel de Vendas", layout="wide")
st.title("Painel de Vendas e Previsão")

clientes = sorted(agrupado['Cliente'].unique())
cliente = st.selectbox("Selecione o Cliente", clientes)

produtos = agrupado[agrupado['Cliente'] == cliente]['Produto'].unique()
produto = st.selectbox("Selecione o Produto", produtos)

df_plot, erro = prever(cliente, produto)

if erro:
    st.error(erro)

if not df_plot.empty:
    fig = px.line(
        df_plot,
        x='AnoMes',
        y='Quantidade',
        color='Previsao',
        title=f"{cliente} - {produto}",
        markers=True,
        color_discrete_map={
            'Histórico': "#080808",
            'Previsão': "#F10707"
        }
    )

    # Força estilo da linha de previsão (independente da formatação do Plotly)
    for trace in fig.data:
        if 'Previsão' in trace.name:
            trace.line.color = "#F10707"
            trace.line.width = 4
            trace.line.dash = 'dot'

    st.plotly_chart(fig, use_container_width=True)

