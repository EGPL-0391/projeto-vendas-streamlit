# streamlit_app.py

import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os

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
df['Grupo'] = df['Grupo'].astype(str)
df = df[df['AnoMes'] >= '2024-01-01']

agrupado = df.groupby(['Grupo', 'Cliente', 'Produto', 'AnoMes'])['Quantidade'].sum().reset_index()

# === Função de previsão ===
def prever(dados, grupo_col, nome_grupo):
    dados = dados.sort_values('AnoMes')
    dados['Previsao'] = 'Histórico'

    if dados.empty:
        return dados, "Sem dados suficientes para previsão."

    serie = dados.set_index('AnoMes')['Quantidade']
    serie.index = pd.date_range(start=serie.index.min(), periods=len(serie), freq='MS')

    try:
        modelo = ExponentialSmoothing(
            serie,
            trend='mul',
            damped_trend=True,
            seasonal=None,
            initialization_method='estimated'
        )
        ajuste = modelo.fit()

        previsao = (ajuste.forecast(6) * 0.9).clip(upper=serie.max() * 1.1).round().astype(int)

        previsao = previsao.reset_index()
        previsao.columns = ['AnoMes', 'Quantidade']
        previsao[grupo_col] = nome_grupo
        previsao['Previsao'] = 'Previsão'

        df_plot = pd.concat([dados, previsao], ignore_index=True)
        return df_plot, None

    except Exception:
        return dados, "Não foi possível gerar previsão (modelo não convergiu)."

# === Interface Streamlit ===
st.set_page_config(page_title="Painel de Vendas", layout="wide")
st.title("Painel de Vendas e Previsão")

# Filtros
grupo_opcoes = sorted(df['Grupo'].unique())
grupo = st.selectbox("Selecione o Grupo", grupo_opcoes)

clientes = sorted(df[df['Grupo'] == grupo]['Cliente'].unique())
cliente = st.selectbox("Selecione o Cliente", clientes)

produtos = df[(df['Grupo'] == grupo) & (df['Cliente'] == cliente)]['Produto'].unique()
produto = st.selectbox("Selecione o Produto", produtos)

# Previsão individual
dados_ind = agrupado[(agrupado['Grupo'] == grupo) & (agrupado['Cliente'] == cliente) & (agrupado['Produto'] == produto)]
df_plot_ind, erro_ind = prever(dados_ind.copy(), 'Produto', produto)

if erro_ind:
    st.error(erro_ind)

if not df_plot_ind.empty:
    fig = px.line(df_plot_ind, x='AnoMes', y='Quantidade', color='Previsao',
                  title=f"{cliente} - {produto}", markers=True)
    fig.update_layout(xaxis_title='Mês', yaxis_title='Quantidade Vendida')
    st.plotly_chart(fig, use_container_width=True)

# Previsão do grupo
dados_grp = agrupado[agrupado['Grupo'] == grupo].groupby('AnoMes')['Quantidade'].sum().reset_index()
dados_grp['Grupo'] = grupo
df_plot_grp, erro_grp = prever(dados_grp.copy(), 'Grupo', grupo)

if not erro_grp and not df_plot_grp.empty:
    fig2 = px.line(df_plot_grp, x='AnoMes', y='Quantidade', color='Previsao',
                   title=f"Previsão Total do Grupo: {grupo}", markers=True)
    fig2.update_layout(xaxis_title='Mês', yaxis_title='Quantidade Vendida')
    st.plotly_chart(fig2, use_container_width=True)