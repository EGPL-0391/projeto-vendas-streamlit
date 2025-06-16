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

df = df[df['AnoMes'] >= '2024-01-01']

agrupado = df.groupby(['Cliente', 'Produto', 'AnoMes'])['Quantidade'].sum().reset_index()

# === Função de previsão ===
def prever(cliente, produto):
    dados = agrupado[(agrupado['Cliente'] == cliente) & (agrupado['Produto'] == produto)].copy()
    dados = dados.sort_values('AnoMes')
    dados['Previsao'] = 'Histórico'

    if dados.empty or len(dados) < 3:
        return dados, "Sem dados suficientes para previsão."

    serie = dados.set_index('AnoMes')['Quantidade']
    serie = serie.asfreq('MS').fillna(0)

    # Tentativa com modelo
    try:
        modelo = ExponentialSmoothing(
            serie,
            trend='add',  # aditivo geralmente mais estável
            damped_trend=True,
            seasonal=None,
            initialization_method='estimated'
        )
        ajuste = modelo.fit()

        ultimo_mes = serie.index.max()
        futuros = pd.date_range(start=ultimo_mes + pd.offsets.MonthBegin(), periods=6, freq='MS')

        previsao = ajuste.forecast(6)

        # Aplica um redutor leve, se desejar (ex: 5%) e evita explosões com clip no máximo +20%
        previsao = (previsao * 0.95).clip(upper=serie.max() * 1.2).round().astype(int)
        previsao.index = futuros

    except Exception as e:
        # Fallback conservador: repetir média dos últimos 3 meses
        media = serie[-3:].mean()
        previsao = pd.Series([int(media)] * 6, index=pd.date_range(start=serie.index.max() + pd.offsets.MonthBegin(), periods=6, freq='MS'))

    previsao = previsao.reset_index()
    previsao.columns = ['AnoMes', 'Quantidade']
    previsao['Cliente'] = cliente
    previsao['Produto'] = produto
    previsao['Previsao'] = 'Previsão'

    df_plot = pd.concat([dados, previsao], ignore_index=True)
    return df_plot, None

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
        df_plot, x='AnoMes', y='Quantidade', color='Previsao',
        title=f"{cliente} - {produto}", markers=True
    )
    fig.update_layout(xaxis_title='Mês', yaxis_title='Quantidade Vendida')
    st.plotly_chart(fig, use_container_width=True)