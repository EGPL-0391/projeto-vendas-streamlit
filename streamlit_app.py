import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os
import unicodedata
import logging

# === Configurações ===
FORECAST_MONTHS = 6
REDUCTION_FACTOR = 0.9
MIN_DATE = '2024-01-01'
logging.getLogger('streamlit.runtime.scriptrunner').setLevel(logging.ERROR)

def remove_acentos(text):
    if not isinstance(text, str):
        return text
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn').strip().lower()

def find_column(df, target):
    target_norm = remove_acentos(target)
    for col in df.columns:
        if remove_acentos(col) == target_norm:
            return col
    return None

def validate_data(df, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"❌ Colunas obrigatórias ausentes: {missing}")
        return False
    if df.empty:
        st.error("❌ DataFrame vazio após limpeza.")
        return False
    return True

def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, 'data', 'base_vendas_24.xlsx')
    if not os.path.exists(path):
        st.error(f"❌ Arquivo não encontrado: {path}")
        st.stop()

    df = pd.read_excel(path, sheet_name='Base vendas', dtype=str)
    df.columns = df.columns.str.strip()
    cols = {}
    for c in ['Emissao', 'Cliente', 'Produto', 'Quantidade']:
        fc = find_column(df, c)
        if not fc:
            st.error(f"❌ Coluna obrigatória '{c}' não encontrada.")
            st.stop()
        cols[c] = fc

    df[cols['Cliente']] = df[cols['Cliente']].astype(str).str.strip().str.upper()
    df[cols['Produto']] = df[cols['Produto']].astype(str).str.strip().str.upper()
    df[cols['Emissao']] = pd.to_datetime(df[cols['Emissao']], errors='coerce')
    df[cols['Quantidade']] = pd.to_numeric(df[cols['Quantidade']], errors='coerce')

    df = df.dropna(subset=[cols['Emissao'], cols['Cliente'], cols['Produto'], cols['Quantidade']])
    df = df[df[cols['Emissao']] >= pd.to_datetime(MIN_DATE)]
    if df.empty:
        st.error("❌ Nenhum dado após filtragem por data.")
        st.stop()

    df['AnoMes'] = df[cols['Emissao']].dt.to_period('M').dt.to_timestamp()

    grupo_col = find_column(df, 'Grupo')
    if grupo_col:
        df['Grupo'] = df[grupo_col].astype(str).str.strip().str.upper()
    else:
        df['Grupo'] = 'SEM GRUPO'

    return df[['Cliente', 'Produto', 'Quantidade', 'AnoMes', 'Grupo']]

def make_forecast_from_series(serie):
    m = ExponentialSmoothing(serie, trend='add', damped_trend=True, seasonal=None, initialization_method='estimated').fit()
    idx = pd.date_range(start=serie.index[-1] + pd.offsets.MonthBegin(), periods=FORECAST_MONTHS, freq='MS')
    fc = (m.forecast(FORECAST_MONTHS) * REDUCTION_FACTOR).round().astype(int)
    fc.index = idx
    df = fc.reset_index()
    df.columns = ['AnoMes', 'Quantidade']
    df['Previsao'] = 'PREVISÃO'
    return df

def create_plot(df, title):
    fig = px.line(df, x='AnoMes', y='Quantidade', color='Previsao', title=title,
                  labels={'AnoMes': 'MÊS', 'Quantidade': 'QUANTIDADE', 'Previsao': 'TIPO'})
    fig.update_layout(xaxis_title='MÊS', yaxis_title='QUANTIDADE', hovermode='x unified')
    return fig

def main():
    st.set_page_config(page_title="PAINEL DE VENDAS", layout="wide")
    st.title("📊 PAINEL DE VENDAS E PREVISÃO")

    @st.cache_data
    def get_data():
        return load_data()
    df = get_data()

    if not validate_data(df, ['Cliente', 'Produto', 'Quantidade', 'AnoMes', 'Grupo']):
        st.stop()

    # == FILTROS CADEIA ==
    grupo = st.selectbox("SELECIONE O GRUPO", ["TODOS"] + sorted(df['Grupo'].unique()))
    dfg = df if grupo == "TODOS" else df[df['Grupo'] == grupo]

    cliente = st.selectbox("SELECIONE O CLIENTE", ["TODOS"] + sorted(dfg['Cliente'].unique()))
    dfc = dfg if cliente == "TODOS" else dfg[dfg['Cliente'] == cliente]

    produto = st.selectbox("SELECIONE O PRODUTO", ["TODOS"] + sorted(dfc['Produto'].unique()))
    dff = dfc if produto == "TODOS" else dfc[dfc['Produto'] == produto]

    if dff.empty:
        st.warning("⚠️ Nenhum dado com os filtros aplicados.")
        return

    # == AGRUPAMENTO HISTÓRICO E PREVISÃO ==
    grouped = dff.groupby('AnoMes', as_index=False)['Quantidade'].sum()
    grouped['Previsao'] = 'HISTÓRICO'
    serie = grouped.set_index('AnoMes')['Quantidade'].sort_index()

    try:
        fc = make_forecast_from_series(serie)
        resultado = pd.concat([grouped, fc], ignore_index=True)
    except Exception as e:
        st.error(f"❌ Erro na previsão: {e}")
        return

    # == TÍTULO INTELIGENTE ==
    if grupo!="TODOS" and cliente=="TODOS" and produto=="TODOS":
        titulo = f"GRUPO {grupo} - CONSOLIDADO"
    elif cliente!="TODOS" and produto=="TODOS":
        titulo = f"{cliente} - TODOS OS PRODUTOS"
    elif cliente=="TODOS" and produto!="TODOS":
        titulo = f"TODOS OS CLIENTES - {produto}"
    elif cliente!="TODOS" and produto!="TODOS":
        titulo = f"{cliente} - {produto}"
    else:
        titulo = "PREVISÃO TOTAL"

    # == GRÁFICO E ESTATÍSTICAS ==
    fig = create_plot(resultado, titulo)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("📈 ESTATÍSTICAS"):
        h = resultado[resultado['Previsao']=='HISTÓRICO']['Quantidade']
        p = resultado[resultado['Previsao']=='PREVISÃO']['Quantidade']
        st.write("📊 HISTÓRICO: total =", h.sum(), "| média =", round(h.mean(),2), "| mediana =", h.median(), "| desvio =", round(h.std(),2))
        st.write("📈 PREVISÃO: total =", p.sum(), "| média =", round(p.mean(),2), "| mediana =", p.median())

if __name__ == "__main__":
    main()
