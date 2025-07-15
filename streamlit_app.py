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
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    ).strip().lower()

def find_column(df, target):
    target_norm = remove_acentos(target)
    for col in df.columns:
        if remove_acentos(col) == target_norm:
            return col
    return None

def validate_data(df, required_cols):
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Colunas obrigatórias ausentes: {missing_cols}")
        return False
    if df.empty:
        st.error("❌ DataFrame vazio após carregamento e limpeza.")
        return False
    return True

def load_data():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, 'data', 'base_vendas_24.xlsx')

        if not os.path.exists(file_path):
            st.error(f"❌ Arquivo não encontrado: {file_path}")
            st.stop()

        df = pd.read_excel(file_path, sheet_name='Base vendas', dtype=str)
        df.columns = df.columns.str.strip()

        cols_map = {}
        for col in ['Emissao', 'Cliente', 'Produto', 'Quantidade']:
            found_col = find_column(df, col)
            if not found_col:
                st.error(f"❌ Coluna obrigatória '{col}' não encontrada.")
                st.stop()
            cols_map[col] = found_col

        df[cols_map['Cliente']] = df[cols_map['Cliente']].astype(str).str.strip().str.upper()
        df[cols_map['Produto']] = df[cols_map['Produto']].astype(str).str.strip().str.upper()
        df[cols_map['Emissao']] = pd.to_datetime(df[cols_map['Emissao']], errors='coerce')
        df[cols_map['Quantidade']] = pd.to_numeric(df[cols_map['Quantidade']], errors='coerce')

        df = df.dropna(subset=[cols_map['Emissao'], cols_map['Cliente'], cols_map['Produto'], cols_map['Quantidade']])
        df = df[df[cols_map['Emissao']] >= pd.to_datetime(MIN_DATE)]

        if df.empty:
            st.error("❌ Nenhum dado válido após filtragem por data.")
            st.stop()

        df['AnoMes'] = df[cols_map['Emissao']].dt.to_period('M').dt.to_timestamp()
        df_grouped = df.groupby([cols_map['Cliente'], cols_map['Produto'], 'AnoMes'])[cols_map['Quantidade']].sum().reset_index()

        df_grouped.rename(columns={
            cols_map['Cliente']: 'Cliente',
            cols_map['Produto']: 'Produto',
            cols_map['Quantidade']: 'Quantidade'
        }, inplace=True)

        grupo_col = find_column(df, 'Grupo')
        if grupo_col:
            df_grouped['Grupo'] = df[[cols_map['Cliente']]].merge(
                df[[cols_map['Cliente'], grupo_col]].drop_duplicates(),
                on=cols_map['Cliente'],
                how='left'
            )[grupo_col].fillna('SEM GRUPO').str.upper()
        else:
            df_grouped['Grupo'] = 'SEM GRUPO'

        return df_grouped

    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {str(e)}")
        st.stop()

def make_forecast_from_series(serie):
    model = ExponentialSmoothing(
        serie,
        trend='add',
        damped_trend=True,
        seasonal=None,
        initialization_method='estimated'
    ).fit()

    forecast_index = pd.date_range(
        start=serie.index[-1] + pd.offsets.MonthBegin(),
        periods=FORECAST_MONTHS,
        freq='MS'
    )

    forecast = (model.forecast(FORECAST_MONTHS) * REDUCTION_FACTOR).round().astype(int)
    forecast.index = forecast_index

    forecast_df = forecast.reset_index()
    forecast_df.columns = ['AnoMes', 'Quantidade']
    forecast_df['Previsao'] = 'PREVISÃO'
    return forecast_df

def create_plot(df, title):
    try:
        fig = px.line(
            df,
            x='AnoMes',
            y='Quantidade',
            color='Previsao',
            title=title.upper(),
            labels={'AnoMes': 'MÊS', 'Quantidade': 'QUANTIDADE', 'Previsao': 'TIPO'}
        )
        fig.update_layout(xaxis_title='MÊS', yaxis_title='QUANTIDADE', hovermode='x unified')
        return fig
    except Exception as e:
        st.error(f"❌ Erro ao criar gráfico: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="PAINEL DE VENDAS", layout="wide")
    st.title("📊 PAINEL DE VENDAS E PREVISÃO")

    @st.cache_data
    def get_data():
        return load_data()

    with st.spinner("CARREGANDO DADOS..."):
        data = get_data()

    if not validate_data(data, ['Cliente', 'Produto', 'AnoMes', 'Quantidade', 'Grupo']):
        st.stop()

    # === FILTROS ===
    grupos = sorted(data['Grupo'].unique())
    grupo_selecionado = st.selectbox("SELECIONE O GRUPO", ["TODOS"] + grupos)

    data_f = data.copy()
    if grupo_selecionado != "TODOS":
        data_f = data_f[data_f['Grupo'] == grupo_selecionado]

    clientes = sorted(data_f['Cliente'].unique())
    cliente = st.selectbox("SELECIONE O CLIENTE", ["TODOS"] + clientes)

    if cliente != "TODOS":
        data_f = data_f[data_f['Cliente'] == cliente]

    produtos = sorted(data_f['Produto'].unique())
    produto = st.selectbox("SELECIONE O PRODUTO", ["TODOS"] + produtos)

    if produto != "TODOS":
        data_f = data_f[data_f['Produto'] == produto]

    if data_f.empty:
        st.warning("⚠️ Nenhum dado disponível com os filtros selecionados.")
        return

    # === AGRUPAMENTO E PREVISÃO ===
    data_agrupada = data_f.groupby('AnoMes')['Quantidade'].sum().reset_index()
    data_agrupada['Previsao'] = 'HISTÓRICO'

    try:
        serie = data_agrupada.set_index('AnoMes')['Quantidade'].sort_index()
        forecast_df = make_forecast_from_series(serie)
        result = pd.concat([data_agrupada, forecast_df], ignore_index=True)

        titulo = "PREVISÃO CONSOLIDADA"
        if cliente != "TODOS" and produto != "TODOS":
            titulo = f"{cliente} - {produto}"
        elif cliente != "TODOS":
            titulo = f"{cliente} - TODOS OS PRODUTOS"
        elif produto != "TODOS":
            titulo = f"CLIENTES - {produto}"
        elif grupo_selecionado != "TODOS":
            titulo = f"GRUPO {grupo_selecionado} - CONSOLIDADO"

        fig = create_plot(result, titulo)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("📈 ESTATÍSTICAS"):
            historico = result[result['Previsao'] == 'HISTÓRICO']
            previsao = result[result['Previsao'] == 'PREVISÃO']

            st.write("📊 HISTÓRICO:")
            st.write("- TOTAL:", historico['Quantidade'].sum())
            st.write("- MÉDIA MENSAL:", historico['Quantidade'].mean().round(2))
            st.write("- MEDIANA:", historico['Quantidade'].median())
            st.write("- DESVIO PADRÃO:", historico['Quantidade'].std().round(2))

            st.write("📈 PREVISÃO:")
            st.write("- TOTAL:", previsao['Quantidade'].sum())
            st.write("- MÉDIA MENSAL:", previsao['Quantidade'].mean().round(2))
            st.write("- MEDIANA:", previsao['Quantidade'].median())

    except Exception as e:
        st.error(f"❌ Erro ao gerar previsão consolidada: {str(e)}")

if __name__ == "__main__":
    main()
