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

        # Mapeamento de colunas obrigatórias
        cols_map = {}
        for col in ['Emissao', 'Cliente', 'Produto', 'Quantidade']:
            found_col = find_column(df, col)
            if not found_col:
                st.error(f"❌ Coluna obrigatória '{col}' não encontrada.")
                st.stop()
            cols_map[col] = found_col

        # Tratamento e padronização
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

        return df_grouped

    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {str(e)}")
        st.stop()

def make_forecast(cliente, produto, data):
    try:
        cliente = cliente.strip().upper()
        produto = produto.strip().upper()

        filtered = data[(data['Cliente'] == cliente) & (data['Produto'] == produto)].copy()
        if filtered.empty:
            return None, f"❌ Nenhum dado para cliente '{cliente}' e produto '{produto}'."

        filtered = filtered.sort_values('AnoMes')
        filtered['Previsao'] = 'HISTÓRICO'
        serie = filtered.set_index('AnoMes')['Quantidade']

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
        forecast_df['Cliente'] = cliente
        forecast_df['Produto'] = produto
        forecast_df['Previsao'] = 'PREVISÃO'

        result = pd.concat([filtered, forecast_df], ignore_index=True)
        return result, None

    except Exception as e:
        return None, f"❌ Erro ao gerar previsão: {str(e)}"

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
        # Define colors for each trace
        colors = {'HISTÓRICO': 'black', 'PREVISÃO': 'red'}
        for trace in fig.data:
            trace.line.color = colors.get(trace.name, 'black')
        
        fig.update_layout(
            xaxis_title='MÊS',
            yaxis_title='QUANTIDADE',
            hovermode='x unified'
        )
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

    if not validate_data(data, ['Cliente', 'Produto', 'AnoMes', 'Quantidade']):
        st.stop()

    # Opções de seleção
    st.sidebar.header("FILTROS")
    clientes = sorted(data['Cliente'].unique())
    grupos = sorted(data['Produto'].str.split('-').str[0].unique())
    
    # Seleção de cliente
    cliente = st.selectbox(
        "Selecione o cliente",
        clientes
    )

    if cliente:
        # Seleção de grupo
        selected_group = st.selectbox(
            "Selecione o grupo",
            ['Todos'] + grupos
        )

        # Seleção de produto
        produtos = ['Todos'] + sorted(data[data['Cliente'] == cliente]['Produto'].unique())
        produto = st.selectbox(
            "Selecione o produto",
            produtos
        )

        # Filtrar dados com base nas seleções
        filtered_data = data[data['Cliente'] == cliente]
        if selected_group != 'Todos':
            filtered_data = filtered_data[filtered_data['Produto'].str.split('-').str[0] == selected_group]

        if filtered_data.empty:
            st.warning("Nenhum dado encontrado com os filtros selecionados.")
            return

        # Se 'Todos' foi selecionado, gerar previsões para todos os produtos
        if produto == 'Todos':
            produtos_cliente = sorted(data[data['Cliente'] == cliente]['Produto'].unique())
            for prod in produtos_cliente:
                with st.spinner(f"GERANDO PREVISÃO PARA {cliente} - {prod}..."):
                    forecast_data, error = make_forecast(cliente, prod, data)
                    if error:
                        st.error(error)
                    if forecast_data is not None:
                        fig = create_plot(forecast_data, f"{cliente} - {prod}")
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
        else:
            with st.spinner(f"GERANDO PREVISÃO PARA {cliente} - {produto}..."):
                forecast_data, error = make_forecast(cliente, produto, data)
                if error:
                    st.error(error)
                if forecast_data is not None:
                    fig = create_plot(forecast_data, f"{cliente} - {produto}")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

            if error:
                st.error(error)
                return

            if forecast_data is not None:
                fig = create_plot(forecast_data, f"{cliente} - {produto}")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                with st.expander("📈 ESTATÍSTICAS"):
                    historico = forecast_data[forecast_data['Previsao'] == 'HISTÓRICO']
                    previsao = forecast_data[forecast_data['Previsao'] == 'PREVISÃO']

                    st.write("📊 HISTÓRICO:")
                    st.write("- TOTAL:", historico['Quantidade'].sum())
                    st.write("- MÉDIA MENSAL:", historico['Quantidade'].mean().round(2))
                    st.write("- MEDIANA:", historico['Quantidade'].median())
                    st.write("- DESVIO PADRÃO:", historico['Quantidade'].std().round(2))

                    st.write("📈 PREVISÃO:")
                    st.write("- TOTAL:", previsao['Quantidade'].sum())
                    st.write("- MÉDIA MENSAL:", previsao['Quantidade'].mean().round(2))
                    st.write("- MEDIANA:", previsao['Quantidade'].median())

if __name__ == "__main__":
    main()
