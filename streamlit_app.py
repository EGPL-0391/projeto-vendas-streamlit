import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os
import logging

# === Configurações ===
FORECAST_MONTHS = 6
REDUCTION_FACTOR = 0.9
MIN_DATE = '2024-01-01'

# Desativa logs de warning do Streamlit
logging.getLogger('streamlit.runtime.scriptrunner').setLevel(logging.ERROR)

# === Validação dos dados ===
def validate_data(df):
    required_columns = ['Emissao', 'Cliente', 'Produto', 'Quantidade']
    if df.empty:
        st.error("❌ O DataFrame está vazio.")
        return False
    actual_columns = df.columns.str.lower()
    missing_cols = [col for col in required_columns if col.lower() not in actual_columns]
    if missing_cols:
        st.error(f"❌ Colunas obrigatórias ausentes: {missing_cols}")
        return False
    return True

# === Carrega e prepara os dados ===
def load_data():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, 'data', 'base_vendas_24.xlsx')

        if not os.path.exists(file_path):
            st.error(f"❌ Arquivo não encontrado: {file_path}")
            st.stop()

        df = pd.read_excel(file_path, sheet_name='Base vendas', dtype={'Cliente': str, 'Produto': str})
        df.columns = df.columns.str.strip()

        # Padronização
        df['Cliente'] = df['Cliente'].str.strip().str.lower()
        df['Produto'] = df['Produto'].str.strip().str.lower()
        df['Emissao'] = pd.to_datetime(df['Emissao'], errors='coerce')
        df = df.dropna(subset=['Emissao', 'Cliente', 'Produto', 'Quantidade'])
        df = df[df['Emissao'] >= MIN_DATE]

        df['AnoMes'] = df['Emissao'].dt.to_period('M').dt.to_timestamp()
        df = df.groupby(['Cliente', 'Produto', 'AnoMes'])['Quantidade'].sum().reset_index()

        if df.empty:
            raise ValueError("❌ Nenhum dado restante após filtragem.")

        return df

    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {str(e)}")
        st.stop()

# === Previsão ===
def make_forecast(cliente, produto, data):
    try:
        cliente = cliente.strip().lower()
        produto = produto.strip().lower()

        filtered = data[(data['Cliente'] == cliente) & (data['Produto'] == produto)].copy()

        if filtered.empty:
            return None, f"❌ Não há dados para cliente '{cliente}' e produto '{produto}'."

        filtered = filtered.sort_values('AnoMes')
        filtered['Previsao'] = 'Histórico'
        serie = filtered.set_index('AnoMes')['Quantidade']

        model = ExponentialSmoothing(
            serie,
            trend='add',
            damped_trend=True,
            seasonal=None,
            initialization_method='estimated'
        ).fit()

        forecast = (model.forecast(FORECAST_MONTHS) * REDUCTION_FACTOR).round().astype(int)
        forecast_df = forecast.reset_index()
        forecast_df.columns = ['AnoMes', 'Quantidade']
        forecast_df['Cliente'] = cliente
        forecast_df['Produto'] = produto
        forecast_df['Previsao'] = 'Previsão'

        result = pd.concat([filtered, forecast_df], ignore_index=True)
        return result, None

    except Exception as e:
        return None, f"❌ Erro ao gerar previsão: {str(e)}"

# === Gráfico ===
def create_plot(df, title):
    try:
        fig = px.line(
            df,
            x='AnoMes',
            y='Quantidade',
            color='Previsao',
            title=title,
            labels={'AnoMes': 'Mês', 'Quantidade': 'Quantidade', 'Previsao': 'Tipo'}
        )
        fig.update_layout(xaxis_title='Mês', yaxis_title='Quantidade', hovermode='x unified')
        return fig
    except Exception as e:
        st.error(f"❌ Erro ao criar gráfico: {str(e)}")
        return None

# === Principal ===
def main():
    st.set_page_config(page_title="Painel de Vendas e Previsão", layout="wide")
    st.title("📊 Painel de Vendas e Previsão")

    st.cache_data.clear()

    @st.cache_data
    def get_data():
        return load_data()

    with st.spinner("Carregando dados..."):
        data = get_data()

    if not validate_data(data):
        st.stop()

    clientes = sorted(data['Cliente'].unique())
    cliente = st.selectbox("Selecione o Cliente", clientes, help="Escolha um cliente")

    if cliente:
        produtos_df = data[data['Cliente'] == cliente]

        if produtos_df.empty:
            st.error(f"❌ Nenhum produto encontrado para o cliente '{cliente}'.")
            st.stop()

        produtos = sorted(produtos_df['Produto'].unique())
        if not produtos:
            st.error(f"❌ Cliente '{cliente}' não possui produtos.")
            st.stop()

        produto = st.selectbox("Selecione o Produto", produtos, help="Escolha um produto")

        if produto:
            with st.spinner(f"Gerando previsão para {cliente} - {produto}..."):
                forecast_data, error = make_forecast(cliente, produto, data)

            if error:
                st.error(error)
            elif forecast_data is not None:
                fig = create_plot(forecast_data, f"{cliente} - {produto}")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                with st.expander("📈 Estatísticas"):
                    historico = forecast_data[forecast_data['Previsao'] == 'Histórico']
                    previsao = forecast_data[forecast_data['Previsao'] == 'Previsão']

                    st.write("📊 Estatísticas Históricas:")
                    st.write("- Total histórico:", historico['Quantidade'].sum())
                    st.write("- Média mensal:", historico['Quantidade'].mean().round(2))
                    st.write("- Mediana mensal:", historico['Quantidade'].median())
                    st.write("- Desvio padrão:", historico['Quantidade'].std().round(2))

                    st.write("📈 Estatísticas da Previsão:")
                    st.write("- Total previsto:", previsao['Quantidade'].sum())
                    st.write("- Média mensal prevista:", previsao['Quantidade'].mean().round(2))
                    st.write("- Mediana prevista:", previsao['Quantidade'].median())

if __name__ == "__main__":
    main()
