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

# Desativa logs de warning do Streamlit
logging.getLogger('streamlit.runtime.scriptrunner').setLevel(logging.ERROR)

# Remove acentos e espaços para facilitar a comparação
def remove_acentos(text):
    if not isinstance(text, str):
        return text
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    ).strip().lower()

# Encontra a coluna do DataFrame que mais se parece com target (sem acento e em minúsculo)
def find_column(df, target):
    target_norm = remove_acentos(target)
    for col in df.columns:
        if remove_acentos(col) == target_norm:
            return col
    return None

# Validação dos dados
def validate_data(df, required_cols):
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Colunas obrigatórias ausentes: {missing_cols}")
        return False
    if df.empty:
        st.error("❌ DataFrame vazio após carregamento e limpeza.")
        return False
    return True

# Carrega e prepara os dados
def load_data():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, 'data', 'base_vendas_24.xlsx')

        if not os.path.exists(file_path):
            st.error(f"❌ Arquivo não encontrado: {file_path}")
            st.stop()

        df = pd.read_excel(file_path, sheet_name='Base vendas', dtype=str)
        df.columns = df.columns.str.strip()

        st.write("Colunas encontradas no arquivo:", df.columns.tolist())

        # Mapear colunas obrigatórias
        cols_map = {}
        for col in ['Emissao', 'Cliente', 'Produto', 'Quantidade']:
            found_col = find_column(df, col)
            if not found_col:
                st.error(f"❌ Coluna obrigatória '{col}' não encontrada no arquivo.")
                st.stop()
            cols_map[col] = found_col

        # Tratamento das colunas
        df[cols_map['Cliente']] = df[cols_map['Cliente']].astype(str).str.strip().str.lower()
        df[cols_map['Produto']] = df[cols_map['Produto']].astype(str).str.strip().str.lower()
        df[cols_map['Emissao']] = pd.to_datetime(df[cols_map['Emissao']], errors='coerce')
        df[cols_map['Quantidade']] = pd.to_numeric(df[cols_map['Quantidade']], errors='coerce')

        # Filtra e remove linhas com dados inválidos
        df = df.dropna(subset=[cols_map['Emissao'], cols_map['Cliente'], cols_map['Produto'], cols_map['Quantidade']])
        df = df[df[cols_map['Emissao']] >= pd.to_datetime(MIN_DATE)]

        if df.empty:
            st.error("❌ Nenhum dado válido após filtragem por data.")
            st.stop()

        # Agrupa dados
        df['AnoMes'] = df[cols_map['Emissao']].dt.to_period('M').dt.to_timestamp()
        df_grouped = df.groupby([cols_map['Cliente'], cols_map['Produto'], 'AnoMes'])[cols_map['Quantidade']].sum().reset_index()

        # Padroniza nomes para usar internamente
        df_grouped.rename(columns={
            cols_map['Cliente']: 'Cliente',
            cols_map['Produto']: 'Produto',
            cols_map['Quantidade']: 'Quantidade'
        }, inplace=True)

        return df_grouped

    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {str(e)}")
        st.stop()

# Gera previsão
def make_forecast(cliente, produto, data):
    try:
        cliente = cliente.strip().lower()
        produto = produto.strip().lower()

        filtered = data[(data['Cliente'] == cliente) & (data['Produto'] == produto)].copy()
        if filtered.empty:
            return None, f"❌ Não foi possível encontrar dados para o cliente '{cliente}' e produto '{produto}'."

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

# Cria gráfico
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

# Função principal
def main():
    st.set_page_config(page_title="Painel de Vendas e Previsão", layout="wide")
    st.title("📊 Painel de Vendas e Previsão")

    @st.cache_data
    def get_data():
        return load_data()

    with st.spinner("Carregando dados..."):
        data = get_data()

    required_cols = ['Cliente', 'Produto', 'AnoMes', 'Quantidade']
    if not validate_data(data, required_cols):
        st.stop()

    clientes = sorted(data['Cliente'].unique())
    cliente = st.selectbox("Selecione o Cliente", clientes, help="Escolha um cliente")

    if cliente:
        produtos_df = data[data['Cliente'] == cliente]
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
                return

            if forecast_data is not None:
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
