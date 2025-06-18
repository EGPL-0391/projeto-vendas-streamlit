# streamlit_app.py

import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os
import logging

# === Configuration ===
FORECAST_MONTHS = 6
REDUCTION_FACTOR = 0.9
MIN_DATE = '2024-01-01'
CACHE_KEY = 'forecast_data'

# Configure logging
logging.getLogger('streamlit.runtime.scriptrunner').setLevel(logging.ERROR)

# === Data Validation ===
def validate_data(df):
    """Valida a estrutura e qualidade dos dados."""
    required_columns = ['Emissao', 'Cliente', 'Produto', 'Quantidade']
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        raise ValueError("Dataframe is empty")
    
    return True

def get_last_emission_date(data):
    """Retorna a data da última emissão."""
    return data['Emissao'].max()

def load_data():
    try:
        # Get the directory where this script is located
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, 'data', 'base_vendas_24.xlsx')
        
        if not os.path.exists(file_path):
            st.error(f"❌ Error: Data file not found at {file_path}")
            st.stop()
            
        df = pd.read_excel(file_path, sheet_name='Base vendas', dtype={'Cliente': str, 'Produto': str})
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Convert date column and drop missing values
        df['Emissao'] = pd.to_datetime(df['Emissao'], errors='coerce')
        df = df.dropna(subset=['Emissao', 'Cliente', 'Produto', 'Quantidade'])
        
        # Filter by date
        df = df[df['Emissao'] >= MIN_DATE]
        
        if df.empty:
            raise ValueError(f"No data found after filtering by date {MIN_DATE}")
        
        # Create month column
        df['AnoMes'] = df['Emissao'].dt.to_period('M').dt.to_timestamp()
        
        # Group and sum quantities
        df = df.groupby(['Cliente', 'Produto', 'AnoMes'])['Quantidade'].sum().reset_index()
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

def make_forecast(cliente, produto, data):
    """
    Gera previsão para um cliente e produto específicos.
    
    Args:
        cliente (str): Nome do cliente
        produto (str): Nome do produto
        data (pd.DataFrame): DataFrame com os dados de vendas
        
    Returns:
        tuple: (DataFrame com previsão, mensagem de erro)
    """
    try:
        if not isinstance(cliente, str) or not isinstance(produto, str):
            return None, "❌ Cliente e produto devem ser strings"
            
        # Filter data for specific client and product
        filtered = data[(data['Cliente'] == cliente) & (data['Produto'] == produto)].copy()
        
        if filtered.empty:
            return None, f"❌ No data found for client: {cliente} and product: {produto}"
        
        # Sort by date
        filtered = filtered.sort_values('AnoMes')
        filtered['Previsao'] = 'Histórico'
        
        # Get last emission date
        last_emission = filtered['AnoMes'].max()
        
        # Create time series from last emission date
        serie = filtered.set_index('AnoMes')['Quantidade']
        serie = serie.reindex(pd.date_range(start=last_emission, 
                                          periods=len(serie), 
                                          freq='MS'))
        
        # Create and fit model
        model = ExponentialSmoothing(
            serie,
            trend='add',
            damped_trend=True,
            seasonal=None,
            initialization_method='estimated'
        )
        
        fitted = model.fit()
        
        # Make forecast (with reduction factor) starting from next month
        forecast_start = last_emission + pd.DateOffset(months=1)
        forecast = (fitted.forecast(FORECAST_MONTHS) * REDUCTION_FACTOR).round().astype(int)
        
        # Prepare forecast data
        forecast_df = forecast.reset_index()
        forecast_df.columns = ['AnoMes', 'Quantidade']
        forecast_df['Cliente'] = cliente
        forecast_df['Produto'] = produto
        forecast_df['Previsao'] = 'Previsão'
        
        # Combine historical and forecast data
        result = pd.concat([filtered, forecast_df], ignore_index=True)
        return result, None
        
    except Exception as e:
        return None, f"❌ Error making forecast: {str(e)}"

def create_plot(df, title):
    """
    Cria gráfico interativo com Plotly.
    
    Args:
        df (pd.DataFrame): DataFrame com dados
        title (str): Título do gráfico
        
    Returns:
        plotly.graph_objs.Figure: Gráfico Plotly
    """
    try:
        if df is None or df.empty:
            return None
            
        fig = px.line(
            df,
            x='AnoMes',
            y='Quantidade',
            color='Previsao',
            title=title,
            markers=True,
            color_discrete_map={
                'Histórico': "#080808",
                'Previsão': "#F10707"
            },
            template="plotly_white"
        )
        
        # Style the forecast line
        for trace in fig.data:
            if 'Previsão' in trace.name:
                trace.line.color = "#F10707"
                trace.line.width = 4
                trace.line.dash = 'dot'
            else:
                trace.line.width = 2
        
        # Add hover information
        fig.update_traces(
            hovertemplate='<b>Mês: %{x}</b><br>'
                         'Quantidade: %{y}<br>'
                         '<extra></extra>'
        )
        
        return fig
    except Exception as e:
        st.error(f"❌ Error creating plot: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Vendas e Previsão", layout="wide")
    st.title("Painel de Vendas e Previsão")
    
    # Load data with caching
    @st.cache_data
    def get_data():
        return load_data()
    
    with st.spinner("Carregando dados..."):
        data = get_data()
    
    if data is None or data.empty:
        st.error("❌ Não foi possível carregar os dados")
        st.stop()
    
    # Get unique clients
    try:
        clientes = data['Cliente'].astype(str).unique()
        clientes = sorted(clientes, key=lambda x: str(x).lower())
    except KeyError as e:
        st.error(f"❌ Erro: Coluna 'Cliente' não encontrada nos dados. Colunas disponíveis: {', '.join(map(str, data.columns))}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Erro ao processar dados: {str(e)}")
        st.stop()
    
    # Create client selector with placeholder
    cliente = st.selectbox(
        "Selecione o Cliente",
        clientes,
        help="Escolha um cliente para visualizar as vendas",
        placeholder="Selecione um cliente..."
    )
    
    if cliente:
        # Get products for selected client
        produtos = data[data['Cliente'] == cliente]['Produto'].unique()
        produtos = sorted(produtos, key=lambda x: str(x).lower())
        
        # Create product selector with placeholder
        produto = st.selectbox(
            "Selecione o Produto",
            produtos,
            help="Escolha um produto para visualizar as vendas",
            placeholder="Selecione um produto..."
        )
        
        if produto:
            # Show loading state
            with st.spinner(f"Gerando previsão para {cliente} - {produto}..."):
                # Make forecast
                forecast_data, error = make_forecast(cliente, produto, data)
                
            if error:
                st.error(error)
            elif forecast_data is not None:
                # Create and display plot
                fig = create_plot(forecast_data, f"{cliente} - {produto}")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show summary statistics
                    with st.expander("📊 Estatísticas"):
                        historico = forecast_data[forecast_data['Previsao'] == 'Histórico']
                        previsao = forecast_data[forecast_data['Previsao'] == 'Previsão']
                        
                        st.write("📊 Estatísticas Históricas:")
                        st.write("- Total histórico:", historico['Quantidade'].sum())
                        st.write("- Média mensal:", historico['Quantidade'].mean().round(2))
                        st.write("- Mediana mensal:", historico['Quantidade'].median())
                        st.write("- Desvio padrão:", historico['Quantidade'].std().round(2))
                        
                        st.write("\n📈 Estatísticas da Previsão:")
                        st.write("- Total previsto:", previsao['Quantidade'].sum())
                        st.write("- Média mensal prevista:", previsao['Quantidade'].mean().round(2))
                        st.write("- Mediana prevista:", previsao['Quantidade'].median())

if __name__ == "__main__":
    main()