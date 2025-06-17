# streamlit_app.py

import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os

# === Configuration ===
FORECAST_MONTHS = 6
REDUCTION_FACTOR = 0.9
MIN_DATE = '2024-01-01'

# === Data Loading and Processing ===
def load_data():
    # Get the directory where this script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Path to the Excel file
    file_path = os.path.join(base_dir, 'data', 'base_vendas_24.xlsx')
    
    # Check if file exists
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        st.stop()
    
    # Read Excel file
    df = pd.read_excel(file_path, sheet_name='Base vendas')
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Convert date column and drop missing values
    df['Emissao'] = pd.to_datetime(df['Emissao'], errors='coerce')
    df = df.dropna(subset=['Emissao', 'Cliente', 'Produto', 'Quantidade'])
    
    # Convert types
    df['Cliente'] = df['Cliente'].astype(str)
    df['Produto'] = df['Produto'].astype(str)
    
    # Filter by date
    df = df[df['Emissao'] >= MIN_DATE]
    
    # Create month column
    df['AnoMes'] = df['Emissao'].dt.to_period('M').dt.to_timestamp()
    
    # Group and sum quantities
    return df.groupby(['Cliente', 'Produto', 'AnoMes'])['Quantidade'].sum().reset_index()

# === Forecasting ===
def make_forecast(cliente, produto, data):
    # Filter data for specific client and product
    filtered = data[(data['Cliente'] == cliente) & (data['Produto'] == produto)].copy()
    
    if filtered.empty:
        return None, "No data available for this client and product"
    
    try:
        # Sort by date
        filtered = filtered.sort_values('AnoMes')
        filtered['Previsao'] = 'Histórico'
        
        # Create time series
        serie = filtered.set_index('AnoMes')['Quantidade']
        serie.index = pd.date_range(start=serie.index.min(), 
                                   periods=len(serie), 
                                   freq='MS')
        
        # Create and fit model
        model = ExponentialSmoothing(
            serie,
            trend='add',
            damped_trend=True,
            seasonal=None,
            initialization_method='estimated'
        )
        
        fitted = model.fit()
        
        # Make forecast (with reduction factor)
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
        return None, f"Error making forecast: {str(e)}"

# === Main Application ===
def main():
    st.set_page_config(page_title="Sales Dashboard", layout="wide")
    st.title("Sales Dashboard and Forecast")
    
    # Load data
    data = load_data()
    
    # Get unique clients
    clientes = sorted(data['Cliente'].unique())
    cliente = st.selectbox("Select Client", clientes)
    
    # Get products for selected client
    produtos = data[data['Cliente'] == cliente]['Produto'].unique()
    produto = st.selectbox("Select Product", produtos)
    
    # Make forecast
    forecast_data, error = make_forecast(cliente, produto, data)
    
    if error:
        st.error(error)
    elif forecast_data is not None:
        # Create plot
        fig = px.line(
            forecast_data,
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
        
        # Style the forecast line
        for trace in fig.data:
            if 'Previsão' in trace.name:
                trace.line.color = "#F10707"
                trace.line.width = 4
                trace.line.dash = 'dot'
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()

