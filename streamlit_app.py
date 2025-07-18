import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os
import unicodedata
import logging
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# === Configurações ===
FORECAST_MONTHS = 6
REDUCTION_FACTOR = 0.9
MIN_DATE = '2024-01-01'
logging.getLogger('streamlit.runtime.scriptrunner').setLevel(logging.ERROR)

# === Função de Autenticação ===
def check_password():
    """Função para verificar login e senha"""
    st.sidebar.markdown("# 🔐 Autenticação")
    username = st.sidebar.text_input("Usuário", type="default")
    password = st.sidebar.text_input("Senha", type="password")
    
    if st.sidebar.button("Entrar"):
        if username == os.getenv("APP_USERNAME") and password == os.getenv("APP_PASSWORD"):
            st.sidebar.success("✅ Login realizado com sucesso!")
            st.session_state['authenticated'] = True
        else:
            st.sidebar.error("❌ Usuário ou senha incorretos")
    
    # Se já autenticado, mostrar botão de logout
    if 'authenticated' in st.session_state and st.session_state['authenticated']:
        if st.sidebar.button("Sair"):
            del st.session_state['authenticated']
            st.rerun()

# === Função para carregar dados ===
def load_data():
    if 'authenticated' not in st.session_state or not st.session_state['authenticated']:
        st.error("❌ Acesso não autorizado. Por favor, faça login.")
        st.stop()

    data_path = os.getenv("DATA_PATH")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, data_path)
    
    if not os.path.exists(path):
        st.error(f"❌ Arquivo não encontrado: {path}")
        st.stop()

    try:
        df = pd.read_excel(path, sheet_name='Base vendas', dtype=str)
        df.columns = df.columns.str.strip()
        cols = {}
        for c in ['Emissao', 'Cliente', 'Produto', 'Quantidade']:
            fc = find_column(df, c)
            if not fc:
                st.error(f"❌ Coluna obrigatória '{c}' não encontrada.")
                st.stop()
            cols[c] = fc

        # Converter colunas para os tipos corretos
        df[cols['Cliente']] = df[cols['Cliente']].astype(str).str.strip().str.upper()
        df[cols['Produto']] = df[cols['Produto']].astype(str).str.strip().str.upper()
        df[cols['Quantidade']] = pd.to_numeric(df[cols['Quantidade']], errors='coerce')
        
        # Converter a coluna de data
        try:
            df = df[df[cols['Emissao']].notna()]
            df[cols['Emissao']] = pd.to_datetime(df[cols['Emissao']], errors='coerce')
            df = df[df[cols['Emissao']].notna()]
            if df.empty:
                st.error("❌ Todas as datas são inválidas após a conversão")
                st.stop()
        except Exception as e:
            st.error(f"❌ Erro ao converter datas: {str(e)}")
            st.stop()

        # Limpar linhas com dados inválidos
        df = df.dropna(subset=[cols['Emissao'], cols['Cliente'], cols['Produto'], cols['Quantidade']])
        
        # Filtrar por data mínima
        try:
            min_date = pd.to_datetime(MIN_DATE)
            df = df[df[cols['Emissao']] >= min_date]
        except Exception as e:
            st.error(f"❌ Erro ao filtrar por data: {str(e)}")
            st.stop()

        if df.empty:
            st.error("❌ Nenhum dado após filtragem por data.")
            st.stop()

        # Criar coluna AnoMes
        df['AnoMes'] = df[cols['Emissao']].dt.to_period('M').dt.to_timestamp()

        grupo_col = find_column(df, 'Grupo')
        if grupo_col:
            df['Grupo'] = df[grupo_col].astype(str).str.strip().str.upper()
        else:
            df['Grupo'] = 'SEM GRUPO'

        return df[['Cliente', 'Produto', 'Quantidade', 'AnoMes', 'Grupo']]
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {str(e)}")
        st.stop()

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
    try:
        fig = px.line(
            df,
            x='AnoMes',
            y='Quantidade',
            color='Previsao',
            title=title.upper(),
            markers=True,
            labels={'AnoMes': 'MÊS', 'Quantidade': 'QUANTIDADE', 'Previsao': 'TIPO'},
            template='plotly_white'  # Usar tema claro
        )

        # Cores: histórico (preto), previsão (vermelho)
        fig.for_each_trace(
            lambda t: t.update(line=dict(color='black')) if t.name == 'HISTÓRICO' else t.update(line=dict(color='red'))
        )

        fig.update_layout(
            title_x=0.5,
            hovermode='x unified',
            plot_bgcolor='white',  # Fundo branco
            paper_bgcolor='white',  # Papel branco
            
            xaxis=dict(
                title='<b>MÊS</b>',
                title_font=dict(size=14, color='black'),
                tickfont=dict(size=12, color='black'),
                gridcolor='lightgray'  # Grade mais suave
            ),
            yaxis=dict(
                title='<b>QUANTIDADE</b>',
                title_font=dict(size=14, color='black'),
                tickfont=dict(size=12, color='black'),
                gridcolor='lightgray'  # Grade mais suave
            ),
            
            # Ajustar layout para dispositivos móveis
            margin=dict(l=20, r=20, t=60, b=20),
            height=600,  # Altura fixa
            width=None,  # Largura automática
            
            # Melhorar legibilidade
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="black"
            )
        )

        return fig
    except Exception as e:
        st.error(f"❌ Erro ao criar gráfico: {str(e)}")
        return None

def main():
    """
    Função principal do aplicativo. Ela verifica se o usuário está autenticado,
    carrega os dados, aplica os filtros selecionados pelo usuário e exibe os resultados
    em forma de gráfico. Além disso, ela também exibe estatísticas detalhadas sobre o
    histórico e a previsão.
    """

if __name__ == "__main__":
    main()