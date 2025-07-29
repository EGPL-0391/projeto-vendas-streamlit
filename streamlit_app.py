import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os
import unicodedata
import logging
import hashlib
import hmac
import base64
from cryptography.fernet import Fernet
from io import BytesIO

# === Configurações ===
FORECAST_MONTHS = 6
REDUCTION_FACTOR = 0.9
MIN_DATE = '2024-01-01'
logging.getLogger('streamlit.runtime.scriptrunner').setLevel(logging.ERROR)

# === Configurações de Segurança ===
def get_encryption_key():
    """Obtém a chave de criptografia do Streamlit Secrets"""
    try:
        return st.secrets["encryption"]["key"].encode()
    except:
        st.error("❌ Chave de criptografia não encontrada nas configurações!")
        st.stop()

def verify_password(username, password):
    """Verifica se o usuário e senha estão corretos"""
    try:
        users = st.secrets["users"]
        if username in users:
            stored_hash = users[username]["password_hash"]
            salt = users[username]["salt"]
            
            # Gera hash da senha fornecida
            password_hash = hashlib.pbkdf2_hmac('sha256', 
                                             password.encode('utf-8'), 
                                             salt.encode('utf-8'), 
                                             100000)
            password_hash_b64 = base64.b64encode(password_hash).decode('ascii')
            
            return hmac.compare_digest(stored_hash, password_hash_b64)
        return False
    except Exception as e:
        st.error(f"❌ Erro na autenticação: {str(e)}")
        return False

def decrypt_data():
    """Descriptografa e carrega os dados"""
    try:
        # Obtém os dados criptografados do secrets
        encrypted_data = base64.b64decode(st.secrets["data"]["encrypted_file"])
        
        # Descriptografa
        key = get_encryption_key()
        fernet = Fernet(key)
        decrypted_data = fernet.decrypt(encrypted_data)
        
        # Carrega como Excel
        df = pd.read_excel(BytesIO(decrypted_data), sheet_name='Base vendas', dtype=str)
        return df
    except Exception as e:
        st.error(f"❌ Erro ao descriptografar dados: {str(e)}")
        st.stop()

def login_form():
    """Formulário de login"""
    st.markdown("""
    <div style="text-align: center; padding: 50px 0;">
        <h1>🔐 PAINEL DE VENDAS</h1>
        <h3>Acesso Restrito</h3>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("login_form"):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            username = st.text_input("👤 Usuário", placeholder="Digite seu usuário")
            password = st.text_input("🔑 Senha", type="password", placeholder="Digite sua senha")
            
            submitted = st.form_submit_button("🚀 ENTRAR", use_container_width=True)
            
            if submitted:
                if username and password:
                    if verify_password(username, password):
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.success("✅ Login realizado com sucesso!")
                        st.rerun()
                    else:
                        st.error("❌ Usuário ou senha incorretos!")
                else:
                    st.warning("⚠️ Preencha todos os campos!")

def logout():
    """Função de logout"""
    if st.sidebar.button("🚪 Sair"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

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
    """Carrega dados descriptografados"""
    df = decrypt_data()
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
    try:
        fig = px.line(
            df,
            x='AnoMes',
            y='Quantidade',
            color='Previsao',
            title=title.upper(),
            markers=True,
            labels={'AnoMes': 'MÊS', 'Quantidade': 'QUANTIDADE', 'Previsao': 'TIPO'}
        )

        fig.for_each_trace(
            lambda t: t.update(line=dict(color='black')) if t.name == 'HISTÓRICO' else t.update(line=dict(color='red'))
        )

        fig.update_layout(
            title_x=0.5,
            hovermode='x unified',
            xaxis=dict(
                title='<b>MÊS</b>',
                title_font=dict(size=14, color='black'),
                tickfont=dict(size=12, color='black')
            ),
            yaxis=dict(
                title='<b>QUANTIDADE</b>',
                title_font=dict(size=14, color='black'),
                tickfont=dict(size=12, color='black')
            )
        )

        return fig
    except Exception as e:
        st.error(f"❌ Erro ao criar gráfico: {str(e)}")
        return None

def main_app():
    """Aplicação principal após autenticação"""
    st.set_page_config(page_title="PAINEL DE VENDAS", layout="wide")
    
    # Header com informações do usuário
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("📊 PAINEL DE VENDAS E PREVISÃO")
    with col2:
        st.markdown(f"**👤 Usuário:** {st.session_state.username}")
        logout()

    @st.cache_data
    def get_data():
        return load_data()
    
    try:
        df = get_data()
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {str(e)}")
        return

    if not validate_data(df, ['Cliente', 'Produto', 'Quantidade', 'AnoMes', 'Grupo']):
        return

    grupo = st.selectbox("SELECIONE A LINHA", ["TODOS"] + sorted(df['Grupo'].unique()))
    dfg = df if grupo == "TODOS" else df[df['Grupo'] == grupo]

    cliente = st.selectbox("SELECIONE O CLIENTE", ["TODOS"] + sorted(dfg['Cliente'].unique()))
    dfc = dfg if cliente == "TODOS" else dfg[dfg['Cliente'] == cliente]

    produto = st.selectbox("SELECIONE O PRODUTO", ["TODOS"] + sorted(dfc['Produto'].unique()))
    dff = dfc if produto == "TODOS" else dfc[dfc['Produto'] == produto]

    if dff.empty:
        st.warning("⚠️ Nenhum dado com os filtros aplicados.")
        return

    grouped = dff.groupby('AnoMes', as_index=False)['Quantidade'].sum()
    grouped['Previsao'] = 'HISTÓRICO'
    serie = grouped.set_index('AnoMes')['Quantidade'].sort_index()

    try:
        fc = make_forecast_from_series(serie)
        resultado = pd.concat([grouped, fc], ignore_index=True)
    except Exception as e:
        st.error(f"❌ Erro na previsão: {e}")
        return

    if grupo != "TODOS" and cliente == "TODOS" and produto == "TODOS":
        titulo = f"GRUPO {grupo} - CONSOLIDADO"
    elif cliente != "TODOS" and produto == "TODOS":
        titulo = f"{cliente} - TODOS OS PRODUTOS"
    elif cliente == "TODOS" and produto != "TODOS":
        titulo = f"TODOS OS CLIENTES - {produto}"
    elif cliente != "TODOS" and produto != "TODOS":
        titulo = f"{cliente} - {produto}"
    else:
        titulo = "PREVISÃO TOTAL"

    st.markdown(f"### 📌 {titulo}")

    fig = create_plot(resultado, titulo)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    with st.expander("📈 ESTATÍSTICAS DETALHADAS", expanded=True):
        historico = resultado[resultado['Previsao'] == 'HISTÓRICO']['Quantidade']
        previsao = resultado[resultado['Previsao'] == 'PREVISÃO']['Quantidade']

        st.subheader("📊 HISTÓRICO")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total", f"{historico.sum():,.0f}")
        col2.metric("Média", f"{historico.mean():.2f}")
        col3.metric("Mediana", f"{historico.median():.0f}")
        col4.metric("Desvio Padrão", f"{historico.std():.2f}")

        st.markdown("")

        st.subheader("📈 PREVISÃO")
        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Total Previsto", f"{previsao.sum():,.0f}")
        col6.metric("Média Prevista", f"{previsao.mean():.2f}")
        col7.metric("Mediana Prevista", f"{previsao.median():.0f}")
        col8.metric("Desvio Padrão", f"{previsao.std():.2f}")

        st.markdown("")
        st.caption("⚠️ Valores previstos foram suavizados com um fator de redução para representar cenários mais conservadores.")

def main():
    """Função principal com controle de autenticação"""
    # Inicializa estado de autenticação
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    # Verifica se está autenticado
    if st.session_state.authenticated:
        main_app()
    else:
        login_form()

if __name__ == "__main__":
    main()