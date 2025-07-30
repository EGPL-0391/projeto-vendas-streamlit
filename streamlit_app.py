import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os
import unicodedata
import logging
from io import BytesIO

# CSS para ocultar o footer do Streamlit
hide_streamlit_style = """
<style>
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# === Configurações ===
FORECAST_MONTHS = 6
REDUCTION_FACTOR = 0.9
MIN_DATE = '2024-01-01'
logging.getLogger('streamlit.runtime.scriptrunner').setLevel(logging.ERROR)

# === Credenciais de Autenticação ===
USUARIOS = {
    "comercial": "cad@2025"
}

def check_authentication():
    """Verifica se o usuário está autenticado"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        show_login_page()
        return False
    return True

def show_login_page():
    """Exibe a página de login"""
    st.set_page_config(page_title="LOGIN - PAINEL DE VENDAS", layout="centered")
    
    # CSS para estilizar o formulário de login
    st.markdown("""
    <style>
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        background-color: #f8f9fa;
    }
    .login-title {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 2rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #3498db;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #2980b9;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="login-title">🔐 ACESSO AO SISTEMA</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="login-title">PAINEL DE VENDAS</h3>', unsafe_allow_html=True)
    
    with st.form("login_form"):
        st.markdown("### 👤 CREDENCIAIS")
        usuario = st.text_input("USUÁRIO", placeholder="Digite seu usuário")
        senha = st.text_input("SENHA", type="password", placeholder="Digite sua senha")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submit_button = st.form_submit_button("🚀 ENTRAR")
        
        if submit_button:
            if authenticate_user(usuario, senha):
                st.session_state.authenticated = True
                st.success("✅ Login realizado com sucesso!")
                st.rerun()
            else:
                st.error("❌ Usuário ou senha incorretos!")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Informações adicionais
    st.markdown("---")
    st.markdown("### ℹ️ INFORMAÇÕES DO SISTEMA")
    st.info("Sistema de análise de vendas e previsões para tomada de decisões comerciais.")

def authenticate_user(usuario, senha):
    """Autentica o usuário"""
    return usuario in USUARIOS and USUARIOS[usuario] == senha

def logout():
    """Realiza o logout do usuário"""
    st.session_state.authenticated = False
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

        # Cores: histórico (preto), previsão (vermelho)
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

def create_export_table(df, selected_date):
    """Cria tabela consolidada por produto para exportação"""
    export_data = []
    produtos = df['Produto'].unique()
    
    for produto in produtos:
        df_produto = df[df['Produto'] == produto]
        
        # Consolidar por mês para o produto
        grouped = df_produto.groupby('AnoMes', as_index=False)['Quantidade'].sum()
        
        if len(grouped) < 2:  # Reduzir requisito mínimo
            continue
        
        # Criar série temporal
        serie = grouped.set_index('AnoMes')['Quantidade'].sort_index()
        
        try:
            # Gerar previsão
            fc = make_forecast_from_series(serie)
            
            # Procurar pela data selecionada
            previsao_mes = fc[fc['AnoMes'] == selected_date]
            
            if not previsao_mes.empty:
                quantidade_prevista = int(previsao_mes['Quantidade'].iloc[0])
                # Só incluir se quantidade > 0
                if quantidade_prevista > 0:
                    export_data.append({
                        'Produto': produto,
                        'Data': selected_date.strftime('%m/%Y'),
                        'Quantidade_Prevista': quantidade_prevista
                    })
        except Exception as e:
            # Debug: mostrar erro no console se necessário
            print(f"Erro na previsão para produto {produto}: {e}")
            continue
    
    return pd.DataFrame(export_data)

def to_excel(df):
    """Converte DataFrame para Excel em memória"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Previsao_Produtos', index=False)
        
        # Formatação básica
        workbook = writer.book
        worksheet = writer.sheets['Previsao_Produtos']
        
        # Formato para números
        number_format = workbook.add_format({'num_format': '#,##0'})
        worksheet.set_column('C:C', 15, number_format)
        
        # Cabeçalho em negrito
        header_format = workbook.add_format({'bold': True})
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
    
    output.seek(0)
    return output

def show_export_section(df):
    """Seção para exportação de previsões por produto"""
    st.markdown("---")
    st.markdown("## 📋 EXPORTAÇÃO DE PREVISÕES POR PRODUTO")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Filtro de Grupo
        grupo_export = st.selectbox(
            "LINHA:", 
            ["TODOS"] + sorted(df['Grupo'].unique()),
            key="grupo_export"
        )
        
        # Filtro por cliente
        dfg_export = df if grupo_export == "TODOS" else df[df['Grupo'] == grupo_export]
        cliente_export = st.selectbox(
            "CLIENTE:", 
            ["TODOS"] + sorted(dfg_export['Cliente'].unique()),
            key="cliente_export"
        )
    
    with col2:
        # Seletor de produtos
        dfc_export = dfg_export if cliente_export == "TODOS" else dfg_export[dfg_export['Cliente'] == cliente_export]
        produtos_disponiveis = ["TODOS"] + sorted(dfc_export['Produto'].unique())
        produtos_selecionados = st.multiselect(
            "PRODUTOS:", 
            produtos_disponiveis,
            default=["TODOS"],
            key="produtos_export"
        )
        
        # Seletor de data - incluindo histórico e previsão
        # Pegar últimos 6 meses históricos
        max_date = df['AnoMes'].max()
        data_options = []
        
        # Adicionar últimos 6 meses históricos
        for i in range(6, 0, -1):
            hist_date = max_date - pd.DateOffset(months=i-1)
            data_options.append(('HISTÓRICO', hist_date))
        
        # Adicionar próximos 6 meses de previsão
        for i in range(1, FORECAST_MONTHS + 1):
            future_date = max_date + pd.DateOffset(months=i)
            data_options.append(('PREVISÃO', future_date))
        
        # Criar selectbox com opções formatadas
        date_labels = [f"{tipo} - {data.strftime('%m/%Y')}" for tipo, data in data_options]
        selected_index = st.selectbox(
            "MÊS/ANO:",
            range(len(date_labels)),
            format_func=lambda x: date_labels[x],
            index=6,  # Começar no primeiro mês de previsão
            key="data_export"
        )
        
        selected_type, selected_date = data_options[selected_index]
    
    # Aplicar filtros
    df_filtered = dfc_export.copy()
    
    if "TODOS" not in produtos_selecionados and produtos_selecionados:
        df_filtered = df_filtered[df_filtered['Produto'].isin(produtos_selecionados)]
    
    if not df_filtered.empty:
        # Gerar tabela de exportação
        if selected_type == 'HISTÓRICO':
            # Para dados históricos, usar dados reais
            hist_data = df_filtered[df_filtered['AnoMes'] == selected_date]
            if not hist_data.empty:
                export_table = hist_data.groupby('Produto', as_index=False)['Quantidade'].sum()
                export_table.columns = ['Produto', 'Quantidade_Prevista']
                export_table['Data'] = selected_date.strftime('%m/%Y')
                export_table = export_table[['Produto', 'Data', 'Quantidade_Prevista']]
            else:
                export_table = pd.DataFrame()
        else:
            # Para previsões, usar a função de previsão
            export_table = create_export_table(df_filtered, selected_date)
        
        if not export_table.empty:
            st.markdown(f"### 📊 PREVIEW DA TABELA - {selected_type}")
            
            # Mostrar resumo
            col1, col2, col3 = st.columns(3)
            col1.metric("Total de Produtos", len(export_table))
            col2.metric("Quantidade Total", f"{export_table['Quantidade_Prevista'].sum():,}")
            col3.metric("Média por Produto", f"{export_table['Quantidade_Prevista'].mean():.0f}")
            
            # Mostrar tabela
            st.dataframe(
                export_table.sort_values('Quantidade_Prevista', ascending=False),
                use_container_width=True
            )
            
            # Botão de download
            excel_file = to_excel(export_table)
            filename = f"{selected_type.lower()}_produtos_{selected_date.strftime('%m_%Y')}.xlsx"
            
            st.download_button(
                label="📥 BAIXAR EXCEL",
                data=excel_file,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
            
        else:
            st.warning(f"⚠️ Nenhum dado disponível para {selected_type.lower()} em {selected_date.strftime('%m/%Y')}.")
    else:
        st.warning("⚠️ Nenhum dado disponível com os filtros aplicados.")

def show_dashboard():
    """Exibe o dashboard principal após autenticação"""
    st.set_page_config(page_title="PAINEL DE VENDAS", layout="wide")
    
    # Header com botão de logout
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("📊 PAINEL DE VENDAS E PREVISÃO")
    with col2:
        st.markdown("### 👤 Usuário: comercial")
        if st.button("🚪 SAIR", type="secondary"):
            logout()

    @st.cache_data
    def get_data():
        return load_data()
    
    df = get_data()

    if not validate_data(df, ['Cliente', 'Produto', 'Quantidade', 'AnoMes', 'Grupo']):
        st.stop()

    # === SEÇÃO PRINCIPAL DE GRÁFICOS ===
    st.markdown("## 📈 ANÁLISE GRÁFICA")
    
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

    # === NOVA SEÇÃO DE EXPORTAÇÃO ===
    show_export_section(df)

def main():
    """Função principal que controla o fluxo da aplicação"""
    # Verifica autenticação
    if not check_authentication():
        return
    
    # Se autenticado, mostra o dashboard
    show_dashboard()

if __name__ == "__main__":
    main()