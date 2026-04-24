import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os
import unicodedata
import logging
from io import BytesIO

# ============================================================
# TEMA VISUAL — Dark Industrial
# ============================================================
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@400;500;600;700&display=swap');

:root {
    --bg-main:      #0d1117;
    --bg-card:      #161b22;
    --bg-card2:     #1c2333;
    --accent:       #f97316;
    --accent2:      #38bdf8;
    --success:      #22c55e;
    --warning:      #eab308;
    --danger:       #ef4444;
    --text-primary: #f0f6fc;
    --text-muted:   #8b949e;
    --border:       #30363d;
}

/* Base */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-main) !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stSidebar"] { background-color: var(--bg-card) !important; }

/* Título principal */
h1 { 
    font-family: 'Space Mono', monospace !important;
    color: var(--text-primary) !important;
    letter-spacing: -1px;
    border-bottom: 2px solid var(--accent);
    padding-bottom: 0.4rem;
}
h2, h3 {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-primary) !important;
}

/* Métricas */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 1rem 1.2rem !important;
    border-left: 3px solid var(--accent) !important;
}
[data-testid="stMetricLabel"] > div {
    color: var(--text-muted) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] > div {
    color: var(--text-primary) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 1.6rem !important;
}

/* Selectbox */
[data-baseweb="select"] > div {
    background-color: var(--bg-card2) !important;
    border-color: var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}
[data-baseweb="select"] span { color: var(--text-primary) !important; }

/* Labels */
label[data-testid="stWidgetLabel"] > div > p {
    color: var(--text-muted) !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    font-weight: 600 !important;
}

/* Botões */
button[kind="primary"], [data-testid="baseButton-primary"] {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: 0.03em !important;
}
button[kind="secondary"], [data-testid="baseButton-secondary"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* Alertas */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    border: none !important;
    font-size: 0.9rem !important;
}
[data-testid="stAlert"][data-baseweb="notification"] {
    background: rgba(34, 197, 94, 0.12) !important;
    border-left: 3px solid var(--success) !important;
    color: #bbf7d0 !important;
}

/* Info box */
div[data-testid="stInfo"] {
    background: rgba(56, 189, 248, 0.1) !important;
    border-left: 3px solid var(--accent2) !important;
    color: #bae6fd !important;
    border-radius: 8px !important;
}

/* Warning */
div[data-testid="stWarning"] {
    background: rgba(234, 179, 8, 0.1) !important;
    border-left: 3px solid var(--warning) !important;
    color: #fef08a !important;
    border-radius: 8px !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* Caption */
[data-testid="stCaptionContainer"] p { color: var(--text-muted) !important; font-size: 0.78rem !important; }

/* Input text */
input[type="text"], input[type="password"] {
    background-color: var(--bg-card2) !important;
    color: var(--text-primary) !important;
    border-color: var(--border) !important;
    border-radius: 8px !important;
}

/* Form */
[data-testid="stForm"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 1.5rem !important;
}

/* Download button */
[data-testid="stDownloadButton"] button {
    background: var(--accent) !important;
    color: #fff !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    border: none !important;
    width: 100% !important;
    font-size: 1rem !important;
    padding: 0.6rem 1rem !important;
}

/* Subheader */
[data-testid="stHeadingWithActionElements"] h2,
[data-testid="stHeadingWithActionElements"] h3 {
    color: var(--text-primary) !important;
}

/* Balloons / misc */
footer { visibility: hidden; }
</style>
"""

# ============================================================
# PLOTLY THEME — Dark
# ============================================================
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#f0f6fc', family='DM Sans'),
    title_font=dict(color='#f0f6fc', size=15),
    xaxis=dict(
        gridcolor='#21262d', gridwidth=1,
        linecolor='#30363d', tickcolor='#30363d',
        title_font=dict(color='#8b949e', size=12),
        tickfont=dict(color='#8b949e', size=11)
    ),
    yaxis=dict(
        gridcolor='#21262d', gridwidth=1,
        linecolor='#30363d', tickcolor='#30363d',
        title_font=dict(color='#8b949e', size=12),
        tickfont=dict(color='#8b949e', size=11)
    ),
    legend=dict(
        bgcolor='rgba(22,27,34,0.8)',
        bordercolor='#30363d', borderwidth=1,
        font=dict(color='#f0f6fc')
    ),
    hoverlabel=dict(
        bgcolor='#1c2333', bordercolor='#30363d',
        font=dict(color='#f0f6fc')
    )
)

# === Configurações ===
FORECAST_MONTHS  = 6
MIN_POINTS_MODEL = 12
MIN_DATE         = '2024-01-01'
logging.getLogger('streamlit.runtime.scriptrunner').setLevel(logging.ERROR)
USUARIOS = {"comercial": "cad@2025"}

# ============================================================
# AUTENTICAÇÃO
# ============================================================
def check_authentication():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        show_login_page()
        return False
    return True

def show_login_page():
    st.set_page_config(page_title="LOGIN - PAINEL DE VENDAS", layout="centered")
    st.markdown(THEME_CSS, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; padding: 2rem 0 1rem;">
        <p style="font-family:'Space Mono',monospace; font-size:2.5rem; color:#f97316; margin:0;">⬡</p>
        <h1 style="font-family:'Space Mono',monospace; font-size:1.4rem; color:#f0f6fc; margin:0.5rem 0 0;">
            PAINEL DE VENDAS
        </h1>
        <p style="color:#8b949e; font-size:0.85rem; margin-top:0.3rem;">Sistema de Análise e Previsão Comercial</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("login_form"):
        usuario = st.text_input("USUÁRIO", placeholder="Digite seu usuário")
        senha   = st.text_input("SENHA", type="password", placeholder="Digite sua senha")
        submit  = st.form_submit_button("🚀 ENTRAR", type="primary", use_container_width=True)
        if submit:
            if usuario in USUARIOS and USUARIOS[usuario] == senha:
                st.session_state.authenticated = True
                st.success("✅ Login realizado com sucesso!")
                st.balloons()
                st.rerun()
            else:
                st.error("❌ Usuário ou senha incorretos!")

def logout():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.authenticated = False
    st.rerun()

# ============================================================
# UTILITÁRIOS
# ============================================================
def remove_acentos(text):
    if not isinstance(text, str):
        return text
    return ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn').strip().lower()

def find_column(df, target):
    t = remove_acentos(target)
    for col in df.columns:
        if remove_acentos(col) == t:
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

# ============================================================
# CARGA DE DADOS
# ============================================================
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
            st.error(f"❌ Coluna '{c}' não encontrada.")
            st.stop()
        cols[c] = fc

    df[cols['Cliente']]    = df[cols['Cliente']].astype(str).str.strip().str.upper()
    df[cols['Produto']]    = df[cols['Produto']].astype(str).str.strip().str.upper()
    df[cols['Emissao']]    = pd.to_datetime(df[cols['Emissao']], errors='coerce')
    df[cols['Quantidade']] = pd.to_numeric(df[cols['Quantidade']], errors='coerce')
    df = df.dropna(subset=list(cols.values()))
    df = df[df[cols['Emissao']] >= pd.to_datetime(MIN_DATE)]
    if df.empty:
        st.error("❌ Nenhum dado após filtragem.")
        st.stop()

    df['AnoMes'] = df[cols['Emissao']].dt.to_period('M').dt.to_timestamp()
    gc = find_column(df, 'Grupo')
    df['Grupo'] = df[gc].astype(str).str.strip().str.upper() if gc else 'SEM GRUPO'
    return df[['Cliente', 'Produto', 'Quantidade', 'AnoMes', 'Grupo']]

# ============================================================
# MODELO DE PREVISÃO
# ============================================================
def make_forecast_from_series(serie):
    serie = serie.sort_index()
    n     = len(serie)
    if n < MIN_POINTS_MODEL:
        return None

    seasonal         = 'add' if n >= 24 else None
    seasonal_periods = 12    if n >= 24 else None

    try:
        model = ExponentialSmoothing(
            serie, trend='add', damped_trend=True,
            seasonal=seasonal, seasonal_periods=seasonal_periods,
            initialization_method='estimated'
        ).fit(optimized=True)
    except Exception:
        try:
            model = ExponentialSmoothing(
                serie, trend='add', damped_trend=True,
                seasonal=None, initialization_method='estimated'
            ).fit(optimized=True)
        except Exception:
            return None

    idx = pd.date_range(
        start=serie.index[-1] + pd.offsets.MonthBegin(),
        periods=FORECAST_MONTHS, freq='MS'
    )
    fc = model.forecast(FORECAST_MONTHS).round().clip(lower=0).astype(int)
    fc.index = idx
    result = fc.reset_index()
    result.columns = ['AnoMes', 'Quantidade']
    result['Previsao'] = 'PREVISÃO'
    return result

# ============================================================
# ACURÁCIA
# ============================================================
def calcular_acuracia(serie):
    n = len(serie)
    if n < MIN_POINTS_MODEL:
        return None

    n_test  = max(3, n // 3)
    n_train = n - n_test
    if n_train < 6:
        return None

    s_train = serie.iloc[:n_train]
    s_test  = serie.iloc[n_train:]
    seasonal, sp = ('add', 12) if n_train >= 24 else (None, None)

    try:
        m = ExponentialSmoothing(
            s_train, trend='add', damped_trend=True,
            seasonal=seasonal, seasonal_periods=sp,
            initialization_method='estimated'
        ).fit(optimized=True)
        pred = m.forecast(n_test).clip(lower=0)
    except Exception:
        try:
            m = ExponentialSmoothing(
                s_train, trend='add', damped_trend=True,
                seasonal=None, initialization_method='estimated'
            ).fit(optimized=True)
            pred = m.forecast(n_test).clip(lower=0)
        except Exception:
            return None

    real = s_test.values
    prev = pred.values
    mask = real > 0
    if mask.sum() == 0:
        return None

    mape = np.mean(np.abs((real[mask] - prev[mask]) / real[mask])) * 100
    mae  = np.mean(np.abs(real - prev))
    rmse = np.sqrt(np.mean((real - prev) ** 2))

    if mape <= 15:
        cls, cor = "🟢 ÓTIMA",   "#22c55e"
    elif mape <= 30:
        cls, cor = "🟡 REGULAR", "#eab308"
    else:
        cls, cor = "🔴 BAIXA",   "#ef4444"

    return dict(mape=round(mape,1), mae=round(mae,1), rmse=round(rmse,1),
                classificacao=cls, cor=cor, n_train=n_train, n_test=n_test,
                real=real, prev=prev, datas_teste=s_test.index)

def show_acuracia_panel(serie, titulo):
    st.markdown("---")
    st.markdown("## 🎯 ACURÁCIA DO MODELO")
    acc = calcular_acuracia(serie)

    if acc is None:
        st.warning(f"⚠️ Dados insuficientes para calcular acurácia (mínimo {MIN_POINTS_MODEL} meses).")
        return

    # Badge de qualidade
    st.markdown(
        f"<div style='display:inline-block; background:{acc['cor']}22; border:1px solid {acc['cor']}; "
        f"color:{acc['cor']}; border-radius:6px; padding:4px 14px; font-size:0.85rem; font-weight:700; margin-bottom:1rem;'>"
        f"QUALIDADE: {acc['classificacao']}</div>",
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAPE",   f"{acc['mape']}%",    help="Erro percentual médio — quanto menor, melhor")
    c2.metric("MAE",    f"{acc['mae']:.0f}",  help="Erro absoluto médio em unidades")
    c3.metric("RMSE",   f"{acc['rmse']:.0f}", help="Raiz do erro quadrático médio")
    c4.metric("VALIDAÇÃO", f"{acc['n_test']} meses")

    st.caption(f"Treino: {acc['n_train']} meses | Validação: últimos {acc['n_test']} meses (walk-forward)")

    # Gráfico real vs previsto
    df_comp = pd.DataFrame({
        'Data':    acc['datas_teste'],
        'Real':    acc['real'],
        'Previsto': acc['prev'].round().astype(int)
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_comp['Data'], y=df_comp['Real'],
        name='Real', mode='lines+markers',
        line=dict(color='#f0f6fc', width=2),
        marker=dict(size=7, color='#f0f6fc')
    ))
    fig.add_trace(go.Scatter(
        x=df_comp['Data'], y=df_comp['Previsto'],
        name='Previsto', mode='lines+markers',
        line=dict(color='#f97316', width=2, dash='dash'),
        marker=dict(size=7, symbol='x', color='#f97316')
    ))
    fig.update_layout(
        title=f"REAL vs PREVISTO — VALIDAÇÃO | {titulo.upper()}",
        hovermode='x unified',
        xaxis_title='MÊS', yaxis_title='QUANTIDADE',
        legend=dict(orientation='h', y=-0.25),
        **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ Como interpretar o MAPE?"):
        st.markdown("""
| MAPE | Qualidade | O que significa |
|------|-----------|-----------------|
| ≤ 15% | 🟢 Ótima | Previsões confiáveis para tomada de decisão |
| 16–30% | 🟡 Regular | Use com cautela, revise trimestralmente |
| > 30% | 🔴 Baixa | Vendas muito irregulares — evite decisões críticas baseadas só na previsão |
        """)

    with st.expander("📋 TABELA REAL vs PREVISTO"):
        df_comp['Erro Abs'] = (df_comp['Real'] - df_comp['Previsto']).abs()
        df_comp['Erro %']   = np.where(
            df_comp['Real'] > 0,
            ((df_comp['Real'] - df_comp['Previsto']).abs() / df_comp['Real'] * 100).round(1),
            np.nan
        )
        df_comp['Data'] = df_comp['Data'].dt.strftime('%m/%Y')
        st.dataframe(df_comp.set_index('Data'), use_container_width=True)

# ============================================================
# GRÁFICOS
# ============================================================
def create_plot(df, title):
    fig = go.Figure()
    hist = df[df['Previsao'] == 'HISTÓRICO']
    prev = df[df['Previsao'] == 'PREVISÃO']

    fig.add_trace(go.Scatter(
        x=hist['AnoMes'], y=hist['Quantidade'],
        name='HISTÓRICO', mode='lines+markers',
        line=dict(color='#38bdf8', width=2.5),
        marker=dict(size=6, color='#38bdf8')
    ))
    if not prev.empty:
        # Conecta histórico ao primeiro ponto da previsão
        if not hist.empty:
            conn_x = [hist['AnoMes'].iloc[-1], prev['AnoMes'].iloc[0]]
            conn_y = [hist['Quantidade'].iloc[-1], prev['Quantidade'].iloc[0]]
            fig.add_trace(go.Scatter(
                x=conn_x, y=conn_y, mode='lines',
                line=dict(color='#f97316', width=2, dash='dot'),
                showlegend=False, hoverinfo='skip'
            ))
        fig.add_trace(go.Scatter(
            x=prev['AnoMes'], y=prev['Quantidade'],
            name='PREVISÃO', mode='lines+markers',
            line=dict(color='#f97316', width=2.5, dash='dash'),
            marker=dict(size=7, color='#f97316', symbol='diamond')
        ))

    fig.update_layout(
        title=title.upper(), hovermode='x unified',
        xaxis_title='MÊS', yaxis_title='QUANTIDADE',
        **PLOTLY_LAYOUT
    )
    return fig

def create_bar_chart(df, grupo_atual, cliente_atual, produto_atual):
    try:
        dfg = df if grupo_atual == "TODOS" else df[df['Grupo'] == grupo_atual]
        dfc = dfg if cliente_atual == "TODOS" else dfg[dfg['Cliente'] == cliente_atual]
        dff = dfc if produto_atual == "TODOS" else dfc[dfc['Produto'] == produto_atual]
        if dff.empty:
            return None

        if cliente_atual != "TODOS" and produto_atual == "TODOS":
            grouped = dff.groupby('Produto')['Quantidade'].sum().reset_index()
            titulo, x_label = f"PRODUTOS MAIS VENDIDOS — {cliente_atual}", "PRODUTO"
        elif grupo_atual != "TODOS" and cliente_atual == "TODOS" and produto_atual == "TODOS":
            grouped = dff.groupby('Cliente')['Quantidade'].sum().reset_index()
            titulo, x_label = f"CLIENTES — LINHA {grupo_atual}", "CLIENTE"
        elif produto_atual != "TODOS" and cliente_atual == "TODOS":
            grouped = dff.groupby('Cliente')['Quantidade'].sum().reset_index()
            titulo, x_label = f"CLIENTES — {produto_atual}", "CLIENTE"
        elif cliente_atual == "TODOS" and produto_atual == "TODOS" and grupo_atual == "TODOS":
            grouped = dff.groupby('Grupo')['Quantidade'].sum().reset_index()
            titulo, x_label = "RANKING POR LINHA", "LINHA"
        else:
            grouped = dff.groupby('AnoMes')['Quantidade'].sum().reset_index()
            grouped['Produto'] = grouped['AnoMes'].dt.strftime('%m/%Y')
            titulo, x_label = f"VENDAS MENSAIS — {cliente_atual} / {produto_atual}", "MÊS"

        grouped = grouped.sort_values('Quantidade', ascending=True).tail(20)
        grouped['Label'] = grouped.iloc[:, 0]

        # Escala de cor laranja → azul
        fig = px.bar(
            grouped, x='Quantidade', y='Label', orientation='h',
            title=titulo.upper(),
            labels={'Quantidade': 'QUANTIDADE', 'Label': x_label},
            color='Quantidade',
            color_continuous_scale=[[0, '#1e3a5f'], [0.5, '#0ea5e9'], [1, '#f97316']]
        )
        fig.update_layout(
            height=max(380, len(grouped) * 28),
            xaxis_title='QUANTIDADE VENDIDA', yaxis_title=x_label,
            showlegend=False, coloraxis_showscale=False,
            **PLOTLY_LAYOUT
        )
        fig.update_traces(texttemplate='%{x:,.0f}', textposition='outside',
                          textfont=dict(color='#f0f6fc', size=11))
        return fig
    except Exception as e:
        st.error(f"❌ Erro ao criar gráfico de barras: {e}")
        return None

# ============================================================
# EXPORTAÇÃO
# ============================================================
def create_all_forecasts_table(df):
    all_forecasts  = []
    max_date       = df['AnoMes'].max()
    forecast_dates = [max_date + pd.DateOffset(months=i) for i in range(1, FORECAST_MONTHS + 1)]
    for produto in df['Produto'].unique():
        serie = df[df['Produto'] == produto].groupby('AnoMes')['Quantidade'].sum().sort_index()
        fc    = make_forecast_from_series(serie)
        if fc is None:
            continue
        for fd in forecast_dates:
            row = fc[fc['AnoMes'] == fd]
            if not row.empty and int(row['Quantidade'].iloc[0]) > 0:
                all_forecasts.append({
                    'Produto':            produto,
                    'Data':               fd.strftime('%m/%Y'),
                    'AnoMes':             fd,
                    'Quantidade_Prevista': int(row['Quantidade'].iloc[0])
                })
    return pd.DataFrame(all_forecasts)

def to_excel_single(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Previsoes_Completas', index=False)
        wb, ws = writer.book, writer.sheets['Previsoes_Completas']
        ws.set_column('A:A', 30)
        ws.set_column('B:B', 10)
        ws.set_column('C:C', 15, wb.add_format({'num_format': '#,##0'}))
        hfmt = wb.add_format({'bold': True})
        for i, col in enumerate(df.columns.values):
            ws.write(0, i, col, hfmt)
    output.seek(0)
    return output

def show_export_section(df, grupo_atual, cliente_atual, produto_atual):
    st.markdown("---")
    st.markdown("## 📋 EXPORTAÇÃO DE PREVISÕES")
    st.info(f"📊 **Filtros:** Linha: {grupo_atual} | Cliente: {cliente_atual} | Produto: {produto_atual}")

    dfg = df if grupo_atual == "TODOS" else df[df['Grupo'] == grupo_atual]
    dfc = dfg if cliente_atual == "TODOS" else dfg[dfg['Cliente'] == cliente_atual]
    dff = dfc if produto_atual == "TODOS" else dfc[dfc['Produto'] == produto_atual]

    if dff.empty:
        st.warning("⚠️ Nenhum dado disponível.")
        return

    all_forecasts = create_all_forecasts_table(dff)
    if all_forecasts.empty:
        st.warning("⚠️ Nenhuma previsão disponível (dados insuficientes).")
        return

    datas = sorted(all_forecasts['Data'].unique(), key=lambda d: pd.to_datetime(d, format='%m/%Y'))

    col_de, col_ate = st.columns(2)
    with col_de:
        data_inicio = st.selectbox("🗓️ DE",  datas, index=0, key="export_inicio")
    with col_ate:
        opcoes_ate = [d for d in datas if pd.to_datetime(d, format='%m/%Y') >= pd.to_datetime(data_inicio, format='%m/%Y')]
        data_fim   = st.selectbox("🗓️ ATÉ", opcoes_ate, index=len(opcoes_ate)-1, key="export_fim")

    dt_i = pd.to_datetime(data_inicio, format='%m/%Y')
    dt_f = pd.to_datetime(data_fim,    format='%m/%Y')
    df_export = all_forecasts[(all_forecasts['AnoMes'] >= dt_i) & (all_forecasts['AnoMes'] <= dt_f)]

    if df_export.empty:
        st.warning("⚠️ Nenhuma previsão no intervalo selecionado.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("🎯 PRODUTOS",        len(df_export['Produto'].unique()))
    c2.metric("📅 MESES",           len(df_export['Data'].unique()))
    c3.metric("📊 TOTAL PREVISÕES", len(df_export))

    if grupo_atual != "TODOS" and cliente_atual == "TODOS":
        suffix = f"grupo_{grupo_atual.replace(' ','_')}"
    elif cliente_atual != "TODOS" and produto_atual == "TODOS":
        suffix = f"cliente_{cliente_atual.replace(' ','_')}"
    elif cliente_atual == "TODOS" and produto_atual != "TODOS":
        suffix = f"produto_{produto_atual.replace(' ','_')}"
    elif cliente_atual != "TODOS":
        suffix = f"{cliente_atual.replace(' ','_')}_{produto_atual.replace(' ','_')}"
    else:
        suffix = "todos"

    periodo  = f"{data_inicio.replace('/','')}_a_{data_fim.replace('/','')}"
    filename = f"previsoes_{suffix}_{periodo}.xlsx"

    df_ord = (df_export[['Produto','Data','Quantidade_Prevista']]
              .sort_values(['Data','Quantidade_Prevista'], ascending=[True,False]))

    st.download_button("📥 BAIXAR PREVISÕES", data=to_excel_single(df_ord),
                       file_name=filename, type="primary",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with st.expander("👀 PREVIEW"):
        st.dataframe(df_ord, use_container_width=True)

# ============================================================
# DASHBOARD PRINCIPAL
# ============================================================
def show_dashboard():
    st.set_page_config(page_title="PAINEL DE VENDAS", layout="wide")
    st.markdown(THEME_CSS, unsafe_allow_html=True)

    col1, col2 = st.columns([5, 1])
    with col1:
        st.title("📊 PAINEL DE VENDAS E PREVISÃO")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚪 SAIR", type="secondary", key="logout_btn", use_container_width=True):
            logout()

    @st.cache_data
    def get_data():
        return load_data()

    df = get_data()
    if not validate_data(df, ['Cliente','Produto','Quantidade','AnoMes','Grupo']):
        st.stop()

    # === FILTROS ===
    st.markdown("## 📈 ANÁLISE GRÁFICA")
    for k, v in [('grupo_selecionado','TODOS'),('cliente_selecionado','TODOS'),('produto_selecionado','TODOS')]:
        if k not in st.session_state:
            st.session_state[k] = v

    grupos  = ["TODOS"] + sorted(df['Grupo'].unique())
    grupo   = st.selectbox("LINHA", grupos,
                           index=grupos.index(st.session_state.grupo_selecionado)
                           if st.session_state.grupo_selecionado in grupos else 0,
                           key="grupo_select")
    st.session_state.grupo_selecionado = grupo

    dfg      = df if grupo == "TODOS" else df[df['Grupo'] == grupo]
    clientes = ["TODOS"] + sorted(dfg['Cliente'].unique())
    if st.session_state.cliente_selecionado not in clientes:
        st.session_state.cliente_selecionado = "TODOS"
    cliente  = st.selectbox("CLIENTE", clientes,
                            index=clientes.index(st.session_state.cliente_selecionado),
                            key="cliente_select")
    st.session_state.cliente_selecionado = cliente

    dfc      = dfg if cliente == "TODOS" else dfg[dfg['Cliente'] == cliente]
    produtos = ["TODOS"] + sorted(dfc['Produto'].unique())
    if st.session_state.produto_selecionado not in produtos:
        st.session_state.produto_selecionado = "TODOS"
    produto  = st.selectbox("PRODUTO", produtos,
                            index=produtos.index(st.session_state.produto_selecionado),
                            key="produto_select")
    st.session_state.produto_selecionado = produto

    dff = dfc if produto == "TODOS" else dfc[dfc['Produto'] == produto]
    if dff.empty:
        st.warning("⚠️ Nenhum dado com os filtros aplicados.")
        return

    grouped = dff.groupby('AnoMes', as_index=False)['Quantidade'].sum()
    grouped['Previsao'] = 'HISTÓRICO'
    serie   = grouped.set_index('AnoMes')['Quantidade'].sort_index()
    n       = len(serie)

    # Título dinâmico
    if grupo != "TODOS" and cliente == "TODOS" and produto == "TODOS":
        titulo = f"GRUPO {grupo} — CONSOLIDADO"
    elif cliente != "TODOS" and produto == "TODOS":
        titulo = f"{cliente} — TODOS OS PRODUTOS"
    elif cliente == "TODOS" and produto != "TODOS":
        titulo = f"TODOS OS CLIENTES — {produto}"
    elif cliente != "TODOS" and produto != "TODOS":
        titulo = f"{cliente} — {produto}"
    else:
        titulo = "VISÃO GERAL"

    # Badge do modelo
    if n >= 24:
        st.success(f"✅ Modelo com **sazonalidade ativada** ({n} meses de histórico)")
    elif n >= MIN_POINTS_MODEL:
        st.info(f"ℹ️ Modelo **sem sazonalidade** — precisa de 24 meses; disponível: {n}")
    else:
        st.warning(f"⚠️ Apenas {n} meses — mínimo para previsão é {MIN_POINTS_MODEL}")

    fc = make_forecast_from_series(serie)
    resultado = pd.concat([grouped, fc], ignore_index=True) if fc is not None else grouped

    st.markdown(f"### 📌 {titulo}")
    st.plotly_chart(create_plot(resultado, titulo), use_container_width=True)

    # === BARRAS ===
    st.markdown("---")
    st.markdown("## 📊 RANKING DE VENDAS")
    bar_fig = create_bar_chart(df, grupo, cliente, produto)
    if bar_fig:
        st.plotly_chart(bar_fig, use_container_width=True)
    else:
        st.warning("⚠️ Não foi possível gerar o gráfico de barras com os filtros aplicados.")

    st.divider()

    # === ESTATÍSTICAS ===
    with st.expander("📈 ESTATÍSTICAS DETALHADAS", expanded=True):
        hist_q = grouped['Quantidade']
        st.subheader("📊 HISTÓRICO")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total",         f"{hist_q.sum():,.0f}")
        c2.metric("Média",         f"{hist_q.mean():.1f}")
        c3.metric("Mediana",       f"{hist_q.median():.0f}")
        c4.metric("Desvio Padrão", f"{hist_q.std():.1f}")

        if fc is not None:
            st.markdown("")
            st.subheader("📈 PREVISÃO")
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Total Previsto",   f"{fc['Quantidade'].sum():,.0f}")
            c6.metric("Média Prevista",   f"{fc['Quantidade'].mean():.1f}")
            c7.metric("Mediana Prevista", f"{fc['Quantidade'].median():.0f}")
            c8.metric("Desvio Padrão",    f"{fc['Quantidade'].std():.1f}")
            st.caption("Sazonalidade ativada automaticamente com 24+ meses de dados.")

    # === ACURÁCIA ===
    show_acuracia_panel(serie, titulo)

    # === EXPORTAÇÃO ===
    show_export_section(df, grupo, cliente, produto)

# ============================================================
# MAIN
# ============================================================
def main():
    if not check_authentication():
        return
    show_dashboard()

if __name__ == "__main__":
    main()