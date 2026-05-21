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

hide_streamlit_style = """
<style>
footer {visibility: hidden;}

/* Métricas — fundo levemente destacado e borda lateral colorida */
[data-testid="metric-container"] {
    background-color: #f0f4f8;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    border-left: 4px solid #2563eb;
}
[data-testid="stMetricLabel"] > div  { color: #374151; font-weight: 600; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; }
[data-testid="stMetricValue"] > div  { color: #111827; font-size: 1.5rem; font-weight: 700; }
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# === Configurações ===
FORECAST_MONTHS  = 6
MIN_POINTS_MODEL = 12
MIN_DATE         = '2024-01-01'


logging.getLogger('streamlit.runtime.scriptrunner').setLevel(logging.ERROR)

# === Credenciais ===
USUARIOS = {"comercial": "cad@2025"}

def check_authentication():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        show_login_page()
        return False
    return True

def show_login_page():
    st.set_page_config(page_title="LOGIN - PAINEL DE VENDAS", layout="centered")
    st.markdown("""
    <style>
    .stButton > button { width: 100%; background-color: #3498db; color: white;
        border: none; padding: 0.5rem 1rem; border-radius: 5px; font-weight: bold; }
    .stButton > button:hover { background-color: #2980b9; }
    </style>""", unsafe_allow_html=True)

    st.markdown('<h1 style="text-align:center">🔐 ACESSO AO SISTEMA</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align:center">PAINEL DE VENDAS</h3>', unsafe_allow_html=True)

    with st.form("login_form"):
        st.markdown("### 👤 CREDENCIAIS")
        usuario = st.text_input("USUÁRIO", placeholder="Digite seu usuário")
        senha   = st.text_input("SENHA", type="password", placeholder="Digite sua senha")
        _, col2, _ = st.columns([1, 2, 1])
        with col2:
            submit = st.form_submit_button("🚀 ENTRAR")
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

def remove_acentos(text):
    if not isinstance(text, str):
        return text
    return ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn').strip().lower()

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

    df[cols['Cliente']]    = df[cols['Cliente']].astype(str).str.strip().str.upper()
    df[cols['Produto']]    = df[cols['Produto']].astype(str).str.strip().str.upper()
    df[cols['Emissao']]    = pd.to_datetime(df[cols['Emissao']], errors='coerce')
    df[cols['Quantidade']] = pd.to_numeric(df[cols['Quantidade']], errors='coerce')

    df = df.dropna(subset=[cols['Emissao'], cols['Cliente'], cols['Produto'], cols['Quantidade']])
    df = df[df[cols['Emissao']] >= pd.to_datetime(MIN_DATE)]
    if df.empty:
        st.error("❌ Nenhum dado após filtragem por data.")
        st.stop()

    df['AnoMes'] = df[cols['Emissao']].dt.to_period('M').dt.to_timestamp()

    grupo_col = find_column(df, 'Grupo')
    df['Grupo'] = df[grupo_col].astype(str).str.strip().str.upper() if grupo_col else 'SEM GRUPO'

    # ── SETOR (novo campo) ────────────────────────────────────────────────────
    setor_col = find_column(df, 'Setor')
    df['Setor'] = df[setor_col].astype(str).str.strip().str.upper() if setor_col else 'SEM SETOR'

    return df[['Cliente', 'Produto', 'Quantidade', 'AnoMes', 'Grupo', 'Setor']]


def make_forecast_from_series(serie):
    serie = serie.sort_index()
    n     = len(serie)

    if n < MIN_POINTS_MODEL:
        return None

    if n >= 24:
        seasonal         = 'add'
        seasonal_periods = 12
    else:
        seasonal         = None
        seasonal_periods = None

    try:
        model = ExponentialSmoothing(
            serie,
            trend='add',
            damped_trend=True,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            initialization_method='estimated'
        ).fit(optimized=True)
    except Exception:
        model = ExponentialSmoothing(
            serie, trend='add', damped_trend=True,
            seasonal=None, initialization_method='estimated'
        ).fit(optimized=True)

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

# ── Histórico de previsões ────────────────────────────────────────────────────
HISTORICO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'historico_previsoes.csv')

MESES_PT = {
    'jan': 1, 'fev': 2, 'mar': 3, 'abr': 4, 'mai': 5, 'jun': 6,
    'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12
}
MESES_PT_INV = {v: k for k, v in MESES_PT.items()}

def parse_data_pt(val):
    """Converte 'jan/26' ou '01/2026' para Timestamp."""
    val = str(val).strip().lower()
    if '/' not in val:
        return pd.NaT
    partes = val.split('/')
    mes_str, ano_str = partes[0], partes[1]
    ano = int(ano_str) + 2000 if len(ano_str) == 2 else int(ano_str)
    if mes_str in MESES_PT:
        mes = MESES_PT[mes_str]
    else:
        try:
            mes = int(mes_str)
        except ValueError:
            return pd.NaT
    return pd.Timestamp(ano, mes, 1)

def fmt_data_pt(ts):
    """Converte Timestamp para 'jan/26'."""
    return f"{MESES_PT_INV[ts.month]}/{str(ts.year)[2:]}"

def carregar_historico():
    if not os.path.exists(HISTORICO_PATH):
        return pd.DataFrame(columns=['Produto', 'AnoMes', 'Quantidade_Prevista', 'Data_Extracao'])
    df = pd.read_csv(HISTORICO_PATH, parse_dates=['AnoMes', 'Data_Extracao'])
    return df

def salvar_historico(df_hist):
    df_hist.to_csv(HISTORICO_PATH, index=False)

def snapshot_proximo_mes(df):
    """
    Gera a previsão do próximo mês para todos os produtos (todos os clientes
    consolidados) e salva no histórico. Sobrescreve se já existir entrada para
    a mesma Data_Extracao + AnoMes.
    """
    max_date  = df['AnoMes'].max()
    proximo   = max_date + pd.DateOffset(months=1)
    hoje      = pd.Timestamp.today().normalize()

    registros = []
    for produto in df['Produto'].unique():
        serie = df[df['Produto'] == produto].groupby('AnoMes')['Quantidade'].sum().sort_index()
        fc    = make_forecast_from_series(serie)
        if fc is None:
            continue
        row = fc[fc['AnoMes'] == proximo]
        if row.empty:
            continue
        qty = int(row['Quantidade'].iloc[0])
        if qty > 0:
            registros.append({
                'Produto':             produto,
                'AnoMes':              proximo,
                'Quantidade_Prevista': qty,
                'Data_Extracao':       hoje
            })

    if not registros:
        return False, "Nenhuma previsão gerada."

    df_novo = pd.DataFrame(registros)
    df_hist = carregar_historico()

    # Remove entradas com mesma data de extração e mesmo mês previsto
    if not df_hist.empty:
        mask = ~(
            (df_hist['Data_Extracao'].dt.normalize() == hoje) &
            (df_hist['AnoMes'] == proximo)
        )
        df_hist = df_hist[mask]

    df_hist = pd.concat([df_hist, df_novo], ignore_index=True)
    salvar_historico(df_hist)
    return True, f"{len(df_novo)} produtos salvos para {fmt_data_pt(proximo)}."

def importar_excel_historico(uploaded_file, data_extracao):
    """
    Importa Excel no formato do usuário: Produto | Data (jan/26) | Quantidade_Prevista
    """
    try:
        df_imp = pd.read_excel(uploaded_file)
    except Exception as e:
        return False, f"Erro ao ler arquivo: {e}"

    required = ['Produto', 'Data', 'Quantidade_Prevista']
    missing  = [c for c in required if c not in df_imp.columns]
    if missing:
        return False, f"Colunas não encontradas: {missing}"

    df_imp['AnoMes'] = df_imp['Data'].apply(parse_data_pt)
    invalidos = df_imp['AnoMes'].isna().sum()
    if invalidos > 0:
        return False, f"{invalidos} linha(s) com data inválida. Formato esperado: jan/26 ou 01/2026."

    df_imp = df_imp[['Produto', 'AnoMes', 'Quantidade_Prevista']].copy()
    df_imp['Produto']             = df_imp['Produto'].astype(str).str.strip().str.upper()
    df_imp['Quantidade_Prevista'] = pd.to_numeric(df_imp['Quantidade_Prevista'], errors='coerce').fillna(0).astype(int)
    df_imp['Data_Extracao']       = pd.Timestamp(data_extracao).normalize()

    df_hist = carregar_historico()
    dt_ext  = pd.Timestamp(data_extracao).normalize()

    # Substitui entradas da mesma data de extração
    if not df_hist.empty:
        df_hist = df_hist[df_hist['Data_Extracao'].dt.normalize() != dt_ext]

    df_hist = pd.concat([df_hist, df_imp], ignore_index=True)
    salvar_historico(df_hist)

    meses = df_imp['AnoMes'].nunique()
    prods = df_imp['Produto'].nunique()
    return True, f"{prods} produtos importados para {meses} {'mês' if meses == 1 else 'meses'} (extração: {dt_ext.strftime('%d/%m/%Y')})."


def show_auditoria_panel(df, grupo_atual, produto_atual):
    st.markdown("---")
    st.markdown("## 🎯 AUDITORIA DE PREVISÕES")

    st.info(
        "A auditoria compara o que foi **previsto e salvo** com o **realizado** da base. "
        "Previsões consolidadas por produto (todos os clientes)."
    )

    # ── Importar histórico ────────────────────────────────────────────────────
    with st.expander("📥 IMPORTAR PREVISÕES HISTÓRICAS"):
        st.markdown(
            "Carregue seus arquivos de previsões anteriores. "
            "Formato esperado: **Produto | Data (jan/26) | Quantidade_Prevista**"
        )
        col_dt, col_up = st.columns([1, 2])
        with col_dt:
            data_imp = st.date_input(
                "📅 Data da extração",
                key="imp_data",
                help="Mês em que você gerou essas previsões"
            )
        with col_up:
            arq_imp = st.file_uploader(
                "Arquivo (.xlsx)", type=["xlsx"], key="imp_arquivo"
            )
        if st.button("💾 IMPORTAR", key="btn_importar"):
            if arq_imp is None:
                st.warning("⚠️ Selecione um arquivo.")
            else:
                ok, msg = importar_excel_historico(arq_imp, data_imp)
                st.success(f"✅ {msg}") if ok else st.error(f"❌ {msg}")
                if ok:
                    st.rerun()

    st.markdown("---")

    # ── Lê histórico ─────────────────────────────────────────────────────────
    df_hist = carregar_historico()

    if df_hist.empty:
        st.info("📭 Nenhuma previsão salva. Importe seus arquivos históricos acima ou use o botão **💾 Salvar Previsão** na seção de exportação.")
        return

    # Extrações disponíveis
    extrações = sorted(df_hist['Data_Extracao'].dt.normalize().unique(), reverse=True)
    opcoes_ext = [fmt_data_pt(e) + f" ({e.strftime('%d/%m/%Y')})" for e in extrações]

    col1, col2 = st.columns(2)
    with col1:
        escolha_ext = st.selectbox(
            "📂 Extração", opcoes_ext, key="audit_extracao",
            help="Mês em que a previsão foi gerada"
        )
    idx_ext    = opcoes_ext.index(escolha_ext)
    dt_ext     = pd.Timestamp(extrações[idx_ext]).normalize()
    df_prev    = df_hist[df_hist['Data_Extracao'].dt.normalize() == dt_ext].copy()

    # Meses previstos nessa extração
    meses_prev = sorted(df_prev['AnoMes'].unique())
    opcoes_mes = [fmt_data_pt(pd.Timestamp(m)) for m in meses_prev]

    with col2:
        escolha_mes = st.selectbox(
            "📅 Mês auditado", opcoes_mes,
            index=len(opcoes_mes) - 1,
            key="audit_mes",
            help="Mês previsto que você quer comparar com o realizado"
        )

    mes_auditado = parse_data_pt(escolha_mes)
    df_prev_mes  = df_prev[df_prev['AnoMes'] == mes_auditado][['Produto', 'Quantidade_Prevista']].copy()

    if df_prev_mes.empty:
        st.warning(f"⚠️ Nenhuma previsão salva para {escolha_mes} nesta extração.")
        return

    # Aplica filtros
    if produto_atual != "TODOS":
        df_prev_mes = df_prev_mes[df_prev_mes['Produto'] == produto_atual]
    if grupo_atual != "TODOS":
        prods_grupo = df[df['Grupo'] == grupo_atual]['Produto'].unique()
        df_prev_mes = df_prev_mes[df_prev_mes['Produto'].isin(prods_grupo)]

    # Realizado consolidado do mês auditado
    df_real_base = df[df['AnoMes'] == mes_auditado]
    if grupo_atual != "TODOS":
        df_real_base = df_real_base[df_real_base['Grupo'] == grupo_atual]
    if produto_atual != "TODOS":
        df_real_base = df_real_base[df_real_base['Produto'] == produto_atual]

    df_real = (
        df_real_base.groupby('Produto')['Quantidade']
        .sum().reset_index()
        .rename(columns={'Quantidade': 'Realizado'})
    )

    df_comp = df_prev_mes.merge(df_real, on='Produto', how='inner')

    if df_comp.empty:
        st.warning(
            f"⚠️ Nenhum produto com previsão salva E realizado disponível para {escolha_mes}. "
            f"Verifique se o mês já fechou na base de vendas."
        )
        return

    # ── Métricas ──────────────────────────────────────────────────────────────
    mask  = df_comp['Realizado'] > 0
    mape  = np.mean(np.abs(
        (df_comp.loc[mask, 'Realizado'] - df_comp.loc[mask, 'Quantidade_Prevista'])
        / df_comp.loc[mask, 'Realizado']
    )) * 100
    mae   = np.mean(np.abs(df_comp['Realizado'] - df_comp['Quantidade_Prevista']))
    rmse  = np.sqrt(np.mean((df_comp['Realizado'] - df_comp['Quantidade_Prevista']) ** 2))
    classificacao = "🟢 ÓTIMA" if mape <= 15 else "🟡 REGULAR" if mape <= 30 else "🔴 BAIXA"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAPE",      f"{mape:.1f}%", help="Erro percentual médio real")
    c2.metric("MAE",       f"{mae:.0f}",   help="Erro absoluto médio em unidades")
    c3.metric("RMSE",      f"{rmse:.0f}",  help="Raiz do erro quadrático médio")
    c4.metric("QUALIDADE", classificacao)
    st.caption(
        f"📐 Extração de **{dt_ext.strftime('%d/%m/%Y')}** | "
        f"**{df_comp['Produto'].nunique()} produtos** auditados em **{escolha_mes}**."
    )

    # ── Gráfico ───────────────────────────────────────────────────────────────
    df_plot = df_comp.sort_values('Realizado', ascending=False).head(20)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Previsto', x=df_plot['Produto'], y=df_plot['Quantidade_Prevista'],
        marker_color='#ea580c', opacity=0.85
    ))
    fig.add_trace(go.Bar(
        name='Realizado', x=df_plot['Produto'], y=df_plot['Realizado'],
        marker_color='#1d4ed8'
    ))
    fig.update_layout(
        title=f"PREVISTO vs REALIZADO — {escolha_mes.upper()} | TOP 20 PRODUTOS",
        title_x=0.5, barmode='group', hovermode='x unified',
        xaxis=dict(title='<b>PRODUTO</b>', title_font=dict(color='#111827'),
                   tickfont=dict(color='#111827'), tickangle=-35),
        yaxis=dict(title='<b>QUANTIDADE</b>', title_font=dict(color='#111827'),
                   tickfont=dict(color='#111827')),
        hoverlabel=dict(bgcolor='#1e293b', bordercolor='#334155', font=dict(color='#f8fafc', size=13)),
        legend=dict(orientation='h', y=-0.3, font=dict(color='#111827'))
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Tabela ────────────────────────────────────────────────────────────────
    with st.expander("📋 TABELA COMPLETA POR PRODUTO"):
        df_tab = df_comp.copy()
        df_tab['Erro Abs'] = (df_tab['Realizado'] - df_tab['Quantidade_Prevista']).abs()
        df_tab['Erro (%)'] = np.where(
            df_tab['Realizado'] > 0,
            ((df_tab['Realizado'] - df_tab['Quantidade_Prevista']).abs()
             / df_tab['Realizado'] * 100).round(1),
            np.nan
        )
        df_tab = df_tab[['Produto', 'Realizado', 'Quantidade_Prevista', 'Erro Abs', 'Erro (%)']].sort_values('Erro (%)')
        st.dataframe(df_tab.set_index('Produto'), use_container_width=True)


def create_plot(df, title):
    try:
        fig = px.line(
            df, x='AnoMes', y='Quantidade', color='Previsao',
            title=title.upper(), markers=True,
            labels={'AnoMes': 'MÊS', 'Quantidade': 'QUANTIDADE', 'Previsao': 'TIPO'}
        )
        fig.for_each_trace(
            lambda t: t.update(line=dict(color='#1d4ed8', width=2.5))
            if t.name == 'HISTÓRICO'
            else t.update(line=dict(color='#ea580c', width=2.5, dash='dash'))
        )
        fig.update_layout(
            title_x=0.5, hovermode='x unified',
            xaxis=dict(title='<b>MÊS</b>',        title_font=dict(color='#111827'), tickfont=dict(color='#111827')),
            yaxis=dict(title='<b>QUANTIDADE</b>',  title_font=dict(color='#111827'), tickfont=dict(color='#111827')),
            hoverlabel=dict(bgcolor='#1e293b', bordercolor='#334155', font=dict(color='#f8fafc', size=13)),
            legend=dict(font=dict(color='#111827'))
        )
        return fig
    except Exception as e:
        st.error(f"❌ Erro ao criar gráfico: {e}")
        return None

def create_bar_chart(df, setor_atual, grupo_atual, cliente_atual, produto_atual):
    try:
        dfs = df if setor_atual == "TODOS" else df[df['Setor'] == setor_atual]
        dfg = dfs if grupo_atual == "TODOS" else dfs[dfs['Grupo'] == grupo_atual]
        dfc = dfg if cliente_atual == "TODOS" else dfg[dfg['Cliente'] == cliente_atual]
        df_filtered = dfc if produto_atual == "TODOS" else dfc[dfc['Produto'] == produto_atual]

        if df_filtered.empty:
            return None

        if cliente_atual != "TODOS" and produto_atual == "TODOS":
            grouped = df_filtered.groupby('Produto')['Quantidade'].sum().reset_index()
            titulo, x_label = f"PRODUTOS MAIS VENDIDOS - {cliente_atual}", "PRODUTO"
        elif grupo_atual != "TODOS" and cliente_atual == "TODOS" and produto_atual == "TODOS":
            grouped = df_filtered.groupby('Cliente')['Quantidade'].sum().reset_index()
            titulo, x_label = f"CLIENTES QUE MAIS COMPRARAM - LINHA {grupo_atual}", "CLIENTE"
        elif produto_atual != "TODOS" and cliente_atual == "TODOS":
            grouped = df_filtered.groupby('Cliente')['Quantidade'].sum().reset_index()
            titulo, x_label = f"CLIENTES QUE MAIS COMPRARAM - {produto_atual}", "CLIENTE"
        elif cliente_atual == "TODOS" and produto_atual == "TODOS" and grupo_atual == "TODOS" and setor_atual == "TODOS":
            grouped = df_filtered.groupby('Grupo')['Quantidade'].sum().reset_index()
            titulo, x_label = "LINHAS QUE MAIS VENDEM", "LINHA"
        elif setor_atual != "TODOS" and grupo_atual == "TODOS" and cliente_atual == "TODOS" and produto_atual == "TODOS":
            grouped = df_filtered.groupby('Grupo')['Quantidade'].sum().reset_index()
            titulo, x_label = f"LINHAS QUE MAIS VENDEM - SETOR {setor_atual}", "LINHA"
        else:
            grouped = df_filtered.groupby('AnoMes')['Quantidade'].sum().reset_index()
            grouped['Mes_Ano'] = grouped['AnoMes'].dt.strftime('%m/%Y')
            grouped = grouped.rename(columns={'Mes_Ano': 'Produto'})
            titulo, x_label = f"VENDAS MENSAIS - {cliente_atual} - {produto_atual}", "MÊS"

        grouped = grouped.sort_values('Quantidade', ascending=True)
        if len(grouped) > 20:
            grouped = grouped.tail(20)
        grouped['Label'] = grouped.iloc[:, 0]

        fig = px.bar(
            grouped, x='Quantidade', y='Label', orientation='h',
            title=titulo.upper(),
            labels={'Quantidade': 'QUANTIDADE VENDIDA', 'Label': x_label},
            color='Quantidade', color_continuous_scale='Blues'
        )
        fig.update_layout(
            title_x=0.5, height=max(400, len(grouped) * 25),
            xaxis=dict(title='<b>QUANTIDADE VENDIDA</b>', title_font=dict(color='#111827'), tickfont=dict(color='#111827')),
            yaxis=dict(title=f'<b>{x_label}</b>',         title_font=dict(color='#111827'), tickfont=dict(color='#111827')),
            hoverlabel=dict(bgcolor='#1e293b', bordercolor='#334155', font=dict(color='#f8fafc', size=13)),
            showlegend=False
        )
        fig.update_traces(texttemplate='%{x:,.0f}', textposition='outside')
        return fig
    except Exception as e:
        st.error(f"❌ Erro ao criar gráfico de barras: {e}")
        return None

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

def show_export_section(df, setor_atual, grupo_atual, cliente_atual, produto_atual):
    st.markdown("---")
    st.markdown("## 📋 EXPORTAÇÃO DE PREVISÕES POR PRODUTO")
    st.info(
        f"📊 **Filtros Aplicados:** Setor: {setor_atual} | "
        f"Linha: {grupo_atual} | Cliente: {cliente_atual} | Produto: {produto_atual}"
    )

    dfs = df if setor_atual == "TODOS" else df[df['Setor'] == setor_atual]
    dfg = dfs if grupo_atual == "TODOS" else dfs[dfs['Grupo'] == grupo_atual]
    dfc = dfg if cliente_atual == "TODOS" else dfg[dfg['Cliente'] == cliente_atual]
    df_filtered = dfc if produto_atual == "TODOS" else dfc[dfc['Produto'] == produto_atual]

    if df_filtered.empty:
        st.warning("⚠️ Nenhum dado disponível com os filtros aplicados.")
        return

    all_forecasts = create_all_forecasts_table(df_filtered)
    if all_forecasts.empty:
        st.warning("⚠️ Nenhuma previsão disponível (dados insuficientes para os produtos filtrados).")
        return

    datas_disponiveis = sorted(
        all_forecasts['Data'].unique(),
        key=lambda d: pd.to_datetime(d, format='%m/%Y')
    )

    st.markdown("### 📅 PERÍODO DE EXPORTAÇÃO")
    col_de, col_ate = st.columns(2)
    with col_de:
        data_inicio = st.selectbox("🗓️ DE",  datas_disponiveis, index=0, key="export_data_inicio")
    with col_ate:
        opcoes_ate = [d for d in datas_disponiveis
                      if pd.to_datetime(d, format='%m/%Y') >= pd.to_datetime(data_inicio, format='%m/%Y')]
        data_fim   = st.selectbox("🗓️ ATÉ", opcoes_ate, index=len(opcoes_ate)-1, key="export_data_fim")

    dt_inicio = pd.to_datetime(data_inicio, format='%m/%Y')
    dt_fim    = pd.to_datetime(data_fim,    format='%m/%Y')
    df_export = all_forecasts[(all_forecasts['AnoMes'] >= dt_inicio) & (all_forecasts['AnoMes'] <= dt_fim)]

    if df_export.empty:
        st.warning("⚠️ Nenhuma previsão no intervalo selecionado.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("🎯 PRODUTOS",        len(df_export['Produto'].unique()))
    col2.metric("📅 MESES",           len(df_export['Data'].unique()))
    col3.metric("📊 TOTAL PREVISÕES", len(df_export))

    if setor_atual != "TODOS" and grupo_atual == "TODOS" and cliente_atual == "TODOS" and produto_atual == "TODOS":
        suffix = f"setor_{setor_atual.replace(' ', '_')}"
    elif grupo_atual != "TODOS" and cliente_atual == "TODOS" and produto_atual == "TODOS":
        suffix = f"grupo_{grupo_atual.replace(' ', '_')}"
    elif cliente_atual != "TODOS" and produto_atual == "TODOS":
        suffix = f"cliente_{cliente_atual.replace(' ', '_')}"
    elif cliente_atual == "TODOS" and produto_atual != "TODOS":
        suffix = f"produto_{produto_atual.replace(' ', '_')}"
    elif cliente_atual != "TODOS" and produto_atual != "TODOS":
        suffix = f"{cliente_atual.replace(' ', '_')}_{produto_atual.replace(' ', '_')}"
    else:
        suffix = "todos"

    periodo  = f"{data_inicio.replace('/', '-')}_a_{data_fim.replace('/', '-')}"
    filename = f"previsoes_{suffix}_{periodo}.xlsx"

    df_ordenado = (
        df_export[['Produto', 'Data', 'Quantidade_Prevista']]
        .sort_values(['Data', 'Quantidade_Prevista'], ascending=[True, False])
    )
    st.download_button(
        label="📥 BAIXAR PREVISÕES", data=to_excel_single(df_ordenado),
        file_name=filename, type="primary",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    with st.expander("👀 PREVIEW DOS DADOS"):
        st.dataframe(df_ordenado, use_container_width=True)

def show_dashboard():
    st.set_page_config(page_title="PAINEL DE VENDAS", layout="wide")

    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("📊 PAINEL DE VENDAS E PREVISÃO")
    with col2:
        st.markdown("### 👤 Usuário: comercial")
        if st.button("🚪 SAIR", type="secondary", key="logout_btn"):
            logout()

    @st.cache_data
    def get_data():
        return load_data()

    df = get_data()
    if not validate_data(df, ['Cliente', 'Produto', 'Quantidade', 'AnoMes', 'Grupo', 'Setor']):
        st.stop()

    st.markdown("## 📈 ANÁLISE GRÁFICA")

    # ── Inicializa session_state (inclui setor) ───────────────────────────────
    for k, v in [
        ('setor_selecionado',   'TODOS'),
        ('grupo_selecionado',   'TODOS'),
        ('cliente_selecionado', 'TODOS'),
        ('produto_selecionado', 'TODOS'),
    ]:
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Filtro 1: SETOR ───────────────────────────────────────────────────────
    setores = ["TODOS"] + sorted(df['Setor'].unique())
    setor   = st.selectbox(
        "SELECIONE O SETOR", setores,
        index=setores.index(st.session_state.setor_selecionado)
        if st.session_state.setor_selecionado in setores else 0,
        key="setor_select"
    )
    st.session_state.setor_selecionado = setor

    # ── Filtro 2: GRUPO (cascata de setor) ────────────────────────────────────
    dfs    = df if setor == "TODOS" else df[df['Setor'] == setor]
    grupos = ["TODOS"] + sorted(dfs['Grupo'].unique())
    if st.session_state.grupo_selecionado not in grupos:
        st.session_state.grupo_selecionado = "TODOS"
    grupo  = st.selectbox(
        "SELECIONE A LINHA", grupos,
        index=grupos.index(st.session_state.grupo_selecionado),
        key="grupo_select"
    )
    st.session_state.grupo_selecionado = grupo

    # ── Filtro 3: CLIENTE (cascata de setor + grupo) ──────────────────────────
    dfg      = dfs if grupo == "TODOS" else dfs[dfs['Grupo'] == grupo]
    clientes = ["TODOS"] + sorted(dfg['Cliente'].unique())
    if st.session_state.cliente_selecionado not in clientes:
        st.session_state.cliente_selecionado = "TODOS"
    cliente  = st.selectbox(
        "SELECIONE O CLIENTE", clientes,
        index=clientes.index(st.session_state.cliente_selecionado),
        key="cliente_select"
    )
    st.session_state.cliente_selecionado = cliente

    # ── Filtro 4: PRODUTO (cascata de setor + grupo + cliente) ───────────────
    dfc      = dfg if cliente == "TODOS" else dfg[dfg['Cliente'] == cliente]
    produtos = ["TODOS"] + sorted(dfc['Produto'].unique())
    if st.session_state.produto_selecionado not in produtos:
        st.session_state.produto_selecionado = "TODOS"
    produto  = st.selectbox(
        "SELECIONE O PRODUTO", produtos,
        index=produtos.index(st.session_state.produto_selecionado),
        key="produto_select"
    )
    st.session_state.produto_selecionado = produto

    # ── Dados filtrados ───────────────────────────────────────────────────────
    dff = dfc if produto == "TODOS" else dfc[dfc['Produto'] == produto]
    if dff.empty:
        st.warning("⚠️ Nenhum dado com os filtros aplicados.")
        return

    grouped = dff.groupby('AnoMes', as_index=False)['Quantidade'].sum()
    grouped['Previsao'] = 'HISTÓRICO'
    serie   = grouped.set_index('AnoMes')['Quantidade'].sort_index()
    n       = len(serie)

    # ── Título do gráfico ─────────────────────────────────────────────────────
    if setor != "TODOS" and grupo == "TODOS" and cliente == "TODOS" and produto == "TODOS":
        titulo = f"SETOR {setor} - CONSOLIDADO"
    elif grupo != "TODOS" and cliente == "TODOS" and produto == "TODOS":
        titulo = f"GRUPO {grupo} - CONSOLIDADO"
    elif cliente != "TODOS" and produto == "TODOS":
        titulo = f"{cliente} - TODOS OS PRODUTOS"
    elif cliente == "TODOS" and produto != "TODOS":
        titulo = f"TODOS OS CLIENTES - {produto}"
    elif cliente != "TODOS" and produto != "TODOS":
        titulo = f"{cliente} - {produto}"
    else:
        titulo = "PREVISÃO TOTAL"

    if n >= 24:
        st.success(f"✅ Modelo com **sazonalidade ativada** ({n} meses de dados disponíveis)")
    elif n >= MIN_POINTS_MODEL:
        st.info(f"ℹ️ Modelo **sem sazonalidade** (necessário 24 meses; disponível: {n})")
    else:
        st.warning(f"⚠️ Apenas {n} meses de dados — mínimo para previsão é {MIN_POINTS_MODEL}")

    fc = make_forecast_from_series(serie)

    if fc is None:
        st.warning("⚠️ Não foi possível gerar previsão com os dados disponíveis.")
        fig = create_plot(grouped, titulo)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    else:
        resultado = pd.concat([grouped, fc], ignore_index=True)
        st.markdown(f"### 📌 {titulo}")
        fig = create_plot(resultado, titulo)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("## 📊 ANÁLISE DE VENDAS POR RANKING")
    bar_fig = create_bar_chart(df, setor, grupo, cliente, produto)
    if bar_fig:
        st.plotly_chart(bar_fig, use_container_width=True)
    else:
        st.warning("⚠️ Não foi possível gerar o gráfico de barras com os filtros aplicados.")

    st.divider()

    with st.expander("📈 ESTATÍSTICAS DETALHADAS", expanded=True):
        historico = grouped['Quantidade']
        st.subheader("📊 HISTÓRICO")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total",         f"{historico.sum():,.0f}")
        c2.metric("Média",         f"{historico.mean():.2f}")
        c3.metric("Mediana",       f"{historico.median():.0f}")
        c4.metric("Desvio Padrão", f"{historico.std():.2f}")

        if fc is not None:
            st.markdown("")
            st.subheader("📈 PREVISÃO")
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Total Previsto",   f"{fc['Quantidade'].sum():,.0f}")
            c6.metric("Média Prevista",   f"{fc['Quantidade'].mean():.2f}")
            c7.metric("Mediana Prevista", f"{fc['Quantidade'].median():.0f}")
            c8.metric("Desvio Padrão",    f"{fc['Quantidade'].std():.2f}")
            st.caption("ℹ️ Sazonalidade ativada automaticamente com 24+ meses de dados.")

    show_auditoria_panel(df, grupo, produto)
    show_export_section(df, setor, grupo, cliente, produto)

def main():
    if not check_authentication():
        return
    show_dashboard()

if __name__ == "__main__":
    main()