import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.theme import get_theme_colors
from utils.css_manager import apply_css
from components.header import create_page_header
from utils.icons import get_icon


# =============================================================================
# UTILIDADES DE DATOS
# =============================================================================

def clean_currency(x):
    """Limpia strings de moneda a float"""
    if isinstance(x, str):
        clean_str = x.replace('$', '').replace(',', '').strip()
        try:
            return float(clean_str)
        except ValueError:
            return 0.0
    return x


@st.cache_data
def load_sucursales_data(file_path):
    df = None
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            break
        except:
            continue

    if df is None:
        st.error("❌ Error al cargar el archivo. Verifica el formato CSV.")
        return None

    df.columns = df.columns.str.replace('\n', '').str.replace(' ', '').str.strip()

    rename_map = {
        'SaldoInsolutoActual': 'Saldo_Actual',
        'SaldoInsolutoVencidoActual': 'Saldo_Vencido',
        'Region': 'Región'
    }
    df.rename(columns=rename_map, inplace=True)

    numeric_cols = ['Saldo_Actual', 'Saldo_Vencido', 'CapitalDispersadoActual', 'CastigosActual']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_currency)

    avg_saldo = df['Saldo_Actual'].mean()

    df['Promedio_Cartera'] = avg_saldo
    df['Performance_Vs_Avg'] = np.where(
        avg_saldo > 0,
        (df['Saldo_Actual'] - avg_saldo) / avg_saldo,
        0.0
    )

    df['ICV'] = np.where(
        df['Saldo_Actual'] > 0,
        df['Saldo_Vencido'] / df['Saldo_Actual'],
        0.0
    )

    return df


def create_metric_card(label, value, delta=None, delta_positive=True):
    colors = get_theme_colors()
    delta_color = colors['success'] if delta_positive else colors['error']
    delta_html = f'<div style="font-size:0.85rem; color:{delta_color}; margin-top:6px;">{delta}</div>' if delta else ''

    return f"""
        <div class="metric-card" style="padding:12px;">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
    """


# =============================================================================
#     RENDER PRINCIPAL (SE LLAMA DESDE main.py)
# =============================================================================
def render():

    apply_css("sucursales")
    st.set_page_config(page_title="Tablero de Sucursales", page_icon=get_icon("tablero_de_sucursales_gris"), layout="wide")
    header_icon = get_icon("tablero_de_sucursales")
    create_page_header(
        f"{header_icon} Tablero de Sucursales",
        "Análisis de saldos, riesgo y desempeño operativo."
    )

    if 'df_sucursales_' not in st.session_state:
        with st.spinner("Cargando datos de sucursales..."):
            df = load_sucursales_data('./DashBoard/Base_de_datos_Dimex.csv')
            st.session_state['df_sucursales_'] = df

    df = st.session_state['df_sucursales_']

    if df is None:
        st.info("Esperando carga de datos…")
        return

    colors = get_theme_colors()

    # -------------------- KPIs --------------------
    col1, col2, col3, col4 = st.columns(4)

    total_saldo = df['Saldo_Actual'].sum()
    total_vencido = df['Saldo_Vencido'].sum()
    imor_global = (total_vencido / total_saldo * 100) if total_saldo > 0 else 0
    top_sucursal = df.loc[df['Saldo_Actual'].idxmax()]

    with col1:
        st.markdown(create_metric_card("Saldo Total Cartera", f"${total_saldo:,.0f}"), unsafe_allow_html=True)
    with col2:
        st.markdown(create_metric_card("Saldo Vencido Total", f"${total_vencido:,.0f}", f"{imor_global:.2f}% IMOR", False), unsafe_allow_html=True)
    with col3:
        st.markdown(create_metric_card("Sucursal Top", top_sucursal['Sucursal'], f"${top_sucursal['Saldo_Actual']:,.0f}", True), unsafe_allow_html=True)
    with col4:
        st.markdown(create_metric_card("Total Sucursales", f"{len(df)}", "Activas"), unsafe_allow_html=True)

    st.markdown("---")

    # -------------------- FILTRO --------------------
    regiones = ['Todas'] + sorted(df['Región'].unique().tolist())
    selected_region = st.selectbox("Filtrar por Región:", regiones)

    df_view = df[df['Región'] == selected_region] if selected_region != 'Todas' else df.copy()

    st.markdown(f"### {get_icon('detalle_de_sucursales')} Detalle de Sucursales", unsafe_allow_html=True)

    max_saldo = df['Saldo_Actual'].max()

    st.dataframe(
        df_view[['Región', 'Sucursal', 'Saldo_Actual', 'Saldo_Vencido', 'ICV', 'Performance_Vs_Avg']]
        .sort_values('Saldo_Actual', ascending=False),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # -------------------- GRÁFICO --------------------
    st.markdown(f"### {get_icon('distribucion_por_region')} Distribución por Región", unsafe_allow_html=True)

    df_grouped = df.groupby('Región')[['Saldo_Actual', 'Saldo_Vencido']].sum().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_grouped['Región'],
        y=df_grouped['Saldo_Actual'],
        name='Saldo Vigente',
        marker_color=colors['accent']
    ))
    fig.add_trace(go.Bar(
        x=df_grouped['Región'],
        y=df_grouped['Saldo_Vencido'],
        name='Saldo Vencido',
        marker_color=colors['error']
    ))

    fig.update_layout(
        barmode='stack',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=colors['text_primary'])
    )

    st.plotly_chart(fig, use_container_width=True)
