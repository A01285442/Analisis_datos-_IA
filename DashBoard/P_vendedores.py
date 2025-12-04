import streamlit as st
import pandas as pd
import numpy as np
import re

from utils.theme import get_theme_colors
from utils.css_manager import apply_css
from components.header import create_page_header
from utils.icons import get_icon

# =========================================================
# UTILIDADES
# =========================================================

def clean_currency(x):
    if isinstance(x, str):
        clean_str = x.replace('$', '').replace(',', '').strip()
        try:
            return float(clean_str)
        except ValueError:
            return 0.0
    return x


@st.cache_data
def load_data(file_path):
    df = None
    encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            sample_str = str(df.columns) + str(df.head(5))
            if "Ã" in sample_str:
                raise ValueError("Bad encoding")
            break
        except Exception:
            continue
    
    if df is None:
        st.error(f"❌ Error leyendo '{file_path}'")
        return None

    df.columns = df.columns.str.strip()

    cols_to_rename = {}
    for col in df.columns:
        if col.startswith("Regi") or col.startswith("Sucu"):
            cols_to_rename[col] = "Sucursal"
    
    if cols_to_rename:
        df.rename(columns=cols_to_rename, inplace=True)

    cols_saldo = [c for c in df.columns if "Saldo Insoluto" in c]
    for col in cols_saldo:
        df[col] = df[col].apply(clean_currency)

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


# =========================================================
# RENDER PRINCIPAL (ANTES TODO ESTABA SUELTO)
# =========================================================

def render():
    # CSS específico de vendedores
    apply_css("vendedores")
    st.set_page_config(page_title="Métricas de Vendedores", page_icon=get_icon("metricas_principales_gris"), layout="wide")
    # Header
    header_icon = get_icon("metricas_principales")
    create_page_header(
        f"{header_icon} Métricas de Vendedores",
        "Tabla de rendimiento individual y tendencias históricas."
    )

    # ----------------------------
    # CARGA DE DATOS
    # ----------------------------

    if 'df_vendedores' not in st.session_state or st.session_state['df'] is None:
        with st.spinner("Procesando datos..."):
            df_raw = load_data('Reto_limpio.csv')
            
            if df_raw is not None:
                cols_present = [c for c in df_raw.columns if re.search(r"Saldo Insoluto T-\d+", c)]

                if cols_present:
                    df_raw["Promedio_Hist_12m"] = df_raw[cols_present].mean(axis=1)
                else:
                    df_raw["Promedio_Hist_12m"] = 0.0

                df_raw["Variacion_Pct"] = np.where(
                    df_raw["Promedio_Hist_12m"] > 0,
                    ((df_raw["Saldo Insoluto Actual"] - df_raw["Promedio_Hist_12m"]) / df_raw["Promedio_Hist_12m"]),
                    0.0
                )
                
                st.session_state['df_vendedores'] = df_raw
            else:
                st.stop()

    df = st.session_state['df_vendedores']

    if df['Vendedor'].astype(str).str.contains("Ã").any():
        st.cache_data.clear()
        del st.session_state["df_vendedores"]
        st.rerun()

    if "Sucursal" not in df.columns:
        st.error("⚠️ Error: No se encontró la columna 'Sucursal'.")
        st.stop()

    # ----------------------------
    # KPIs
    # ----------------------------
    col1, col2, col3 = st.columns(3)
    with col1:
        saldo_total = df["Saldo Insoluto Actual"].sum()
        st.markdown(create_metric_card("Saldo Total Cartera", f"${saldo_total:,.2f}"), unsafe_allow_html=True)

    with col2:
        if not df.empty:
            top_v = df.loc[df["Saldo Insoluto Actual"].idxmax()]
            st.markdown(create_metric_card("Top Vendedor", top_v["Vendedor"], f"${top_v['Saldo Insoluto Actual']:,.0f}"), unsafe_allow_html=True)

    with col3:
        crec_prom = df["Variacion_Pct"].mean() * 100
        st.markdown(create_metric_card("Tendencia Promedio", f"{crec_prom:+.2f}%", "vs Año Anterior", crec_prom > 0), unsafe_allow_html=True)

    st.markdown("---")

    # ----------------------------
    # TABLA DETALLADA
    # ----------------------------
    st.markdown(f"### {get_icon('listado_de_vendedores')} Listado de Vendedores", unsafe_allow_html=True)

    max_val = df["Saldo Insoluto Actual"].max()
    if pd.isna(max_val) or max_val <= 0:
        max_val = 1

    cols_to_show = ["Sucursal", "Vendedor", "Saldo Insoluto Actual", "Promedio_Hist_12m", "Variacion_Pct"]
    cols_existing = [c for c in cols_to_show if c in df.columns]

    st.dataframe(
        df[cols_existing].sort_values("Saldo Insoluto Actual", ascending=False),
        use_container_width=True,
        column_config={
            "Sucursal": st.column_config.TextColumn("Sucursal"),
            "Vendedor": st.column_config.TextColumn("Nombre Vendedor"),
            "Saldo Insoluto Actual": st.column_config.ProgressColumn(
                "Saldo Actual", format="$%.2f", min_value=0, max_value=float(max_val)
            ),
            "Promedio_Hist_12m": st.column_config.NumberColumn("Promedio (12 Meses)", format="$%.2f"),
            "Variacion_Pct": st.column_config.NumberColumn("Tendencia", format="%.2f %%"),
        },
        hide_index=True,
        height=700
    )
