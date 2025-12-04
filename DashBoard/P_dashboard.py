import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Importaciones del Template
from utils.theme import get_theme_colors
from utils.css_manager import apply_css
from components.header import create_page_header
from utils.icons import get_icon

# =============================================================================
# CONFIG PAGE (keep at top level)
# =============================================================================
st.set_page_config(
    page_title="Dashboard Financiero Estratégico",
    page_icon=get_icon("exploracion_de_datos_gris"),
    layout="wide"
)

# =============================================================================
# HELPERS / DATA LOAD
# =============================================================================
def init_memory():
    if 'kpi_memory' not in st.session_state:
        st.session_state['kpi_memory'] = {}

def get_kpi_config(kpi_id):
    if kpi_id not in st.session_state['kpi_memory']:
        st.session_state['kpi_memory'][kpi_id] = {
            "breakdown": "Global",
            "chart_type": "Distribución (Hist)"
        }
    return st.session_state['kpi_memory'][kpi_id]

def update_kpi_setting():
    for key in st.session_state:
        if key.startswith("setting_"):
            parts = key.split("_")
            if len(parts) >= 3:
                setting_type = parts[1]
                kpi_id = parts[2]
                if kpi_id in st.session_state.get('kpi_memory', {}):
                    st.session_state['kpi_memory'][kpi_id][setting_type] = st.session_state[key]

def format_big_number(num):
    if num is None: return "$0.00"
    if abs(num) >= 1_000_000: return f"${num/1_000_000:.2f}M"
    elif abs(num) >= 1_000: return f"${num/1_000:.2f}k"
    return f"${num:,.2f}"

def format_percent(num):
    if num is None: return "0.00%"
    return f"{num:.2f}%"

def fix_text_encoding(text):
    if not isinstance(text, str): return text
    text = text.strip()
    replacements = {
        'RegiÃ³n': 'Region', 'Región': 'Region', 'Regi?n': 'Region',
        'Ã¡': 'á', 'Ã©': 'é', 'Ãed': 'í', 'Ã³': 'ó', 'Ãº': 'ú', 'Ã±': 'ñ', 'Â': ''
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text

def clean_column_name(col_name):
    return col_name.replace('\n', '').replace(' ', '').replace('.', '').replace('%', '').strip()

@st.cache_data
def load_data(file_path='Base_Con_NA_Historico.csv'):
    try:
        df = pd.read_csv(file_path, encoding='latin-1')
    except:
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except Exception as e:
            st.error(f"❌ Error crítico: {e}")
            return pd.DataFrame()

    df.columns = [clean_column_name(c) for c in df.columns]

    col_sucursal = next((c for c in df.columns if 'sucursal' in c.lower()), 'Sucursal')
    col_region = next((c for c in df.columns if 'region' in c.lower() or 'regi' in c.lower()), 'Region')
    df.rename(columns={col_sucursal: 'Sucursal', col_region: 'Region'}, inplace=True)

    for col in ['Sucursal', 'Region']:
        if col in df.columns:
            df[col] = df[col].apply(fix_text_encoding)

    # normalize numeric columns
    for col in df.columns:
        if col not in ['Sucursal', 'Region']:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '').str.replace(' ', '')
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # helper to find columns by keywords
    def find_col(keywords):
        for c in df.columns:
            if all(k in c.lower() for k in keywords): return c
        return None

    c_insoluto = find_col(['saldoinsoluto', 'actual'])
    if c_insoluto and 'vencido' in c_insoluto.lower():
        c_insoluto = None
    if not c_insoluto:
        c_insoluto = next((c for c in df.columns if 'saldoinsoluto' in c.lower() and 'actual' in c.lower() and 'vencido' not in c.lower()), None)

    c_vencido = find_col(['vencido', 'actual'])
    c_fpd = find_col(['fpd', 'actual'])
    c_dispersado = find_col(['dispersado', 'actual'])
    c_3089 = find_col(['3089', 'actual'])

    c_quitas = find_col(['quita', 'actual'])
    c_castigos = find_col(['castigo', 'actual'])
    c_liquidado = find_col(['liquidado', 'actual'])

    df['Saldo_Calc'] = df[c_insoluto] if c_insoluto else 0
    df['Dispersado_Calc'] = df[c_dispersado] if c_dispersado else 0
    df['Perdidas_Calc'] = (df[c_quitas] if c_quitas else 0) + (df[c_castigos] if c_castigos else 0) + (df[c_liquidado] if c_liquidado else 0)

    if c_insoluto and c_vencido:
        df['ICV_Calc'] = np.where(df['Saldo_Calc'] > 0, (df[c_vencido] / df['Saldo_Calc']) * 100, 0)
    else:
        df['ICV_Calc'] = 0

    if c_insoluto and c_3089:
        df['Ratio_30_89_Calc'] = np.where(df['Saldo_Calc'] > 0, (df[c_3089] / df['Saldo_Calc']) * 100, 0)
    else:
        df['Ratio_30_89_Calc'] = 0

    if c_fpd:
        if df[c_fpd].mean() < 1: df['FPD_Calc'] = df[c_fpd] * 100
        else: df['FPD_Calc'] = df[c_fpd]
    else:
        df['FPD_Calc'] = 0

    def clasificar_riesgo(row):
        score = sum([row['ICV_Calc'] > 5.0, row['Ratio_30_89_Calc'] > 3.0, row['FPD_Calc'] > 6.0])
        if score == 3: return "Riesgo Alto"
        elif score == 2: return "Riesgo Medio"
        return "Saludable"

    df['Nivel_Riesgo'] = df.apply(clasificar_riesgo, axis=1)
    return df

def get_trend_series(df, kpi_type, limit=24):
    root_map = {'Saldo': 'SaldoInsoluto', 'ICV': 'SaldoInsolutoVencido', 'FPD': 'FPD', 'Dispersado': 'CapitalDispersado', 'Perdidas': 'Castigos'}
    root = root_map.get(kpi_type, 'SaldoInsoluto')
    cols = []
    labels = []
    for i in range(limit, 0, -1):
        patterns = [f"{root}T{i}", f"{root}T{i:02d}", f"{root}T-{i}"]
        match = next((c for c in df.columns for p in patterns if p.lower() == c.lower()), None)
        if match:
            cols.append(match)
            labels.append(f"T-{i}")
    col_act = next((c for c in df.columns if root.lower() in c.lower() and 'actual' in c.lower() and ('vencido' not in c.lower() if root=='SaldoInsoluto' else True)), None)
    if col_act:
        cols.append(col_act)
        labels.append("Actual")
    return cols, labels

# =============================================================================
# RENDER FUNCTION
# =============================================================================
def render(df=None, colors=None):
    """
    Main render function. If df is None, it will load data from default path.
    If colors is None, it will compute them via get_theme_colors().
    """
    init_memory()

    # load data if needed
    if df is None:
        if 'df_main' not in st.session_state:
            st.session_state['df_main'] = load_data()
        df = st.session_state['df_main']

    if colors is None:
        colors = get_theme_colors()

    # basic UI setup
    apply_css("dashboard")

    header_icon = get_icon("exploracion_de_datos")
    create_page_header(
        f"{header_icon} Tablero Financiero Estratégico",
        "Análisis Histórico y Distribución de Riesgo."
    )

    # consolidated CSS (kept minimal and not duplicated)
    st.markdown(f"""
    <style>
    .filter-label {{
        display: flex;
        align-items: center;
        gap: 0.4rem;
        font-weight: 600;
        color: {colors['text_primary']};
        margin-bottom: 0.25rem;
        font-size: 0.95rem;
    }}
    .filter-label svg {{ width: 18px; height: 18px; }}
    div.stButton > button {{ border-radius:12px !important; }}
    .filter-container {{ background-color: {colors['bg_secondary']}; padding: 1rem; border-radius:10px; border:1px solid {colors['border']}; margin-bottom:1.5rem; }}
    </style>
    """, unsafe_allow_html=True)

    if df is None or df.empty:
        st.warning("No hay datos para mostrar.")
        return
    # --- FILTERS ---
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        counts_region = df['Region'].value_counts().to_dict()
        def fmt_reg(opt): return f"Todas ({len(df)})" if opt == "Todas" else f"{opt} ({counts_region.get(opt, 0)})"
        regiones = ["Todas"] + sorted(list(df['Region'].unique()))
        st.markdown(f"<div class='filter-label'>{get_icon('region')} Región</div>", unsafe_allow_html=True)
        sel_region = st.multiselect(label="", options=regiones, default=["Todas"], key="fr", format_func=fmt_reg, label_visibility="collapsed")

    with c2:
        df_f = df.copy()
        if "Todas" not in sel_region:
            df_f = df_f[df_f['Region'].isin(sel_region)]
        sucursales = ["Todas"] + sorted(list(df_f['Sucursal'].unique()))
        st.markdown(f"<div class='filter-label'>{get_icon('tablero_de_sucursales')} Sucursal</div>", unsafe_allow_html=True)
        sel_sucursal = st.multiselect(label="", options=sucursales, default=["Todas"], key="fs", label_visibility="collapsed")

    with c3:
        counts_risk = df['Nivel_Riesgo'].value_counts().to_dict()
        def fmt_risk(opt): return f"Todos ({len(df)})" if opt == "Todos" else f"{opt} ({counts_risk.get(opt, 0)})"
        riesgos = ["Todos"] + sorted(list(df['Nivel_Riesgo'].unique()))
        st.markdown(f"<div class='filter-label'>{get_icon('nivel_de_riesgo')} Nivel de Riesgo</div>", unsafe_allow_html=True)
        sel_riesgo = st.selectbox(label="", options=riesgos, key="frisk", format_func=fmt_risk, label_visibility="collapsed")

    with c4:
        st.markdown(f"<div class='filter-label'>{get_icon('modo_visualizacion')} Modo de Visualización</div>", unsafe_allow_html=True)
        if 'global_view_mode' not in st.session_state:
            st.session_state['global_view_mode'] = "Total (Suma)"
        view_mode = st.radio(label="", options=["Total (Suma)", "Promedio por Sucursal"], index=0 if st.session_state['global_view_mode']=="Total (Suma)" else 1, key="global_view_mode", label_visibility="collapsed")

    st.markdown('</div>', unsafe_allow_html=True)

    # filtered dataframe
    df_view = df.copy()
    if "Todas" not in sel_region:
        df_view = df_view[df_view['Region'].isin(sel_region)]
    if "Todas" not in sel_sucursal:
        df_view = df_view[df_view['Sucursal'].isin(sel_sucursal)]
    if sel_riesgo != "Todos":
        df_view = df_view[df_view['Nivel_Riesgo'] == sel_riesgo]

    # --- KPIS ---
    kpis = [
        {"id": "Saldo", "label": "Saldo Insoluto", "col": "Saldo_Calc", "type": "money"},
        {"id": "ICV", "label": "ICV (Riesgo)", "col": "ICV_Calc", "type": "percent"},
        {"id": "FPD", "label": "% FPD", "col": "FPD_Calc", "type": "percent"},
        {"id": "Dispersado", "label": "Cap. Dispersado", "col": "Dispersado_Calc", "type": "money"},
        {"id": "Perdidas", "label": "Pérdidas Totales", "col": "Perdidas_Calc", "type": "money"},
    ]

    if 'kpi_selected' not in st.session_state:
        st.session_state['kpi_selected'] = kpis[0]

    current_id = st.session_state['kpi_selected']['id']
    kpi_state = get_kpi_config(current_id)
    use_sum = (view_mode == "Total (Suma)")

    cols_kpi = st.columns(5)
    for i, kpi in enumerate(kpis):
        if use_sum:
            val = df_view[kpi['col']].sum()
            if kpi['type'] == 'percent':
                val = df_view[kpi['col']].mean()
        else:
            val = df_view[kpi['col']].mean()
        val_fmt = format_percent(val) if kpi['type']=='percent' else format_big_number(val)

        is_active = (current_id == kpi['id'])
        with cols_kpi[i]:
            if st.button(f"{kpi['label']}\n\n{val_fmt}", key=f"btn_{kpi['id']}", type="primary" if is_active else "secondary", use_container_width=True):
                st.session_state['kpi_selected'] = kpi
                st.rerun()

    cur_kpi = st.session_state['kpi_selected']

    # --- HISTORICO / DISTRIBUCION ---
    cols_hist, labels_hist = get_trend_series(df_view, cur_kpi['id'], limit=24)
    cols_den = []
    if cur_kpi['id'] == 'ICV':
        cols_den, _ = get_trend_series(df_view, 'Saldo', limit=24)

    st.markdown("<br>", unsafe_allow_html=True)
    col_izq, col_der = st.columns([3, 2], gap="large")

    def calculate_trend_values(sub_df):
        if cur_kpi['id'] == 'ICV' and cols_den:
            num = sub_df[cols_hist].sum().values
            den = sub_df[cols_den].sum().values
            with np.errstate(divide='ignore', invalid='ignore'):
                val = (num / den) * 100
            return np.nan_to_num(val)
        elif cur_kpi['type'] == 'percent':
            # CORRECCIÓN: Si es porcentaje, usamos el promedio y multiplicamos por 100 para escalar
            return sub_df[cols_hist].mean().values * 100
        elif use_sum:
            return sub_df[cols_hist].sum().values
        else:
            return sub_df[cols_hist].mean().values

    with col_izq:
        st.markdown(f"#### {get_icon('evolucion_por_riesgo')} Evolución por Riesgo", unsafe_allow_html=True)
        if cols_hist:
            fig = go.Figure()
            y_global = calculate_trend_values(df_view)
            tooltip_fmt = format_percent if cur_kpi['type'] == 'percent' else format_big_number

            fig.add_trace(go.Scatter(
                x=labels_hist, y=y_global,
                mode='lines+markers', name='Global (Ref)',
                line=dict(color='#2b6cb0', width=4, dash='dash'),
                marker=dict(size=6, color='#2b6cb0'),
                text=[tooltip_fmt(v) for v in y_global]
            ))

            risk_colors = {"Saludable": "#63AB32", "Riesgo Medio": "#F6AD55", "Riesgo Alto": "#EF5350"}
            for risk in ["Saludable", "Riesgo Medio", "Riesgo Alto"]:
                sub = df_view[df_view['Nivel_Riesgo'] == risk]
                if sub.empty: continue
                y_risk = calculate_trend_values(sub)
                fig.add_trace(go.Scatter(
                    x=labels_hist, y=y_risk,
                    mode='lines+markers', name=risk,
                    line=dict(width=3, color=risk_colors[risk]),
                    marker=dict(size=7),
                    text=[tooltip_fmt(v) for v in y_risk]
                ))

            yaxis_config = dict(showgrid=True, gridcolor=colors['border'])
            if cur_kpi['type'] == 'percent': yaxis_config['ticksuffix'] = " %"

            fig.update_layout(
                paper_bgcolor=colors['bg_card'], plot_bgcolor=colors['bg_card'],
                font=dict(color=colors['text_primary']), height=400,
                margin=dict(l=20,r=20,t=10,b=20), hovermode="x unified",
                legend=dict(orientation="h", y=1.1), yaxis=yaxis_config
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sin histórico disponible")

    with col_der:
        c_d1, c_d2 = st.columns([2, 1])
        with c_d1:
            st.markdown(f"#### {get_icon('distribucion_por_region')} Distribución por Riesgo", unsafe_allow_html=True)
        with c_d2:
            opts_ct = ["Distribución (Hist)", "Composición (Pie)"]
            idx_ct = opts_ct.index(kpi_state['chart_type']) if kpi_state['chart_type'] in opts_ct else 0
            chart_type = st.selectbox("Tipo:", opts_ct, index=idx_ct, key=f"setting_charttype_{cur_kpi['id']}", on_change=update_kpi_setting, label_visibility="collapsed")

        risk_colors_map = {"Saludable": "#63AB32", "Riesgo Medio": "#F6AD55", "Riesgo Alto": "#EF5350"}

        if "Pie" in chart_type:
            group_col = "Nivel_Riesgo"
            if use_sum:
                df_pie = df_view.groupby(group_col)[cur_kpi['col']].sum().reset_index()
                if cur_kpi['type'] == 'percent':
                     df_pie = df_view.groupby(group_col)[cur_kpi['col']].mean().reset_index()
            else:
                df_pie = df_view.groupby(group_col)[cur_kpi['col']].mean().reset_index()

            fig_pie = px.pie(df_pie, values=cur_kpi['col'], names=group_col, hole=0.4, color=group_col,
                             color_discrete_map=risk_colors_map)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(paper_bgcolor=colors['bg_card'], plot_bgcolor=colors['bg_card'], font=dict(color=colors['text_primary']), height=370, margin=dict(l=20,r=20,t=10,b=20), showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            fig_hist = px.histogram(
                df_view, x=cur_kpi['col'], color="Nivel_Riesgo",
                nbins=20, barmode="overlay",
                color_discrete_map=risk_colors_map,
                opacity=0.75
            )
            fig_hist.update_layout(
                paper_bgcolor=colors['bg_card'], plot_bgcolor=colors['bg_card'],
                font=dict(color=colors['text_primary']), height=400,
                margin=dict(l=20,r=20,t=10,b=20), dragmode='select',
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    # --- BUBBLE ANALYSIS ---
    st.markdown("---")
    st.markdown(f"### {get_icon('analisis_multivariable')} Análisis Multivariable (Burbujas)", unsafe_allow_html=True)

    c_bubble1, c_bubble2, c_bubble3, c_bubble4 = st.columns(4)
    bubble_opts = [
        {"label": "Saldo Insoluto", "col": "Saldo_Calc"},
        {"label": "ICV (%)", "col": "ICV_Calc"},
        {"label": "% FPD", "col": "FPD_Calc"},
        {"label": "Capital Dispersado", "col": "Dispersado_Calc"},
        {"label": "Pérdidas Totales", "col": "Perdidas_Calc"}
    ]
    bubble_labels = [opt["label"] for opt in bubble_opts]

    with c_bubble1:
        x_axis_label = st.selectbox("Eje X:", bubble_labels, index=0, key="bubble_x")
    with c_bubble2:
        y_axis_label = st.selectbox("Eje Y:", bubble_labels, index=1, key="bubble_y")
    with c_bubble3:
        size_axis_label = st.selectbox("Tamaño (Burbuja):", ["Ninguno"] + bubble_labels, index=4, key="bubble_size")
    with c_bubble4:
        color_axis_label = st.selectbox("Color (Agrupación):", ["Nivel de Riesgo", "Región"], index=0, key="bubble_color")

    x_col = next(opt["col"] for opt in bubble_opts if opt["label"] == x_axis_label)
    y_col = next(opt["col"] for opt in bubble_opts if opt["label"] == y_axis_label)
    size_col = None
    if size_axis_label != "Ninguno":
        size_col = next(opt["col"] for opt in bubble_opts if opt["label"] == size_axis_label)
    color_col = "Nivel_Riesgo" if color_axis_label == "Nivel de Riesgo" else "Region"

    if not df_view.empty:
        if size_col:
            df_view[size_col] = df_view[size_col].fillna(0)
            df_view[size_col] = df_view[size_col].clip(lower=0)

        if color_col == "Nivel_Riesgo":
            color_map = {"Saludable": "#63AB32", "Riesgo Medio": "#F6AD55", "Riesgo Alto": "#EF5350"}
        else:
            color_map = px.colors.qualitative.Plotly

        try:
            fig_bubble = px.scatter(
                df_view,
                x=x_col, y=y_col,
                size=size_col if size_col else None,
                color=color_col,
                hover_name="Sucursal",
                hover_data={"Region": True, "Nivel_Riesgo": True, x_col: True, y_col: True},
                color_discrete_map=color_map if isinstance(color_map, dict) else None,
                size_max=40
            )
            
            # --- CÁLCULO MANUAL DE TENDENCIA (Sin Statsmodels) ---
            # CORRECCIÓN: Usar numpy para la regresión y evitar errores
            x_vals = df_view[x_col].values
            y_vals = df_view[y_col].values
            
            # Filtrar NaNs e Infinitos
            mask = np.isfinite(x_vals) & np.isfinite(y_vals)
            x_clean = x_vals[mask]
            y_clean = y_vals[mask]

            title_text = "Correlación de Variables"
            if len(x_clean) > 1:
                # Regresión Lineal Grado 1
                slope, intercept = np.polyfit(x_clean, y_clean, 1)
                line_y = slope * x_clean + intercept
                
                # Calcular R2
                correlation_matrix = np.corrcoef(x_clean, y_clean)
                correlation_xy = correlation_matrix[0,1]
                r_squared = correlation_xy**2
                
                title_text = f"Correlación de Variables (R² = {r_squared:.4f})"
                
                # Agregar la línea al gráfico
                fig_bubble.add_trace(
                    go.Scatter(
                        x=x_clean, y=line_y, mode="lines",
                        name="Tendencia", line=dict(color="gray", dash="dash")
                    )
                )

            fig_bubble.update_layout(
                title=dict(text=title_text, font=dict(size=16)),
                paper_bgcolor=colors['bg_card'],
                plot_bgcolor=colors['bg_card'],
                font=dict(color=colors['text_primary']),
                height=500,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title=x_axis_label,
                yaxis_title=y_axis_label,
                legend=dict(orientation="h", y=1.1)
            )

            if not size_col:
                fig_bubble.update_traces(marker=dict(size=12))

            st.plotly_chart(fig_bubble, use_container_width=True)

        except Exception as e:
            st.warning(f"Error al generar gráfico: {e}")
    else:
        st.info("No hay datos para mostrar en el gráfico de burbujas con los filtros actuales.")

    # --- TABLE ---
    st.markdown(f"### {get_icon('detalle_de_sucursales')}  Detalle de Sucursales", unsafe_allow_html=True)
    df_table = df_view.copy()

    # Show table (selection-by-chart removed for simplicity / reliability)
    max_col_value = float(df[cur_kpi['col']].max()) if cur_kpi['col'] in df.columns else 1.0
    st.dataframe(
        df_table[['Region', 'Sucursal', 'Nivel_Riesgo', cur_kpi['col']]].sort_values(cur_kpi['col'], ascending=(cur_kpi['type']!='money')),
        use_container_width=True, hide_index=True,
        column_config={cur_kpi['col']: st.column_config.ProgressColumn(cur_kpi['label'], format="$%.2f" if cur_kpi['type']=='money' else "%.2f%%", min_value=0, max_value=max_col_value)}
    )

# =============================================================================
# run render when module executed
# =============================================================================
if __name__ == "__main__":
    render()