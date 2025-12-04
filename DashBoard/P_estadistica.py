# P_estadistica.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)

# Se importan funcion de tema y sidebar unificado
from utils.theme import get_theme_colors
from utils.css_manager import apply_css
from components.header import create_page_header
from utils.icons import get_icon
from urllib.parse import quote  # <-- nuevo import


# ---------------------------------------------------------------------
# Nota: este módulo expone `render()` que debes llamar desde el router
# ej: P_estadistica.render()
# ---------------------------------------------------------------------

# --------- Helpers internos ----------
def _safe_ratio(numerador, denominador):
    num = pd.to_numeric(numerador, errors='coerce')
    den = pd.to_numeric(denominador, errors='coerce')
    return np.where((den == 0) | (pd.isna(den)) | (pd.isna(num)), np.nan, num / den)

@st.cache_data
def _load_data_from_file(archivo='Base_de_datos_Dimex.csv'):
    """Carga archivo Excel/CSV desde ruta (retorna DataFrame)"""
    try:
        if archivo.endswith(('.xlsx', '.xls')):
            df_local = pd.read_excel(archivo)
            return df_local
        elif archivo.endswith('.csv'):
            try:
                return pd.read_csv(archivo, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    return pd.read_csv(archivo, encoding='latin-1')
                except:
                    return pd.read_csv(archivo, encoding='ISO-8859-1')
        else:
            raise ValueError("Formato no soportado")
    except FileNotFoundError:
        raise
    except Exception as e:
        raise

def _render_metric_card(title, value, hint="", colors=None):
    c = colors or get_theme_colors()
    hint_html = f'<div style="font-size:0.85rem; color:{c["text_muted"]}; margin-top:6px;">{hint}</div>' if hint else ""
    st.markdown(f"""
        <div class="metric-card" style="padding:12px;">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
            {hint_html}
        </div>
    """, unsafe_allow_html=True)

def _svg_to_data_uri(svg: str) -> str:
    """Convierte el SVG de get_icon en data URI para usarlo en CSS."""
    if not svg:
        return ""
    svg_clean = svg.replace("\n", "").replace('"', "'")
    return "data:image/svg+xml;utf8," + quote(svg_clean)

# --------- Render principal ----------
def render():
    st.set_page_config(page_title="Estadísticas y Modelos", page_icon=get_icon("metricas_principales_gris"), layout="wide")
    # Inicializar UI base
    colors = get_theme_colors()
    apply_css("statistics")


    # Inicializar UI base
    colors = get_theme_colors()
    apply_css("statistics")
    header_icon = get_icon("analisis_estadistico")  
    create_page_header(
       f"{header_icon}Análisis Estadístico y Modelos Predictivos",
        "Sistema de detección de deterioro crediticio"
    )

    # Estado modal / usuario (prefijado para seguridad)
    st.session_state.setdefault("pest_show_config_modal", False)
    st.session_state.setdefault("pest_show_user_modal", False)
    st.session_state.setdefault("pest_show_notifications_modal", False)
    st.session_state.setdefault("pest_user_name", "Usuario DIMEX")
    st.session_state.setdefault("pest_user_email", "usuario@dimex.com")
    st.session_state.setdefault("pest_user_password", "********")

    # Forzar el background del contenedor principal (simple safeguard)
    st.markdown(
        f"""
        <style>
            [data-testid="stAppViewContainer"] {{
                background-color: {colors['bg_primary']};
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # -------------------------
    # CARGA DE DATOS (panel recarga)
    # -------------------------
    if 'pest_df' not in st.session_state or st.session_state.get('pest_df') is None:
        with st.expander("Opciones de carga (ruta del archivo)", expanded=False):
            archivo_input = st.text_input("Ruta del archivo (ej: Base_de_datos_Dimex.csv)", value="Base_de_datos_Dimex.csv", key="pest_archivo_input")
            cargar_btn = st.button("Recargar archivo", key="pest_recargar")

        try:
            with st.spinner("Cargando datos..."):
                archivo = archivo_input if archivo_input else "Base_de_datos_Dimex.csv"
                st.session_state['pest_df'] = _load_data_from_file(archivo)
                st.success(f"Archivo cargado: {archivo}")
        except FileNotFoundError:
            st.error(f"No se encontró: {archivo}")
            st.stop()
        except Exception as e:
            st.error(f"Error cargando archivo: {e}")
            st.stop()

    df = st.session_state['pest_df']
    colors = get_theme_colors()

    # -------------------------
    # KPIs INICIALES
    # -------------------------
    col1, col2, col3 = st.columns(3)
    with col1:
        _render_metric_card("Total de Registros", f"{df.shape[0]:,}", colors=colors)
    with col2:
        _render_metric_card("Columnas", f"{df.shape[1]}", colors=colors)
    with col3:
        valores_nulos = int(df.isnull().sum().sum())
        _render_metric_card("Valores Nulos", f"{valores_nulos:,}", colors=colors)

    st.markdown("---")

    with st.expander("Ver primeras filas del dataset", expanded=False):
        st.dataframe(df.head(10), use_container_width=True, height=300)

    st.markdown("---")

    # -------------------------
    # INGENIERÍA DE CARACTERÍSTICAS
    # -------------------------
    st.markdown(f'<div class="section-title"> {get_icon("ingenieria_de_caracteristicas")} Ingeniería de Características</div>', unsafe_allow_html=True)

    with st.spinner("Creando variables predictoras..."):
        # Crear variables si existen
        if 'SaldoInsolutoVencidoActual' in df.columns and 'SaldoInsolutoActual' in df.columns:
            df['ICV'] = _safe_ratio(df['SaldoInsolutoVencidoActual'], df['SaldoInsolutoActual'])

        if 'CapitalLiquidadoActual' in df.columns and 'CapitalDispersadoActual' in df.columns:
            df['Ratio_Recuperacion'] = _safe_ratio(df['CapitalLiquidadoActual'], df['CapitalDispersadoActual'])

        if 'QuitasActual' in df.columns and 'CastigosActual' in df.columns and 'SaldoInsolutoActual' in df.columns:
            df['Perdidas_Total'] = df['QuitasActual'].fillna(0) + df['CastigosActual'].fillna(0)
            df['Ratio_Perdidas'] = _safe_ratio(df['Perdidas_Total'], df['SaldoInsolutoActual'])

        if '%FPDActual' in df.columns:
            df['FPD_Actual'] = pd.to_numeric(df['%FPDActual'], errors='coerce')

        # Buscar columna 30-89
        col_3089 = None
        for col in df.columns:
            if '3089' in str(col) and 'Actual' in str(col):
                col_3089 = col
                break

        if col_3089 and 'SaldoInsolutoActual' in df.columns:
            df['Ratio_30_89'] = _safe_ratio(df[col_3089], df['SaldoInsolutoActual'])

    # Crear target (mismo criterio usado antes)
    def _create_target(df_local):
        cond1 = (df_local['ICV'] > 0.05) if 'ICV' in df_local.columns else pd.Series([False] * len(df_local))
        cond2 = (df_local['Ratio_30_89'] > 0.03) if 'Ratio_30_89' in df_local.columns else pd.Series([False] * len(df_local))
        cond3 = (df_local['FPD_Actual'] > 0.06) if 'FPD_Actual' in df_local.columns else pd.Series([False] * len(df_local))
        target = (cond1 & cond2 & cond3).astype(int)
        return target, int(cond1.sum()), int(cond2.sum()), int(cond3.sum())

    df['Deterioro_Crediticio'], n1, n2, n3 = _create_target(df)

    # Mostrar métricas de ingeniería
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
            <div class="metric-card" style="padding:10px;">
                <div class="metric-label">Variables Creadas</div>
                <div class="metric-value">5</div>
                <div style="font-size:0.85rem; color:{colors['success']}; margin-top:6px;">✓ ICV, Recuperación, Pérdidas, FPD, 30-89</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        casos_deterioro = int(df['Deterioro_Crediticio'].sum())
        st.markdown(f"""
            <div class="metric-card" style="padding:10px;">
                <div class="metric-label">Casos con Deterioro</div>
                <div class="metric-value">{casos_deterioro}</div>
                <div style="font-size:0.85rem; color:{colors['error']}; margin-top:6px;">⚠ Requiere atención</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        proporcion = df['Deterioro_Crediticio'].mean() * 100
        st.markdown(f"""
            <div class="metric-card" style="padding:10px;">
                <div class="metric-label">Proporción Deterioro</div>
                <div class="metric-value">{proporcion:.1f}%</div>
                <div style="font-size:0.85rem; color:{'#ef4444' if proporcion>10 else '#16a34a'}; margin-top:6px;">{'⚠ Alto' if proporcion>10 else '✓ Controlado'}</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        casos_sanos = len(df) - casos_deterioro
        st.markdown(f"""
            <div class="metric-card" style="padding:10px;">
                <div class="metric-label">Casos Sanos</div>
                <div class="metric-value">{casos_sanos}</div>
                <div style="font-size:0.85rem; color:{colors['success']}; margin-top:6px;">✓ {(casos_sanos/len(df)*100):.1f}% de la cartera</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    with st.expander("Ver estadísticas de variables creadas"):
        feature_columns = ['ICV', 'Ratio_Recuperacion', 'Ratio_Perdidas', 'FPD_Actual', 'Ratio_30_89']
        feature_columns = [col for col in feature_columns if col in df.columns]
        if feature_columns:
            st.dataframe(df[feature_columns].describe().round(4), use_container_width=True)

    st.markdown("---")

    # -------------------------
    # VISUALIZACIONES (tabs)
    # -------------------------
    st.markdown(
    f'<div class="section-title">{get_icon("analisis_visual")} Análisis Exploratorio Visual</div>',
    unsafe_allow_html=True
)

    # ====== ÍCONOS PARA LAS TABS (usando tus SVG) ======
    # Cambia los nombres "distribuciones", "correlaciones", etc.
    # por los que tengas definidos en utils.icons si son diferentes.
    tab1_icon_uri = _svg_to_data_uri(get_icon("distribuciones"))
    tab2_icon_uri = _svg_to_data_uri(get_icon("correlaciones"))
    tab3_icon_uri = _svg_to_data_uri(get_icon("por_clase"))
    tab4_icon_uri = _svg_to_data_uri(get_icon("target"))
    tab5_icon_uri = _svg_to_data_uri(get_icon("semaforo"))

    current_theme = st.session_state.get("theme", "dark")
    icon_filter_value = "brightness(0) invert(1)" if current_theme == "dark" else "none"

    st.markdown(f"""
    <style>
    /* Hacemos que el contenido del tab sea flex para icono + texto */
    .stTabs [data-baseweb="tab-list"] button > div {{
        display: flex;
        align-items: center;
        gap: 6px;
    }}

    /* ------- TAB 1: Distribuciones ------- */
    .stTabs [data-baseweb="tab-list"] button:nth-child(1) > div {{
        padding-left: 26px;
        position: relative;
    }}
    .stTabs [data-baseweb="tab-list"] button:nth-child(1) > div::before {{
        content: "";
        position: absolute;
        left: 4px;
        top: 50%;
        transform: translateY(-50%);
        width: 18px;
        height: 18px;
        background-image: url("{tab1_icon_uri}");
        background-size: contain;
        background-repeat: no-repeat;
        filter: {icon_filter_value};
    }}

    /* ------- TAB 2: Correlaciones ------- */
    .stTabs [data-baseweb="tab-list"] button:nth-child(2) > div {{
        padding-left: 26px;
        position: relative;
    }}
    .stTabs [data-baseweb="tab-list"] button:nth-child(2) > div::before {{
        content: "";
        position: absolute;
        left: 4px;
        top: 50%;
        transform: translateY(-50%);
        width: 18px;
        height: 18px;
        background-image: url("{tab2_icon_uri}");
        background-size: contain;
        background-repeat: no-repeat;
        filter: {icon_filter_value};
    }}

    /* ------- TAB 3: Por Clase ------- */
    .stTabs [data-baseweb="tab-list"] button:nth-child(3) > div {{
        padding-left: 26px;
        position: relative;
    }}
    .stTabs [data-baseweb="tab-list"] button:nth-child(3) > div::before {{
        content: "";
        position: absolute;
        left: 4px;
        top: 50%;
        transform: translateY(-50%);
        width: 18px;
        height: 18px;
        background-image: url("{tab3_icon_uri}");
        background-size: contain;
        background-repeat: no-repeat;
        filter: {icon_filter_value};
    }}

    /* ------- TAB 4: Target ------- */
    .stTabs [data-baseweb="tab-list"] button:nth-child(4) > div {{
        padding-left: 26px;
        position: relative;
    }}
    .stTabs [data-baseweb="tab-list"] button:nth-child(4) > div::before {{
        content: "";
        position: absolute;
        left: 4px;
        top: 50%;
        transform: translateY(-50%);
        width: 18px;
        height: 18px;
        background-image: url("{tab4_icon_uri}");
        background-size: contain;
        background-repeat: no-repeat;
        filter: {icon_filter_value};
    }}

    /* ------- TAB 5: Semáforo ------- */
    .stTabs [data-baseweb="tab-list"] button:nth-child(5) > div {{
        padding-left: 26px;
        position: relative;
    }}
    .stTabs [data-baseweb="tab-list"] button:nth-child(5) > div::before {{
        content: "";
        position: absolute;
        left: 4px;
        top: 50%;
        transform: translateY(-50%);
        width: 18px;
        height: 18px;
        background-image: url("{tab5_icon_uri}");
        background-size: contain;
        background-repeat: no-repeat;
        filter: {icon_filter_value};
    }}
    </style>
    """, unsafe_allow_html=True)

    # Tabs REALES de Streamlit (ya SIN emojis, solo texto)
    tabs = st.tabs(["Distribuciones", "Correlaciones", "Por Clase", "Target", "Semáforo"])

    # Tab 0: Distribuciones
    with tabs[0]:
        st.markdown(f"<h3 style='color:{colors['text_primary']};'>Distribución de Variables Predictoras</h3>", unsafe_allow_html=True)
        feature_columns = [col for col in ['ICV', 'Ratio_Recuperacion', 'Ratio_Perdidas', 'FPD_Actual', 'Ratio_30_89'] if col in df.columns]
        if feature_columns:
            cols = 3
            rows = 2
            fig = make_subplots(rows=rows, cols=cols, subplot_titles=feature_columns, vertical_spacing=0.15, horizontal_spacing=0.08)
            plot_colors = [colors['success'], '#16a34a', '#34d399', '#60a5fa', '#7c3aed']
            for i, col in enumerate(feature_columns):
                row = i // cols + 1
                col_pos = i % cols + 1
                fig.add_trace(go.Histogram(x=df[col].dropna(), name=col, nbinsx=30, marker=dict(color=plot_colors[i % len(plot_colors)], line=dict(width=0)), opacity=0.9), row=row, col=col_pos)
            fig.update_layout(height=600, showlegend=False, title=dict(text="Distribución de Variables", font=dict(color=colors['text_primary'], size=18)), paper_bgcolor=colors['bg_primary'], plot_bgcolor=colors['bg_card'], font=dict(color=colors['text_primary']), margin=dict(t=100, l=60, r=40, b=50))
            fig.update_xaxes(showgrid=True, gridcolor=colors['grid'], tickfont=dict(size=10))
            fig.update_yaxes(showgrid=True, gridcolor=colors['grid'], tickfont=dict(size=10))
            for annotation in fig.layout.annotations:
                annotation.update(font=dict(size=13, color=colors['text_primary']), y=annotation.y + 0.02)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Tab 1: Correlaciones
    with tabs[1]:
        st.markdown(f"<h3 style='color:{colors['text_primary']};'>Correlación con Deterioro Crediticio</h3>", unsafe_allow_html=True)
        feature_columns = [col for col in ['ICV', 'Ratio_Recuperacion', 'Ratio_Perdidas', 'FPD_Actual', 'Ratio_30_89'] if col in df.columns]
        if feature_columns:
            correlaciones = df[feature_columns].corrwith(df['Deterioro_Crediticio']).sort_values()
            colors_corr = [colors['error'] if x < 0 else colors['success'] for x in correlaciones.values]
            fig = go.Figure(go.Bar(x=correlaciones.values, y=correlaciones.index, orientation='h', marker=dict(color=colors_corr), text=[f"{val:.3f}" for val in correlaciones.values], textposition='outside', textfont=dict(color=colors['text_primary'], size=12)))
            fig.update_layout(title=dict(text="Correlación con Target", font=dict(color=colors['text_primary'], size=16)), height=420, paper_bgcolor=colors['bg_primary'], plot_bgcolor=colors['bg_card'], font=dict(color=colors['text_primary']), xaxis=dict(title="", tickfont=dict(color=colors['text_primary'], size=11)), yaxis=dict(title="", tickfont=dict(color=colors['text_primary'], size=11)))
            fig.add_vline(x=0, line_dash="dash", line_color=colors['border'])
            fig.update_xaxes(showgrid=True, gridcolor=colors['grid'])
            st.plotly_chart(fig, use_container_width=True)

    # Tab 2: Distribución por Clase
    with tabs[2]:
        st.markdown(f"<h3 style='color:{colors['text_primary']};'>Distribución por Clase</h3>", unsafe_allow_html=True)
        feature_columns = [col for col in ['ICV', 'Ratio_Recuperacion', 'Ratio_Perdidas', 'FPD_Actual', 'Ratio_30_89'] if col in df.columns]
        if feature_columns:
            selected_var = st.selectbox("Selecciona variable:", feature_columns, key='pest_boxplot_var')
            fig = go.Figure()
            colors_box = [colors['success'], colors['error']]
            for i, clase in enumerate([0, 1]):
                data = df[df['Deterioro_Crediticio'] == clase][selected_var].dropna()
                fig.add_trace(go.Box(y=data, name=f"{'Sin' if clase == 0 else 'Con'} Deterioro", boxmean='sd', marker_color=colors_box[i]))
            fig.update_layout(title=dict(text=f"Distribución de {selected_var} por Clase", font=dict(color=colors['text_primary'], size=16)), height=460, paper_bgcolor=colors['bg_primary'], plot_bgcolor=colors['bg_card'], font=dict(color=colors['text_primary']), xaxis=dict(tickfont=dict(color=colors['text_primary'], size=11)), yaxis=dict(title=selected_var, tickfont=dict(color=colors['text_primary'], size=11)), legend=dict(font=dict(color=colors['text_primary'], size=11)))
            fig.update_yaxes(showgrid=True, gridcolor=colors['grid'])
            st.plotly_chart(fig, use_container_width=True)

    # Tab 3: Distribución del Target
    with tabs[3]:
        st.markdown(f"<h3 style='color:{colors['text_primary']};'>Distribución del Target</h3>", unsafe_allow_html=True)
        counts = df['Deterioro_Crediticio'].value_counts().reindex([0, 1]).fillna(0)
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(data=[go.Pie(labels=['Sin Deterioro', 'Con Deterioro'], values=counts.values, hole=.4, marker=dict(colors=[colors['success'], colors['error']]), textfont_size=14)])
            fig.update_layout(title=dict(text="Proporción de Deterioro", font=dict(color=colors['text_primary'], size=16)), height=380, paper_bgcolor=colors['bg_primary'], font=dict(color=colors['text_primary']))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = go.Figure(data=[go.Bar(x=['Sin Deterioro', 'Con Deterioro'], y=counts.values, marker_color=[colors['success'], colors['error']], text=counts.values, textposition='outside')])
            fig.update_layout(title=dict(text="Conteo de Clases", font=dict(color=colors['text_primary'], size=16)), height=380, paper_bgcolor=colors['bg_primary'], plot_bgcolor=colors['bg_card'], font=dict(color=colors['text_primary']))
            fig.update_yaxes(showgrid=True, gridcolor=colors['grid'])
            st.plotly_chart(fig, use_container_width=True)

    # Tab 4: Semáforo
    with tabs[4]:
        st.markdown(f"<h3 style='color:{colors['text_primary']};'>Semáforo de Sucursales</h3>", unsafe_allow_html=True)
        def clasificar_semaforo(row):
            if pd.isna(row.get('ICV')):
                return 'Precaucion'
            icv = row.get('ICV', 0)
            fpd = row.get('FPD_Actual', 0)
            if icv > 0.05 or fpd > 0.06:
                return 'Saludable'
            elif icv > 0.03 or fpd > 0.04:
                return 'Precaucion'
            else:
                return 'Deterioro'
        df['Semaforo'] = df.apply(clasificar_semaforo, axis=1)
        col1, col2 = st.columns([2,1])
        with col1:
            semaforo_counts = df['Semaforo'].value_counts()
            labels = semaforo_counts.index.tolist()
            vals = semaforo_counts.values.tolist()
            color_map = {'Saludable': colors['success'], 'Precaucion': colors['warning'], 'Deterioro': colors['error']}
            pie_colors = [color_map.get(x, colors['border']) for x in labels]
            fig = go.Figure(data=[go.Pie(labels=labels, values=vals, hole=.4, marker=dict(colors=pie_colors), textfont_size=13)])
            fig.update_layout(paper_bgcolor=colors['bg_primary'], plot_bgcolor=colors['bg_primary'], font=dict(color=colors['text_primary']), title=dict(text="Distribución por Nivel de Riesgo", font=dict(color=colors['text_primary'], size=16)), height=420)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown(f"<h4 style='color:{colors['text_primary']};'>Resumen</h4>", unsafe_allow_html=True)
            for color in ['Saludable','Precaucion','Deterioro']:
                count = semaforo_counts.get(color, 0)
                pct = (count / len(df) * 100) if len(df) > 0 else 0
                emoji = '🟢' if color=='Saludable' else '🟡' if color=='Precaucion' else '🔴'
                border = {'Saludable': colors['success'], 'Precaucion': colors['warning'], 'Deterioro': colors['error']}[color]
                st.markdown(f"""
                    <div style="background:{colors['bg_card']};padding:12px;border-radius:8px;border-left:4px solid {border};margin-bottom:10px;color:{colors['text_primary']};">
                        <div style="display:flex; gap:10px; align-items:center;">
                            <div style="font-size:1.6rem;">{emoji}</div>
                            <div>
                                <div style="font-size:0.85rem; color:{colors['text_muted']};">{color}</div>
                                <div style="font-weight:700; font-size:1.25rem;">{count}</div>
                                <div style="font-size:0.85rem; color:{colors['text_muted']};">{pct:.1f}% del total</div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # -------------------------
    # ENTRENAMIENTO DE MODELOS
    # -------------------------
    st.markdown(f'<div class="section-title">{get_icon("train")} Entrenamiento de Modelos Predictivos</div>', unsafe_allow_html=True)
    st.markdown(f"""<div style="background:{colors['bg_card']};padding:12px 14px;border-radius:8px;border-left:4px solid {colors['success']};margin-bottom:12px;color:{colors['text_primary']};font-size:0.95rem;">
        <strong>Modelos Seleccionados:</strong> Decision Tree y Gradient Boosting.
    </div>""", unsafe_allow_html=True)

    # Botón con key único
    if st.button("Entrenar modelos", key="pest_train_models_btn", type="primary"):
        feature_columns = ['ICV', 'Ratio_Recuperacion', 'Ratio_Perdidas', 'FPD_Actual', 'Ratio_30_89']
        feature_columns = [col for col in feature_columns if col in df.columns and df[col].notna().sum() > 0]

        if len(feature_columns) == 0:
            st.warning("No hay variables predictoras disponibles.")
        else:
            X = df[feature_columns].fillna(df[feature_columns].median())
            y = df['Deterioro_Crediticio']

            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            except Exception:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            st.success(f"Datos: {len(X_train)} train / {len(X_test)} test")

            progress_bar = st.progress(0)
            status_text = st.empty()
            resultados = []

            # Decision Tree
            status_text.info("Entrenando Decision Tree...")
            dt_model = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced')
            dt_model.fit(X_train_scaled, y_train)
            y_pred_dt = dt_model.predict(X_test_scaled)
            y_pred_proba_dt = dt_model.predict_proba(X_test_scaled)[:, 1]

            resultados.append({
                'Modelo': 'Decision Tree',
                'Accuracy': accuracy_score(y_test, y_pred_dt),
                'Precision': precision_score(y_test, y_pred_dt, zero_division=0),
                'Recall': recall_score(y_test, y_pred_dt, zero_division=0),
                'F1-Score': f1_score(y_test, y_pred_dt, zero_division=0),
                'AUC-ROC': roc_auc_score(y_test, y_pred_proba_dt) if len(np.unique(y_test)) > 1 else 0
            })
            progress_bar.progress(50)

            # Gradient Boosting
            status_text.info("Entrenando Gradient Boosting...")
            gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
            gb_model.fit(X_train_scaled, y_train)
            y_pred_gb = gb_model.predict(X_test_scaled)
            y_pred_proba_gb = gb_model.predict_proba(X_test_scaled)[:, 1]

            resultados.append({
                'Modelo': 'Gradient Boosting',
                'Accuracy': accuracy_score(y_test, y_pred_gb),
                'Precision': precision_score(y_test, y_pred_gb, zero_division=0),
                'Recall': recall_score(y_test, y_pred_gb, zero_division=0),
                'F1-Score': f1_score(y_test, y_pred_gb, zero_division=0),
                'AUC-ROC': roc_auc_score(y_test, y_pred_proba_gb) if len(np.unique(y_test)) > 1 else 0
            })
            progress_bar.progress(100)
            status_text.success("Entrenamiento completado!")

            # Guardar en session_state con prefijos (recomendado para aislar entre páginas)
            st.session_state['pest_resultados'] = pd.DataFrame(resultados)
            st.session_state['pest_models_trained'] = True
            st.session_state['pest_y_test'] = y_test
            st.session_state['pest_predictions'] = {
                'dt': (y_pred_dt, y_pred_proba_dt),
                'gb': (y_pred_gb, y_pred_proba_gb)
            }
            st.session_state['pest_feature_columns'] = feature_columns
            st.session_state['pest_importances'] = {
                'dt': dt_model.feature_importances_,
                'gb': gb_model.feature_importances_
            }
            # Guardar modelos en session_state (nota: puede usar joblib.dump para persistir en disco)
            st.session_state['pest_dt_model'] = dt_model
            st.session_state['pest_gb_model'] = gb_model

    # -------------------------
    # RESULTADOS (si entrenados)
    # -------------------------
    if st.session_state.get('pest_models_trained'):
        st.markdown("---")
        st.markdown(f'<div class="section-title">{get_icon("results")} Resultados y Comparación de Modelos</div>', unsafe_allow_html=True)

        df_resultados = st.session_state['pest_resultados']
        mejor_idx = df_resultados['F1-Score'].idxmax()
        mejor_modelo = df_resultados.iloc[mejor_idx]

        st.markdown(f"<h3 style='color:{colors['text_primary']};'>{get_icon("model")} Mejor Modelo</h3>", unsafe_allow_html=True)

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Modelo</div><div class="metric-value" style="font-size:1.1rem;">{mejor_modelo['Modelo']}</div></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Accuracy</div><div class="metric-value">{mejor_modelo['Accuracy']:.3f}</div></div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Precision</div><div class="metric-value">{mejor_modelo['Precision']:.3f}</div></div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Recall</div><div class="metric-value">{mejor_modelo['Recall']:.3f}</div></div>""", unsafe_allow_html=True)
        with col5:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">AUC-ROC</div><div class="metric-value">{mejor_modelo['AUC-ROC']:.3f}</div></div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f"<h3 style='color:{colors['text_primary']};'>{get_icon("table")}Tabla Comparativa</h3>", unsafe_allow_html=True)
        st.dataframe(df_resultados.round(4), use_container_width=True, height=180)

        # Gráfico comparativo
        st.markdown(f"<h3 style='color:{colors['text_primary']};'>{get_icon("compare")}Comparación Visual</h3>", unsafe_allow_html=True)
        fig = go.Figure()
        metricas = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
        colors_metrics = [colors['success'], '#4ade80', '#86efac', '#34d399', '#10b981']
        for i, metrica in enumerate(metricas):
            fig.add_trace(go.Bar(name=metrica, x=df_resultados['Modelo'], y=df_resultados[metrica], text=df_resultados[metrica].round(3), textposition='outside', marker_color=colors_metrics[i % len(colors_metrics)], textfont=dict(color=colors['text_primary'], size=11)))
        fig.update_layout(title=dict(text="Comparación de Métricas", font=dict(color=colors['text_primary'], size=18)), barmode='group', height=480, paper_bgcolor=colors['bg_primary'], plot_bgcolor=colors['bg_card'], font=dict(color=colors['text_primary']))
        fig.update_yaxes(range=[0,1.05], showgrid=True, gridcolor=colors['grid'])
        st.plotly_chart(fig, use_container_width=True)

        predictions = st.session_state['pest_predictions']
        y_test = st.session_state['pest_y_test']

        # Matrices de confusión
        st.markdown(f"<h3 style='color:{colors['text_primary']};'>{get_icon("matriz")} Matrices de Confusión</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            cm_dt = confusion_matrix(y_test, predictions['dt'][0])
            fig = go.Figure(data=go.Heatmap(z=cm_dt, x=['Pred:Sin','Pred:Con'], y=['Real:Sin','Real:Con'], colorscale=[[0, colors['bg_card']],[1, colors['success']]], text=cm_dt, texttemplate='%{text}', textfont=dict(color=colors['text_primary'], size=14), showscale=False))
            fig.update_layout(title=dict(text="Decision Tree", font=dict(color=colors['text_primary'], size=14)), height=380, paper_bgcolor=colors['bg_primary'], plot_bgcolor=colors['bg_card'])
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            cm_gb = confusion_matrix(y_test, predictions['gb'][0])
            fig = go.Figure(data=go.Heatmap(z=cm_gb, x=['Pred:Sin','Pred:Con'], y=['Real:Sin','Real:Con'], colorscale=[[0, colors['bg_card']],[1, colors['error']]], text=cm_gb, texttemplate='%{text}', textfont=dict(color=colors['text_primary'], size=14), showscale=False))
            fig.update_layout(title=dict(text="Gradient Boosting", font=dict(color=colors['text_primary'], size=14)), height=380, paper_bgcolor=colors['bg_primary'], plot_bgcolor=colors['bg_card'])
            st.plotly_chart(fig, use_container_width=True)

        # Importancia de variables
        st.markdown(f"<h3 style='color:{colors['text_primary']};'>{get_icon("importancia")} Importancia de Variables</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        feature_columns = st.session_state['pest_feature_columns']
        with col1:
            importances_dt = st.session_state['pest_importances']['dt']
            imp_df_dt = pd.DataFrame({'Variable': feature_columns, 'Importancia': importances_dt}).sort_values('Importancia', ascending=True)
            fig = go.Figure(go.Bar(x=imp_df_dt['Importancia'], y=imp_df_dt['Variable'], orientation='h', marker_color=colors['success'], text=imp_df_dt['Importancia'].round(3), textposition='outside', textfont=dict(color=colors['text_primary'], size=11)))
            fig.update_layout(title=dict(text="Decision Tree", font=dict(color=colors['text_primary'], size=14)), height=380, paper_bgcolor=colors['bg_primary'], plot_bgcolor=colors['bg_card'])
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            importances_gb = st.session_state['pest_importances']['gb']
            imp_df_gb = pd.DataFrame({'Variable': feature_columns, 'Importancia': importances_gb}).sort_values('Importancia', ascending=True)
            fig = go.Figure(go.Bar(x=imp_df_gb['Importancia'], y=imp_df_gb['Variable'], orientation='h', marker_color='#16a34a', text=imp_df_gb['Importancia'].round(3), textposition='outside', textfont=dict(color=colors['text_primary'], size=11)))
            fig.update_layout(title=dict(text="Gradient Boosting", font=dict(color=colors['text_primary'], size=14)), height=380, paper_bgcolor=colors['bg_primary'], plot_bgcolor=colors['bg_card'])
            st.plotly_chart(fig, use_container_width=True)

        # Árbol de decisión (matplotlib)
        st.markdown(f"<h3 style='color:{colors['text_primary']};'>{get_icon("tree")} Visualización del Árbol de Decisión</h3>", unsafe_allow_html=True)
        figplt, ax = plt.subplots(figsize=(16,8))
        figplt.patch.set_facecolor('#C1C4C0' if st.session_state.get('theme') == 'light' else '#1e293b')
        try:
            plot_tree(st.session_state['pest_dt_model'], feature_names=feature_columns, class_names=['Sin Deterioro','Con Deterioro'], filled=True, rounded=True, fontsize=9, ax=ax)
            plt.tight_layout()
            st.pyplot(figplt)
        except Exception as e:
            st.warning(f"No se pudo dibujar el árbol: {e}")

        # Reportes de clasificación
        st.markdown(f"<h3 style='color:{colors['text_primary']};'>{get_icon("report")} Reportes Detallados</h3>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["Decision Tree", "Gradient Boosting"])
        with tab1:
            report_dt = classification_report(y_test, predictions['dt'][0], target_names=['Sin Deterioro','Con Deterioro'], output_dict=True)
            st.dataframe(pd.DataFrame(report_dt).transpose(), use_container_width=True)
        with tab2:
            report_gb = classification_report(y_test, predictions['gb'][0], target_names=['Sin Deterioro','Con Deterioro'], output_dict=True)
            st.dataframe(pd.DataFrame(report_gb).transpose(), use_container_width=True)

        # Descargas
        st.markdown("---")
        st.markdown(f"<h3 style='color:{colors['text_primary']};'>{get_icon("download")} Descargar Resultados</h3>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            csv = df_resultados.to_csv(index=False).encode('utf-8')
            st.download_button(label="Tabla Comparativa (CSV)", data=csv, file_name="comparacion_modelos.csv", mime="text/csv", key="pest_dbtn_comp_csv")
        with col2:
            reporte = f"""REPORTE EJECUTIVO - MODELOS PREDICTIVOS
=========================================
MEJOR MODELO: {mejor_modelo['Modelo']}

MÉTRICAS:
- Accuracy:  {mejor_modelo['Accuracy']:.4f}
- Precision: {mejor_modelo['Precision']:.4f}
- Recall:    {mejor_modelo['Recall']:.4f}
- F1-Score:  {mejor_modelo['F1-Score']:.4f}
- AUC-ROC:   {mejor_modelo['AUC-ROC']:.4f}

COMPARACIÓN:
{df_resultados.to_string(index=False)}

VARIABLES MÁS IMPORTANTES (Gradient Boosting):
{imp_df_gb.sort_values('Importancia', ascending=False).to_string(index=False)}
"""
            st.download_button(label="Reporte Ejecutivo (TXT)", data=reporte, file_name="reporte_ejecutivo.txt", mime="text/plain", key="pest_dbtn_report_txt")
        with col3:
            importancias_csv = imp_df_gb.to_csv(index=False).encode('utf-8')
            st.download_button(label="Importancia Variables (CSV)", data=importancias_csv, file_name="importancia_variables.csv", mime="text/csv", key="pest_dbtn_imp_csv")

    # end if models_trained

# end render()
