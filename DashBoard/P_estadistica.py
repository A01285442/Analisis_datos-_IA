import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)

# =============================================================================
# CONFIGURACIÓN DE PÁGINA
# =============================================================================
st.set_page_config(page_title="Estadísticas y Modelos", page_icon="📈", layout="wide")

# =============================================================================
# FUNCIONES DE TEMA Y ESTILOS
# =============================================================================

def get_theme_colors():
    """Retorna colores según el tema activo (Verde/Oscuro y Verde/Claro)"""
    if 'theme' not in st.session_state:
        st.session_state['theme'] = 'dark'
    
    if st.session_state['theme'] == 'dark':
        # Tema Oscuro: Fondo Gris Oscuro, Acento Verde Esmeralda
        return {
            'bg_primary': '#0f1116',
            'bg_secondary': '#1a1d24',
            'bg_card': '#1e293b',
            'text_primary': '#e2e8f0',
            'text_secondary': '#cbd5e0',
            'text_muted': '#94a3b8',
            'border': '#2d3748',
            'accent': '#63AB32',  # Verde Esmeralda Oscuro para acento
            'accent_secondary': '#2FEB00',  # Verde más oscuro para gradiente
            'success': '#22c55e',
            'warning': '#eab308',
            'error': '#ef4444',
            'grid': '#334155'
        }
    else:
        # Tema Claro: Fondo Blanco, Acento Verde Menta
        return {
            'bg_primary': '#ffffff',
            'bg_secondary': '#f8fafc',
            'bg_card': '#ffffff',
            'text_primary': '#1e293b',
            'text_secondary': '#475569',
            'text_muted': '#64748b',
            'border': '#e2e8f0',
            'accent': '#2FEB00',  # Verde Menta para acento
            'accent_secondary': '#2FEB00',  # Verde ligeramente más oscuro para gradiente
            'success': '#16a34a',
            'warning': '#ca8a04',
            'error': '#dc2626',
            'grid': '#e2e8f0'
        }

def apply_custom_css():
    """Aplica CSS personalizado según el tema"""
    colors = get_theme_colors()
    
    # Forzar el background del contenedor principal
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
    
    css = f"""
    <style>
    /* =============== GLOBAL =============== */
    .main, .block-container {{
        background-color: {colors['bg_primary']} !important;
        color: {colors['text_primary']} !important;
    }}
    
    h1, h2, h3, h4, h5, h6, label, p, span {{
        color: {colors['text_primary']} !important;
    }}
    
    /* =============== SIDEBAR (VERDE) =============== */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {colors['accent']} 0%, {colors['accent_secondary']} 100%) !important;
    }}
    
    /* Regla general para widgets de Streamlit en sidebar */
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {{
        color: white !important;
    }}

    /* CORRECCIÓN: Excepción para tarjetas personalizadas */
    .dimex-sidebar-card * {{
        color: {colors['text_primary']} !important;
    }}
    
    /* =============== TITLES =============== */
    .section-title {{
        color: {colors['text_primary']} !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        padding: 10px 0;
        border-bottom: 3px solid {colors['accent']};
        margin-bottom: 1.5rem;
    }}
    
    /* =============== CARDS =============== */
    .metric-card {{
        background: {colors['bg_card']} !important;
        color: {colors['text_primary']} !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
        border-left: 5px solid {colors['accent']} !important;
        transition: transform 0.2s;
    }}
    
    .metric-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }}
    
    .metric-label {{
        color: {colors['text_muted']} !important;
        font-size: 0.85rem !important;
    }}
    
    .metric-value {{
        color: {colors['text_primary']} !important;
        font-size: 1.5rem !important;
        font-weight: bold !important;
    }}
    
    /* =============== ALERTS =============== */
    .alert-box {{
        background-color: {colors['bg_card']} !important;
        border: 1px solid {colors['border']} !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        color: {colors['text_primary']} !important;
    }}
    
    /* =============== TABLES =============== */
    thead tr th {{
        background-color: {colors['bg_secondary']} !important;
        color: {colors['text_primary']} !important;
    }}
    
    tbody tr {{
        background-color: {colors['bg_card']} !important;
        color: {colors['text_primary']} !important;
    }}
    
    /* =============== INPUTS =============== */
    .stSelectbox div, .stTextInput input, .stNumberInput input {{
        background-color: {colors['bg_secondary']} !important;
        color: {colors['text_primary']} !important;
        border: 1px solid {colors['border']} !important;
    }}
    
    /* =============== BUTTONS =============== */
    .stButton > button {{
        background: linear-gradient(135deg, {colors['accent']} 0%, {colors['accent_secondary']} 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s !important;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px {colors['accent']}40 !important;
    }}
    
    /* =============== DOWNLOAD BUTTONS =============== */
    .stDownloadButton > button {{
        background: linear-gradient(135deg, {colors['accent']} 0%, {colors['accent_secondary']} 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s !important;
    }}
    
    .stDownloadButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px {colors['accent']}40 !important;
    }}
    
    /* Forzar color de texto en botones de descarga */
    .stDownloadButton > button > div {{
        color: white !important;
    }}
    
    .stDownloadButton > button > div > span {{
        color: white !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# =============================================================================
# SIDEBAR UNIFICADO
# =============================================================================

def create_dimex_sidebar():
    """Crea el sidebar estándar de DIMEX con toggle de tema"""
    colors = get_theme_colors()
    
    with st.sidebar:
        # Logo/Header con clase protectora
        st.markdown(f"""
        <div class="dimex-sidebar-card" style="text-align: center; padding: 1rem; 
             background: {colors['bg_card']}; 
             border-radius: 10px; margin-bottom: 1rem;">
            <h1 style="color: {colors['accent']} !important; margin: 0; font-size: 1.8rem;">DIMEX</h1>
            <p style="color: {colors['text_muted']} !important; margin: 0.2rem 0 0 0; font-size: 0.8rem;">Sistema Predictivo</p>
            <hr style="border: none; border-top: 1px solid {colors['border']}; margin: 0.2rem 0;">
            <p style="color: {colors['text_secondary']} !important; font-size: 0.75rem; margin: 0;">Módulo Estadísticas</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Toggle de tema
        st.markdown("### ⚙️ Configuración")
        theme_option = st.radio(
            "Tema de color:",
            options=['Oscuro', 'Claro'],
            index=0 if st.session_state.get('theme', 'dark') == 'dark' else 1,
            key='theme_selector'
        )
        
        new_theme = 'dark' if theme_option == 'Oscuro' else 'light'
        if st.session_state.get('theme') != new_theme:
            st.session_state['theme'] = new_theme
            st.rerun()
        
        st.markdown("---")
        
        # Navegación con clase protectora
        st.markdown("### 📚 Navegación")
        st.markdown(f"""
        <div class="dimex-sidebar-card" style="background: {colors['bg_card']};
             padding: 1rem; border-radius: 8px;">
            <ul style="margin: 0; padding-left: 1.5rem; line-height: 2;">
                <li><strong style="color: {colors['text_primary']} !important;">Home</strong></li>
                <li><strong style="color: {colors['text_primary']} !important;">Dashboard</strong></li>
                <li><strong style="color: {colors['accent']} !important;">→ Estadísticas</strong></li>
                <li><strong style="color: {colors['text_primary']} !important;">Asistente IA</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Información del archivo cargado con clase protectora
        if 'df' in st.session_state and st.session_state['df'] is not None:
            df = st.session_state['df']
            st.markdown("### 📊 Datos Cargados")
            st.markdown(f"""
            <div class="dimex-sidebar-card" style="background: {colors['bg_card']}; 
                 padding: 1rem; border-radius: 8px;">
                <p style="margin: 0.5rem 0; color: {colors['text_secondary']} !important;">
                    <strong>Registros:</strong> 
                    <span style="color: {colors['text_primary']} !important;">{df.shape[0]:,}</span>
                </p>
                <p style="margin: 0.5rem 0; color: {colors['text_secondary']} !important;">
                    <strong>Columnas:</strong> 
                    <span style="color: {colors['text_primary']} !important;">{df.shape[1]}</span>
                </p>
                <p style="margin: 0.5rem 0; color: {colors['text_secondary']} !important;">
                    <strong>Nulos:</strong> 
                    <span style="color: {colors['text_primary']} !important;">{df.isnull().sum().sum():,}</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Recursos con clase protectora
        st.markdown("### 📚 Recursos")
        st.markdown(f"""
        <div class="dimex-sidebar-card" style="background: {colors['bg_card']};
             padding: 1rem; border-radius: 8px;">
            <ul style="margin: 0; padding-left: 1.5rem; line-height: 2;">
                <li><a href="#" style="color: {colors['accent']} !important; text-decoration: none;">📖 Documentación</a></li>
                <li><a href="#" style="color: {colors['accent']} !important; text-decoration: none;">💬 Soporte</a></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# CARGA DE DATOS
# =============================================================================

@st.cache_data
def load_data_from_file(archivo='Base_Con_NA.xlsx'):
    """Carga archivo Excel/CSV desde ruta"""
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

# =============================================================================
# HEADER DE PÁGINA
# =============================================================================

def create_page_header(title, subtitle):
    """Crea un header consistente"""
    colors = get_theme_colors()
    st.markdown(f"""
        <div style="
            background: linear-gradient(120deg, {colors['accent']} 70%, {colors['accent_secondary']} 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        ">
            <h1 style="margin: 0; font-size: 1.8rem; font-weight: 700; color: white !important;">{title}</h1>
            <p style="margin: 0.25rem 0 0 0; color: #f0f0f0;">{subtitle}</p>
        </div>
    """, unsafe_allow_html=True)

# =============================================================================
# APLICAR TEMA Y SIDEBAR
# =============================================================================

apply_custom_css()
create_dimex_sidebar()
create_page_header(
    "📈 Análisis Estadístico y Modelos Predictivos",
    "Sistema de detección de deterioro crediticio"
)

# =============================================================================
# CARGA DE DATOS
# =============================================================================

# Cargar datos en session_state si no existe
if 'df' not in st.session_state or st.session_state['df'] is None:
    with st.expander("⚙️ Opciones de carga (ruta del archivo)", expanded=False):
        archivo_input = st.text_input("Ruta del archivo (ej: Base_Con_NA.xlsx)", value="Base_Con_NA.xlsx")
        cargar_btn = st.button("🔁 Recargar archivo", key="recargar")

    # Intentar cargar
    try:
        with st.spinner("Cargando datos..."):
            archivo = archivo_input if 'archivo_input' in locals() and archivo_input else "Base_Con_NA.xlsx"
            st.session_state['df'] = load_data_from_file(archivo)
            st.success(f"✅ Archivo cargado: {archivo}")
    except FileNotFoundError:
        st.error(f"⚠️ No se encontró: {archivo}")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error cargando archivo: {e}")
        st.stop()

df = st.session_state['df']
colors = get_theme_colors()

# =============================================================================
# KPIs INICIALES
# =============================================================================

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
        <div class="metric-card" style="padding:12px;">
            <div class="metric-label">Total de Registros</div>
            <div class="metric-value">{df.shape[0]:,}</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-card" style="padding:12px;">
            <div class="metric-label">Columnas</div>
            <div class="metric-value">{df.shape[1]}</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    valores_nulos = df.isnull().sum().sum()
    st.markdown(f"""
        <div class="metric-card" style="padding:12px;">
            <div class="metric-label">Valores Nulos</div>
            <div class="metric-value">{valores_nulos:,}</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

with st.expander("Ver primeras filas del dataset", expanded=False):
    st.dataframe(df.head(10), use_container_width=True, height=300)

st.markdown("---")

# =============================================================================
# INGENIERÍA DE CARACTERÍSTICAS
# =============================================================================

st.markdown('<div class="section-title">🔧 Ingeniería de Características</div>', unsafe_allow_html=True)

with st.spinner("Creando variables predictoras..."):
    def safe_ratio(numerador, denominador):
        num = pd.to_numeric(numerador, errors='coerce')
        den = pd.to_numeric(denominador, errors='coerce')
        return np.where((den == 0) | (pd.isna(den)) | (pd.isna(num)), np.nan, num / den)

    # Crear variables
    if 'SaldoInsolutoVencidoActual' in df.columns and 'SaldoInsolutoActual' in df.columns:
        df['ICV'] = safe_ratio(df['SaldoInsolutoVencidoActual'], df['SaldoInsolutoActual'])

    if 'CapitalLiquidadoActual' in df.columns and 'CapitalDispersadoActual' in df.columns:
        df['Ratio_Recuperacion'] = safe_ratio(df['CapitalLiquidadoActual'], df['CapitalDispersadoActual'])

    if 'QuitasActual' in df.columns and 'CastigosActual' in df.columns and 'SaldoInsolutoActual' in df.columns:
        df['Perdidas_Total'] = df['QuitasActual'].fillna(0) + df['CastigosActual'].fillna(0)
        df['Ratio_Perdidas'] = safe_ratio(df['Perdidas_Total'], df['SaldoInsolutoActual'])

    if '%FPDActual' in df.columns:
        df['FPD_Actual'] = pd.to_numeric(df['%FPDActual'], errors='coerce')

    # Buscar columna 30-89
    col_3089 = None
    for col in df.columns:
        if '3089' in str(col) and 'Actual' in str(col):
            col_3089 = col
            break

    if col_3089 and 'SaldoInsolutoActual' in df.columns:
        df['Ratio_30_89'] = safe_ratio(df[col_3089], df['SaldoInsolutoActual'])

# Crear target
def create_target(df):
    cond1 = (df['ICV'] > 0.05) if 'ICV' in df.columns else pd.Series([False] * len(df))
    cond2 = (df['Ratio_30_89'] > 0.03) if 'Ratio_30_89' in df.columns else pd.Series([False] * len(df))
    cond3 = (df['FPD_Actual'] > 0.06) if 'FPD_Actual' in df.columns else pd.Series([False] * len(df))
    target = (cond1 & cond2 & cond3).astype(int)
    return target, cond1.sum(), cond2.sum(), cond3.sum()

df['Deterioro_Crediticio'], n1, n2, n3 = create_target(df)

# Mostrar métricas
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

# =============================================================================
# VISUALIZACIONES
# =============================================================================

st.markdown('<div class="section-title">📊 Análisis Exploratorio Visual</div>', unsafe_allow_html=True)

tabs = st.tabs(["📈 Distribuciones", "🔗 Correlaciones", "📦 Por Clase", "🎯 Target", "🌐 Semáforo"])

with tabs[0]:
    st.markdown(f"<h3 style='color:{colors['text_primary']};'>Distribución de Variables Predictoras</h3>", unsafe_allow_html=True)
    
    feature_columns = [col for col in ['ICV', 'Ratio_Recuperacion', 'Ratio_Perdidas', 'FPD_Actual', 'Ratio_30_89'] if col in df.columns]
    
    if feature_columns:
        cols = 3
        rows = 2
        fig = make_subplots(
            rows=rows, 
            cols=cols, 
            subplot_titles=feature_columns, 
            vertical_spacing=0.15,  # Aumentado de 0.10 a 0.15 para más espacio
            horizontal_spacing=0.08  # También aumentado ligeramente
        )
        
        plot_colors = [colors['success'], '#16a34a', '#34d399', '#60a5fa', '#7c3aed']
        
        for i, col in enumerate(feature_columns):
            row = i // cols + 1
            col_pos = i % cols + 1
            
            fig.add_trace(
                go.Histogram(
                    x=df[col].dropna(),
                    name=col,
                    nbinsx=30,
                    marker=dict(color=plot_colors[i % len(plot_colors)], line=dict(width=0)),
                    opacity=0.9
                ),
                row=row, col=col_pos
            )
        
        fig.update_layout(
            height=600,  # Aumentado de 560 a 600 para dar más espacio vertical
            showlegend=False,
            title=dict(text="Distribución de Variables", font=dict(color=colors['text_primary'], size=18)),
            paper_bgcolor=colors['bg_primary'],
            plot_bgcolor=colors['bg_card'],
            font=dict(color=colors['text_primary']),
            margin=dict(t=100, l=60, r=40, b=50)  # Aumentado el margen superior (t) de 80 a 100
        )
        
        fig.update_xaxes(showgrid=True, gridcolor=colors['grid'], tickfont=dict(size=10))
        fig.update_yaxes(showgrid=True, gridcolor=colors['grid'], tickfont=dict(size=10))
        
        # Ajustar los títulos de los subplots para mejor espaciado
        for annotation in fig.layout.annotations:
            annotation.update(
                font=dict(size=13, color=colors['text_primary']),
                y=annotation.y + 0.02  # Mover los títulos un poco más arriba
            )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

with tabs[1]:
    st.markdown(f"<h3 style='color:{colors['text_primary']};'>Correlación con Deterioro Crediticio</h3>", unsafe_allow_html=True)
    
    feature_columns = [col for col in ['ICV', 'Ratio_Recuperacion', 'Ratio_Perdidas', 'FPD_Actual', 'Ratio_30_89'] if col in df.columns]
    
    if feature_columns:
        correlaciones = df[feature_columns].corrwith(df['Deterioro_Crediticio']).sort_values()
        colors_corr = [colors['error'] if x < 0 else colors['success'] for x in correlaciones.values]
        
        fig = go.Figure(go.Bar(
            x=correlaciones.values,
            y=correlaciones.index,
            orientation='h',
            marker=dict(color=colors_corr),
            text=[f"{val:.3f}" for val in correlaciones.values],
            textposition='outside',
            textfont=dict(color=colors['text_primary'], size=12)  # AÑADIDO: color de texto
        ))
        
        fig.update_layout(
            title=dict(text="Correlación con Target", font=dict(color=colors['text_primary'], size=16)),
            height=420,
            paper_bgcolor=colors['bg_primary'],
            plot_bgcolor=colors['bg_card'],
            font=dict(color=colors['text_primary']),
            xaxis=dict(
                title="",
                tickfont=dict(color=colors['text_primary'], size=11)  # AÑADIDO
            ),
            yaxis=dict(
                title="",
                tickfont=dict(color=colors['text_primary'], size=11)  # AÑADIDO
            )
        )
        
        fig.add_vline(x=0, line_dash="dash", line_color=colors['border'])
        fig.update_xaxes(showgrid=True, gridcolor=colors['grid'])
        
        st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.markdown(f"<h3 style='color:{colors['text_primary']};'>Distribución por Clase</h3>", unsafe_allow_html=True)
    
    feature_columns = [col for col in ['ICV', 'Ratio_Recuperacion', 'Ratio_Perdidas', 'FPD_Actual', 'Ratio_30_89'] if col in df.columns]
    
    if feature_columns:
        selected_var = st.selectbox("Selecciona variable:", feature_columns, key='boxplot_var')
        
        fig = go.Figure()
        colors_box = [colors['success'], colors['error']]
        
        for i, clase in enumerate([0, 1]):
            data = df[df['Deterioro_Crediticio'] == clase][selected_var].dropna()
            fig.add_trace(go.Box(
                y=data,
                name=f"{'Sin' if clase == 0 else 'Con'} Deterioro",
                boxmean='sd',
                marker_color=colors_box[i]
            ))
        
        fig.update_layout(
            title=dict(text=f"Distribución de {selected_var} por Clase", font=dict(color=colors['text_primary'], size=16)),
            height=460,
            paper_bgcolor=colors['bg_primary'],
            plot_bgcolor=colors['bg_card'],
            font=dict(color=colors['text_primary']),
            xaxis=dict(
                tickfont=dict(color=colors['text_primary'], size=11)  # AÑADIDO
            ),
            yaxis=dict(
                title=selected_var,
                tickfont=dict(color=colors['text_primary'], size=11)  # AÑADIDO
            ),
            legend=dict(
                font=dict(color=colors['text_primary'], size=11)  # AÑADIDO
            )
        )
        
        fig.update_yaxes(showgrid=True, gridcolor=colors['grid'])
        st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    st.markdown(f"<h3 style='color:{colors['text_primary']};'>Distribución del Target</h3>", unsafe_allow_html=True)
    
    counts = df['Deterioro_Crediticio'].value_counts().reindex([0, 1]).fillna(0)
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(data=[go.Pie(
            labels=['Sin Deterioro', 'Con Deterioro'],
            values=counts.values,
            hole=.4,
            marker=dict(colors=[colors['success'], colors['error']]),
            textfont_size=14
        )])
        
        fig.update_layout(
            title=dict(text="Proporción de Deterioro", font=dict(color=colors['text_primary'], size=16)),
            height=380,
            paper_bgcolor=colors['bg_primary'],
            font=dict(color=colors['text_primary'])
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure(data=[go.Bar(
            x=['Sin Deterioro', 'Con Deterioro'],
            y=counts.values,
            marker_color=[colors['success'], colors['error']],
            text=counts.values,
            textposition='outside'
        )])
        
        fig.update_layout(
            title=dict(text="Conteo de Clases", font=dict(color=colors['text_primary'], size=16)),
            height=380,
            paper_bgcolor=colors['bg_primary'],
            plot_bgcolor=colors['bg_card'],
            font=dict(color=colors['text_primary'])
        )
        
        fig.update_yaxes(showgrid=True, gridcolor=colors['grid'])
        st.plotly_chart(fig, use_container_width=True)

with tabs[4]:
    st.markdown(f"<h3 style='color:{colors['text_primary']};'>Semáforo de Sucursales</h3>", unsafe_allow_html=True)
    
    def clasificar_semaforo(row):
        if pd.isna(row.get('ICV')):
            return 'Amarillo'
        icv = row.get('ICV', 0)
        fpd = row.get('FPD_Actual', 0)
        if icv > 0.05 or fpd > 0.06:
            return 'Rojo'
        elif icv > 0.03 or fpd > 0.04:
            return 'Amarillo'
        else:
            return 'Verde'
    
    df['Semaforo'] = df.apply(clasificar_semaforo, axis=1)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        semaforo_counts = df['Semaforo'].value_counts()
        labels = semaforo_counts.index.tolist()
        vals = semaforo_counts.values.tolist()
        
        color_map = {'Verde': colors['success'], 'Amarillo': colors['warning'], 'Rojo': colors['error']}
        pie_colors = [color_map.get(x, colors['border']) for x in labels]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=vals,
            hole=.4,
            marker=dict(colors=pie_colors),
            textfont_size=13
        )])
        
        fig.update_layout(
            paper_bgcolor=colors['bg_primary'],
            plot_bgcolor=colors['bg_primary'],
            font=dict(color=colors['text_primary']),
            title=dict(text="Distribución por Nivel de Riesgo", font=dict(color=colors['text_primary'], size=16)),
            height=420
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown(f"<h4 style='color:{colors['text_primary']};'>Resumen</h4>", unsafe_allow_html=True)
        
        for color in ['Verde', 'Amarillo', 'Rojo']:
            count = semaforo_counts.get(color, 0)
            pct = (count / len(df) * 100) if len(df) > 0 else 0
            emoji = '🟢' if color == 'Verde' else '🟡' if color == 'Amarillo' else '🔴'
            border = {'Verde': colors['success'], 'Amarillo': colors['warning'], 'Rojo': colors['error']}[color]
            
            st.markdown(f"""
                <div style="
                    background:{colors['bg_card']};
                    padding:12px;
                    border-radius:8px;
                    border-left:4px solid {border};
                    margin-bottom:10px;
                    color:{colors['text_primary']};
                ">
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

# =============================================================================
# ENTRENAMIENTO DE MODELOS
# =============================================================================

st.markdown('<div class="section-title">🤖 Entrenamiento de Modelos Predictivos</div>', unsafe_allow_html=True)

st.markdown(f"""
<div style="background:{colors['bg_card']};padding:12px 14px;border-radius:8px;border-left:4px solid {colors['success']};margin-bottom:12px;color:{colors['text_primary']};font-size:0.95rem;">
    <strong>Modelos Seleccionados:</strong> Decision Tree y Gradient Boosting.
</div>
""", unsafe_allow_html=True)

if st.button("🚀 Entrenar modelos", type="primary"):
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

        st.success(f"📊 Datos: {len(X_train)} train / {len(X_test)} test")

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
        status_text.success("✅ Entrenamiento completado!")

        # Guardar en session state
        st.session_state['resultados'] = pd.DataFrame(resultados)
        st.session_state['models_trained'] = True
        st.session_state['y_test'] = y_test
        st.session_state['predictions'] = {
            'dt': (y_pred_dt, y_pred_proba_dt),
            'gb': (y_pred_gb, y_pred_proba_gb)
        }
        st.session_state['feature_columns'] = feature_columns
        st.session_state['importances'] = {
            'dt': dt_model.feature_importances_,
            'gb': gb_model.feature_importances_
        }
        st.session_state['dt_model'] = dt_model
        st.session_state['gb_model'] = gb_model

# =============================================================================
# RESULTADOS
# =============================================================================

if 'models_trained' in st.session_state and st.session_state['models_trained']:
    st.markdown("---")
    st.markdown('<div class="section-title">📊 Resultados y Comparación de Modelos</div>', unsafe_allow_html=True)
    
    df_resultados = st.session_state['resultados']
    mejor_idx = df_resultados['F1-Score'].idxmax()
    mejor_modelo = df_resultados.iloc[mejor_idx]

    st.markdown(f"<h3 style='color:{colors['text_primary']};'>🏆 Mejor Modelo</h3>", unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Modelo</div>
                <div class="metric-value" style="font-size:1.1rem;">{mejor_modelo['Modelo']}</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value">{mejor_modelo['Accuracy']:.3f}</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Precision</div>
                <div class="metric-value">{mejor_modelo['Precision']:.3f}</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Recall</div>
                <div class="metric-value">{mejor_modelo['Recall']:.3f}</div>
            </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">AUC-ROC</div>
                <div class="metric-value">{mejor_modelo['AUC-ROC']:.3f}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"<h3 style='color:{colors['text_primary']};'>📋 Tabla Comparativa</h3>", unsafe_allow_html=True)
    st.dataframe(df_resultados.round(4), use_container_width=True, height=180)

    # Gráfico comparativo
    st.markdown(f"<h3 style='color:{colors['text_primary']};'>📊 Comparación Visual</h3>", unsafe_allow_html=True)
    
    fig = go.Figure()
    metricas = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    colors_metrics = [colors['success'], '#4ade80', '#86efac', '#34d399', '#10b981']
    
    for i, metrica in enumerate(metricas):
        fig.add_trace(go.Bar(
            name=metrica,
            x=df_resultados['Modelo'],
            y=df_resultados[metrica],
            text=df_resultados[metrica].round(3),
            textposition='outside',
            marker_color=colors_metrics[i % len(colors_metrics)],
            textfont=dict(color=colors['text_primary'], size=11)  # AÑADIDO
        ))
    
    fig.update_layout(
        title=dict(text="Comparación de Métricas", font=dict(color=colors['text_primary'], size=18)),
        barmode='group',
        height=480,
        paper_bgcolor=colors['bg_primary'],
        plot_bgcolor=colors['bg_card'],
        font=dict(color=colors['text_primary']),
        xaxis=dict(
            tickfont=dict(color=colors['text_primary'], size=11)  # AÑADIDO
        ),
        yaxis=dict(
            tickfont=dict(color=colors['text_primary'], size=11)  # AÑADIDO
        ),
        legend=dict(
            font=dict(color=colors['text_primary'], size=11)  # AÑADIDO
        )
    )
    
    fig.update_yaxes(range=[0, 1.05], showgrid=True, gridcolor=colors['grid'])
    st.plotly_chart(fig, use_container_width=True)

    # Curvas ROC
    st.markdown(f"<h3 style='color:{colors['text_primary']};'>📈 Curvas ROC</h3>", unsafe_allow_html=True)
    
    predictions = st.session_state['predictions']
    y_test = st.session_state['y_test']
    
    fig = go.Figure()
    
    try:
        fpr_dt, tpr_dt, _ = roc_curve(y_test, predictions['dt'][1])
        auc_dt = roc_auc_score(y_test, predictions['dt'][1])
        fig.add_trace(go.Scatter(
            x=fpr_dt, y=tpr_dt,
            name=f'Decision Tree (AUC={auc_dt:.3f})',
            mode='lines',
            line=dict(width=3, color='#4ade80')
        ))
    except Exception:
        pass
    
    try:
        fpr_gb, tpr_gb, _ = roc_curve(y_test, predictions['gb'][1])
        auc_gb = roc_auc_score(y_test, predictions['gb'][1])
        fig.add_trace(go.Scatter(
            x=fpr_gb, y=tpr_gb,
            name=f'Gradient Boosting (AUC={auc_gb:.3f})',
            mode='lines',
            line=dict(width=3, color=colors['success'])
        ))
    except Exception:
        pass
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name='Aleatorio',
        mode='lines',
        line=dict(dash='dash', color=colors['text_muted'])
    ))
    
    fig.update_layout(
        title=dict(text="Curvas ROC", font=dict(color=colors['text_primary'], size=16)),
        height=460,
        paper_bgcolor=colors['bg_primary'],
        plot_bgcolor=colors['bg_card'],
        font=dict(color=colors['text_primary']),
        xaxis=dict(
            title="Tasa de Falsos Positivos",
            tickfont=dict(color=colors['text_primary'], size=11)
        ),
        yaxis=dict(
            title="Tasa de Verdaderos Positivos",
            tickfont=dict(color=colors['text_primary'], size=11)
        ),
        legend=dict(
            font=dict(color=colors['text_primary'], size=11)
        )
    )
    
    fig.update_xaxes(range=[-0.02, 1.02], showgrid=True, gridcolor=colors['grid'])
    fig.update_yaxes(range=[-0.02, 1.02], showgrid=True, gridcolor=colors['grid'])
    st.plotly_chart(fig, use_container_width=True)

    # Matrices de confusión
    st.markdown(f"<h3 style='color:{colors['text_primary']};'>🔲 Matrices de Confusión</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        cm_dt = confusion_matrix(y_test, predictions['dt'][0])
        fig = go.Figure(data=go.Heatmap(
            z=cm_dt,
            x=['Pred:Sin','Pred:Con'],
            y=['Real:Sin','Real:Con'],
            colorscale=[[0, colors['bg_card']],[1, colors['success']]],
            text=cm_dt,
            texttemplate='%{text}',
            textfont=dict(color=colors['text_primary'], size=14),  # AÑADIDO
            showscale=False
        ))
        fig.update_layout(
            title=dict(text="Decision Tree", font=dict(color=colors['text_primary'], size=14)),
            height=380,
            paper_bgcolor=colors['bg_primary'],
            plot_bgcolor=colors['bg_card'],
            font=dict(color=colors['text_primary']),
            xaxis=dict(
                tickfont=dict(color=colors['text_primary'], size=11),
                side='bottom'
            ),
            yaxis=dict(
                tickfont=dict(color=colors['text_primary'], size=11)
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        cm_gb = confusion_matrix(y_test, predictions['gb'][0])
        fig = go.Figure(data=go.Heatmap(
            z=cm_gb,
            x=['Pred:Sin','Pred:Con'],
            y=['Real:Sin','Real:Con'],
            colorscale=[[0, colors['bg_card']],[1, colors['error']]],
            text=cm_gb,
            texttemplate='%{text}',
            textfont=dict(color=colors['text_primary'], size=14),  # AÑADIDO
            showscale=False
        ))
        fig.update_layout(
            title=dict(text="Gradient Boosting", font=dict(color=colors['text_primary'], size=14)),
            height=380,
            paper_bgcolor=colors['bg_primary'],
            plot_bgcolor=colors['bg_card'],
            font=dict(color=colors['text_primary']),
            xaxis=dict(
                tickfont=dict(color=colors['text_primary'], size=11),
                side='bottom'
            ),
            yaxis=dict(
                tickfont=dict(color=colors['text_primary'], size=11)
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    # Importancia de variables
    st.markdown(f"<h3 style='color:{colors['text_primary']};'>⭐ Importancia de Variables</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    feature_columns = st.session_state['feature_columns']
    
    with col1:
        importances_dt = st.session_state['importances']['dt']
        imp_df_dt = pd.DataFrame({
            'Variable': feature_columns,
            'Importancia': importances_dt
        }).sort_values('Importancia', ascending=True)
        
        fig = go.Figure(go.Bar(
            x=imp_df_dt['Importancia'],
            y=imp_df_dt['Variable'],
            orientation='h',
            marker_color=colors['success'],
            text=imp_df_dt['Importancia'].round(3),
            textposition='outside',
            textfont=dict(color=colors['text_primary'], size=11)  # AÑADIDO
        ))
        
        fig.update_layout(
            title=dict(text="Decision Tree", font=dict(color=colors['text_primary'], size=14)),
            height=380,
            paper_bgcolor=colors['bg_primary'],
            plot_bgcolor=colors['bg_card'],
            font=dict(color=colors['text_primary']),
            xaxis=dict(
                tickfont=dict(color=colors['text_primary'], size=11)
            ),
            yaxis=dict(
                tickfont=dict(color=colors['text_primary'], size=11)
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        importances_gb = st.session_state['importances']['gb']
        imp_df_gb = pd.DataFrame({
            'Variable': feature_columns,
            'Importancia': importances_gb
        }).sort_values('Importancia', ascending=True)
        
        fig = go.Figure(go.Bar(
            x=imp_df_gb['Importancia'],
            y=imp_df_gb['Variable'],
            orientation='h',
            marker_color='#16a34a',
            text=imp_df_gb['Importancia'].round(3),
            textposition='outside',
            textfont=dict(color=colors['text_primary'], size=11)  # AÑADIDO
        ))
        
        fig.update_layout(
            title=dict(text="Gradient Boosting", font=dict(color=colors['text_primary'], size=14)),
            height=380,
            paper_bgcolor=colors['bg_primary'],
            plot_bgcolor=colors['bg_card'],
            font=dict(color=colors['text_primary']),
            xaxis=dict(
                tickfont=dict(color=colors['text_primary'], size=11)
            ),
            yaxis=dict(
                tickfont=dict(color=colors['text_primary'], size=11)
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    # Árbol de decisión
    st.markdown(f"<h3 style='color:{colors['text_primary']};'>🌳 Visualización del Árbol de Decisión</h3>", unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor('#C1C4C0' if st.session_state['theme'] == 'light' else '#1e293b')
    plot_tree(
        st.session_state['dt_model'],
        feature_names=feature_columns,
        class_names=['Sin Deterioro', 'Con Deterioro'],
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax
    )
    plt.tight_layout()
    st.pyplot(fig)

    # Reportes de clasificación
    st.markdown(f"<h3 style='color:{colors['text_primary']};'>📄 Reportes Detallados</h3>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Decision Tree", "Gradient Boosting"])
    
    with tab1:
        report_dt = classification_report(
            y_test,
            predictions['dt'][0],
            target_names=['Sin Deterioro', 'Con Deterioro'],
            output_dict=True
        )
        st.dataframe(pd.DataFrame(report_dt).transpose(), use_container_width=True)
    
    with tab2:
        report_gb = classification_report(
            y_test,
            predictions['gb'][0],
            target_names=['Sin Deterioro', 'Con Deterioro'],
            output_dict=True
        )
        st.dataframe(pd.DataFrame(report_gb).transpose(), use_container_width=True)

    # Descargas
    st.markdown("---")
    st.markdown(f"<h3 style='color:{colors['text_primary']};'>💾 Descargar Resultados</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = df_resultados.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📊 Tabla Comparativa (CSV)",
            data=csv,
            file_name="comparacion_modelos.csv",
            mime="text/csv"
        )
    
    with col2:
        reporte = f"""
REPORTE EJECUTIVO - MODELOS PREDICTIVOS
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
        st.download_button(
            label="📄 Reporte Ejecutivo (TXT)",
            data=reporte,
            file_name="reporte_ejecutivo.txt",
            mime="text/plain"
        )
    
    with col3:
        importancias_csv = imp_df_gb.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⭐ Importancia Variables (CSV)",
            data=importancias_csv,
            file_name="importancia_variables.csv",
            mime="text/csv"
        )