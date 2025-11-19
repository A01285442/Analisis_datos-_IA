"""
TEMPLATE REUTILIZABLE PARA NUEVOS MÓDULOS DIMEX
================================================
Copia este archivo y personaliza las secciones marcadas con # TODO

Características:
- Toggle Dark/Light mode automático
- Sidebar unificado con navegación
- Sistema de colores adaptativo
- Carga de datos centralizada
- Gráficos con tema automático
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
# CONFIGURACIÓN DE PÁGINA (Solo si es página principal)
# =============================================================================
st.set_page_config(
    page_title="Nombre del Módulo",  # TODO: Cambiar nombre
    page_icon="📊",  # TODO: Cambiar icono
    layout="wide"
)

# =============================================================================
# SISTEMA DE TEMAS (COPIALO TAL CUAL)
# =============================================================================

def get_theme_colors():
    """Retorna diccionario de colores según tema activo"""
    if 'theme' not in st.session_state:
        st.session_state['theme'] = 'dark'
    
    if st.session_state['theme'] == 'dark':
        return {
            'bg_primary': '#0f1116',
            'bg_secondary': '#1a1d24',
            'bg_card': '#1e293b',
            'text_primary': '#e2e8f0',
            'text_secondary': '#cbd5e0',
            'text_muted': '#94a3b8',
            'border': '#2d3748',
            'accent': "#63AB32", # Verde Esmeralda Oscuro para acento
            'accent_secondary': '#2FEB00', # Verde más oscuro para gradiente
            'success': '#22c55e',
            'warning': '#eab308',
            'error': '#ef4444',
            'grid': '#334155'
        }
    else:
        return {
            'bg_primary': '#ffffff',
            'bg_secondary': '#f8fafc',
            'bg_card': '#ffffff',
            'text_primary': '#1e293b',
            'text_secondary': '#475569',
            'text_muted': '#64748b',
            'border': '#e2e8f0',
            'accent': "#63AB32", # Verde Esmeralda Oscuro para acento
            'accent_secondary': '#2FEB00', # Verde más oscuro para gradiente
            'success': '#16a34a',
            'warning': '#ca8a04',
            'error': '#dc2626',
            'grid': '#e2e8f0'
        }

def apply_custom_css():
    """Aplica CSS dinámico según el tema"""
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
    
    /* Regla general para widgets de Streamlit en sidebar (Radio buttons, etc.) */
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {{
        color: white !important;
    }}

    /* CORRECCIÓN IMPORTANTE: Excepción para nuestras tarjetas personalizadas */
    /* Esto asegura que el texto DENTRO de las tarjetas blancas sea del color correcto */
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
    
    .alert-info {{ border-left: 5px solid #4299e1 !important; }}
    .alert-success {{ border-left: 5px solid {colors['success']} !important; }}
    .alert-warning {{ border-left: 5px solid {colors['warning']} !important; }}
    .alert-error {{ border-left: 5px solid {colors['error']} !important; }}
    
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
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# =============================================================================
# SIDEBAR UNIFICADO (COPIALO TAL CUAL)
# =============================================================================

def create_dimex_sidebar():
    """Sidebar estándar con logo, toggle de tema y navegación"""
    colors = get_theme_colors()
    
    with st.sidebar:
        # Logo DIMEX con clase protectora
        st.markdown(f"""
        <div class="dimex-sidebar-card" style="text-align: center; padding: 1rem; 
             background: {colors['bg_card']}; 
             border-radius: 10px; margin-bottom: 1rem;">
            <h1 style="color: {colors['accent']} !important; margin: 0; font-size: 1.8rem;">DIMEX</h1>
            <p style="color: {colors['text_muted']} !important; margin: 0.2rem 0 0 0; font-size: 0.8rem;">Sistema Predictivo</p>
            <hr style="border: none; border-top: 1px solid {colors['border']}; margin: 0.2rem 0;">
            <p style="color: {colors['text_secondary']} !important; font-size: 0.75rem; margin: 0;">Tu Módulo v1.0</p>
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
                <li><strong style="color: {colors['text_primary']} !important;">Estadísticas</strong></li>
                <li><strong style="color: {colors['text_primary']} !important;">Asistente IA</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Información de datos cargados con clase protectora
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
# UTILIDADES
# =============================================================================

def create_page_header(title, subtitle):
    """Crea header consistente para la página"""
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

@st.cache_data
def load_excel_data(file_path='Base_Con_NA.xlsx'):
    """Carga datos desde Excel/CSV con manejo robusto de errores"""
    try:
        if file_path.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            try:
                return pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    return pd.read_csv(file_path, encoding='latin-1')
                except:
                    return pd.read_csv(file_path, encoding='ISO-8859-1')
        else:
            raise ValueError("Formato no soportado. Use .xlsx, .xls o .csv")
    except FileNotFoundError:
        st.error(f"⚠️ Archivo no encontrado: {file_path}")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error al cargar: {e}")
        st.stop()

def create_plotly_figure(**layout_kwargs):
    """Crea figura de Plotly con tema automático"""
    colors = get_theme_colors()
    plotly_template = 'plotly_dark' if st.session_state.get('theme') == 'dark' else 'plotly_white'
    
    fig = go.Figure()
    
    fig.update_layout(
        template=plotly_template,
        paper_bgcolor=colors['bg_primary'],
        plot_bgcolor=colors['bg_card'],
        font=dict(color=colors['text_primary']),
        title_font=dict(color=colors['text_primary']),
        **layout_kwargs
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=colors['grid'],
        zerolinecolor=colors['border'],
        tickfont=dict(size=14, color=colors['text_primary'])
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=colors['grid'],
        zerolinecolor=colors['border'],
        tickfont=dict(size=14, color=colors['text_primary'])
    )
    
    return fig

def create_metric_card(label, value, delta=None, delta_positive=True):
    """Crea una tarjeta de métrica con estilo consistente"""
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
# INICIALIZACIÓN
# =============================================================================

# Aplicar tema y sidebar
apply_custom_css()
create_dimex_sidebar()

# Header de la página
create_page_header(
    "Título del Módulo",  # TODO: Cambiar
    "Descripción breve del módulo"  # TODO: Cambiar
)

# Obtener colores del tema actual
colors = get_theme_colors()

# =============================================================================
# CARGA DE DATOS
# =============================================================================

# Cargar datos en session_state si no existen
if 'df' not in st.session_state or st.session_state['df'] is None:
    with st.spinner("Cargando datos..."):
        st.session_state['df'] = load_excel_data('Base_Con_NA.xlsx')  # TODO: Cambiar ruta
        st.success("✅ Datos cargados correctamente")

df = st.session_state['df']

# =============================================================================
# SECCIÓN 1: KPIs PRINCIPALES
# =============================================================================

st.markdown('<div class="section-title">📊 Métricas Principales</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(create_metric_card(
        "Total Registros",
        f"{df.shape[0]:,}",
        "✓ Actualizado"
    ), unsafe_allow_html=True)

with col2:
    st.markdown(create_metric_card(
        "Columnas",
        df.shape[1],
        f"✓ {df.shape[1]} variables"
    ), unsafe_allow_html=True)

with col3:
    nulos = df.isnull().sum().sum()
    st.markdown(create_metric_card(
        "Valores Nulos",
        f"{nulos:,}",
        f"{(nulos/df.size*100):.1f}% del total"
    ), unsafe_allow_html=True)

with col4:
    memoria = df.memory_usage(deep=True).sum() / 1024**2
    st.markdown(create_metric_card(
        "Memoria Utilizada",
        f"{memoria:.1f} MB",
        "✓ Optimizado"
    ), unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# SECCIÓN 2: VISUALIZACIONES
# =============================================================================

st.markdown('<div class="section-title">📈 Análisis Visual</div>', unsafe_allow_html=True)

# TODO: Agregar tabs según necesites
tabs = st.tabs(["📊 Tab 1", "📈 Tab 2", "🔍 Tab 3"])

with tabs[0]:
    st.markdown(f"<h3 style='color:{colors['text_primary']};'>Gráfico de Ejemplo</h3>", unsafe_allow_html=True)
    
    # Ejemplo de gráfico
    fig = create_plotly_figure(
        title=dict(text="Mi Gráfico", font=dict(size=16)),
        height=400
    )
    
    # TODO: Agregar tus datos
    fig.add_trace(go.Bar(
        x=['A', 'B', 'C', 'D'],
        y=[10, 25, 15, 30],
        marker_color=colors['accent'],
        text=[10, 25, 15, 30],
        textposition='outside'
    ))
    
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.markdown(f"<h3 style='color:{colors['text_primary']};'>Segundo Análisis</h3>", unsafe_allow_html=True)
    # TODO: Agregar contenido

with tabs[2]:
    st.markdown(f"<h3 style='color:{colors['text_primary']};'>Tercer Análisis</h3>", unsafe_allow_html=True)
    # TODO: Agregar contenido

st.markdown("---")

# =============================================================================
# SECCIÓN 3: TABLAS Y DATOS
# =============================================================================

st.markdown('<div class="section-title">📋 Exploración de Datos</div>', unsafe_allow_html=True)

with st.expander("Ver primeras 20 filas", expanded=False):
    st.dataframe(df.head(20), use_container_width=True, height=400)

with st.expander("Estadísticas descriptivas"):
    st.dataframe(df.describe(), use_container_width=True)

st.markdown("---")

# =============================================================================
# SECCIÓN 4: ALERTAS E INFORMACIÓN
# =============================================================================

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div class="alert-box alert-success">
        <h4 style="margin-top:0; color:{colors['text_primary']};">✅ Estado del Sistema</h4>
        <p style="margin-bottom:0;">Todos los sistemas operando normalmente.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="alert-box alert-info">
        <h4 style="margin-top:0; color:{colors['text_primary']};">ℹ️ Información</h4>
        <p style="margin-bottom:0;">Los datos se actualizan automáticamente cada hora.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# FOOTER
# =============================================================================

st.markdown(f"""
<div style="text-align: center; padding: 2rem; color: {colors['text_muted']}; border-top: 1px solid {colors['border']};">
    <p><strong style="color: {colors['text_primary']};">Sistema DIMEX v1.0</strong></p>
    <p style="font-size: 0.875rem;">Desarrollado con Streamlit y Plotly</p>
</div>
""", unsafe_allow_html=True)