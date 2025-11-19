import streamlit as st
import pandas as pd
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="Dashboard Dimex",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"  # Siempre expandido
)

# CSS personalizado estilo Dimex
st.markdown("""
    <style>
    /* Importar fuente */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Variables globales */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Ocultar elementos de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    

    [data-testid="stSidebar"] {
        min-width: 250px !important;
        width: 250px !important;
    }
    
    [data-testid="stSidebar"][aria-expanded="true"] {
        min-width: 250px !important;
        width: 250px !important;
    }
    
    [data-testid="stSidebar"][aria-expanded="false"] {
        width: 0 !important;
        min-width: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
}
    
    /* Fondo principal */
    .main {
        background-color: #F5F7FA;
    }
    
    /* Header principal con gradiente */
    .dimex-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .dimex-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    .dimex-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.95;
    }
    
    /* Tarjetas de módulos */
    .module-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        border-left: 5px solid;
        height: 100%;
    }
    
    .module-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }
    
    .module-card.dashboard {
        border-left-color: #667eea;
    }
    
    .module-card.stats {
        border-left-color: #4CAF50;
    }
    
    .module-card.ai {
        border-left-color: #764ba2;
    }
    
    .module-card h3 {
        margin: 0 0 1rem 0;
        font-size: 1.5rem;
        font-weight: 600;
        color: #2D3748;
    }
    
    .module-card p {
        color: #718096;
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    
    .module-card ul {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .module-card ul li {
        padding: 0.5rem 0;
        color: #4A5568;
        font-size: 0.95rem;
    }
    
    .module-card ul li:before {
        content: "✓";
        color: #4CAF50;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    
    /* Tarjetas de métricas */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2D3748;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    
    .metric-delta {
        font-size: 0.875rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    .delta-positive {
        color: #4CAF50;
    }
    
    .delta-negative {
        color: #EF5350;
    }
    
    /* Secciones */
    .section-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #363E66;
        margin: 2.5rem 0 1.5rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid #667eea;
    }
    
    /* Cajas de información */
    .info-box {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #42A5F5;
    }
    
    .info-box h4 {
        margin: 0 0 1rem 0;
        color: #2D3748;
        font-weight: 600;
    }
    
    .info-box ul {
        margin: 0;
        padding-left: 1.5rem;
        color: #4A5568;
        line-height: 1.8;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: white;
    }
    
    [data-testid="stSidebar"] h2 {
        color: white;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #718096;
        margin-top: 3rem;
        border-top: 2px solid #E2E8F0;
    }
    
    .footer strong {
        color: #2D3748;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .dimex-header h1 {
            font-size: 1.75rem;
        }
        
        .metric-value {
            font-size: 1.75rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Header principal
st.markdown("""
    <div class="dimex-header">
        <h1>Sistema de Análisis Predictivo Dimex</h1>
        <p>Plataforma integral para la detección y prevención de deterioro Sucursales</p>
    </div>
""", unsafe_allow_html=True)

# Sección de bienvenida
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### Bienvenido al Sistema Integrado
    
    Esta plataforma combina análisis estadístico avanzado, modelos de Machine Learning 
    y asistencia de Inteligencia Artificial para proporcionar una visión completa del 
    estado de la cartera crediticia.
    
    **Navegue por los módulos usando el menú lateral** para acceder a las diferentes 
    funcionalidades del sistema.
    """)

with col2:
    st.markdown("""
    <div class="info-box">
        <h4>Inicio Rápido</h4>
        <ul>
            <li>Explore el Dashboard</li>
            <li>Analice Estadísticas</li>
            <li>Consulte con IA</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section-title">Módulos del Sistema</div>', unsafe_allow_html=True)

# Tarjetas de módulos
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="module-card dashboard">
        <h3>📊 Dashboard</h3>
        <p>Vista general con métricas clave y resumen ejecutivo del estado de la cartera crediticia.</p>
        <ul>
            <li>Indicadores en tiempo real</li>
            <li>Alertas de riesgo</li>
            <li>Tendencias principales</li>
            <li>Semáforo de sucursales</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="module-card stats">
        <h3>📈 Estadísticas</h3>
        <p>Análisis estadístico completo y modelos predictivos de Machine Learning.</p>
        <ul>
            <li>Decision Tree optimizado</li>
            <li>Gradient Boosting</li>
            <li>Visualizaciones avanzadas</li>
            <li>Comparación de desempeño</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="module-card ai">
        <h3>🤖 Asistente IA</h3>
        <p>Chatbot inteligente con Gemini para consultas sobre los datos y análisis.</p>
        <ul>
            <li>Consultas en lenguaje natural</li>
            <li>4 roles especializados</li>
            <li>Integración con datos</li>
            <li>Análisis personalizado</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section-title"> Métricas del Sistema</div>', unsafe_allow_html=True)

# Métricas rápidas
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Modelos Disponibles</div>
        <div class="metric-value">2</div>
        <div class="metric-delta delta-positive">✓ Decision Tree + Gradient Boosting</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Mejor F1-Score</div>
        <div class="metric-value">0.87</div>
        <div class="metric-delta delta-positive">+6.1% vs baseline</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Variables Predictoras</div>
        <div class="metric-value">5</div>
        <div class="metric-delta delta-positive">✓ Alta correlación</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Precisión Promedio</div>
        <div class="metric-value">84%</div>
        <div class="metric-delta delta-positive">+5.3% optimizado</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section-title">Cómo Usar el Sistema</div>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("""
    ### Flujo de Trabajo Recomendado
    
    1. **Explore el Dashboard**
       - Revise las métricas clave
       - Identifique alertas de riesgo
       - Analice tendencias
    
    2. **Analice Estadísticas**
       - Los datos se cargan automáticamente
       - Visualice distribuciones
       - Compare modelos predictivos
       - Descargue reportes
    
    3. **Consulte con IA**
       - Seleccione un rol especializado
       - Haga preguntas específicas
       - Obtenga insights personalizados
    
    4. **Genere Reportes**
       - Exporte tablas comparativas
       - Descargue visualizaciones
       - Cree reportes ejecutivos
    """)

with col2:
    st.markdown("""
    <div class="info-box" style="border-left-color: #4CAF50;">
        <h4>✨ Características Principales</h4>
        <ul>
            <li><strong>2 Modelos Optimizados:</strong> Decision Tree y Gradient Boosting</li>
            <li><strong>Carga Automática:</strong> Datos pre-configurados</li>
            <li><strong>Visualizaciones Interactivas:</strong> Gráficos con Plotly</li>
            <li><strong>IA Integrada:</strong> Google Gemini para consultas</li>
            <li><strong>Exportación Completa:</strong> CSV, PNG y TXT</li>
        </ul>
    </div>
    
    <div class="info-box" style="margin-top: 1rem; border-left-color: #FFA726;">
        <h4>⚠️ Variables Requeridas</h4>
        <ul style="font-size: 0.85rem;">
            <li>SaldoInsolutoActual</li>
            <li>SaldoInsolutoVencidoActual</li>
            <li>CapitalDispersadoActual</li>
            <li>CapitalLiquidadoActual</li>
            <li>%FPDActual</li>
            <li>QuitasActual + CastigosActual</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section-title"> Ventajas Competitivas</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-box" style="border-left-color: #667eea;">
        <h4>⚡ Velocidad</h4>
        <p style="color: #4A5568; margin: 0;">
            Análisis completo en <strong>menos de 2 minutos</strong>. 
            Desde carga de datos hasta reportes finales.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-box" style="border-left-color: #4CAF50;">
        <h4>🎯 Precisión</h4>
        <p style="color: #4A5568; margin: 0;">
            Modelos con <strong>84% de accuracy</strong> promedio. 
            Detección temprana de deterioro crediticio.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-box" style="border-left-color: #764ba2;">
        <h4>🤖 Inteligencia</h4>
        <p style="color: #4A5568; margin: 0;">
            Asistente IA con <strong>4 roles especializados</strong>. 
            Respuestas personalizadas por área.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p><strong>Sistema de Análisis Predictivo Dimex v1.0</strong></p>
    <p>Desarrollado usando Streamlit, Scikit-learn y Google Gemini</p>
    <p style="font-size: 0.875rem; margin-top: 1rem;">© 2024 Dimex - Todos los derechos reservados</p>
</div>
""", unsafe_allow_html=True)

# Sidebar con información
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: white; border-radius: 10px; margin-bottom: 1rem;">
        <h1 style="color: #667eea; margin: 0; font-size: 1.8rem;">DIMEX</h1>
        <p style="color: #718096; margin: 0 rem 0 0 0; font-size: 0.8rem;">Sistema Predictivo</p>
        <hr style="border: none; border-top: 1px solid #E2E8F0; margin: 0.2rem 0;">
        <p style="color: #4A5568; font-size: 0.75rem; margin: 0;">Dashboard v1.0</p>
    </div>
""", unsafe_allow_html=True)

    
    st.markdown("### Navegación")
    st.markdown("""
    <div style="background: white; padding: 1rem; border-radius: 8px; color: #2D3748;">
        <p style="margin: 0 0 0.5rem 0; font-weight: 600;">Selecciona un módulo:</p>
        <ul style="margin: 0; padding-left: 1.5rem; line-height: 2;">
            <li><strong>Dashboard</strong></li>
            <li><strong>Estadísticas</strong></li>
            <li><strong>Asistente IA</strong></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📚 Recursos")
    st.markdown("""
    <div style="background: white; padding: 1rem; border-radius: 8px; color: #2D3748;">
        <ul style="margin: 0; padding-left: 1.5rem; line-height: 2;">
            <li><a href="#" style="color: #667eea; text-decoration: none;">📖 Documentación</a></li>
            <li><a href="#" style="color: #667eea; text-decoration: none;">🔧 GitHub</a></li>
            <li><a href="#" style="color: #667eea; text-decoration: none;">💬 Soporte</a></li>
            <li><a href="#" style="color: #667eea; text-decoration: none;">📧 Contacto</a></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)