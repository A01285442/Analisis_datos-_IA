import streamlit as st
from components.sidebar import render_dimex_modals
from utils.icons import get_icon
from utils.theme import get_theme_colors

# Importa tus páginas
import P_vendedores
import P_sucursales
import P_dashboard
import P_estadistica
import P_asistenteAI

# -----------------------------
# Sidebar con navegación integrada
# -----------------------------
navegacion = get_icon("navegacion")
def sidebar():
    """Crea el sidebar de DIMEX con navegación integrada"""
    colors = get_theme_colors()
    
    with st.sidebar:
        # Logo DIMEX
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
        
        st.markdown("---")
        
        # NAVEGACIÓN (después del logo, antes de datos y botones)
        st.markdown(f"###{navegacion} Navegación", unsafe_allow_html=True)
        menu = st.radio(
            "Selecciona una página",
            [
                "Dashboard",
                "Estadística",
                "Sucursales",
                "Asistente IA",
            ],
            label_visibility="collapsed"
        )
        
        
        # Información de datos cargados
        if 'df' in st.session_state and st.session_state['df'] is not None:
            from utils.icons import get_icon
            df = st.session_state['df']
            st.markdown(f"### {get_icon('datos_cargados')} Datos Cargados", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="dimex-sidebar-card" style="background: {colors['bg_card']}; 
                 padding: 1rem; border-radius: 8px;">
                <p style="margin: 0.5rem 0; color: {colors['text_secondary']} !important;">
                    <strong>Registros:</strong> 
                    <span class="value-highlight" style="color:{colors['accent']}; font-weight:700;">{df.shape[0]:,}</span>
                </p>
                <p style="margin: 0.5rem 0; color: {colors['text_secondary']} !important;">
                    <strong>Columnas:</strong> 
                    <span class="value-highlight" style="color:{colors['accent']}; font-weight:700;">{df.shape[1]}</span>
                </p>
                <p style="margin: 0.5rem 0; color: {colors['text_secondary']} !important;">
                    <strong>Nulos:</strong> 
                    <span class="value-highlight" style="color:{colors['accent']}; font-weight:700;">{df.isnull().sum().sum():,}</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # BOTONES DE ACCIÓN (al final)
        HAS_RERUN = hasattr(st, "rerun")

        if st.button("Usuario", use_container_width=True, key="btn_user"):
            st.session_state["show_user_modal"] = True
            st.session_state["show_config_modal"] = False
            st.session_state["show_notifications_modal"] = False
            if HAS_RERUN:
                st.rerun()

        if st.button("Configuración", use_container_width=True, key="btn_config"):
            st.session_state["show_config_modal"] = True
            st.session_state["show_user_modal"] = False
            st.session_state["show_notifications_modal"] = False
            if HAS_RERUN:
                st.rerun()
    
    return menu

# -----------------------------
# Router: carga la página seleccionada
# -----------------------------
def router(page):
    if page == "Dashboard":
        P_dashboard.render()
    elif page == "Estadística":
        P_estadistica.render()
    elif page == "Vendedores":
        P_vendedores.render()
    elif page == "Sucursales":
        P_sucursales.render()
    elif page == "Asistente IA":
        P_asistenteAI.render()
    else:
        st.error("Página no encontrada")

# -----------------------------
# MAIN APP
# -----------------------------
def main():
    st.set_page_config(page_title="DIMEX", layout="wide")
    
    # Inicializar tema si no existe
    if 'theme' not in st.session_state:
        st.session_state['theme'] = 'dark'
    
    # Renderizar modales primero (si están activos, detienen la ejecución)
    render_dimex_modals()
    
    # Crear sidebar y obtener página seleccionada
    page = sidebar()
    
    # Renderizar la página
    router(page)

# Ejecutar app
if __name__ == "__main__":
    main()