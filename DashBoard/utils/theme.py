import streamlit as st
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
            'bg_sidebar': '#111824',
            'bg_sidebar_button': '#1B2332',
            'bg_hover': '#263044',
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
            'bg_hover': '#0f1a2b', 
            'bg_sidebar':'#F3F5F7',
            'bg_sidebar_button': '#E9ECF0',
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
