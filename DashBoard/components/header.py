import streamlit as st
from utils.theme import get_theme_colors

def create_page_header(title, subtitle):
    """Crea un header consistente para todas las páginas"""
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
            <p style="margin: 0.25rem 0 0 0; color: #f0f0f0 !important; opacity: 0.95;">{subtitle}</p>
        </div>
    """, unsafe_allow_html=True)