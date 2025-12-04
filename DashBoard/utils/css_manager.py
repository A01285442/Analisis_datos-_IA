import streamlit as st
from pathlib import Path
from utils.theme import get_theme_colors

# Obtener la ruta correcta de la carpeta css
CSS_FOLDER = Path(__file__).parent.parent / "css"

def load_css_file(filename: str, colors: dict) -> str:
    """Carga y procesa un archivo CSS con tokens de color"""
    path = CSS_FOLDER / filename
    
    if not path.exists():
        st.error(f"⚠️ Archivo CSS no encontrado: {path}")
        st.warning(f"Ruta buscada: {path.absolute()}")
        return ""
    
    try:
        # Leer eliminando BOM
        css = path.read_text(encoding="utf-8").lstrip("\ufeff").strip()

        # Reemplazar tokens
        for key, value in colors.items():
            css = css.replace(f"{{{key}}}", value)

        return css  # Retornar sin tags <style>, lo agregamos después
    except Exception as e:
        st.error(f"Error leyendo CSS {filename}: {e}")
        return ""

def apply_css(section: str = "template"):
    """
    Aplica CSS global + CSS específico por sección.
    Secciones: template, assistant, statistics
    """
    colors = get_theme_colors()

    # PRIMERO: Forzar background del contenedor principal
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-color: {colors['bg_primary']} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # CSS base (siempre se aplica)
    css_base = load_css_file("base.css", colors)

    # CSS específico por sección
    section_map = {
        "dashboard": "dashboard.css",
        "template": "template.css",
        "assistant": "assistant.css",
        "statistics": "statistics.css",
        "sucursales": "sucursales.css",  
        "vendedores": "vendedores.css",
    }

    css_section = load_css_file(section_map.get(section, "template.css"), colors)

    # Combinar todo el CSS
    combined_css = f"""
    <style>
    {css_base}
    
    {css_section}
    </style>
    """

    # Inyectar en Streamlit
    st.markdown(combined_css, unsafe_allow_html=True)


## ✅ **Solución 2: Verificar estructura de carpetas**