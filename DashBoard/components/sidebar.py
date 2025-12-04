import streamlit as st
import pandas as pd
import numpy as np
from utils.theme import get_theme_colors
from utils.icons import get_icon
from urllib.parse import quote

# Variables Necesarias
HAS_RERUN = hasattr(st, "rerun")

def svg_to_data_uri(svg: str) -> str:
    svg_clean = svg.replace("\n", "").replace('"', "'")
    return "data:image/svg+xml;utf8," + quote(svg_clean)

def get_clean_icon(name: str) -> str:
    """
    Toma el icono desde utils.icons.get_icon(name)
    y devuelve solo el fragmento <svg>...</svg>.
    Así evitamos spans/divs o texto extra que rompa el HTML.
    """
    raw = get_icon(name)
    if not raw:
        return ""

    start = raw.find("<svg")
    end = raw.rfind("</svg>")
    if start != -1 and end != -1:
        return raw[start:end + len("</svg>")]

    # Si no encuentra svg, devolvemos tal cual (por si algunos ya están limpios)
    return raw

# =============================================================================
# PANTALLAS MODALES (NOTIFICACIONES, USUARIO, CONFIGURACIÓN)
# =============================================================================
def _safe_ratio_notif(numerador, denominador):
    """Calcula ratio seguro evitando división por cero"""
    num = pd.to_numeric(numerador, errors='coerce')
    den = pd.to_numeric(denominador, errors='coerce')
    return np.where((den == 0) | (pd.isna(den)) | (pd.isna(num)), np.nan, num / den)

def _create_target_for_notifications(df_notif: pd.DataFrame):
    """Crea variable de deterioro crediticio para notificaciones"""
    df_temp = df_notif.copy()

    if 'SaldoInsolutoVencidoActual' in df_temp.columns and 'SaldoInsolutoActual' in df_temp.columns:
        df_temp['ICV'] = _safe_ratio_notif(df_temp['SaldoInsolutoVencidoActual'], df_temp['SaldoInsolutoActual'])

    if '%FPDActual' in df_temp.columns:
        df_temp['FPD_Actual'] = pd.to_numeric(df_temp['%FPDActual'], errors='coerce')

    col_3089 = None
    for col in df_temp.columns:
        if '3089' in str(col) and 'Actual' in str(col):
            col_3089 = col
            break
    if col_3089 and 'SaldoInsolutoActual' in df_temp.columns:
        df_temp['Ratio_30_89'] = _safe_ratio_notif(df_temp[col_3089], df_temp['SaldoInsolutoActual'])

    def create_target_internal(df_):
        cond1 = (df_['ICV'] > 0.05) if 'ICV' in df_.columns else pd.Series([False] * len(df_))
        cond2 = (df_['Ratio_30_89'] > 0.03) if 'Ratio_30_89' in df_.columns else pd.Series([False] * len(df_))
        cond3 = (df_['FPD_Actual'] > 0.06) if 'FPD_Actual' in df_.columns else pd.Series([False] * len(df_))
        target_arr = (cond1 & cond2 & cond3).astype(float)
        return pd.Series(target_arr.values, index=df_.index)

    try:
        return create_target_internal(df_temp)
    except Exception:
        return pd.Series([np.nan] * len(df_notif), index=df_notif.index)

def get_risk_counts_for_notifications(df: pd.DataFrame):
    """Devuelve conteos de riesgo (Rojo/Amarillo/Verde)"""
    total = len(df)
    if total == 0:
        return {"rojo": (0, 0.0), "amarillo": (0, 0.0), "verde": (0, 0.0), "total": 0}

    if 'Deterioro_Crediticio' in df.columns:
        target = pd.to_numeric(df['Deterioro_Crediticio'], errors='coerce')
    else:
        target = _create_target_for_notifications(df)

    rojo = int((target == 1).sum())
    verde = int((target == 0).sum())
    amarillo = int(target.isna().sum())

    def pct(x):
        return float(x) * 100.0 / total if total > 0 else 0.0

    return {
        "rojo": (rojo, pct(rojo)),
        "amarillo": (amarillo, pct(amarillo)),
        "verde": (verde, pct(verde)),
        "total": total
    }

# --- PANTALLA NOTIFICACIONES ---
def render_notifications_modal():
    if st.session_state.get("show_notifications_modal", False):
        colors = get_theme_colors()
    
        if 'df' in st.session_state and st.session_state['df'] is not None:
            stats = get_risk_counts_for_notifications(st.session_state['df'])
            (rojo_count, rojo_pct) = stats["rojo"]
            (verde_count, verde_pct) = stats["verde"]
        else:
            rojo_count, rojo_pct = 0, 0.0
            verde_count, verde_pct = 0, 0.0

        st.markdown("<br>", unsafe_allow_html=True)
        col_left, col_center, col_right = st.columns([1, 2, 1])

        with col_center:
            st.markdown(f"""
            <div style="
                background:{colors['bg_card']};
                border-radius:12px;
                border:1px solid {colors['border']};
                padding:1.5rem 2rem;
                box-shadow:0 8px 24px rgba(0,0,0,0.4);
                margin-bottom:1.5rem;
                text-align:center;
            ">
                <h2 style="margin-top:0; color:{colors['text_primary']};">
                    🔔 Centro de Notificaciones
                </h2>
                <p style="margin:0; color:{colors['text_secondary']}; font-size:0.95rem;">
                    Resumen de sucursales en riesgo y niveles de salud del portafolio
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="
                background:{colors['bg_card']};
                border-radius:10px;
                border-left:5px solid {colors['error']};
                padding:0.9rem 1.2rem;
                margin-bottom:0.8rem;
                color:{colors['text_primary']};
            ">
                <strong>🔴 Sucursales en riesgo (Rojo):</strong><br>
                {rojo_count} sucursal(es), equivalentes al {rojo_pct:.1f}% del portafolio.
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="
                background:{colors['bg_card']};
                border-radius:10px;
                border-left:5px solid {colors['success']};
                padding:0.9rem 1.2rem;
                margin-bottom:0.8rem;
                color:{colors['text_primary']};
            ">
                <strong>🟢 Sucursales sanas (Verde):</strong><br>
                {verde_count} sucursal(es), equivalentes al {verde_pct:.1f}% del portafolio.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Cerrar notificaciones", use_container_width=True, key="close_notif"):
                st.session_state["show_notifications_modal"] = False
                if HAS_RERUN:
                    st.rerun()

        st.stop()

# --- PANTALLA USUARIO ---
def render_user_modal():
    if st.session_state.get("show_user_modal", False):
        colors = get_theme_colors()

        icon = get_clean_icon("usuario")
        if "<svg" in icon:
            icon = icon.replace(
                "<svg",
                "<svg style='width:24px; height:24px; display:block;'",
                1
            )

        st.markdown("<br>", unsafe_allow_html=True)
        col_left, col_center, col_right = st.columns([1, 2, 1])

        with col_center:
            st.markdown(f"""
            <div style="
                background:{colors['bg_card']};
                border-radius:12px;
                border:1px solid {colors['border']};
                padding:1.5rem 2rem;
                box-shadow:0 8px 24px rgba(0,0,0,0.4);
                margin-bottom:1.5rem;
            ">
                <div style="
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    gap:0.75rem;
                ">
                    <span style="
                        display:inline-flex;
                        align-items:center;
                        justify-content:center;
                        width:32px;
                        height:32px;
                    ">
                        {icon}
                    </span>
                    <h2 style="margin:0; color:{colors['text_primary']};">
                        Perfil de Usuario
                    </h2>
                </div>
            </div>
            """, unsafe_allow_html=True)

            avatar_col, form_col = st.columns([1, 2])

            with avatar_col:
                icon_svg = get_icon("usuario")
                icon_svg = icon_svg.replace("\n", " ")

                if "<svg" in icon_svg:
                    icon_svg = icon_svg.replace(
                        "<svg",
                        "<svg style='width:64px; height:64px; display:block; margin:auto;'",
                        1
                    )

                st.markdown(f"""
                <div style="
                    display:flex;
                    justify-content:center;
                    align-items:center;
                    margin-top:0.5rem;
                ">
                    <div style="
                        width:120px;
                        height:120px;
                        border-radius:50%;
                        background:{colors['bg_secondary']};
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        box-shadow:0 4px 15px rgba(0,0,0,0.4);
                    ">
                        {icon_svg}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(
                    f"<p style='text-align:center; margin-top:0.6rem; font-weight:600; color:{colors['text_primary']};'>"
                    f"{st.session_state.get('user_name', 'Usuario DIMEX')}</p>"
                    f"<p style='text-align:center; margin-top:-0.4rem; font-size:0.85rem; color:{colors['text_muted']};'>"
                    f"{st.session_state.get('user_email', 'usuario@dimex.com')}</p>",
                    unsafe_allow_html=True
                )

            with form_col:
                st.markdown("### Datos de la cuenta")
                st.text_input("Nombre", key="user_name")
                st.text_input("Correo electrónico", key="user_email")
                st.text_input("Contraseña", key="user_password", type="password")

                if st.button("Guardar cambios", use_container_width=True, key="save_user"):
                    st.success("✅ Datos de usuario actualizados")

                st.markdown("<br>", unsafe_allow_html=True)

                if st.button("Cerrar usuario", use_container_width=True, key="close_user"):
                    st.session_state["show_user_modal"] = False
                    if HAS_RERUN:
                        st.rerun()

        st.stop()

# --- PANTALLA CONFIGURACIÓN ---
def render_config_modal():
    if st.session_state.get("show_config_modal", False):
        colors = get_theme_colors()

        st.markdown("<br>", unsafe_allow_html=True)
        col_left, col_center, col_right = st.columns([1, 2, 1])

        with col_center:
            st.markdown(f"""
            <div style="
                background:{colors['bg_card']};
                border-radius:12px;
                border:1px solid {colors['border']};
                padding:1.5rem 2rem;
                box-shadow:0 8px 24px rgba(0,0,0,0.4);
                margin-bottom:1.5rem;
                text-align:center;
            ">
               <h1 style="margin-top:0; color:{colors['accent']} !important;">{get_icon("configuracion")} Configuración del Sistema</h1> 
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### Tema del sistema")

            theme_current = st.session_state.get("theme_option", "Oscuro")
            theme_selected = st.radio(
                "Tema",
                ["Oscuro", "Claro"],
                index=0 if theme_current == "Oscuro" else 1,
                key="theme_selector"
            )

            st.session_state["theme_option"] = theme_selected

            new_theme = "dark" if theme_selected == "Oscuro" else "light"
            if st.session_state.get("theme") != new_theme:
                st.session_state["theme"] = new_theme
                st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Cerrar configuración", use_container_width=True, key="close_config"):
                st.session_state["show_config_modal"] = False
                if HAS_RERUN:
                    st.rerun()

        st.stop()

def render_dimex_modals():
    """Renderiza las pantallas modales si están activas."""
    
    if st.session_state.get("show_notifications_modal", False):
        render_notifications_modal()
        st.stop()

    if st.session_state.get("show_user_modal", False):
        render_user_modal()
        st.stop()

    if st.session_state.get("show_config_modal", False):
        render_config_modal()
        st.stop()