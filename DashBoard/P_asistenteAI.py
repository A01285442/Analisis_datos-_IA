import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

# =============================================================================
# CONFIGURACIÓN DE PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Asistente IA - Gemini", 
    page_icon="🤖",
    layout="wide"
)

# =============================================================================
# SISTEMA DE TEMAS
# =============================================================================

def get_theme_colors():
    """Retorna colores según el tema activo (Verde/Oscuro y Verde/Claro)"""
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
            'accent': '#63AB32',
            'accent_secondary': '#2FEB00',
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
            'accent': '#2FEB00',
            'accent_secondary': '#2FEB00',
            'success': '#16a34a',
            'warning': '#ca8a04',
            'error': '#dc2626',
            'grid': '#e2e8f0'
        }

def apply_custom_css():
    """Aplica CSS dinámico según el tema"""
    colors = get_theme_colors()
    
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
    
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {{
        color: white !important;
    }}

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
    
    /* =============== CHAT MESSAGES =============== */
    .stChatMessage {{
        background-color: {colors['bg_card']} !important;
        border: 1px solid {colors['border']} !important;
        border-radius: 10px !important;
        color: {colors['text_primary']} !important;
    }}
    
    /* Forzar color de texto dentro de mensajes */
    .stChatMessage p, .stChatMessage span, .stChatMessage div {{
        color: {colors['text_primary']} !important;
    }}
    
    /* =============== CHAT INPUT =============== */
    .stChatInputContainer {{
        background-color: {colors['bg_secondary']} !important;
        border: 1px solid {colors['border']} !important;
        border-radius: 10px !important;
    }}
    
    .stChatInput textarea {{
        background-color: {colors['bg_secondary']} !important;
        color: {colors['text_primary']} !important;
        border: 1px solid {colors['border']} !important;
    }}
    
    /* Placeholder del chat input */
    .stChatInput textarea::placeholder {{
        color: {colors['text_muted']} !important;
        opacity: 0.7;
    }}
    
    /* Forzar color en el área de escritura */
    [data-testid="stChatInput"] textarea {{
        background-color: {colors['bg_secondary']} !important;
        color: {colors['text_primary']} !important;
    }}
    
    [data-testid="stChatInput"] {{
        background-color: {colors['bg_secondary']} !important;
    }}
    
    /* =============== FILE UPLOADER =============== */
    .stFileUploader {{
        background-color: {colors['bg_card']} !important;
        border: 1px dashed {colors['border']} !important;
        border-radius: 8px !important;
    }}
    
    .stFileUploader label {{
        color: {colors['text_primary']} !important;
    }}
    
    /* =============== SELECTBOX =============== */
    .stSelectbox div {{
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
# SIDEBAR UNIFICADO
# =============================================================================

def create_dimex_sidebar():
    """Sidebar estándar de DIMEX"""
    colors = get_theme_colors()
    
    with st.sidebar:
        # Logo
        st.markdown(f"""
        <div class="dimex-sidebar-card" style="text-align: center; padding: 1rem; 
             background: {colors['bg_card']}; 
             border-radius: 10px; margin-bottom: 1rem;">
            <h1 style="color: {colors['accent']} !important; margin: 0; font-size: 1.8rem;">DIMEX</h1>
            <p style="color: {colors['text_muted']} !important; margin: 0.2rem 0 0 0; font-size: 0.8rem;">Sistema Predictivo</p>
            <hr style="border: none; border-top: 1px solid {colors['border']}; margin: 0.2rem 0;">
            <p style="color: {colors['text_secondary']} !important; font-size: 0.75rem; margin: 0;">Asistente IA</p>
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
        
        # Navegación
        st.markdown("### 📚 Navegación")
        st.markdown(f"""
        <div class="dimex-sidebar-card" style="background: {colors['bg_card']};
             padding: 1rem; border-radius: 8px;">
            <ul style="margin: 0; padding-left: 1.5rem; line-height: 2;">
                <li><strong style="color: {colors['text_primary']} !important;">Home</strong></li>
                <li><strong style="color: {colors['text_primary']} !important;">Dashboard</strong></li>
                <li><strong style="color: {colors['text_primary']} !important;">Estadísticas</strong></li>
                <li><strong style="color: {colors['accent']} !important;">→ Asistente IA</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Información de datos cargados
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
                    <strong>Contexto:</strong> 
                    <span style="color: {colors['success']} !important;">✓ Activo</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
        
        # Recursos
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
# HEADER
# =============================================================================

def create_page_header(title, subtitle):
    """Header consistente"""
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
# INICIALIZACIÓN DE GEMINI
# =============================================================================

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

try:
    if not api_key:
        st.error("Error: La clave API (GEMINI_API_KEY) no fue encontrada en las variables de entorno.")
        st.stop()
    client = genai.Client(api_key=api_key) 
except Exception as e:
    st.error(f"Error al inicializar el cliente Gemini: {e}")
    st.stop()

# =============================================================================
# ROLES DEL ASISTENTE
# =============================================================================

ROLES = {
    "Riesgo": "Eres un analista de riesgos experto. Tu tarea es resumir datos de portafolios y destacar cualquier anomalía o alerta de cambio significativa. Responde de forma concisa y profesional.",
    "Cobranza": "Eres un asesor de cobranza. Tu objetivo es sugerir acciones de cobro y priorizar cuentas basándote en la información proporcionada. Usa un tono motivacional y directo.",
    "Servicio": "Eres un especialista de servicio al cliente. Responde a consultas frecuentes sobre productos Dimex de manera clara, amable y precisa. Si no conoces la respuesta, indica que la buscarás.",
    "Fraude": "Eres un experto en prevención de fraude. Identifica patrones sospechosos en los datos y valida la información dinámicamente. Pide más detalles si es necesario para la validación.",
}

# =============================================================================
# CARGA DE DATOS (IGUAL QUE ESTADÍSTICAS)
# =============================================================================

@st.cache_data
def load_excel_data(file_path='Base_Con_NA.xlsx'):
    """Carga datos desde Excel/CSV con manejo de errores (igual que estadísticas)"""
    try:
        if file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
            return df
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
        return None
    except Exception as e:
        st.error(f"⚠️ Error al cargar: {e}")
        return None

def prepare_knowledge_context(df):
    """Prepara el contexto desde el DataFrame completo"""
    if df is not None:
        # Usar las primeras 10 filas y todas las columnas relevantes
        df_limited = df.head(10)
        context_string = f"DATOS DE CONOCIMIENTO:\n\n{df_limited.to_markdown(index=False)}"
        return context_string
    return ""

def generate_response(role_key, user_prompt, context_data):
    """Genera respuesta con Gemini"""
    system_instruction = ROLES.get(role_key, ROLES["Servicio"])
    full_prompt = f"{context_data}\n\n[INSTRUCCIÓN: {role_key}]\n\nPregunta del Usuario: {user_prompt}"

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=full_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )
        return response.text
    except Exception as e:
        return f"❌ ERROR AL LLAMAR A LA API: {e}"

# =============================================================================
# APLICAR TEMA Y SIDEBAR
# =============================================================================

apply_custom_css()
create_dimex_sidebar()
create_page_header(
    "🤖 Asistente Inteligente con Gemini",
    "Consulta especializada por área con IA generativa"
)

colors = get_theme_colors()

# =============================================================================
# CARGAR DATOS AL INICIO (IGUAL QUE ESTADÍSTICAS)
# =============================================================================

# Cargar datos en session_state si no existe
if 'df' not in st.session_state or st.session_state['df'] is None:
    with st.expander("⚙️ Opciones de carga (ruta del archivo)", expanded=False):
        archivo_input = st.text_input(
            "Ruta del archivo (ej: Base_Con_NA.xlsx)", 
            value="Base_Con_NA.xlsx",
            key="archivo_path"
        )
        cargar_btn = st.button("🔁 Cargar/Recargar archivo", key="recargar_datos")

    # Cargar automáticamente o con botón
    if cargar_btn or ('df' not in st.session_state):
        with st.spinner("Cargando datos..."):
            archivo = archivo_input if archivo_input else "Base_Con_NA.xlsx"
            df_loaded = load_excel_data(archivo)
            if df_loaded is not None:
                st.session_state['df'] = df_loaded
                st.session_state['knowledge_context'] = prepare_knowledge_context(df_loaded)
                st.success(f"✅ Archivo cargado: {archivo}")
            else:
                st.session_state['df'] = None
                st.session_state['knowledge_context'] = ""
else:
    # Ya está cargado, asegurar contexto
    if 'knowledge_context' not in st.session_state or not st.session_state['knowledge_context']:
        st.session_state['knowledge_context'] = prepare_knowledge_context(st.session_state['df'])

st.markdown("---")

# =============================================================================
# CONFIGURACIÓN DEL ASISTENTE
# =============================================================================

st.markdown('<div class="section-title">⚙️ Configuración del Asistente</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f"""
    <div class="alert-box alert-info">
        <h4 style="margin-top:0; color:{colors['text_primary']};">ℹ️ Instrucciones</h4>
        <ol style="margin-bottom:0; color:{colors['text_secondary']};">
            <li>Selecciona un <strong>área especializada</strong> (Riesgo, Cobranza, Servicio o Fraude)</li>
            <li>Opcionalmente, sube un archivo Excel para agregar contexto</li>
            <li>Escribe tu consulta en el chat</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Selector de Rol
    selected_role = st.selectbox(
        "📋 Selecciona el Área:",
        list(ROLES.keys()),
        key="role_selector"
    )
    
    st.markdown(f"""
    <div class="metric-card" style="margin-top:1rem;">
        <div class="metric-label">Rol Activo</div>
        <div class="metric-value" style="font-size:1.2rem;">{selected_role}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Información adicional sobre el archivo cargado
if st.session_state.get('df') is not None:
    with st.expander("📊 Información del archivo cargado", expanded=False):
        df = st.session_state['df']
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Registros</div>
                <div class="metric-value">{df.shape[0]:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Columnas</div>
                <div class="metric-value">{df.shape[1]}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Contexto</div>
                <div class="metric-value">Activo</div>
                <div style="font-size:0.85rem; color:{colors['success']}; margin-top:6px;">✓ 10 filas disponibles</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("**Vista previa de datos:**")
        st.dataframe(df.head(5), use_container_width=True, height=200)

st.markdown("---")

# Opción para cargar archivo adicional (opcional)
with st.expander("📤 Cargar archivo adicional (opcional)", expanded=False):
    st.markdown(f"""
    <div class="alert-box alert-info">
        <p style="margin:0; color:{colors['text_primary']};">
            Si deseas agregar contexto adicional, puedes subir otro archivo Excel aquí. 
            El sistema ya tiene cargado el archivo principal automáticamente.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Archivo Excel adicional (.xlsx):",
        type=["xlsx"],
        help="Este archivo complementará el contexto existente",
        key="archivo_adicional"
    )
    
    if uploaded_file:
        try:
            df_extra = pd.read_excel(uploaded_file)
            extra_context = f"\n\nDATOS ADICIONALES:\n\n{df_extra.head(5).to_markdown(index=False)}"
            st.session_state['knowledge_context'] += extra_context
            st.success(f"✅ Archivo adicional cargado: {uploaded_file.name}")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

st.markdown("---")

# =============================================================================
# CHAT INTERFACE
# =============================================================================

st.markdown('<div class="section-title">💬 Conversación</div>', unsafe_allow_html=True)

# Inicializar historial
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": f"¡Hola! Soy tu asistente de **{selected_role}**. ¿En qué puedo ayudarte hoy?"}
    ]

# Mostrar mensajes
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input del usuario
if prompt := st.chat_input("💭 Escribe tu consulta aquí..."):
    # Agregar mensaje del usuario
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generar respuesta
    with st.spinner(f"🤔 El asistente de {selected_role} está procesando..."):
        response_text = generate_response(
            selected_role, 
            prompt, 
            st.session_state.get("knowledge_context", "")
        )
    
    # Mostrar respuesta
    with st.chat_message("assistant"):
        st.markdown(response_text)
    
    # Agregar respuesta al historial
    st.session_state["messages"].append({"role": "assistant", "content": response_text})

# =============================================================================
# INFORMACIÓN ADICIONAL
# =============================================================================

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Roles Disponibles</div>
        <div class="metric-value">4</div>
        <div style="font-size:0.85rem; color:{colors['success']}; margin-top:6px;">✓ Especializados</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Modelo IA</div>
        <div class="metric-value" style="font-size:1.1rem;">Gemini 2.5</div>
        <div style="font-size:0.85rem; color:{colors['success']}; margin-top:6px;">✓ Flash</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    mensajes = len(st.session_state.get("messages", []))
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Mensajes</div>
        <div class="metric-value">{mensajes}</div>
        <div style="font-size:0.85rem; color:{colors['text_muted']}; margin-top:6px;">En esta sesión</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    tiene_contexto = bool(st.session_state.get("knowledge_context"))
    contexto = "✓ Cargado" if tiene_contexto else "Sin datos"
    color_contexto = colors['success'] if tiene_contexto else colors['warning']
    
    # Mostrar cantidad de registros si hay datos
    if st.session_state.get('df') is not None:
        num_registros = len(st.session_state['df'])
        detalle = f"{num_registros:,} registros"
    else:
        detalle = "No disponible"
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Contexto de Datos</div>
        <div class="metric-value" style="font-size:1.1rem;">{contexto}</div>
        <div style="font-size:0.85rem; color:{color_contexto}; margin-top:6px;">{detalle}</div>
    </div>
    """, unsafe_allow_html=True)