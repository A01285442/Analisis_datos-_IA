import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

# --- Configuración Inicial ---
st.set_page_config(
    page_title="Gemini Chat Dimex", 
    layout="wide"
)

# 1. Cargar variables de entorno (asumiendo que tu archivo .env tiene GEMINI_API_KEY o GOOGLE_API_KEY)
load_dotenv()

# Determinar qué variable de entorno usar
api_key = os.getenv("GEMINI_API_KEY")

# 2. Inicializar cliente Gemini
try:
    if not api_key:
        st.error("Error: La clave API (GEMINI_API_KEY) no fue encontrada en las variables de entorno.")
        st.stop()
        
    # Inicialización explícita con la clave obtenida
    client = genai.Client(api_key=api_key) 
    
except Exception as e:
    st.error(f"Error al inicializar el cliente Gemini: {e}")
    st.stop()

# --- Definición de Roles (Instrucciones del Sistema) ---

ROLES = {
    "Riesgo": "Eres un analista de riesgos experto. Tu tarea es resumir datos de portafolios y destacar cualquier anomalía o alerta de cambio significativa. Responde de forma concisa y profesional.",
    "Cobranza": "Eres un asesor de cobranza. Tu objetivo es sugerir acciones de cobro y priorizar cuentas basándote en la información proporcionada. Usa un tono motivacional y directo.",
    "Servicio": "Eres un especialista de servicio al cliente. Responde a consultas frecuentes sobre productos Dimex de manera clara, amable y precisa. Si no conoces la respuesta, indica que la buscarás.",
    "Fraude": "Eres un experto en prevención de fraude. Identifica patrones sospechosos en los datos y valida la información dinámicamente. Pide más detalles si es necesario para la validación.",
}

# --- Funciones de Lógica de la Aplicación ---

# def load_and_prepare_knowledge(uploaded_file):
#     """Carga un archivo Excel y prepara el contenido como string de contexto."""
#     if uploaded_file is not None:
#         try:
#             # Leer la primera hoja del Excel
#             df = pd.read_excel(uploaded_file)
            
#             # Convertir las primeras 10 filas del DataFrame a un formato de texto (CSV o Markdown)
#             # Esto SIMULA el proceso de RAG, donde solo inyectamos datos relevantes.
#             context_string = f"DATOS DE CONOCIMIENTO:\n\n{df.head(10).to_markdown(index=False)}"
#             st.success("Datos cargados exitosamente. El modelo usará las primeras 10 filas como contexto.")
#             return context_string
#         except Exception as e:
#             st.error(f"Error al leer el archivo Excel: {e}")
#             return ""
#     return ""

# def load_and_prepare_knowledge(uploaded_file):
#     """Carga un archivo Excel y limita el contenido para evitar sobrecarga."""
#     if uploaded_file is not None:
#         try:
#             df = pd.read_excel(uploaded_file)
            
#             # **NUEVO:** Limitar a, por ejemplo, 5 filas y las primeras 5 columnas
#             df_limited = df.head(5).iloc[:, :5] 
            
#             context_string = f"DATOS DE CONOCIMIENTO:\n\n{df_limited.to_markdown(index=False)}"
            
#             # **NUEVO:** Añadir una verificación de longitud simple (por ejemplo, 1000 caracteres)
#             if len(context_string) > 1000:
#                 st.warning("El contexto generado es muy largo. Solo se usarán los primeros 1000 caracteres.")
#                 context_string = context_string[:1000]

#             st.success("Datos cargados exitosamente y limitados para el contexto.")
#             return context_string
#         except Exception as e:
#             st.error(f"Error al leer el archivo Excel: {e}")
#             return ""
#     return ""

def load_and_prepare_knowledge(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            
            # *** CAMBIO CRUCIAL: Solo las 2 primeras filas y 3 primeras columnas ***
            df_limited = df.head(2).iloc[:, :3] 
            
            context_string = f"DATOS DE CONOCIMIENTO (LIMITADOS):\n\n{df_limited.to_markdown(index=False)}"
            
            st.success("Datos cargados exitosamente. Contexto MUY LIMITADO para prueba.")
            return context_string
        except Exception as e:
            st.error(f"Error al leer el archivo Excel: {e}")
            return ""
    return ""
def generate_response(role_key, user_prompt, context_data):
    """Genera una respuesta usando el modelo de Gemini, inyectando el rol y el contexto."""
    
    # 1. Definir la instrucción del sistema (el comportamiento de la 'Gem')
    system_instruction = ROLES.get(role_key, ROLES["Servicio"])
    
    # 2. Construir el prompt final inyectando el contexto
    full_prompt = f"{context_data}\n\n[INSTRUCCIÓN ESPECÍFICA DE LA TAREA: {role_key}]\n\nPregunta del Usuario: {user_prompt}"

    try:
        # Llamada a la API de Gemini
        response = client.models.generate_content(
            model='gemini-2.5-flash', # Un modelo rápido y eficiente
            contents=full_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )
        return response.text
    except Exception as e:
        return f"ERROR AL LLAMAR A LA API: {e}"

# --- Interfaz de Streamlit ---

st.title("🤖 Asistente Dimex Personalizado (Gem-Simulado)")
st.subheader("Selecciona un área para personalizar el comportamiento del chat.")

# Área de selección de Rol y subida de Archivo (Sidebar)
with st.sidebar:
    st.header("⚙️ Configuración del Asistente")
    
    # Selector de Rol
    selected_role = st.selectbox(
        "Selecciona el Área (Comportamiento del Chat):",
        list(ROLES.keys()),
        key="role_selector"
    )
    st.info(f"Comportamiento Actual: **{selected_role}**")
    
    st.header("📤 Cargar Conocimiento (Excel)")
    uploaded_file = st.file_uploader(
        "Sube un archivo Excel (.xlsx) para inyectar datos de conocimiento:", 
        type=["xlsx"]
    )
    
    # Cargar y preparar el contexto al cambiar el archivo
    knowledge_context = load_and_prepare_knowledge(uploaded_file)
    st.session_state["knowledge_context"] = knowledge_context


# Inicializar el historial de chat si no existe
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hola! Selecciona un área y sube un archivo (opcional) para empezar."}]

# Mostrar mensajes anteriores
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capturar la entrada del usuario
if prompt := st.chat_input("Escribe tu consulta..."):
    # Añadir el mensaje del usuario al historial
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generar la respuesta
    with st.spinner(f"El asistente de {selected_role} está pensando..."):
        # La función clave llama a generate_response
        response_text = generate_response(selected_role, prompt, st.session_state["knowledge_context"])
    
    # Mostrar la respuesta del asistente
    with st.chat_message("assistant"):
        st.markdown(response_text)
    
    # Añadir la respuesta del asistente al historial
    st.session_state["messages"].append({"role": "assistant", "content": response_text})