
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Configuración de la app
st.set_page_config(page_title="Chat con Gemini", page_icon="🌈", layout="centered")
st.title("🌈 Chat con Gemini (Streamlit + Google AI)")

# Inicializar historial
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial previo
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Entrada del usuario
if prompt := st.chat_input("Escribe tu mensaje..."):
    # Mostrar mensaje del usuario
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generar respuesta con Gemini
    model = genai.GenerativeModel("gemini-2.5-flash")

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            response = model.generate_content(prompt)
            answer = response.text
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
