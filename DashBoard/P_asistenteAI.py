import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
from dotenv import load_dotenv
import numpy as np
import os
import re

# Componentes DIMEX
from utils.theme import get_theme_colors
from utils.css_manager import apply_css
from components.header import create_page_header
from utils.icons import get_icon

# =============================================================================
# RENDER PRINCIPAL DE LA PÁGINA (INTEGRADO AL ROUTER)
# =============================================================================
def render():

    # --------------------------- CONFIG DE PÁGINA -----------------------------
    st.set_page_config(
        page_title="Asistente IA - Gemini", 
        page_icon=get_icon("chatbot_gris"),
        layout="wide"
    )

    HAS_RERUN = hasattr(st, "rerun")

    # -------------------------- INICIALIZAR GEMINI ---------------------------
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    try:
        if not api_key:
            st.error("Error: No se encontró GEMINI_API_KEY.")
            st.stop()
        client = genai.Client(api_key=api_key) 
    except Exception as e:
        st.error(f"Error al inicializar Gemini: {e}")
        st.stop()

    # ------------------------- ROLES DEL ASISTENTE ---------------------------
    ROLES_DEFINITIONS = """
Eres un asistente inteligente de DIMEX que adopta diferentes roles según la consulta:

[Rol: Riesgo] → Análisis de cartera, deterioro, IMOR, alertas, tendencias.  
[Rol: Cobranza] → Estrategias, priorización, seguimientos.  
[Rol: Servicio] → Procesos, productos, información general.  
[Rol: Fraude] → Anomalías, validación sospechosa, patrones extraños.

GLOSARIO DE MÉTRICAS FINANCIERAS:
- ICV (Índice de Cartera Vencida): Porcentaje de saldo vencido respecto al saldo total. Se calcula como (Saldo_Vencido / Saldo_Actual) * 100. Un ICV alto indica mayor riesgo.
- IMOR (Índice de Morosidad): Similar al ICV, mide el porcentaje de cartera en mora. Es sinónimo de ICV en este contexto.
- ICV_Crecimiento_6M: Porcentaje de crecimiento del ICV en los últimos 6 meses. Se calcula como ((ICV_Actual - ICV_T06) / ICV_T06) * 100.
  * Valores positivos: El ICV está AUMENTANDO (⚠️ deterioro, más riesgo)
  * Valores negativos: El ICV está DISMINUYENDO (✅ mejora, menos riesgo)
  * Ejemplo: +25% significa que el ICV creció 25% en 6 meses (señal de alerta)
- FPD (First Payment Default): Porcentaje de créditos que caen en mora en su primer pago. Indicador temprano de calidad de originación.
- Ratio 30-89: Porcentaje de cartera con mora entre 30 y 89 días. Zona crítica antes de deterioro severo.
- Nivel_Riesgo: Clasificación automática basada en ICV, FPD y Ratio 30-89:
  * Riesgo Alto: 3 indicadores por encima de umbrales (ICV>5%, Ratio 30-89>3%, FPD>6%)
  * Riesgo Medio: 2 indicadores por encima de umbrales
  * Saludable: 0-1 indicadores por encima de umbrales

INSTRUCCIONES:
1. Detecta automáticamente el rol apropiado.
2. Indica el rol al inicio de la respuesta.
3. Usa SIEMPRE los datos proporcionados en el CONTEXTO DE DATOS.
4. Si una métrica aparece en los datos (como ICV, IMOR, FPD, ICV_Crecimiento_6M), úsala directamente.
5. Cuando hables de crecimiento del ICV, aclara si es positivo (deterioro) o negativo (mejora).
6. Explica de forma profesional y concisa.
7. Si realmente faltan datos críticos, solicita detalles específicos.
"""

    # ======================================================================
    # FUNCIONES DE EXTRACCIÓN / FILTRO Y ENRIQUECIMIENTO DE DATOS
    # ======================================================================

    def enrich_dataframe(df):
        """Agrega columnas calculadas de riesgo al DataFrame"""
        if df is None or df.empty:
            return df
        
        # Detectar nombres de columnas originales
        col_saldo_actual = next((c for c in df.columns if c in ['SaldoInsolutoActual', 'Saldo_Actual', 'Saldo Insoluto Actual']), None)
        col_saldo_vencido = next((c for c in df.columns if c in ['SaldoInsolutoVencidoActual', 'Saldo_Vencido', 'Saldo Insoluto Vencido']), None)
        
        # Calcular ICV (Índice de Cartera Vencida)
        if col_saldo_actual and col_saldo_vencido:
            df['ICV'] = np.where(
                df[col_saldo_actual] > 0,
                (df[col_saldo_vencido] / df[col_saldo_actual]) * 100,
                0.0
            )
        elif 'ICV_Calc' in df.columns:
            df['ICV'] = df['ICV_Calc']
        
        # Calcular ICV de hace 6 meses (T06) y el % de crecimiento
        col_saldo_t06 = next((c for c in df.columns if 'SaldoInsolutoT06' in c or 'SaldoInsolutoT6' in c), None)
        col_vencido_t06 = next((c for c in df.columns if 'SaldoInsolutoVencidoT06' in c or 'SaldoInsolutoVencidoT6' in c), None)
        
        if col_saldo_t06 and col_vencido_t06:
            # Calcular ICV de hace 6 meses
            df['ICV_T06'] = np.where(
                df[col_saldo_t06] > 0,
                (df[col_vencido_t06] / df[col_saldo_t06]) * 100,
                0.0
            )
            
            # Calcular % de Crecimiento del ICV (últimos 6 meses)
            # Fórmula: ((ICV_Actual - ICV_T06) / ICV_T06) * 100
            df['ICV_Crecimiento_6M'] = np.where(
                df['ICV_T06'] > 0,
                ((df['ICV'] - df['ICV_T06']) / df['ICV_T06']) * 100,
                0.0
            )
        else:
            # Si no hay datos T06, intentar calcular con las columnas disponibles
            # Buscar cualquier columna T06
            saldo_t06_alt = next((c for c in df.columns if 'T06' in c and 'Vencido' not in c), None)
            vencido_t06_alt = next((c for c in df.columns if 'T06' in c and 'Vencido' in c), None)
            
            if saldo_t06_alt and vencido_t06_alt:
                df['ICV_T06'] = np.where(
                    df[saldo_t06_alt] > 0,
                    (df[vencido_t06_alt] / df[saldo_t06_alt]) * 100,
                    0.0
                )
                df['ICV_Crecimiento_6M'] = np.where(
                    df['ICV_T06'] > 0,
                    ((df['ICV'] - df['ICV_T06']) / df['ICV_T06']) * 100,
                    0.0
                )
        
        # Calcular Ratio 30-89
        if 'Ratio_30_89_Calc' not in df.columns:
            col_3089 = next((c for c in df.columns if '3089' in c.lower()), None)
            if col_3089 and col_saldo_actual:
                df['Ratio_30_89_Calc'] = np.where(
                    df[col_saldo_actual] > 0,
                    (df[col_3089] / df[col_saldo_actual]) * 100,
                    0.0
                )
        
        # Calcular FPD
        if 'FPD_Calc' not in df.columns:
            col_fpd = next((c for c in df.columns if 'fpd' in c.lower() and 'actual' in c.lower()), None)
            if col_fpd:
                df['FPD_Calc'] = df[col_fpd] if df[col_fpd].mean() >= 1 else df[col_fpd] * 100
        
        # Clasificar Nivel de Riesgo
        def clasificar_riesgo(row):
            icv_val = row.get('ICV', 0)
            ratio_30_89 = row.get('Ratio_30_89_Calc', 0)
            fpd_val = row.get('FPD_Calc', 0)
            
            score = sum([
                icv_val > 5.0,
                ratio_30_89 > 3.0,
                fpd_val > 6.0
            ])
            
            if score == 3:
                return "Riesgo Alto"
            elif score == 2:
                return "Riesgo Medio"
            return "Saludable"
        
        df['Nivel_Riesgo'] = df.apply(clasificar_riesgo, axis=1)
        return df

    def generate_data_summary(df):
        summary_parts = []
        summary_parts.append(f"Total de registros: {len(df)}")

        # Agregar información de nivel de riesgo si existe
        if 'Nivel_Riesgo' in df.columns:
            risk_counts = df['Nivel_Riesgo'].value_counts()
            summary_parts.append("\nDistribución de Riesgo:")
            for risk, count in risk_counts.items():
                summary_parts.append(f"  - {risk}: {count} sucursales ({count/len(df)*100:.1f}%)")

        # Agregar promedios de métricas clave si existen
        if 'ICV' in df.columns:
            icv_avg = df['ICV'].mean() * 100
            summary_parts.append(f"\nICV Promedio: {icv_avg:.2f}%")
        if 'ICV_Calc' in df.columns:
            icv_calc_avg = df['ICV_Calc'].mean()
            summary_parts.append(f"\nICV Promedio: {icv_calc_avg:.2f}%")
        if 'FPD_Calc' in df.columns:
            fpd_avg = df['FPD_Calc'].mean()
            summary_parts.append(f"FPD Promedio: {fpd_avg:.2f}%")
        if 'Ratio_30_89_Calc' in df.columns:
            ratio_avg = df['Ratio_30_89_Calc'].mean()
            summary_parts.append(f"Ratio 30-89 Promedio: {ratio_avg:.2f}%")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if any(k in col.lower() for k in ["saldo", "capital"]):
                summary_parts.append(
                    f"\n{col}: Total=${df[col].sum():,.0f}, "
                    f"Promedio=${df[col].mean():,.0f}, "
                    f"Máximo=${df[col].max():,.0f}"
                )
        return "\n".join(summary_parts)
    def keyword_in_query(keyword, text):
    # Palabra completa: evita coincidencias dentro de otras palabras
        return re.search(rf"\b{re.escape(keyword)}\b", text) is not None

    def extract_relevant_data(df, query, max_rows=10):
        if df is None or df.empty:
            return "No hay datos disponibles."

        query_lower = query.lower()
        relevant_cols = []

        # Detectar nombres de columnas originales disponibles
        col_saldo_actual = next((c for c in df.columns if c in ['SaldoInsolutoActual', 'Saldo_Actual', 'Saldo Insoluto Actual']), None)
        col_saldo_vencido = next((c for c in df.columns if c in ['SaldoInsolutoVencidoActual', 'Saldo_Vencido', 'Saldo Insoluto Vencido']), None)

        keyword_mapping = {
            'saldo': ['Saldo Insoluto Actual', 'SaldoInsolutoActual', 'Saldo_Actual'],
            'vencido': ['Saldo Insoluto Vencido', 'SaldoInsolutoVencidoActual', 'Saldo_Vencido'],
            'sucursal': ['Sucursal', 'Region', 'Región'],
            'vendedor': ['Vendedor', 'Nombre Vendedor'],
            'capital': ['Capital Dispersado', 'CapitalDispersadoActual','CapitalLiquidadoActual'],
            'castigo': ['Castigos', 'CastigosActual'],
            'fpd': ['%FPD', 'FPD', 'FPDActual', 'FPD_Calc'],
            'imor': ['IMOR', 'Porcentaje_Vencido', 'ICV', 'ICV_Calc', 'SaldoInsolutoVencidoActual', 'SaldoInsolutoActual', 'ICV_Crecimiento_6M'],
            'icv': ['ICV', 'ICV_Calc', 'SaldoInsolutoVencidoActual', 'SaldoInsolutoActual', 'ICV_Crecimiento_6M', 'ICV_T06'],
            '3089': ['3089', 'Saldo30-89', 'Ratio_30_89_Calc'],
            'riesgo': ['Nivel_Riesgo'],
            'cluster': ['Nivel_Riesgo'],
            'crecimiento': ['ICV_Crecimiento_6M', 'ICV', 'ICV_T06'],
            'tendencia': ['ICV_Crecimiento_6M', 'ICV', 'ICV_T06'],
            'evolución': ['ICV_Crecimiento_6M', 'ICV', 'ICV_T06'],
        }
        relevant_cols = set()
        for keyword, cols in keyword_mapping.items():
            if keyword in query_lower:    # tu lógica actual
                for c in cols:
                    if c in df.columns:
                        relevant_cols.add(c) 
        relevant_cols = list(relevant_cols)

        # SIEMPRE incluir columnas de identificación
        id_cols = ['Sucursal', 'Vendedor', 'Región', 'Region', 'Nivel_Riesgo']
        relevant_cols += [c for c in id_cols if c in df.columns and c not in relevant_cols]

        # Para consultas de ICV/IMOR, FORZAR inclusión de TODAS las columnas necesarias
        if any(w in query_lower for w in ['icv', 'imor', 'morosidad', 'cartera vencida', 'crecimiento', 'tendencia', 'evolución']):
            # Lista EXPLÍCITA de columnas que DEBEN estar si existen
            required_for_icv = ['ICV', 'ICV_Calc', 'ICV_Crecimiento_6M', 'ICV_T06',
                                'SaldoInsolutoVencidoActual', 'SaldoInsolutoActual', 
                                'Saldo_Vencido', 'Saldo_Actual', 'Nivel_Riesgo', 'Sucursal', 'Region', 'Región']
            for col in required_for_icv:
                if col in df.columns and col not in relevant_cols:
                    relevant_cols.append(col)

        # Si no hay columnas relevantes, usar las importantes
        if not relevant_cols:
            important = ['Sucursal', 'Saldo Insoluto Actual', 'SaldoInsolutoActual', 'Saldo_Actual', 'Vendedor', 'Nivel_Riesgo']
            relevant_cols = [c for c in important if c in df.columns]

        try:
            df_filtered = df[relevant_cols].copy()
        except Exception as e:
            # Si falla, incluir todas las columnas
            st.warning(f"Error al filtrar columnas: {e}. Usando todas las columnas.")
            df_filtered = df.copy()

        # Ordenar por la métrica relevante si existe
        if 'ICV' in df_filtered.columns and any(w in query_lower for w in ['icv', 'imor']):
            df_filtered = df_filtered.sort_values('ICV', ascending=False)
        elif 'ICV_Calc' in df_filtered.columns and any(w in query_lower for w in ['icv', 'imor']):
            df_filtered = df_filtered.sort_values('ICV_Calc', ascending=False)
        else:
            if col_saldo_actual and col_saldo_actual in df_filtered.columns:
                df_filtered = df_filtered.sort_values(col_saldo_actual, ascending=False)

        df_filtered = df_filtered.head(max_rows)
        summary = generate_data_summary(df_filtered)

        # Agregar información sobre columnas calculadas disponibles
        available_metrics = []
        if 'ICV' in df.columns:
            available_metrics.append("ICV (Índice de Cartera Vencida) - COLUMNA PRESENTE")
        if 'ICV_Calc' in df.columns:
            available_metrics.append("ICV_Calc - COLUMNA PRESENTE")
        if 'ICV_Crecimiento_6M' in df.columns:
            available_metrics.append("ICV_Crecimiento_6M (% Crecimiento ICV últimos 6 meses) - COLUMNA PRESENTE")
        if 'FPD_Calc' in df.columns:
            available_metrics.append("FPD (First Payment Default)")
        if 'Ratio_30_89_Calc' in df.columns:
            available_metrics.append("Ratio 30-89 días")
        if 'Nivel_Riesgo' in df.columns:
            available_metrics.append("Nivel de Riesgo")

        metrics_info = f"\n\nMÉTRICAS DISPONIBLES EN LOS DATOS: {', '.join(available_metrics)}" if available_metrics else ""

        # Lista explícita de columnas presentes
        columns_present = f"\n\nCOLUMNAS PRESENTES EN LA TABLA: {', '.join(df_filtered.columns.tolist())}"

        return f"""
RESUMEN ESTADÍSTICO:
{summary}{metrics_info}{columns_present}

DATOS COMPLETOS (Top {len(df_filtered)} ordenados por relevancia):
{df_filtered.to_markdown(index=False, floatfmt=".4f")}

⚠️ INSTRUCCIÓN CRÍTICA: La tabla anterior contiene TODAS las columnas necesarias.
- Si ves la columna "ICV", úsala DIRECTAMENTE para reportar el Índice de Cartera Vencida.
- Si ves "SaldoInsolutoVencidoActual" y "SaldoInsolutoActual", puedes calcular ICV = (Vencido/Actual)*100.
- NO digas que el ICV no está disponible si la columna ICV o los datos para calcularlo están en la tabla.
- Analiza TODAS las columnas mostradas antes de responder.
"""

    def detect_intent_and_filter(query, df):
        q = query.lower()

        # Detectar nombres de columnas originales
        col_saldo_actual = next((c for c in df.columns if c in ['SaldoInsolutoActual', 'Saldo_Actual', 'Saldo Insoluto Actual']), None)
        col_saldo_vencido = next((c for c in df.columns if c in ['SaldoInsolutoVencidoActual', 'Saldo_Vencido', 'Saldo Insoluto Vencido']), None)

        # Detectar consultas sobre ICV/IMOR específicamente
        if any(w in q for w in ['icv', 'imor', 'índice de cartera vencida', 'morosidad']):
            # Asegurar que existe la columna ICV
            if 'ICV' not in df.columns and col_saldo_vencido and col_saldo_actual:
                df = df.copy()
                df['ICV'] = np.where(
                    df[col_saldo_actual] > 0,
                    (df[col_saldo_vencido] / df[col_saldo_actual]) * 100,
                    0
                )
            
            if 'ICV' in df.columns:
                if any(w in q for w in ['mayor', 'alto', 'más', 'peor', 'top']):
                    return df.nlargest(15, 'ICV')
                elif any(w in q for w in ['menor', 'bajo', 'mejor']):
                    return df.nsmallest(15, 'ICV')
                else:
                    return df.nlargest(15, 'ICV')  # Por defecto muestra los mayores
            elif 'ICV_Calc' in df.columns:
                if any(w in q for w in ['mayor', 'alto', 'más', 'peor', 'top']):
                    return df.nlargest(15, 'ICV_Calc')
                elif any(w in q for w in ['menor', 'bajo', 'mejor']):
                    return df.nsmallest(15, 'ICV_Calc')
                else:
                    return df.nlargest(15, 'ICV_Calc')

        if any(w in q for w in ['top', 'mayor', 'más alto']) and 'icv' not in q and 'imor' not in q:
            if col_saldo_actual:
                return df.nlargest(10, col_saldo_actual)

        if any(w in q for w in ['bajo', 'menor', 'peor']) and 'icv' not in q and 'imor' not in q:
            if col_saldo_actual:
                return df.nsmallest(10, col_saldo_actual)

        if any(w in q for w in ['riesgo', 'vencido', 'deterioro', 'alto riesgo']):
            # Priorizar filtrado por Nivel_Riesgo si existe
            if 'Nivel_Riesgo' in df.columns and any(w in q for w in ['alto', 'crítico', 'peligro']):
                return df[df['Nivel_Riesgo'] == 'Riesgo Alto'].head(15)
            
            # Calcular IMOR si no existe
            if col_saldo_vencido and col_saldo_actual:
                df2 = df.copy()
                df2['IMOR'] = np.where(
                    df2[col_saldo_actual] > 0,
                    (df2[col_saldo_vencido] / df2[col_saldo_actual]) * 100,
                    0
                )
                return df2.nlargest(15, 'IMOR')

        return df

    def generate_response(prompt, context):
        full_prompt = f"""
{ROLES_DEFINITIONS}

CONTEXTO DE DATOS:
{context}

PREGUNTA: {prompt}

Responde de manera clara y profesional.
"""

        try:
            result = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=2048,
                )
            )
            return result.text
        except Exception as e:
            return f"ERROR: {e}"

    # ======================================================================
    # CARGA DE DATOS
    # ======================================================================
    @st.cache_data
    def load_excel_data(path='./DashBoard/Base_Con_NA_Historico.csv'):
        try:
            df = None
            if path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(path)
            elif path.endswith('.csv'):
                for enc in ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252']:
                    try:
                        df = pd.read_csv(path, encoding=enc)
                        break
                    except:
                        pass
                if df is None:
                    raise ValueError("No encoding compatible")
            else:
                raise ValueError("Formato no soportado")
            
            # Enriquecer DataFrame con columnas calculadas
            return enrich_dataframe(df)
            
        except Exception as e:
            st.error(f"⚠️ {e}")
            return None

    # ======================================================================
    # UI DIMEX
    # ======================================================================
    apply_css("assistant")

    create_page_header(
        f"{get_icon("icon_chatbot")}Asistente Inteligente DIMEX",
        "Consulta especializada con selección automática de rol experto"
    )

    colors = get_theme_colors()

    # ------------------- Cargar Datos -------------------
    if "df" not in st.session_state:
        st.session_state["df"] = None

    with st.expander("Configuración de datos", expanded=False):
        archivo = st.text_input("Archivo:", "./DashBoard/Base_Con_NA_Historico.csv")
        if st.button("Cargar archivo"):
            with st.spinner("Cargando..."):
                df_loaded = load_excel_data(archivo)
                if df_loaded is not None:
                    st.session_state["df"] = df_loaded
                    st.success(f"✅ {len(df_loaded)} registros cargados")
                    if HAS_RERUN:
                        st.rerun()

    st.markdown("---")

    # ======================================================================
    # CHAT
    # ======================================================================

    # (opcional) Si quieres forzar a limpiar el chat cuando cambias el layout
    # puedes cambiar el número de versión
    if "assistant_version" not in st.session_state:
        st.session_state["assistant_version"] = 1

    if "messages" not in st.session_state or not st.session_state["messages"]:
        mensaje_bienvenida = f"""
<div style="font-size:0.95rem;">
    {get_icon("chatbot")}
  <div style="display:flex;align-items:center;gap:0.4rem;margin-bottom:0.6rem;">
    Hola, soy tu asistente inteligente DIMEX.
  </div>

  <p>Puedo ayudarte con:</p>
  <ul style="list-style:none;padding-left:0;margin:0;">
    <li style="display:flex;align-items:center;gap:0.35rem;">
        <span>{get_icon("metricas_principales")} Riesgo</span>
    </li>
    <li style="display:flex;align-items:center;gap:0.35rem;">
        <span>{get_icon("cobranza")} Cobranza</span>
    </li>
    <li style="display:flex;align-items:center;gap:0.35rem;">
        <span>{get_icon("servicio")} Servicio</span>
    </li>
    <li style="display:flex;align-items:center;gap:0.35rem;">
        <span>{get_icon("fraude")} Fraude</span>
    </li>
  </ul>

  <p style="margin-top:0.8rem;">Solo escribe tu consulta.</p>
</div>
"""
        st.session_state["messages"] = [
            {"role": "assistant", "content": mensaje_bienvenida}
        ]

    # Mostrar historial de mensajes
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                st.markdown(msg["content"], unsafe_allow_html=True)
            else:
                st.markdown(msg["content"])

    # Input de chat
    if prompt := st.chat_input("Escribe tu consulta..."):

        if st.session_state["df"] is None:
            st.error("⚠️ Primero carga un archivo.")
            st.stop()

        # Mensaje del usuario
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Respuesta del modelo
        with st.spinner("🤖 Analizando y extrayendo datos relevantes..."):
            df = st.session_state["df"]
            
            # DEBUG: Mostrar columnas disponibles
            
            df_filtered = detect_intent_and_filter(prompt, df)
            
            # DEBUG: Mostrar columnas después del filtro
            
            context = extract_relevant_data(df_filtered, prompt)
            
            # DEBUG: Mostrar contexto que se envía a Gemini
            
            response = generate_response(prompt, context)

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state["messages"].append({"role": "assistant", "content": response})

    st.markdown("---")
    st.caption(f" Mensajes: {len(st.session_state['messages'])} |  Rol automático activo")