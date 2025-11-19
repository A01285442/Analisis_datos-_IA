
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuración de página
st.set_page_config(page_title="Dashboard de Vendedores", layout="wide")
st.title("🧾 Dashboard de Ventas por Región y Vendedor")
st.caption("Adaptado al formato de archivo de **Adrián Alejandro Galván Salgado - A01285442**")

# ---------------------- CARGA DE ARCHIVO ----------------------
with st.sidebar:
    st.header("📂 Cargar archivo CSV")
    archivo = st.file_uploader("Sube tu archivo CSV (UTF-8 o Latin-1)", type=["csv"])

def cargar_csv(archivo):
    if archivo is None:
        st.warning("Por favor, sube tu archivo CSV.")
        st.stop()
    try:
        return pd.read_csv(archivo, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(archivo, encoding="latin-1")

df = cargar_csv(archivo)

# ---------------------- LIMPIEZA Y PREPARACIÓN ----------------------
# Renombrar columnas estandarizadas
df.columns = [c.strip().title() for c in df.columns]

# Intentamos detectar las columnas esperadas
esperadas = [
    "Region", "Id", "Nombre", "Apellido", "Salario",
    "Unidades Vendidas", "Ventas Totales", "Porcentaje De Ventas"
]
# Renombramos si coinciden parcialmente
for col in df.columns:
    for exp in esperadas:
        if exp.lower().startswith(col.lower()) or col.lower().startswith(exp.lower()):
            df = df.rename(columns={col: exp})

# Fusionar nombre completo
if "Nombre" in df.columns and "Apellido" in df.columns:
    df["Vendedor"] = df["Nombre"].astype(str).str.strip() + " " + df["Apellido"].astype(str).str.strip()
else:
    df["Vendedor"] = df["Nombre"]

# Normalizamos texto
df["Region"] = df["Region"].astype(str).str.title().str.strip()

# Aseguramos tipos numéricos
for col in ["Unidades Vendidas", "Ventas Totales", "Salario"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if "Porcentaje De Ventas" not in df.columns and "Ventas Totales" in df.columns:
    df["Porcentaje De Ventas"] = (df["Ventas Totales"] / df["Ventas Totales"].sum()).round(4)

# ---------------------- FILTROS ----------------------
st.sidebar.header("🔎 Filtros")
regiones = sorted(df["Region"].dropna().unique().tolist())
region_sel = st.sidebar.multiselect("Filtrar por Región", regiones, default=regiones)
df_f = df[df["Region"].isin(region_sel)].copy()

# ---------------------- TABLA FILTRADA ----------------------
st.subheader("📋 Datos filtrados por Región")
if df_f.empty:
    st.info("No hay registros con el filtro actual.")
    st.stop()

st.dataframe(df_f, use_container_width=True)

# ---------------------- RESUMEN POR REGIÓN ----------------------
st.subheader("📑 Resumen por Región")
resumen = (
    df_f.groupby("Region", as_index=False)
    .agg({
        "Unidades Vendidas": "sum",
        "Ventas Totales": "sum",
        "Vendedor": pd.Series.nunique
    })
    .rename(columns={"Vendedor": "Vendedores Distintos"})
)
st.dataframe(resumen, use_container_width=True)
st.markdown("---")

# ---------------------- GRÁFICAS ----------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📦 Unidades Vendidas por Región")
    fig1, ax1 = plt.subplots()
    unidades = df_f.groupby("Region")["Unidades Vendidas"].sum()
    ax1.bar(unidades.index, unidades.values, color="cornflowerblue")
    ax1.set_xlabel("Región")
    ax1.set_ylabel("Unidades Vendidas")
    plt.xticks(rotation=30)
    st.pyplot(fig1)

with col2:
    st.markdown("### 💵 Ventas Totales por Región")
    fig2, ax2 = plt.subplots()
    ventas = df_f.groupby("Region")["Ventas Totales"].sum()
    ax2.bar(ventas.index, ventas.values, color="lightgreen")
    ax2.set_xlabel("Región")
    ax2.set_ylabel("Ventas Totales")
    plt.xticks(rotation=30)
    st.pyplot(fig2)

# ---------------------- PASTELES ----------------------
st.markdown("### 🥧 Porcentaje de Ventas por Región")
pie_region = df_f.groupby("Region")["Ventas Totales"].sum()
fig3, ax3 = plt.subplots()
ax3.pie(pie_region, labels=pie_region.index, autopct="%1.1f%%", startangle=90)
ax3.axis("equal")
st.pyplot(fig3)

st.markdown("### 🥧 Porcentaje de Ventas por Vendedor")
# Top 10 vendedores y agrupar el resto
top_n = 10
pie_vend = df_f.groupby("Vendedor")["Ventas Totales"].sum().sort_values(ascending=False)
otros = pie_vend[top_n:].sum()
pie_vend_top = pie_vend.head(top_n)
if otros > 0:
    pie_vend_top.loc["Otros"] = otros

fig4, ax4 = plt.subplots(figsize=(7, 7))
ax4.pie(
    pie_vend_top,
    labels=pie_vend_top.index,
    autopct="%1.1f%%",
    startangle=90
)
ax4.axis("equal")
st.pyplot(fig4)


# ---------------------- DETALLE DE VENDEDOR ----------------------
st.subheader("👤 Detalle de Vendedor Específico")
vendedores = sorted(df_f["Vendedor"].unique())
vend_sel = st.selectbox("Selecciona un vendedor", vendedores)

df_v = df_f[df_f["Vendedor"] == vend_sel]
c1, c2, c3 = st.columns(3)
c1.metric("Unidades Vendidas", int(df_v["Unidades Vendidas"].sum()))
c2.metric("Ventas Totales", f"${df_v['Ventas Totales'].sum():,.2f}")
c3.metric("Salario Promedio", f"${df_v['Salario'].mean():,.2f}")

st.write("**Registros del vendedor seleccionado:**")
st.dataframe(df_v, use_container_width=True)
