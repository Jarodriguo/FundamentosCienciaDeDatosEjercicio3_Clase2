"""
Dashboard de Streamlit: Texto -> Tabla -> EDA
-----------------------------------------------
Recibe un párrafo con cifras, usa un LLM (API de Groq) para extraer
los datos en formato tabular y genera un análisis exploratorio (EDA)
con gráficos hechos únicamente en seaborn.

Ejecutar con:
    streamlit run app.py
"""

import io
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

try:
    from groq import Groq
except ImportError:
    Groq = None


# --------------------------------------------------------------------------- #
# Configuración general de la página
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Texto → Tabla → EDA",
    page_icon="📊",
    layout="wide",
)

# Paleta de colores personalizada (tonos teal / coral / índigo)
PALETA = ["#0F766E", "#F59E0B", "#6366F1", "#EF4444", "#10B981",
          "#8B5CF6", "#EC4899", "#14B8A6", "#F97316", "#3B82F6"]
COLOR_PRINCIPAL = "#0F766E"
COLOR_ACENTO = "#F59E0B"

# Tema global de seaborn
sns.set_theme(style="whitegrid", font_scale=1.05)
sns.set_palette(sns.color_palette(PALETA))
plt.rcParams.update({
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#E2E8F0",
    "axes.titlecolor": "#0F172A",
    "axes.labelcolor": "#334155",
    "xtick.color": "#475569",
    "ytick.color": "#475569",
    "grid.color": "#EEF2F6",
    "axes.titleweight": "bold",
})

# Estilos CSS ligeros para dar un aire más pulido
st.markdown(
    """
    <style>
        .main > div { padding-top: 1.2rem; }
        h1, h2, h3 { color: #0F172A; }
        div[data-testid="stMetricValue"] { color: #0F766E; }
        .stButton>button {
            background: #0F766E; color: white; border: none;
            border-radius: 8px; padding: 0.55rem 1.3rem; font-weight: 600;
        }
        .stButton>button:hover { background: #0d5f59; color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Extracción de datos con el LLM
# --------------------------------------------------------------------------- #
PROMPT_SISTEMA = """Eres un extractor de datos estructurados. Recibes un párrafo \
de texto en lenguaje natural que contiene cifras y devuelves ÚNICAMENTE un objeto \
JSON válido, sin explicaciones ni bloques de código markdown.

El JSON debe tener esta forma exacta:
{
  "records": [ { "columna_1": valor, "columna_2": valor, ... }, ... ]
}

Reglas:
- Identifica las entidades del texto (filas) y sus atributos (columnas).
- Usa los mismos nombres de columna para todas las filas (esquema homogéneo).
- Las cifras deben ser números (int o float), NO cadenas. Ej: "1,250" -> 1250.
- No inventes datos que no estén en el texto. Si un valor falta, usa null.
- Los nombres de columna deben ser cortos, en minúsculas y con guion_bajo.
- Devuelve SOLO el JSON."""


def extraer_json(texto: str) -> str:
    """Recorta cualquier envoltura y deja solo el objeto JSON."""
    texto = texto.strip()
    texto = re.sub(r"^```(?:json)?|```$", "", texto, flags=re.MULTILINE).strip()
    inicio = texto.find("{")
    fin = texto.rfind("}")
    if inicio != -1 and fin != -1:
        return texto[inicio:fin + 1]
    return texto


def extraer_datos_llm(parrafo: str, api_key: str, modelo: str) -> pd.DataFrame:
    """Llama al LLM (Groq) y convierte la respuesta en un DataFrame."""
    if Groq is None:
        raise RuntimeError("El paquete 'groq' no está instalado.")

    client = Groq(api_key=api_key)
    respuesta = client.chat.completions.create(
        model=modelo,
        max_tokens=2000,
        temperature=0,
        # Modo JSON: obliga al modelo a devolver un objeto JSON válido.
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": PROMPT_SISTEMA},
            {"role": "user", "content": parrafo},
        ],
    )

    texto_salida = respuesta.choices[0].message.content or ""
    crudo = extraer_json(texto_salida)

    try:
        data = json.loads(crudo)
    except json.JSONDecodeError as err:
        raise ValueError(
            f"El LLM no devolvió un JSON válido.\n\nRespuesta:\n{texto_salida}"
        ) from err

    registros = data.get("records", data if isinstance(data, list) else [])
    df = pd.DataFrame(registros)

    # Intentar convertir columnas numéricas escritas como texto
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


# --------------------------------------------------------------------------- #
# Funciones de EDA (gráficos solo en seaborn)
# --------------------------------------------------------------------------- #
def graficar_barras(df, col_cat, col_num):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    datos = df.sort_values(col_num, ascending=False)
    sns.barplot(data=datos, x=col_cat, y=col_num, hue=col_cat,
                palette=PALETA, legend=False, ax=ax)
    ax.set_title(f"{col_num} por {col_cat}")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=35)
    for etiqueta in ax.get_xticklabels():
        etiqueta.set_ha("right")
    sns.despine()
    fig.tight_layout()
    return fig


def graficar_distribucion(df, col_num):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(df[col_num].dropna(), kde=True, color=COLOR_PRINCIPAL,
                 edgecolor="white", ax=ax)
    ax.set_title(f"Distribución de {col_num}")
    sns.despine()
    fig.tight_layout()
    return fig


def graficar_boxplot(df, cols_num):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    datos = df[cols_num].melt(var_name="variable", value_name="valor")
    sns.boxplot(data=datos, x="variable", y="valor", hue="variable",
                palette=PALETA, legend=False, ax=ax)
    ax.set_title("Boxplot de variables numéricas")
    ax.set_xlabel("")
    sns.despine()
    fig.tight_layout()
    return fig


def graficar_correlacion(df, cols_num):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    corr = df[cols_num].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="crest",
                linewidths=0.5, linecolor="white", square=True,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Matriz de correlación")
    fig.tight_layout()
    return fig


def graficar_dispersion(df, x, y):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.scatterplot(data=df, x=x, y=y, s=120, color=COLOR_ACENTO,
                    edgecolor=COLOR_PRINCIPAL, linewidth=1.5, ax=ax)
    ax.set_title(f"{y} vs {x}")
    sns.despine()
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Barra lateral
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Configuración")

    # En despliegue (Streamlit Cloud) la clave se lee de st.secrets;
    # en local o como respaldo, se puede pegar manualmente.
    api_key_secreta = st.secrets.get("GROQ_API_KEY", "")
    if api_key_secreta:
        api_key = api_key_secreta
        st.success("API Key cargada desde secrets ✅")
    else:
        api_key = st.text_input("Groq API Key", type="password",
                                help="Tu clave se usa solo en esta sesión.")

    modelo = st.selectbox(
        "Modelo (gratuito en Groq)",
        [
            "llama-3.3-70b-versatile",   # mejor calidad para extracción
            "llama-3.1-8b-instant",      # más rápido, mayor rate limit
            "openai/gpt-oss-120b",       # alternativa potente
            "qwen/qwen3-32b",
        ],
        index=0,
    )
    st.caption("Consigue una clave gratis en console.groq.com")
    st.divider()
    st.markdown("**Flujo:** párrafo → LLM → tabla → EDA en seaborn.")


# --------------------------------------------------------------------------- #
# Cuerpo principal
# --------------------------------------------------------------------------- #
st.title("📊 De Texto a Tabla + EDA")
st.markdown(
    "Pega un párrafo con cifras. El LLM extrae los datos, los estructura en una "
    "tabla y se genera un análisis exploratorio con gráficos en **seaborn**."
)

ejemplo = (
    "En 2023 la sucursal Norte vendió 1.250 unidades con ingresos de 45.000 USD "
    "y 12 empleados. La sucursal Sur vendió 980 unidades, generó 38.500 USD y "
    "tiene 9 empleados. La sucursal Este alcanzó 1.540 unidades, 52.300 USD y 15 "
    "empleados, mientras que la sucursal Oeste registró 760 unidades, 29.800 USD "
    "y 7 empleados."
)

parrafo = st.text_area("Párrafo de entrada", value=ejemplo, height=160)

if st.button("🚀 Extraer y analizar"):
    if not api_key:
        st.error("Ingresa tu Groq API Key en la barra lateral.")
        st.stop()
    if not parrafo.strip():
        st.error("Escribe o pega un párrafo con cifras.")
        st.stop()

    with st.spinner("Extrayendo datos con el LLM..."):
        try:
            df = extraer_datos_llm(parrafo, api_key, modelo)
        except Exception as err:  # noqa: BLE001
            st.error(f"Error: {err}")
            st.stop()

    if df.empty:
        st.warning("El LLM no encontró datos tabulables en el texto.")
        st.stop()

    st.session_state["df"] = df  # guardamos para no re-llamar al LLM

# Si ya hay datos en sesión, mostramos tabla + EDA
if "df" in st.session_state:
    df = st.session_state["df"]
    cols_num = df.select_dtypes(include=np.number).columns.tolist()
    cols_cat = [c for c in df.columns if c not in cols_num]

    st.subheader("🗂️ Tabla extraída")
    st.dataframe(df, use_container_width=True)

    # Métricas rápidas
    c1, c2, c3 = st.columns(3)
    c1.metric("Filas", df.shape[0])
    c2.metric("Columnas", df.shape[1])
    c3.metric("Variables numéricas", len(cols_num))

    st.download_button(
        "⬇️ Descargar CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="datos_extraidos.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("📈 Análisis exploratorio (EDA)")

    if cols_num:
        with st.expander("Resumen estadístico", expanded=True):
            st.dataframe(df[cols_num].describe().T, use_container_width=True)

    # ---- Gráficos ---- #
    g1, g2 = st.columns(2)

    # Barras: categoría vs numérica
    if cols_cat and cols_num:
        with g1:
            col_cat = st.selectbox("Categoría (barras)", cols_cat, key="bc")
            col_num = st.selectbox("Valor (barras)", cols_num, key="bn")
            st.pyplot(graficar_barras(df, col_cat, col_num))

    # Distribución
    if cols_num:
        with g2:
            col_dist = st.selectbox("Variable (distribución)", cols_num, key="dd")
            st.pyplot(graficar_distribucion(df, col_dist))

    g3, g4 = st.columns(2)

    # Boxplot (si hay ≥2 numéricas comparables)
    if len(cols_num) >= 1:
        with g3:
            st.pyplot(graficar_boxplot(df, cols_num))

    # Correlación o dispersión
    if len(cols_num) >= 2:
        with g4:
            st.pyplot(graficar_correlacion(df, cols_num))

        st.markdown("**Relación entre dos variables**")
        d1, d2 = st.columns(2)
        x = d1.selectbox("Eje X", cols_num, key="sx", index=0)
        y = d2.selectbox("Eje Y", cols_num, key="sy",
                         index=min(1, len(cols_num) - 1))
        st.pyplot(graficar_dispersion(df, x, y))
    elif len(cols_num) == 1:
        st.info("Se necesitan al menos 2 variables numéricas para correlación "
                "y dispersión.")
