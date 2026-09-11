"""
Dashboard de Streamlit: Texto -> Tabla -> EDA
-----------------------------------------------
Recibe un párrafo con cifras, usa un LLM (API de Groq) para extraer
los datos en formato tabular y genera un análisis exploratorio (EDA)
con gráficos hechos únicamente en seaborn.

La interfaz está disponible en varios idiomas (ver i18n.py y locales/).

Ejecutar con:
    streamlit run app.py
"""

import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from i18n import idioma_actual, idiomas_disponibles, selector_idioma, t, texto_en

try:
    from groq import Groq
except ImportError:
    Groq = None


# --------------------------------------------------------------------------- #
# Configuración general de la página
# --------------------------------------------------------------------------- #
idioma_actual()  # inicializa el idioma antes de traducir el título de la pestaña
st.set_page_config(
    page_title=t("titulo_pestana"),
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
# Modelos disponibles en Groq (revisado en septiembre de 2026)
# --------------------------------------------------------------------------- #
# Los Llama 3.x y Qwen3-32B que usaba la versión anterior fueron retirados del
# plan gratuito en julio/agosto de 2026. Si alguno de estos deja de funcionar,
# revisa https://console.groq.com/docs/deprecations y actualiza este diccionario
# (o usa la opción "Otro" de la barra lateral, que no requiere tocar código).
#
# "extra" son parámetros propios de cada familia de modelos:
#   - GPT-OSS razona antes de responder; con esfuerzo "low" basta para extraer
#     datos, y include_reasoning=False evita recibir el razonamiento.
#   - Qwen 3.6 permite desactivar el razonamiento con "none".
MODELOS = {
    "openai/gpt-oss-120b": {
        "etiqueta": "modelo_recomendado",
        "extra": {"reasoning_effort": "low", "include_reasoning": False},
    },
    "openai/gpt-oss-20b": {
        "etiqueta": "modelo_rapido",
        "extra": {"reasoning_effort": "low", "include_reasoning": False},
    },
    "qwen/qwen3.6-27b": {
        "etiqueta": "modelo_preview",
        "extra": {"reasoning_effort": "none"},
    },
}
OPCION_OTRO = "__otro__"


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
        raise RuntimeError(t("error_sin_groq"))

    # Los nombres de columna se piden en el idioma de la interfaz.
    prompt = (PROMPT_SISTEMA
              + f"\n- Escribe los nombres de columna en {t('idioma_columnas_llm')}.")
    extra = MODELOS.get(modelo, {}).get("extra", {})

    client = Groq(api_key=api_key)
    try:
        respuesta = client.chat.completions.create(
            model=modelo,
            # Margen amplio: en los modelos de razonamiento este límite
            # incluye también los tokens que el modelo usa para "pensar".
            max_completion_tokens=4096,
            temperature=0,
            # Modo JSON: obliga al modelo a devolver un objeto JSON válido.
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": parrafo},
            ],
            **extra,
        )
    except Exception as err:  # noqa: BLE001
        mensaje = str(err).lower()
        if "decommissioned" in mensaje or "model_not_found" in mensaje:
            raise RuntimeError(t("error_modelo_retirado", modelo=modelo)) from err
        raise

    eleccion = respuesta.choices[0]
    if eleccion.finish_reason == "length":
        raise ValueError(t("error_truncado"))

    texto_salida = eleccion.message.content or ""
    crudo = extraer_json(texto_salida)

    try:
        data = json.loads(crudo)
    except json.JSONDecodeError as err:
        raise ValueError(t("error_json", respuesta=texto_salida)) from err

    registros = data.get("records", data if isinstance(data, list) else [])
    df = pd.DataFrame(registros)

    # Convertir a numérico solo las columnas que lo son de verdad.
    # (errors="ignore" fue eliminado en pandas 3, así que lo hacemos a mano.)
    for col in df.columns:
        convertida = pd.to_numeric(df[col], errors="coerce")
        # Si todos los valores no nulos se convirtieron sin perderse, es numérica.
        if convertida.notna().sum() == df[col].notna().sum():
            df[col] = convertida
    return df


# --------------------------------------------------------------------------- #
# Funciones de EDA (gráficos solo en seaborn)
# --------------------------------------------------------------------------- #
def mostrar(fig):
    """Muestra la figura y la cierra para no acumular memoria entre reruns."""
    st.pyplot(fig)
    plt.close(fig)


def graficar_barras(df, col_cat, col_num):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    datos = df.sort_values(col_num, ascending=False)
    colores = sns.color_palette(PALETA, n_colors=datos[col_cat].nunique())
    sns.barplot(data=datos, x=col_cat, y=col_num, hue=col_cat,
                palette=colores, legend=False, ax=ax)
    ax.set_title(t("graf_barras", num=col_num, cat=col_cat))
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
    ax.set_title(t("graf_distribucion", col=col_num))
    ax.set_ylabel(t("graf_frecuencia"))
    sns.despine()
    fig.tight_layout()
    return fig


def graficar_boxplot(df, cols_num):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    datos = df[cols_num].melt(var_name="variable", value_name="valor")
    colores = sns.color_palette(PALETA, n_colors=len(cols_num))
    sns.boxplot(data=datos, x="variable", y="valor", hue="variable",
                palette=colores, legend=False, ax=ax)
    ax.set_title(t("graf_boxplot"))
    ax.set_xlabel("")
    ax.set_ylabel(t("graf_valor"))
    sns.despine()
    fig.tight_layout()
    return fig


def graficar_correlacion(df, cols_num):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    corr = df[cols_num].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="crest",
                linewidths=0.5, linecolor="white", square=True,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title(t("graf_correlacion"))
    fig.tight_layout()
    return fig


def graficar_dispersion(df, x, y):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.scatterplot(data=df, x=x, y=y, s=120, color=COLOR_ACENTO,
                    edgecolor=COLOR_PRINCIPAL, linewidth=1.5, ax=ax)
    ax.set_title(t("graf_dispersion", x=x, y=y))
    sns.despine()
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Barra lateral
# --------------------------------------------------------------------------- #
def al_cambiar_idioma():
    """Si el párrafo sigue siendo el ejemplo de algún idioma, lo traduce.

    Si el usuario ya escribió su propio texto, se respeta tal cual.
    """
    ejemplos = {texto_en(codigo, "ejemplo") for codigo in idiomas_disponibles()}
    if st.session_state.get("parrafo") in ejemplos:
        st.session_state["parrafo"] = t("ejemplo")


with st.sidebar:
    selector_idioma(al_cambiar=al_cambiar_idioma)
    st.divider()

    st.header(t("config_titulo"))

    # En despliegue (Streamlit Cloud) la clave se lee de st.secrets;
    # en local o como respaldo, se puede pegar manualmente.
    try:
        api_key_secreta = st.secrets.get("GROQ_API_KEY", "")
    except Exception:  # noqa: BLE001  (no existe secrets.toml)
        api_key_secreta = ""
    if api_key_secreta:
        api_key = api_key_secreta
        st.success(t("api_key_cargada"))
    else:
        # key fija: la clave escrita no se pierde al cambiar de idioma.
        api_key = st.text_input(t("api_key_etiqueta"), type="password",
                                help=t("api_key_ayuda"), key="api_key")

    # Se guardan IDs de modelo; el texto visible se traduce con format_func.
    # Las etiquetas se precalculan en un dict (más eficiente y predecible
    # que llamar a t() dentro de una lambda).
    nombres_modelo = {m: f"{m} · {t(info['etiqueta'])}" for m, info in MODELOS.items()}
    nombres_modelo[OPCION_OTRO] = t("modelo_otro")
    eleccion = st.selectbox(
        t("modelo_etiqueta"),
        options=list(nombres_modelo),
        format_func=nombres_modelo.get,
        help=t("modelo_ayuda"),
        key="modelo",
    )
    if eleccion == OPCION_OTRO:
        modelo = st.text_input(t("modelo_id_etiqueta"), help=t("modelo_id_ayuda"),
                               key="modelo_personalizado").strip()
    else:
        modelo = eleccion

    st.caption(t("api_key_nota"))
    st.divider()
    st.markdown(t("flujo"))


# --------------------------------------------------------------------------- #
# Cuerpo principal
# --------------------------------------------------------------------------- #
st.title(t("titulo"))
st.markdown(t("intro"))

# El párrafo vive en session_state (key="parrafo") para que no se borre
# al cambiar de idioma; el valor inicial es el ejemplo del idioma activo.
if "parrafo" not in st.session_state:
    st.session_state["parrafo"] = t("ejemplo")
parrafo = st.text_area(t("parrafo_etiqueta"), height=160, key="parrafo")

if st.button(t("boton_extraer")):
    if not api_key:
        st.error(t("error_sin_api_key"))
        st.stop()
    if not modelo:
        st.error(t("error_sin_modelo"))
        st.stop()
    if not parrafo.strip():
        st.error(t("error_sin_texto"))
        st.stop()

    with st.spinner(t("extrayendo")):
        try:
            df = extraer_datos_llm(parrafo, api_key, modelo)
        except Exception as err:  # noqa: BLE001
            st.error(t("error_generico", error=err))
            st.stop()

    if df.empty:
        st.warning(t("aviso_sin_datos"))
        st.stop()

    st.session_state["df"] = df  # guardamos para no re-llamar al LLM

# Si ya hay datos en sesión, mostramos tabla + EDA
if "df" in st.session_state:
    df = st.session_state["df"]
    cols_num = df.select_dtypes(include=np.number).columns.tolist()
    cols_cat = [c for c in df.columns if c not in cols_num]

    st.subheader(t("tabla_titulo"))
    st.dataframe(df, width="stretch")

    # Métricas rápidas
    c1, c2, c3 = st.columns(3)
    c1.metric(t("metrica_filas"), df.shape[0])
    c2.metric(t("metrica_columnas"), df.shape[1])
    c3.metric(t("metrica_numericas"), len(cols_num))

    st.download_button(
        t("descargar_csv"),
        df.to_csv(index=False).encode("utf-8"),
        file_name=t("nombre_csv"),
        mime="text/csv",
    )

    st.divider()
    st.subheader(t("eda_titulo"))

    if cols_num:
        with st.expander(t("resumen_estadistico"), expanded=True):
            resumen = df[cols_num].describe().T.rename(columns={
                "count": t("est_count"), "mean": t("est_mean"),
                "std": t("est_std"), "min": t("est_min"), "max": t("est_max"),
            })
            st.dataframe(resumen, width="stretch")

    # ---- Gráficos ---- #
    # Todos los selectores tienen key fija: así conservan la selección
    # aunque su etiqueta cambie al cambiar de idioma.
    g1, g2 = st.columns(2)

    # Barras: categoría vs numérica
    if cols_cat and cols_num:
        with g1:
            col_cat = st.selectbox(t("sel_categoria_barras"), cols_cat, key="bc")
            col_num = st.selectbox(t("sel_valor_barras"), cols_num, key="bn")
            mostrar(graficar_barras(df, col_cat, col_num))

    # Distribución
    if cols_num:
        with g2:
            col_dist = st.selectbox(t("sel_distribucion"), cols_num, key="dd")
            mostrar(graficar_distribucion(df, col_dist))

    g3, g4 = st.columns(2)

    # Boxplot (si hay ≥1 numérica)
    if len(cols_num) >= 1:
        with g3:
            mostrar(graficar_boxplot(df, cols_num))

    # Correlación o dispersión
    if len(cols_num) >= 2:
        with g4:
            mostrar(graficar_correlacion(df, cols_num))

        st.markdown(t("relacion_titulo"))
        d1, d2 = st.columns(2)
        x = d1.selectbox(t("eje_x"), cols_num, key="sx", index=0)
        y = d2.selectbox(t("eje_y"), cols_num, key="sy",
                         index=min(1, len(cols_num) - 1))
        mostrar(graficar_dispersion(df, x, y))
    elif len(cols_num) == 1:
        st.info(t("info_dos_numericas"))
