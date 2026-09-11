from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import streamlit as st

CARPETA_LOCALES = Path(__file__).parent / "locales"
IDIOMA_POR_DEFECTO = "es"
CLAVE_ESTADO = "idioma"  # clave en st.session_state y del widget selector


# --------------------------------------------------------------------------- #
# Carga de traducciones
# --------------------------------------------------------------------------- #
def _firma_archivos() -> tuple:
    """Nombre + fecha de modificación de cada JSON.

    Se usa como argumento del caché: si editas un JSON, la firma cambia y
    las traducciones se recargan sin reiniciar la app.
    """
    return tuple(
        (p.name, p.stat().st_mtime) for p in sorted(CARPETA_LOCALES.glob("*.json"))
    )


def _leer_json(firma: tuple) -> dict[str, dict[str, str]]:
    traducciones = {}
    for nombre, _ in firma:
        ruta = CARPETA_LOCALES / nombre
        with open(ruta, encoding="utf-8") as f:
            traducciones[ruta.stem] = json.load(f)
    return traducciones


# Versión cacheada: los JSON se leen una vez y no en cada llamada a t().
_leer_traducciones = st.cache_data(show_spinner=False)(_leer_json)


def cargar_traducciones() -> dict[str, dict[str, str]]:
    """Devuelve {codigo_idioma: {clave: texto}}."""
    return _leer_traducciones(_firma_archivos())


def idiomas_disponibles() -> dict[str, str]:
    """Devuelve {codigo: nombre visible}, p. ej. {"es": "Español", "en": "English"}."""
    return {codigo: datos.get("_nombre", codigo)
            for codigo, datos in cargar_traducciones().items()}


# --------------------------------------------------------------------------- #
# Idioma actual
# --------------------------------------------------------------------------- #
def _idioma_inicial() -> str:
    disponibles = cargar_traducciones()

    # 1) ?lang=xx en la URL
    desde_url = st.query_params.get("lang")
    if desde_url in disponibles:
        return desde_url

    # 2) Idioma del navegador, p. ej. "es-CO" -> "es"
    locale_navegador = getattr(st.context, "locale", None) or ""
    base = locale_navegador.split("-")[0].lower()
    if base in disponibles:
        return base

    # 3) Por defecto
    return IDIOMA_POR_DEFECTO


def idioma_actual() -> str:
    """Código del idioma activo. Lo inicializa la primera vez que se llama.

    Solo lee session_state / URL, así que puede llamarse antes de
    st.set_page_config() para traducir también el título de la pestaña.
    """
    if CLAVE_ESTADO not in st.session_state:
        st.session_state[CLAVE_ESTADO] = _idioma_inicial()
    return st.session_state[CLAVE_ESTADO]


# --------------------------------------------------------------------------- #
# Traducción
# --------------------------------------------------------------------------- #
def texto_en(idioma: str, clave: str, **variables) -> str:
    """Traduce `clave` a un idioma concreto (no necesariamente el activo).

    Si la clave no existe en ese idioma, usa el idioma por defecto; si
    tampoco existe, devuelve la propia clave (así se nota qué falta).
    """
    traducciones = cargar_traducciones()
    texto = traducciones.get(idioma, {}).get(clave)
    if texto is None:
        texto = traducciones.get(IDIOMA_POR_DEFECTO, {}).get(clave, clave)
    return texto.format(**variables) if variables else texto


def t(clave: str, **variables) -> str:
    """Traduce `clave` al idioma activo. Admite variables: t("hola", nombre="Ana")."""
    return texto_en(idioma_actual(), clave, **variables)


# --------------------------------------------------------------------------- #
# Selector de idioma
# --------------------------------------------------------------------------- #
def selector_idioma(contenedor=None, etiqueta: str | None = None,
                    al_cambiar: Callable[[], None] | None = None) -> str:
    """Dibuja el selector de idioma y devuelve el código elegido.

    contenedor: dónde dibujarlo (st.sidebar por defecto; también sirve una
                columna, st.popover(...), etc.).
    etiqueta:   texto del selector. Por defecto usa la clave "idioma_etiqueta"
                del JSON o, si no existe, "🌐 Idioma / Language".
    al_cambiar: función opcional que se ejecuta tras cambiar el idioma
                (útil para refrescar textos que el usuario no ha editado).
    """
    idioma_actual()  # garantiza que el estado existe antes del widget
    contenedor = contenedor or st.sidebar
    opciones = idiomas_disponibles()

    if etiqueta is None:
        etiqueta = t("idioma_etiqueta")
        if etiqueta == "idioma_etiqueta":
            etiqueta = "🌐 Idioma / Language"

    def _callback():
        # Refleja el idioma en la URL para que un enlace compartido lo conserve.
        st.query_params["lang"] = st.session_state[CLAVE_ESTADO]
        if al_cambiar:
            al_cambiar()

    contenedor.selectbox(
        etiqueta,
        options=list(opciones),               # se guardan CÓDIGOS ("es", "en")...
        format_func=lambda c: opciones[c],    # ...y se muestran NOMBRES
        key=CLAVE_ESTADO,
        on_change=_callback,
    )
    return st.session_state[CLAVE_ESTADO]


# --------------------------------------------------------------------------- #
# Utilidad de mantenimiento
# --------------------------------------------------------------------------- #
def verificar_traducciones() -> dict[str, list[str]]:
    """Lista las claves que faltan en cada idioma respecto al resto.

    Úsala al añadir textos nuevos:
        python -c "import i18n; print(i18n.verificar_traducciones())"
    """
    traducciones = _leer_json(_firma_archivos())
    todas = set().union(*(set(d) for d in traducciones.values()))
    return {codigo: sorted(todas - set(datos))
            for codigo, datos in traducciones.items()
            if todas - set(datos)}
