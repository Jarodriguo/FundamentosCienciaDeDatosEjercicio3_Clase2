# De Texto a Tabla + EDA

Dashboard en **Streamlit** que convierte un párrafo escrito en lenguaje natural en una tabla de datos y genera automáticamente un **análisis exploratorio (EDA)** con gráficos en **seaborn**.

Proyecto desarrollado como ejercicio de la asignatura *Fundamentos de Ciencia de Datos* (Ejercicio 3, Clase 2).

---

## ¿En qué consiste?

Mucha información numérica llega como texto corrido: informes, noticias, correos o actas. Este proyecto muestra cómo un modelo de lenguaje (LLM) puede actuar como **extractor de datos estructurados**, para después aplicar sobre esos datos las técnicas clásicas de análisis exploratorio.

Por ejemplo, a partir de este texto:

> En 2023 la sucursal Norte vendió 1.250 unidades con ingresos de 45.000 USD y 12 empleados. La sucursal Sur vendió 980 unidades, generó 38.500 USD y tiene 9 empleados...

la aplicación obtiene una tabla como esta y la analiza:

| sucursal | unidades | ingresos_usd | empleados |
|----------|---------:|-------------:|----------:|
| Norte    |    1250  |       45000  |        12 |
| Sur      |     980  |       38500  |         9 |
| ...      |     ...  |         ...  |       ... |

## ¿Cómo funciona?

```
Párrafo con cifras ──► LLM (Groq) ──► JSON ──► DataFrame (pandas) ──► EDA (seaborn)
```

1. **Entrada.** El usuario pega un párrafo que contenga cifras.
2. **Extracción con IA.** Se envía el texto a un LLM a través de la API de [Groq](https://console.groq.com) con instrucciones para devolver **solo un objeto JSON** (`{"records": [...]}`), usando el modo JSON de la API, que obliga al modelo a responder en ese formato.
3. **Estructuración.** El JSON se convierte en un `DataFrame` de pandas y cada columna se detecta automáticamente como numérica o categórica.
4. **Análisis exploratorio.** Se muestran métricas rápidas, un resumen estadístico y cinco tipos de gráfico:
   - barras (categoría vs. valor numérico),
   - histograma con curva de densidad (KDE),
   - diagrama de caja (boxplot),
   - matriz de correlación (mapa de calor),
   - dispersión entre dos variables.
5. **Exportación.** La tabla extraída se puede descargar en CSV.

## Funcionalidades

- Extracción de datos con modelos de IA vigentes en Groq, con opción de escribir cualquier otro ID de modelo sin tocar código.
- Interfaz en **español** e **inglés**, con cambio de idioma instantáneo que conserva lo que el usuario ya escribió o seleccionó.
- Los nombres de columna que genera la IA siguen el idioma de la interfaz.
- Enlaces por idioma: `?lang=en` abre la app directamente en inglés; si no se indica, se usa el idioma del navegador.
- La API key se lee de `st.secrets` en despliegue, o se introduce manualmente en local.
- Gráficos generados exclusivamente con seaborn, con una paleta de colores propia.

## Modelos de IA

La versión anterior usaba `llama-3.3-70b-versatile`, que Groq retiró del plan gratuito el 16 de agosto de 2026 (junto con `llama-3.1-8b-instant` y `qwen/qwen3-32b`). La app usa ahora:

| Modelo | Tipo | Comentario |
|---|---|---|
| `openai/gpt-oss-120b` | Producción | **Recomendado.** Reemplazo oficial sugerido por Groq. |
| `openai/gpt-oss-20b` | Producción | Más rápido y económico. |
| `qwen/qwen3.6-27b` | Vista previa | Alternativa; puede retirarse con poco aviso. |
| *Otro (escribir ID)* | — | Permite usar cualquier modelo vigente sin modificar el código. |

Los modelos GPT-OSS "razonan" antes de responder. Para esta tarea de extracción basta con un esfuerzo de razonamiento bajo (`reasoning_effort="low"`), lo que reduce la latencia y el consumo de tokens.

> Groq retira modelos periódicamente. Si un modelo deja de funcionar, la app muestra un aviso claro. Para actualizar la lista, consulta la [página de modelos obsoletos de Groq](https://console.groq.com/docs/deprecations) y edita el diccionario `MODELOS` en `app.py`.

## Estructura del proyecto

```
.
├── app.py              # Aplicación Streamlit (interfaz, llamada al LLM y EDA)
├── i18n.py             # Módulo reutilizable de traducción (selector de idioma y t())
├── locales/
│   ├── es.json         # Textos de la interfaz en español
│   └── en.json         # Textos de la interfaz en inglés
├── requirements.txt    # Dependencias
└── README.md
```

## Instalación y ejecución local

Requisitos: Python 3.10 o superior y una API key gratuita de Groq ([console.groq.com/keys](https://console.groq.com/keys)).

```bash
git clone https://github.com/Jarodriguo/FundamentosCienciaDeDatosEjercicio3_Clase2.git
cd FundamentosCienciaDeDatosEjercicio3_Clase2

python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

La app se abre en `http://localhost:8501`. Puedes pegar la API key en la barra lateral o guardarla en `.streamlit/secrets.toml` para no escribirla cada vez:

```toml
GROQ_API_KEY = "gsk_..."
```

> No subas `secrets.toml` a GitHub. Añádelo a tu `.gitignore`.

## Despliegue en Streamlit Community Cloud

1. Entra en [share.streamlit.io](https://share.streamlit.io) y crea una app a partir de este repositorio, con `app.py` como archivo principal.
2. En **Settings → Secrets**, añade `GROQ_API_KEY = "gsk_..."`.
3. Si la app ya estaba desplegada, pulsa **Reboot app** para que se instalen las nuevas versiones de `requirements.txt`.

## Añadir otro idioma

1. Copia `locales/es.json` como `locales/<código>.json` (por ejemplo, `pt.json`).
2. Cambia `"_nombre"` por el nombre del idioma (`"Português"`) y traduce los valores. **No cambies las claves** ni los marcadores entre llaves como `{col}`.
3. Comprueba que no falte ninguna clave:
   ```bash
   python -c "import i18n; print(i18n.verificar_traducciones() or 'Todo completo')"
   ```

El nuevo idioma aparece automáticamente en el selector, sin tocar `app.py`.

## Tecnologías

[Streamlit](https://streamlit.io) · [pandas](https://pandas.pydata.org) · [NumPy](https://numpy.org) · [seaborn](https://seaborn.pydata.org) · [Matplotlib](https://matplotlib.org) · [Groq API](https://console.groq.com/docs)
