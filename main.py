# main.py
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Frases de Riesgo Docente", layout="wide")

# ---- Cargar archivo ----
st.sidebar.title("📁 Cargar archivo")
archivo = st.sidebar.file_uploader("Sube un archivo CSV con comentarios", type=["csv"])

# ---- Diccionario de frases de riesgo ----
palabras_riesgo = {
    "riesgo_psicosocial": [
        "estrés", "ansiedad", "depresión", "cansancio", "agobio",
        "presión", "burnout", "tensión", "desgaste", "agotamiento"
    ],
    "violencia_acoso": [
        "acoso", "hostigamiento", "intimidación", "amenaza", "agresión",
        "violencia", "golpear", "forzar", "manoseo", "imposición"
    ],
    "maltrato_verbal_fisico": [
        "gritar", "insultar", "ofender", "ridiculizar", "menospreciar",
        "burlarse", "humillar", "descalificar", "pegar", "empujar"
    ],
    "vulnerabilidad_discriminación": [
        "discriminación", "exclusión", "racismo", "clasismo", "marginación",
        "desigualdad", "inequidad", "vulnerable", "preferencia", "estigmatizar"
    ]
}

# ---- Función para detectar categorías ----
def detectar_categoria(texto):
    categorias_detectadas = set()
    for categoria, palabras in palabras_riesgo.items():
        if any(palabra in texto for palabra in palabras):
            categorias_detectadas.add(categoria)
    return list(categorias_detectadas)

# ---- Si se carga un archivo ----
if archivo is not None:
    df = pd.read_csv(archivo)
    df['comentarios'] = df['comentarios'].astype(str).str.lower().str.strip()
    df['comentario_valido'] = ~df['comentarios'].isin(['.', '-', '', ' '])
    df_validos = df[df['comentario_valido']].copy()

    df_validos['categorias_riesgo'] = df_validos['comentarios'].apply(detectar_categoria)
    df_riesgo = df_validos[df_validos['categorias_riesgo'].apply(lambda x: len(x) > 0)].copy()

    st.title("🧨 Análisis de Frases de Riesgo y Palabras Clave")

    tab1, tab2, tab3 = st.tabs(["🧨 Comentarios con Riesgo", "🔍 Palabra Clave en Comentarios con Riesgo", "📢 Palabra Clave en Todos los Comentarios"])

    with tab1:
        st.subheader("🧨 Comentarios con riesgo por docente")
        agrupado = df_riesgo.groupby('id_docente').agg({
            'comentarios': list,
            'categorias_riesgo': lambda x: sorted(set([cat for sublist in x for cat in sublist])),
            'comentario_valido': 'count'
        }).reset_index()

        agrupado = agrupado.rename(columns={
            'comentario_valido': 'comentarios_riesgo',
            'comentarios': 'comentarios_detectados',
            'categorias_riesgo': 'categorias_identificadas'
        })

        st.dataframe(agrupado.sort_values(by='comentarios_riesgo', ascending=False), use_container_width=True)

    with tab2:
        palabra = st.text_input("🔍 Escribe una palabra clave para buscar en comentarios con riesgo").strip().lower()
        if palabra:
            df_riesgo['coincide_busqueda'] = df_riesgo['comentarios'].str.contains(palabra, case=False)
            df_coincidencia = df_riesgo[df_riesgo['coincide_busqueda']].copy()

            if df_coincidencia.empty:
                st.error(f"No se encontró la palabra '{palabra}' en comentarios con riesgo.")
            else:
                resumen_palabra = df_coincidencia.groupby('id_docente').agg({
                    'comentarios': list,
                    'categorias_riesgo': lambda x: sorted(set([cat for sublist in x for cat in sublist])),
                    'coincide_busqueda': 'count'
                }).reset_index()

                resumen_palabra = resumen_palabra.rename(columns={
                    'coincide_busqueda': f"coincidencias_de_{palabra}",
                    'comentarios': 'comentarios_donde_aparece',
                    'categorias_riesgo': 'categorias_asociadas'
                })

                st.success(f"Se encontraron {len(df_coincidencia)} comentarios con la palabra '{palabra}'.")
                st.dataframe(resumen_palabra.sort_values(by=f"coincidencias_de_{palabra}", ascending=False), use_container_width=True)

    with tab3:
        palabra_general = st.text_input("📢 Escribe una palabra para buscar en todos los comentarios").strip().lower()
        if palabra_general:
            df['comentarios'] = df['comentarios'].astype(str)
            df['coincide_palabra'] = df['comentarios'].str.contains(palabra_general, case=False)
            df_coincidencias = df[df['coincide_palabra']].copy()

            if df_coincidencias.empty:
                st.error(f"No se encontró la palabra '{palabra_general}' en ningún comentario.")
            else:
                resumen_palabra_general = df_coincidencias.groupby('id_docente').agg({
                    'comentarios': list,
                    'coincide_palabra': 'count'
                }).reset_index()

                resumen_palabra_general = resumen_palabra_general.rename(columns={
                    'coincide_palabra': f"coincidencias_de_{palabra_general}",
                    'comentarios': 'comentarios_donde_aparece'
                })

                st.success(f"Se encontraron comentarios con la palabra '{palabra_general}'.")
                st.dataframe(resumen_palabra_general.sort_values(by=f"coincidencias_de_{palabra_general}", ascending=False), use_container_width=True)

else:
    st.warning("⚠️ Por favor, sube un archivo CSV con las columnas `id_docente` y `comentarios`.")
