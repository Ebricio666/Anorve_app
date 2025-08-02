# main.py
import streamlit as st
import pandas as pd

# Configuración inicial
st.set_page_config(page_title="🔍 Comentarios de Riesgo, toxicidad y Búsqueda personalizada", layout="wide")
st.title("🛑 Detección de Comentarios de Riesgo + Búsqueda de Palabras Clave")

# Cargar archivo
archivo = st.sidebar.file_uploader("📂 Sube un archivo CSV con las columnas: id_docente, id_asignatura, comentarios", type=["csv"])

# Palabras clave por categoría
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

def detectar_categoria(texto):
    categorias_detectadas = set()
    for categoria, palabras in palabras_riesgo.items():
        if any(palabra in texto for palabra in palabras):
            categorias_detectadas.add(categoria)
    return list(categorias_detectadas)

if archivo is not None:
    df = pd.read_csv(archivo)
    
    if not {"id_docente", "comentarios", "id_asignatura"}.issubset(df.columns):
        st.error("❌ El archivo debe tener las columnas: id_docente, comentarios, id_asignatura.")
    else:
        # Normalizar
        df["comentarios"] = df["comentarios"].astype(str).str.lower().str.strip()
        df["comentario_valido"] = ~df["comentarios"].isin(["", ".", "-", " "])
        df_validos = df[df["comentario_valido"]].copy()

        # Detectar riesgo
        df_validos["categorias_riesgo"] = df_validos["comentarios"].apply(detectar_categoria)
        df_riesgo = df_validos[df_validos["categorias_riesgo"].apply(lambda x: len(x) > 0)].copy()

        st.subheader("🧠 Comentarios con palabras de riesgo detectadas automáticamente")
        st.dataframe(
            df_riesgo[["id_docente", "id_asignatura", "comentarios", "categorias_riesgo"]],
            use_container_width=True
        )

        # Búsqueda personalizada
        st.subheader("🔍 Búsqueda personalizada en comentarios con riesgo")
        palabra_riesgo = st.text_input("🔍 Escribe una palabra para buscar entre los comentarios con riesgo").strip().lower()

        if palabra_riesgo:
            df_riesgo["coincide_busqueda"] = df_riesgo["comentarios"].str.contains(palabra_riesgo, case=False, na=False)
            resultados_riesgo = df_riesgo[df_riesgo["coincide_busqueda"]].copy()

            if resultados_riesgo.empty:
                st.warning(f"❌ No se encontró la palabra '{palabra_riesgo}' en comentarios con riesgo.")
            else:
                st.success(f"✅ Se encontraron {len(resultados_riesgo)} coincidencias.")
                st.dataframe(resultados_riesgo[["id_docente", "id_asignatura", "comentarios", "categorias_riesgo"]],
                             use_container_width=True)

        # Búsqueda general
        st.subheader("📌 Búsqueda en todos los comentarios")
        palabra_general = st.text_input("📌 Palabra a buscar en todos los comentarios").strip().lower()

        if palabra_general:
            df["comentarios"] = df["comentarios"].astype(str)
            df["coincide_palabra"] = df["comentarios"].str.contains(palabra_general, case=False, na=False)
            df_coincidencias = df[df["coincide_palabra"]].copy()

            if df_coincidencias.empty:
                st.warning(f"❌ No se encontró la palabra '{palabra_general}' en ningún comentario.")
            else:
                st.success(f"✅ Se encontraron {len(df_coincidencias)} coincidencias.")
                st.dataframe(df_coincidencias[["id_docente", "id_asignatura", "comentarios"]],
                             use_container_width=True)

else:
    st.info("💡 Sube un archivo CSV para comenzar.")
