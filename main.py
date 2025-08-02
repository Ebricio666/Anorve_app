# main.py
import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import pipeline

st.set_page_config(page_title="Análisis de Docentes", layout="wide")

# ---- Función para cargar modelo con caché ----
@st.cache_resource
def cargar_modelo():
    return pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        device=0 if torch.cuda.is_available() else -1
    )

# ---- Función para mapear sentimientos ----
def mapear_sentimiento(label):
    estrellas = int(label.split()[0])
    if estrellas <= 2:
        return "NEG"
    elif estrellas == 3:
        return "NEU"
    else:
        return "POS"

# ---- Subida de archivo ----
st.sidebar.title("📁 Cargar archivo")
archivo = st.sidebar.file_uploader("Sube un archivo CSV con comentarios", type=["csv"])
if archivo is not None:
    df = pd.read_csv(archivo)

    menu = st.sidebar.radio("Selecciona una sección", ["📊 Por Departamento", "🧨 Frases de Riesgo"])

    sentiment_pipeline = cargar_modelo()

    # ---- SECCIÓN 1: Por Departamento ----
    if menu == "📊 Por Departamento":
        st.title("📊 Análisis de Sentimientos por Departamento")
        st.write("Muestra de peor a mejor docente basado en comentarios negativos.")

        rango = st.slider("Selecciona el rango de ID de docentes", 
                          int(df['id_docente'].min()), 
                          int(df['id_docente'].max()), 
                          (int(df['id_docente'].min()), int(df['id_docente'].max())))

        df_rango = df[(df['id_docente'] >= rango[0]) & (df['id_docente'] <= rango[1])].copy()
        comentarios_invalidos = ['.', '-', '', ' ']
        df_rango['comentario_valido'] = ~df_rango['comentarios'].astype(str).str.strip().isin(comentarios_invalidos)
        df_validos = df_rango[df_rango['comentario_valido']].copy()

        # Limpieza y truncado
        df_validos['comentario_limpio'] = df_validos['comentarios'].astype(str).str.strip().str.replace(r"[\.\-]", "", regex=True).str.lower().str[:510]

        with st.spinner("🧠 Analizando sentimientos..."):
            predicciones = sentiment_pipeline(df_validos['comentario_limpio'].tolist())
        df_validos['sentimiento'] = [mapear_sentimiento(p['label']) for p in predicciones]

        resumen_list = []
        for docente_id in sorted(df_rango['id_docente'].unique()):
            subset = df_rango[df_rango['id_docente'] == docente_id]
            subset_validos = df_validos[df_validos['id_docente'] == docente_id]
            total_validos = len(subset_validos)
            neg = (subset_validos['sentimiento'] == 'NEG').sum()

            if total_validos > 0:
                proporcion_neg = neg / total_validos
                log_neg = np.log1p(neg)
                indice = proporcion_neg * log_neg
            else:
                proporcion_neg = 0
                indice = 0

            resumen = {
                'id_docente': docente_id,
                'asignaturas_impartidas': subset['id_asignatura'].nunique(),
                'alumnos_atendidos': len(subset),
                'comentarios_validos': total_validos,
                'comentarios_negativos': neg,
                'comentarios_neutros': (subset_validos['sentimiento'] == 'NEU').sum(),
                'comentarios_positivos': (subset_validos['sentimiento'] == 'POS').sum(),
                'proporcion_negativa': round(proporcion_neg, 2),
                'indice_severidad': round(indice, 4)
            }
            resumen_list.append(resumen)

        df_resumen = pd.DataFrame(resumen_list)
        df_resumen = df_resumen.sort_values(by='indice_severidad', ascending=False)
        st.dataframe(df_resumen)

        csv = df_resumen.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Descargar resumen en CSV", csv, file_name='resumen_por_departamento.csv', mime='text/csv')

    # ---- SECCIÓN 2: Frases de Riesgo ----
    elif menu == "🧨 Frases de Riesgo":
        st.title("🧨 Búsqueda de frases de riesgo o toxicidad")

        palabra = st.text_input("🔍 Escribe una palabra para rastrear").strip().lower()

        if palabra:
            df['comentarios'] = df['comentarios'].astype(str)
            df['coincide_palabra'] = df['comentarios'].str.lower().str.contains(palabra)

            df_coincidencias = df[df['coincide_palabra']].copy()

            if df_coincidencias.empty:
                st.error(f"No se encontró la palabra '{palabra}' en ningún comentario.")
            else:
                st.success(f"Se encontraron comentarios con la palabra '{palabra}'.")
                resumen_palabra = df_coincidencias.groupby('id_docente').agg({
                    'comentarios': list,
                    'coincide_palabra': 'count'
                }).reset_index()

                resumen_palabra = resumen_palabra.rename(columns={
                    'coincide_palabra': f"coincidencias_de_{palabra}",
                    'comentarios': 'comentarios_donde_aparece'
                })

                st.dataframe(resumen_palabra.sort_values(by=f"coincidencias_de_{palabra}", ascending=False))

else:
    st.warning("⚠️ Por favor, sube un archivo CSV con las columnas `id_docente`, `id_asignatura` y `comentarios`.")
