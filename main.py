# app.py
import subprocess
import sys

# Instalar transformers si no está disponible
try:
    from transformers import pipeline
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
    from transformers import pipeline
    
import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline
import torch

# Configuración inicial
st.set_page_config(page_title="Análisis Docente", layout="wide")
st.title("📊 Análisis de Comentarios Docentes")

# Cargar modelo de sentimiento
@st.cache_resource
def cargar_modelo():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", device=0 if torch.cuda.is_available() else -1)

sentiment_pipeline = cargar_modelo()

# Cargar archivo CSV
st.sidebar.header("📁 Cargar archivo")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['comentarios'] = df['comentarios'].astype(str)

    # Menú de módulos
    opcion = st.sidebar.radio("📌 Selecciona un módulo:", [
        "1️⃣ Resumen por departamento",
        "2️⃣ Análisis por ID docente",
        "3️⃣ Búsqueda de palabras clave"
    ])

    # -------------------- Módulo 1 --------------------
    if opcion == "1️⃣ Resumen por departamento":
        st.header("📋 Módulo 1: Resumen por departamento")

        rango_inicio = st.number_input("ID inicial del docente", min_value=0, step=1)
        rango_fin = st.number_input("ID final del docente", min_value=rango_inicio, step=1)

        if st.button("Generar resumen"):
            df_rango = df[(df['id_docente'] >= rango_inicio) & (df['id_docente'] <= rango_fin)]

            # Filtrar y limpiar
            comentarios_invalidos = ['.', '-', '', ' ']
            df_rango['comentario_valido'] = ~df_rango['comentarios'].str.strip().isin(comentarios_invalidos)
            df_validos = df_rango[df_rango['comentario_valido']].copy()
            df_validos['comentario_limpio'] = df_validos['comentarios'].str.lower().str.strip().str.replace(r"[\.\-]", "", regex=True)
            df_validos['comentario_limpio'] = df_validos['comentario_limpio'].str[:510]

            st.info("Analizando sentimientos...")

            predicciones = sentiment_pipeline(df_validos['comentario_limpio'].tolist())
            def mapear_sentimiento(label):
                estrellas = int(label.split()[0])
                if estrellas <= 2:
                    return "NEG"
                elif estrellas == 3:
                    return "NEU"
                else:
                    return "POS"

            df_validos['sentimiento'] = [mapear_sentimiento(p['label']) for p in predicciones]

            resumen_list = []
            for docente_id in sorted(df_rango['id_docente'].unique()):
                sub = df_rango[df_rango['id_docente'] == docente_id]
                sub_validos = df_validos[df_validos['id_docente'] == docente_id]
                total = len(sub_validos)
                neg = (sub_validos['sentimiento'] == 'NEG').sum()
                proporcion = neg / total if total else 0
                log_neg = np.log1p(neg)
                indice = proporcion * log_neg
                resumen_list.append({
                    'id_docente': docente_id,
                    'asignaturas_impartidas': sub['id_asignatura'].nunique(),
                    'alumnos_atendidos': len(sub),
                    'comentarios_validos': total,
                    'comentarios_negativos': neg,
                    'comentarios_neutros': (sub_validos['sentimiento'] == 'NEU').sum(),
                    'comentarios_positivos': (sub_validos['sentimiento'] == 'POS').sum(),
                    'proporcion_negativa': round(proporcion, 2),
                    'indice_severidad': round(indice, 4)
                })

            df_resumen = pd.DataFrame(resumen_list).sort_values(by="indice_severidad", ascending=False)
            st.dataframe(df_resumen)

    # -------------------- Módulo 2 --------------------
    elif opcion == "2️⃣ Análisis por ID docente":
        st.header("🧑‍🏫 Módulo 2: Comentarios por docente")
        docente_id = st.number_input("🔍 Ingresa el ID del docente:", step=1, min_value=0)

        if st.button("Buscar docente"):
            resultados = df[df['id_docente'] == docente_id]
            if resultados.empty:
                st.warning("❌ Docente no encontrado.")
            else:
                st.write("📘 Comentarios encontrados:")
                st.dataframe(resultados[['id_asignatura', 'comentarios']])
                
                comentarios_invalidos = ['.', '-', '', ' ']
                resultados['comentario_valido'] = ~resultados['comentarios'].astype(str).str.strip().isin(comentarios_invalidos)
                comentarios_validos = resultados[resultados['comentario_valido']].copy()
                comentarios_validos['comentario_limpio'] = comentarios_validos['comentarios'].str.lower().str.strip().str.replace(r"[\.\-]", "", regex=True)
                comentarios_validos['comentario_limpio'] = comentarios_validos['comentario_limpio'].str[:510]

                st.info("Analizando sentimientos...")
                predicciones = sentiment_pipeline(comentarios_validos['comentario_limpio'].tolist())
                comentarios_validos['sentimiento'] = [mapear_sentimiento(p['label']) for p in predicciones]

                st.subheader("📊 Resumen:")
                st.write(f"👥 Total comentarios válidos: {len(comentarios_validos)}")
                st.write(comentarios_validos['sentimiento'].value_counts())

                st.subheader("💬 Comentarios clasificados:")
                for tipo in ['NEG', 'NEU', 'POS']:
                    subset = comentarios_validos[comentarios_validos['sentimiento'] == tipo]
                    if not subset.empty:
                        st.markdown(f"### {tipo}")
                        for c in subset['comentario_limpio']:
                            st.markdown(f"- {c}")

    # -------------------- Módulo 3 --------------------
    elif opcion == "3️⃣ Búsqueda de palabras clave":
        st.header("🚨 Módulo 3: Búsqueda de palabras específicas o de riesgo")

        palabra = st.text_input("🔍 Ingresa la palabra a rastrear:")
        if palabra:
            df['comentarios'] = df['comentarios'].astype(str)
            df['coincide_palabra'] = df['comentarios'].str.lower().str.contains(palabra.strip().lower())

            df_coincidencias = df[df['coincide_palabra']]
            if df_coincidencias.empty:
                st.warning(f"No se encontraron coincidencias con '{palabra}'")
            else:
                resumen = df_coincidencias.groupby('id_docente').agg(
                    coincidencias=('coincide_palabra', 'sum'),
                    comentarios_donde_aparece=('comentarios', list)
                ).reset_index()
                resumen = resumen.sort_values(by='coincidencias', ascending=False)
                st.success(f"Se encontraron {len(df_coincidencias)} comentarios con la palabra '{palabra}'")
                st.dataframe(resumen)
else:
    st.warning("⚠️ Por favor, carga un archivo CSV con columnas 'id_docente', 'id_asignatura' y 'comentarios'.")
