# 📊 Streamlit app con barra lateral: Sentimientos y Frases de Riesgo
import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline

# Cargar modelo de sentimientos (sin usar torch directamente)
@st.cache_resource
def cargar_modelo_sentimientos():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

sentiment_pipeline = cargar_modelo_sentimientos()

# Subir archivo CSV
st.title("📘 Análisis de Comentarios Docentes")
archivo = st.file_uploader("🔼 Sube tu archivo CSV", type=["csv"])

if archivo:
    df = pd.read_csv(archivo)
    st.success("✅ Archivo cargado correctamente.")
    st.write(df.head())

    # Selección de módulo desde la barra lateral
    opcion = st.sidebar.selectbox("Selecciona una opción:", [
        "Análisis de Sentimientos por Docente",
        "Frases de riesgo y toxicidad"
    ])

    # --------------------------------------
    # 1️⃣ MÓDULO: Análisis de Sentimientos
    # --------------------------------------
    if opcion == "Análisis de Sentimientos por Docente":
        st.header("📈 Análisis de Sentimientos por Docente")
        rango_inicio = st.number_input("ID inicial del docente", min_value=0, step=1)
        rango_fin = st.number_input("ID final del docente", min_value=0, step=1)

        if rango_fin > rango_inicio:
            df = df[(df['id_docente'] >= rango_inicio) & (df['id_docente'] <= rango_fin)]

            # Limpiar comentarios
            comentarios_invalidos = ['.', '-', '', ' ']
            df['comentario_valido'] = ~df['comentarios'].astype(str).str.strip().isin(comentarios_invalidos)
            df_validos = df[df['comentario_valido']].copy()

            df_validos['comentario_limpio'] = (
                df_validos['comentarios']
                .astype(str)
                .str.strip()
                .str.replace(r"[\.\-]", "", regex=True)
                .str.lower()
            )
            df_validos['comentario_limpio'] = df_validos['comentario_limpio'].str[:510]

            with st.spinner("Analizando sentimientos..."):
                predicciones = sentiment_pipeline(df_validos['comentario_limpio'].tolist())

            # Mapear a etiquetas
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

            for docente_id in sorted(df['id_docente'].unique()):
                subset = df[df['id_docente'] == docente_id]
                subset_validos = df_validos[df_validos['id_docente'] == docente_id]

                total_validos = len(subset_validos)
                neg = (subset_validos['sentimiento'] == 'NEG').sum()

                if total_validos > 0:
                    proporcion_neg = neg / total_validos
                    log_neg = np.log1p(neg)
                    indice = proporcion_neg * log_neg
                else:
                    proporcion_neg = 0
                    log_neg = 0
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
            st.subheader("📊 Resumen ordenado por severidad negativa")
            st.dataframe(df_resumen)

    # --------------------------------------
    # 2️⃣ MÓDULO: Frases de riesgo y toxicidad
    # --------------------------------------
    elif opcion == "Frases de riesgo y toxicidad":
        st.header("⚠️ Rastreo de frases de riesgo o toxicidad")

        palabra_busqueda = st.text_input("🔍 Escribe una palabra para rastrear:").strip().lower()

        if palabra_busqueda:
            df['comentarios'] = df['comentarios'].astype(str)
            df['coincide_palabra'] = df['comentarios'].str.lower().str.contains(palabra_busqueda)
            df_coincidencias = df[df['coincide_palabra']].copy()

            if df_coincidencias.empty:
                st.warning(f"❌ No se encontró la palabra '{palabra_busqueda}' en ningún comentario.")
            else:
                st.success(f"✅ Se encontraron comentarios con la palabra '{palabra_busqueda}'.")

                resumen_palabra = df_coincidencias.groupby('id_docente').agg({
                    'comentarios': list,
                    'coincide_palabra': 'count'
                }).reset_index()

                resumen_palabra = resumen_palabra.rename(columns={
                    'coincide_palabra': f"coincidencias_de_{palabra_busqueda}",
                    'comentarios': 'comentarios_donde_aparece'
                })

                st.dataframe(resumen_palabra.sort_values(by=f"coincidencias_de_{palabra_busqueda}", ascending=False))
