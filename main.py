import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import pipeline

# Título principal
st.title("🧑‍🏫 Análisis de Sentimientos por Docente")

# Subir archivo
archivo = st.file_uploader("📤 Sube tu archivo CSV con comentarios", type=["csv"])

if archivo is not None:
    df = pd.read_csv(archivo, encoding='utf-8')
    st.success("📄 Archivo cargado correctamente")
    st.dataframe(df.head())

    # Ingreso de rango de IDs
    rango_inicio = st.number_input("🔸 ID inicial de docente", min_value=int(df['id_docente'].min()), value=int(df['id_docente'].min()))
    rango_fin = st.number_input("🔸 ID final de docente", min_value=int(df['id_docente'].min()), value=int(df['id_docente'].max()))

    if rango_inicio <= rango_fin:
        df = df[(df['id_docente'] >= rango_inicio) & (df['id_docente'] <= rango_fin)]

        # Filtrar comentarios válidos
        comentarios_invalidos = ['.', '-', '', ' ']
        df['comentario_valido'] = ~df['comentarios'].astype(str).str.strip().isin(comentarios_invalidos)
        df_validos = df[df['comentario_valido']].copy()

        # Limpiar texto
        df_validos['comentario_limpio'] = (
            df_validos['comentarios']
            .astype(str)
            .str.strip()
            .str.replace(r"[\.\-]", "", regex=True)
            .str.lower()
        )
        df_validos['comentario_limpio'] = df_validos['comentario_limpio'].str[:510]

        # Análisis de sentimientos
        st.info("🔍 Analizando sentimientos... puede tardar unos minutos.")

        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device=0 if torch.cuda.is_available() else -1
        )

        predicciones = sentiment_pipeline(df_validos['comentario_limpio'].tolist())

        # Mapear etiquetas
        def mapear_sentimiento(label):
            estrellas = int(label.split()[0])
            if estrellas <= 2:
                return "NEG"
            elif estrellas == 3:
                return "NEU"
            else:
                return "POS"

        df_validos['sentimiento'] = [mapear_sentimiento(p['label']) for p in predicciones]

        # Crear resumen
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

        # Mostrar resultados
        df_resumen = pd.DataFrame(resumen_list)
        df_resumen = df_resumen.sort_values(by='indice_severidad', ascending=False)
        st.subheader("📊 Resultados ordenados por índice de severidad")
        st.dataframe(df_resumen)

        # Descargar
        nombre_archivo = f"resumen_docentes_{rango_inicio}_a_{rango_fin}.csv"
        csv = df_resumen.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Descargar resumen CSV", data=csv, file_name=nombre_archivo, mime='text/csv')
    else:
        st.warning("⚠️ El ID inicial debe ser menor o igual al final.")
