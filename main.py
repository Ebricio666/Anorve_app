import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline

# Configurar la app
st.set_page_config(page_title="Análisis de Sentimientos por Docente", layout="wide")

# Cargar modelo con caché
@st.cache_resource

def cargar_modelo():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

modelo_sentimiento = cargar_modelo()

# Subir archivo
archivo = st.sidebar.file_uploader("📁 Subir archivo CSV", type=["csv"])

# Pestañas laterales
menu = st.sidebar.radio("Selecciona una pestaña", ["Por Departamento", "Frases de Riesgo"])

if archivo:
    df = pd.read_csv(archivo)
    df['comentarios'] = df['comentarios'].astype(str)

    if menu == "Por Departamento":
        st.header("📊 Análisis por Departamento")
        ids = df['id_docente'].dropna().unique()
        id_min, id_max = int(min(ids)), int(max(ids))
        rango = st.slider("Selecciona el rango de ID de docentes", id_min, id_max, (id_min, id_min + 5))

        df_rango = df[(df['id_docente'] >= rango[0]) & (df['id_docente'] <= rango[1])].copy()

        comentarios_invalidos = ['.', '-', '', ' ']
        df_rango['comentario_valido'] = ~df_rango['comentarios'].astype(str).str.strip().isin(comentarios_invalidos)
        df_validos = df_rango[df_rango['comentario_valido']].copy()

        df_validos['comentario_limpio'] = df_validos['comentarios'].astype(str).str.strip().str.replace(r"[\.-]", "", regex=True).str.lower().str[:510]

        with st.spinner("🔍 Analizando sentimientos..."):
            predicciones = modelo_sentimiento(df_validos['comentario_limpio'].tolist())

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
                log_neg = 0
                indice = 0

            resumen_list.append({
                'id_docente': docente_id,
                'asignaturas_impartidas': subset['id_asignatura'].nunique(),
                'alumnos_atendidos': len(subset),
                'comentarios_validos': total_validos,
                'comentarios_negativos': neg,
                'comentarios_neutros': (subset_validos['sentimiento'] == 'NEU').sum(),
                'comentarios_positivos': (subset_validos['sentimiento'] == 'POS').sum(),
                'proporcion_negativa': round(proporcion_neg, 2),
                'indice_severidad': round(indice, 4)
            })

        df_resumen = pd.DataFrame(resumen_list).sort_values(by='indice_severidad', ascending=False)
        st.dataframe(df_resumen)

    elif menu == "Frases de Riesgo":
        st.header("⚠️ Búsqueda de Frases de Riesgo")
        palabra_clave = st.text_input("🔍 Escribe una palabra para rastrear", "difícil")

        df['coincide_palabra'] = df['comentarios'].str.lower().str.contains(palabra_clave.lower())
        df_resultado = df[df['coincide_palabra']].copy()

        if df_resultado.empty:
            st.warning(f"No se encontró la palabra '{palabra_clave}' en ningún comentario.")
        else:
            resumen = df_resultado.groupby('id_docente').agg({
                'comentarios': list,
                'coincide_palabra': 'count'
            }).reset_index().rename(columns={
                'coincide_palabra': f"coincidencias_de_{palabra_clave}",
                'comentarios': 'comentarios_donde_aparece'
            })

            st.dataframe(resumen.sort_values(by=f"coincidencias_de_{palabra_clave}", ascending=False))
