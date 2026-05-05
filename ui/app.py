import streamlit as st
import pandas as pd
import joblib
import sys
import os
import plotly.graph_objects as go

# --- CONFIGURACIÓN DE RUTAS ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Importación de la lógica del compañero
from src.mistral_definitivo import extraer_caracteristicas_cualitativas, decision_arbitrada

# --- DISEÑO VISUAL (VUESTRA PALETA) ---
st.set_page_config(page_title="SoccerMon AI", page_icon="⚽", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    h1, h2, h3, p, span, label { color: #FFFFFF !important; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #00f2ff; }
    .stButton>button {
        background-color: #00f2ff;
        color: #0E1117;
        border-radius: 10px;
        font-weight: bold;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# --- NAVEGACIÓN ---
with st.sidebar:
    logo_path = os.path.join(ROOT_DIR, "reports", "logo_tfg.jpg")
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    st.markdown("<h2 style='text-align: center;'>SoccerMon AI</h2>", unsafe_allow_html=True)
    opcion = st.radio("Menú de Navegación", ["🏠 Bienvenida", "📊 Dashboard", "🤖 Analizador Híbrido"])

# --- PANTALLA 1: BIENVENIDA ---
if opcion == "🏠 Bienvenida":
    st.title("⚽ Bienvenido a SoccerMon AI")
    st.subheader("Sistema de Inteligencia Artificial para la Salud Deportiva")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("""
        Esta plataforma integra **Machine Learning Avanzado** con **Procesamiento de Lenguaje Natural** para reducir el riesgo de lesiones en futbolistas profesionales.
        
        - **Nave 1:** Análisis cualitativo de reportes médicos (Mistral AI).
        - **Nave 2:** Predicción cuantitativa mediante Stacking Ensemble.
        - **Nave 3:** Interfaz de control y toma de decisiones.
        """)
    with col2:
        shap_path = os.path.join(ROOT_DIR, "reports", "shap_importance.png")
        if os.path.exists(shap_path):
            st.image(shap_path, caption="Importancia de Variables (SHAP)")

# --- PANTALLA 2: DASHBOARD ---
elif opcion == "📊 Dashboard":
    st.title("📊 Dashboard de Estado del Equipo")
    csv_path = os.path.join(ROOT_DIR, "data", "dataset_sintetico_soccermon.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path).head(15)
        st.dataframe(df.style.background_gradient(cmap='Blues'))
    else:
        st.error("Archivo de datos no encontrado.")

# --- PANTALLA 3: ANALIZADOR ---
else:
    st.title("🤖 Asistente de Diagnóstico Híbrido")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 📥 Datos del GPS")
        dist = st.number_input("Distancia Total (m)", value=8000)
        acwr_val = st.slider("ACWR (Carga Aguda/Crónica)", 0.5, 2.0, 1.1)
    
    with col_b:
        st.markdown("### 💬 Reporte del Fisioterapeuta")
        prompt = st.text_area("Introduce observaciones:", placeholder="Ej: Molestias en sóleo...")

    if st.button("ANALIZAR AHORA"):
        with st.spinner("Sincronizando Naves..."):
            # 1. Llamada a Mistral
            features_llm = extraer_caracteristicas_cualitativas(prompt)
            
            # 2. Simulación de ML (aquí cargarías el modelo .pkl)
            prob_ml = 0.45 
            
            # 3. Arbitraje Final
            res = decision_arbitrada(prob_ml, features_llm, prompt, acwr_val)
            
            st.markdown(f"## Decisión: **{res['decision']}**")
            
            # Gráfico Gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = res['risk_score'] * 100,
                gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#00f2ff"}}
            ))
            st.plotly_chart(fig)
            st.info(f"**Razonamiento IA:** {res['razonamiento']}")