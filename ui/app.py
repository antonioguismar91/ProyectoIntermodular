# Ubicación: ProyectoIntermodular/ui/app.py
import streamlit as st
import pandas as pd
import joblib
import sys
import os
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime

# --- CONFIGURACIÓN DE RUTAS ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Importar generador de informes PDF
try:
    from generar_informe import generar_informe_pdf
except ImportError:
    generar_informe_pdf = None

# ── Historial de análisis ─────────────────────────────────────────────────────
HISTORIAL_PATH = os.path.join(ROOT_DIR, "data", "historial_analisis.csv")
HISTORIAL_COLS = [
    "fecha", "jugador", "posicion", "decision", "risk_score",
    "acwr", "distancia_m", "fatiga_subj", "horas_sueno",
    "fc_max", "sprints", "tiempo_recuperacion", "fuente"
]

def guardar_historial(row: dict):
    """Añade una fila al CSV de historial, creándolo si no existe."""
    import csv
    existe = os.path.exists(HISTORIAL_PATH)
    with open(HISTORIAL_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORIAL_COLS, extrasaction="ignore")
        if not existe:
            writer.writeheader()
        writer.writerow(row)

def cargar_historial() -> pd.DataFrame:
    """Carga el CSV de historial o devuelve DataFrame vacío."""
    if os.path.exists(HISTORIAL_PATH):
        return pd.read_csv(HISTORIAL_PATH, parse_dates=["fecha"])
    return pd.DataFrame(columns=HISTORIAL_COLS)

# Importación de la lógica del compañero
try:
    from src.mistral_definitivo import extraer_caracteristicas_cualitativas, decision_arbitrada
except ImportError:
    # Stubs para desarrollo/demo
    def extraer_caracteristicas_cualitativas(prompt):
        return {"fatiga": 0.5, "dolor": 0.3, "alarma": False}
    def decision_arbitrada(prob_ml, features_llm, prompt, acwr):
        score = (prob_ml + features_llm.get("fatiga", 0.5)) / 2
        if score > 0.7:
            decision = "🔴 ROJO"
        elif score > 0.4:
            decision = "🟡 ÁMBAR"
        else:
            decision = "🟢 VERDE"
        return {"decision": decision, "risk_score": score, "razonamiento": "Análisis completado correctamente."}

# ─────────────────────────────────────────────────────────────
# DISEÑO VISUAL
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Greenlight", page_icon="⚽", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&display=swap');

/* ── Base ── */
.stApp { background-color: #060b14; }
html, body { background-color: #060b14; }
h1, h2, h3, h4, p, span, label, div { color: #FFFFFF; }
.stMarkdown p { color: #c8d6e5 !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, #060b14 100%);
    border-right: 1px solid rgba(0,242,255,0.2);
}
[data-testid="stSidebar"] * { color: #fff !important; }
.stRadio > label { color: #00f2ff !important; font-family: 'Rajdhani', sans-serif; font-size: 1.05rem; }
.stRadio div[role="radiogroup"] label {
    padding: 8px 12px;
    border-radius: 8px;
    transition: background 0.2s;
}
.stRadio div[role="radiogroup"] label:hover { background: rgba(0,242,255,0.08); }

/* ── Botones ── */
.stButton > button {
    background: linear-gradient(135deg, #00f2ff 0%, #0080ff 100%);
    color: #060b14 !important;
    border-radius: 10px;
    font-weight: 900;
    font-family: 'Orbitron', sans-serif;
    font-size: 0.95rem;
    letter-spacing: 2px;
    border: none;
    padding: 14px 0;
    width: 100%;
    transition: box-shadow 0.3s, transform 0.2s;
}
.stButton > button:hover {
    box-shadow: 0 0 30px rgba(0,242,255,0.6);
    transform: translateY(-2px);
}

/* ── Inputs ── */
.stNumberInput input, .stTextArea textarea, .stSelectbox select {
    background-color: #0d1526 !important;
    color: #fff !important;
    border: 1px solid rgba(0,242,255,0.3) !important;
    border-radius: 8px !important;
}
.stSlider .st-bx { color: #00f2ff; }

/* ── Métricas nativas ── */
[data-testid="stMetricValue"] { color: #00f2ff !important; font-family: 'Orbitron', sans-serif; font-size: 1.8rem !important; }
[data-testid="stMetricDelta"] { color: #7ecfff !important; }
[data-testid="stMetricLabel"] { color: #8899aa !important; font-family: 'Rajdhani', sans-serif; }

/* ── Cards genéricas ── */
.metric-card {
    background: linear-gradient(135deg, #0d1526 0%, #111d30 100%);
    border: 1px solid rgba(0,242,255,0.18);
    border-radius: 16px;
    padding: 22px 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s, box-shadow 0.3s;
}
.metric-card:hover {
    border-color: rgba(0,242,255,0.5);
    box-shadow: 0 0 20px rgba(0,242,255,0.12);
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00f2ff, transparent);
}
.metric-card .card-value {
    font-family: 'Orbitron', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #00f2ff;
    margin: 8px 0 4px;
    line-height: 1;
}
.metric-card .card-label {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.82rem;
    color: #8899aa;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}
.metric-card .card-icon { font-size: 2rem; line-height: 1; }
.metric-card .card-delta {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.8rem;
    color: #4ade80;
    margin-top: 4px;
}

/* ── Nave cards (pantalla bienvenida) ── */
.nave-card {
    background: #0d1526;
    border-left: 4px solid #00f2ff;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 14px;
    transition: transform 0.2s, box-shadow 0.3s;
}
.nave-card:hover {
    transform: translateX(4px);
    box-shadow: -4px 0 20px rgba(0,242,255,0.2);
}
.nave-card h4 { color: #00f2ff !important; margin: 0 0 6px; font-family: 'Orbitron', sans-serif; font-size: 0.9rem; }
.nave-card p  { color: #c8d6e5 !important; margin: 0; font-size: 0.88rem; font-family: 'Rajdhani', sans-serif; }

/* ── Semáforo ── */
.semaforo-wrapper {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    margin: 20px auto;
}
.semaforo-carcasa {
    background: #111;
    border: 3px solid #333;
    border-radius: 60px;
    padding: 18px 22px;
    display: flex;
    flex-direction: column;
    gap: 18px;
    box-shadow: 0 0 40px rgba(0,0,0,0.8), inset 0 0 10px rgba(0,0,0,0.5);
    width: 110px;
}
.luz {
    width: 66px;
    height: 66px;
    border-radius: 50%;
    border: 3px solid #222;
    transition: box-shadow 0.5s, opacity 0.5s;
}
.luz-roja   { background: #3a0000; }
.luz-ambar  { background: #3a2800; }
.luz-verde  { background: #003a00; }
.luz-roja.activa   { background: #ff2222; box-shadow: 0 0 30px #ff0000, 0 0 60px rgba(255,0,0,0.4); }
.luz-ambar.activa  { background: #ffaa00; box-shadow: 0 0 30px #ffaa00, 0 0 60px rgba(255,170,0,0.4); }
.luz-verde.activa  { background: #22ff44; box-shadow: 0 0 30px #00ff44, 0 0 60px rgba(0,255,68,0.4); }
.semaforo-label {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    text-align: center;
    letter-spacing: 3px;
    margin-top: 8px;
}
.semaforo-label.rojo  { color: #ff4444; text-shadow: 0 0 20px rgba(255,68,68,0.6); }
.semaforo-label.ambar { color: #ffaa00; text-shadow: 0 0 20px rgba(255,170,0,0.6); }
.semaforo-label.verde { color: #22ff44; text-shadow: 0 0 20px rgba(34,255,68,0.6); }

/* ── Input section card ── */
.input-section {
    background: #0d1526;
    border: 1px solid rgba(0,242,255,0.15);
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 16px;
}
.input-section h3 { font-family: 'Orbitron', sans-serif; font-size: 0.9rem; color: #00f2ff !important; letter-spacing: 2px; margin-bottom: 14px; }

/* ── Resultado card ── */
.resultado-card {
    background: linear-gradient(135deg, #0d1526 0%, #0a1020 100%);
    border: 1px solid rgba(0,242,255,0.25);
    border-radius: 16px;
    padding: 28px;
}
.resultado-card h4 { font-family: 'Rajdhani', sans-serif; font-size: 0.8rem; color: #8899aa !important; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px; }
.resultado-card p  { color: #c8d6e5 !important; font-family: 'Rajdhani', sans-serif; font-size: 1rem; line-height: 1.6; }

/* ── Títulos principales ── */
.page-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.8rem;
    font-weight: 900;
    color: #00f2ff !important;
    letter-spacing: 3px;
    margin-bottom: 4px;
}
.page-subtitle {
    font-family: 'Rajdhani', sans-serif;
    color: #8899aa !important;
    font-size: 1rem;
    letter-spacing: 1px;
    margin-bottom: 24px;
}

/* ── Divider ── */
.glowing-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #00f2ff, transparent);
    margin: 28px 0;
    border: none;
}

/* ── Jugador card en dashboard ── */
.player-row {
    background: #0d1526;
    border: 1px solid rgba(0,242,255,0.1);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 16px;
    font-family: 'Rajdhani', sans-serif;
}

/* ── Spinner override ── */
.stSpinner > div { border-top-color: #00f2ff !important; }

/* ── Logo sin fondo blanco (blend mode sobre fondo oscuro) ── */
.logo-sidebar img, [data-testid="stSidebar"] img {
    mix-blend-mode: multiply;
    border-radius: 50%;
    background: transparent;
}
.logo-hero img {
    mix-blend-mode: multiply;
    background: transparent;
    border-radius: 50%;
    filter: drop-shadow(0 0 18px rgba(0,242,255,0.25));
}

/* ── Scrollbar personalizada ── */
::-webkit-scrollbar             { width: 6px; height: 6px; }
::-webkit-scrollbar-track       { background: #060b14; }
::-webkit-scrollbar-thumb       { background: #1e2e45; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #00f2ff; }

/* ── Download button ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #00f2ff 0%, #0080ff 100%) !important;
    color: #060b14 !important;
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 900 !important;
    border: none !important;
    border-radius: 10px !important;
    letter-spacing: 2px !important;
}
.stDownloadButton > button:hover {
    box-shadow: 0 0 30px rgba(0,242,255,0.6) !important;
    transform: translateY(-2px) !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo corporativo Greenlight — tamaño reducido y centrado
    logo_path = os.path.join(ROOT_DIR, "reports", "logo_greenlight.jpg")
    _sb_l, _sb_c, _sb_r = st.columns([1, 2, 1])
    with _sb_c:
        if os.path.exists(logo_path):
            st.markdown('<div class="logo-sidebar">', unsafe_allow_html=True)
            st.image(logo_path, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center; padding: 14px 0 8px;">
                <div style="font-size:2.5rem;">🟢</div>
                <div style="font-family:'Orbitron',sans-serif; font-size:0.85rem; color:#00f2ff; letter-spacing:3px;">GREENLIGHT</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(0,242,255,0.15); margin:10px 0 20px;'>", unsafe_allow_html=True)
    opcion = st.radio(
        "NAVEGACIÓN",
        ["🏠  Bienvenida", "📊  Dashboard", "🤖  Analizador Híbrido", "📋  Historial"],
        label_visibility="visible"
    )
    st.markdown("<hr style='border-color:rgba(0,242,255,0.15); margin:20px 0 10px;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Rajdhani',sans-serif; font-size:0.75rem; color:#8899aa; text-align:center; line-height:1.8;">
        Temporada 2024/25<br>
        <span style="color:#00f2ff;">v2.1.0</span> · Elite Edition
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PANTALLA 1: BIENVENIDA
# ─────────────────────────────────────────────────────────────
if opcion == "🏠  Bienvenida":

    # Hero header con logo corporativo
    _logo_hero = os.path.join(ROOT_DIR, "reports", "logo_greenlight.jpg")
    _col_logo, _col_titulo = st.columns([0.65, 2.2], gap="large")
    with _col_logo:
        st.markdown("<div style='padding: 22px 0 0;'>", unsafe_allow_html=True)
        if os.path.exists(_logo_hero):
            st.markdown('<div class="logo-hero">', unsafe_allow_html=True)
            st.image(_logo_hero, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-size:4rem;text-align:center;'>🟢</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with _col_titulo:
        st.markdown("""
        <div style="padding: 28px 0 10px;">
            <div style="font-family:'Rajdhani',sans-serif; font-size:0.85rem; color:#8899aa; letter-spacing:4px; margin-bottom:10px;">
                PLATAFORMA DE RENDIMIENTO ÉLITE
            </div>
            <h1 style="font-family:'Orbitron',sans-serif; font-size:2.8rem; font-weight:900; color:#00f2ff !important;
                       text-shadow: 0 0 40px rgba(0,242,255,0.4); margin:0 0 12px; letter-spacing:5px;">GREENLIGHT</h1>
            <p style="font-family:'Rajdhani',sans-serif; font-size:1.1rem; color:#c8d6e5; margin:0; opacity:0.85; line-height:1.6;">
                Inteligencia Híbrida para la<br>
                <strong style="color:#00f2ff;">Prevención de Lesiones</strong> en el Fútbol Élite
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='glowing-divider'></div>", unsafe_allow_html=True)

    # Métricas impactantes
    st.markdown("<div style='font-family:Orbitron,sans-serif; font-size:0.75rem; color:#8899aa; letter-spacing:3px; margin-bottom:16px;'>INDICADORES CLAVE</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("📈", "AUC 0.82", "CAPACIDAD PREDICTIVA", "Stacking Ensemble"),
        ("🎯", "71.5%", "RECALL DEL MODELO", "Lesiones detectadas"),
        ("🗄️", "30.000", "REGISTROS ENTRENAMIENTO", "Dataset sintético calibrado"),
        ("🧱", "5 capas", "ARQUITECTURA DSS", "ML + NLP + Reglas clínicas"),
    ]
    for col, (icon, val, label, delta) in zip([c1, c2, c3, c4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-icon">{icon}</div>
                <div class="card-value">{val}</div>
                <div class="card-label">{label}</div>
                <div class="card-delta">{delta}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='glowing-divider'></div>", unsafe_allow_html=True)

    # Hero body: descripción + SHAP
    col_izq, col_der = st.columns([1.4, 1], gap="large")

    with col_izq:
        st.markdown("""
        <h2 style="font-family:'Orbitron',sans-serif; font-size:1.1rem; color:#00f2ff !important; letter-spacing:2px; margin-bottom:18px;">
            🚀 EL ECOSISTEMA HÍBRIDO
        </h2>
        <p style="font-family:'Rajdhani',sans-serif; font-size:1.05rem; color:#c8d6e5; line-height:1.7; margin-bottom:24px;">
            Greenlight combina la <strong style="color:#00f2ff;">precisión matemática</strong> del Machine Learning
            con la <strong style="color:#00f2ff;">comprensión contextual</strong> de la IA Generativa,
            creando el primer sistema de arbitraje inteligente para la gestión de carga en fútbol profesional.
        </p>
        """, unsafe_allow_html=True)

        naves = [
            ("🛰️", "Análisis Cognitivo", "Procesamiento de reportes médicos mediante Mistral AI para detectar fatiga subjetiva, señales de alarma clínica y patrones de riesgo en lenguaje natural."),
            ("🧠", "Motor Predictivo", "Algoritmo Stacking Ensemble que integra ACWR, distancias GPS, sprints de alta intensidad y métricas biométricas en una probabilidad de riesgo calibrada."),
            ("💻", "Centro de Mando", "Interfaz de decisión inmediata diseñada para el cuerpo técnico, con semáforo de riesgo, explicabilidad SHAP y trazabilidad de cada análisis."),
        ]
        for icon, titulo, texto in naves:
            st.markdown(f"""
            <div class="nave-card">
                <h4>{icon} {titulo}</h4>
                <p>{texto}</p>
            </div>
            """, unsafe_allow_html=True)

    with col_der:
        shap_path = os.path.join(ROOT_DIR, "reports", "shap_importance.png")
        if os.path.exists(shap_path):
            st.image(shap_path, use_container_width=True)
            st.markdown("<p style='text-align:center; font-family:Rajdhani,sans-serif; color:#8899aa; font-size:0.82rem; letter-spacing:1px;'>INTERPRETABILIDAD DEL MODELO · SHAP VALUES</p>", unsafe_allow_html=True)
        else:
            # Demo radar chart si no hay imagen SHAP
            cats = ["ACWR", "Sprint HD", "Distancia", "Fatiga", "Sueño", "Carga Aguda"]
            fig = go.Figure(go.Scatterpolar(
                r=[0.85, 0.72, 0.90, 0.65, 0.78, 0.88],
                theta=cats, fill='toself',
                line_color='#00f2ff',
                fillcolor='rgba(0,242,255,0.12)',
                name='Importancia SHAP'
            ))
            fig.update_layout(
                polar=dict(
                    bgcolor='rgba(0,0,0,0)',
                    radialaxis=dict(visible=True, range=[0,1], color='#8899aa', gridcolor='rgba(255,255,255,0.1)'),
                    angularaxis=dict(color='#c8d6e5', gridcolor='rgba(255,255,255,0.1)')
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#c8d6e5',
                margin=dict(l=40, r=40, t=30, b=30),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.markdown("<p style='text-align:center; font-family:Rajdhani,sans-serif; color:#8899aa; font-size:0.82rem; letter-spacing:1px;'>IMPORTANCIA DE VARIABLES · RADAR SHAP</p>", unsafe_allow_html=True)

    st.markdown("<div class='glowing-divider'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0d1526,#0a1824); border:1px solid rgba(0,242,255,0.2);
                border-radius:12px; padding:18px 22px; text-align:center;">
        <span style="font-family:'Orbitron',sans-serif; font-size:0.8rem; color:#00f2ff; letter-spacing:2px;">💡 PRÓXIMO PASO</span>
        <p style="font-family:'Rajdhani',sans-serif; font-size:1rem; color:#c8d6e5; margin:8px 0 0;">
            Navega al <strong style="color:#00f2ff;">Analizador Híbrido</strong> para evaluar el estado de un jugador en tiempo real.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PANTALLA 2: DASHBOARD
# ─────────────────────────────────────────────────────────────
elif opcion == "📊  Dashboard":

    st.markdown("<div class='page-title'>DASHBOARD DEL EQUIPO</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Estado de carga y riesgo · Temporada 2024/25</div>", unsafe_allow_html=True)

    # ── Cargar dataset real ──
    # Columnas: player_id, date, distance_m, hsr_distance_m, acwr, ctl, atl,
    #           fatigue_score, sleep_quality, stress, muscle_soreness, injury_risk (0/1)
    csv_path = os.path.join(ROOT_DIR, "data", "dataset_sintetico_soccermon.csv")
    if os.path.exists(csv_path):
        df_full = pd.read_csv(csv_path)
    else:
        np.random.seed(42)
        n = 200
        df_full = pd.DataFrame({
            "player_id":       [f"P{str(i).zfill(4)}" for i in range(n)],
            "date":            pd.date_range("2023-01-01", periods=n, freq="D").astype(str),
            "distance_m":      np.random.randint(4500, 13000, n),
            "hsr_distance_m":  np.random.randint(200, 1200, n),
            "acwr":            np.round(np.random.uniform(0.8, 2.1, n), 3),
            "ctl":             np.round(np.random.uniform(40, 90, n), 2),
            "atl":             np.round(np.random.uniform(40, 95, n), 2),
            "fatigue_score":   np.random.randint(1, 11, n),
            "sleep_quality":   np.random.randint(1, 11, n),
            "stress":          np.random.randint(1, 11, n),
            "muscle_soreness": np.random.randint(1, 11, n),
            "injury_risk":     np.random.choice([0, 1], n, p=[0.83, 0.17]),
        })

    # Un registro por jugador (último disponible)
    df_last = df_full.drop_duplicates(subset="player_id", keep="last").head(20).copy()

    # Score continuo de riesgo compuesto para visualizaciones
    df_last["risk_score"] = (
        df_last["injury_risk"] * 0.55 +
        ((df_last["acwr"] - 0.8) / 1.5).clip(0, 1) * 0.25 +
        (df_last["fatigue_score"] / 10) * 0.20
    ).round(3)

    df_last["estado"] = df_last.apply(
        lambda r: "🔴 ALTO"  if r["injury_risk"] == 1 or r["acwr"] > 1.5
             else "🟡 MEDIO" if r["acwr"] > 1.2 or r["fatigue_score"] >= 7
             else "🟢 BAJO",
        axis=1
    )

    # ── ALERTAS AUTOMÁTICAS ───────────────────────────────────────────────────
    _alerta_critica = df_last[(df_last["acwr"] > 1.5) & (df_last["fatigue_score"] >= 7)]
    _alerta_acwr    = df_last[(df_last["acwr"] > 1.5) & (df_last["fatigue_score"] < 7)]
    _alerta_fatiga  = df_last[(df_last["acwr"] <= 1.5) & (df_last["fatigue_score"] >= 8)]
    _alerta_sueno   = df_last[df_last["sleep_quality"] <= 3] if "sleep_quality" in df_last.columns else df_last.iloc[0:0]
    _alerta_stress  = df_last[df_last["stress"] >= 8]        if "stress"         in df_last.columns else df_last.iloc[0:0]

    _hay_alertas = any([len(_alerta_critica)>0, len(_alerta_acwr)>0,
                        len(_alerta_fatiga)>0,  len(_alerta_sueno)>0, len(_alerta_stress)>0])

    def _alerta_html(emoji, color, titulo, n_jug, ids, mensaje):
        return f"""
        <div style="background:linear-gradient(135deg,rgba(0,0,0,0.6),rgba(0,0,0,0.4));
             border:1px solid {color};border-left:5px solid {color};
             border-radius:12px;padding:14px 18px;margin-bottom:8px;">
            <div style="display:flex;align-items:center;gap:12px;">
                <div style="font-size:1.5rem;">{emoji}</div>
                <div>
                    <div style="font-family:Orbitron,sans-serif;font-size:0.72rem;
                         color:{color};letter-spacing:2px;margin-bottom:3px;">{titulo}</div>
                    <div style="font-family:Rajdhani,sans-serif;font-size:0.92rem;color:#c8d6e5;">
                        <strong style="color:{color};">{n_jug} jugador(es):</strong> {ids} — {mensaje}
                    </div>
                </div>
            </div>
        </div>"""

    if _hay_alertas:
        with st.expander("🚨  ALERTAS ACTIVAS — Jugadores que requieren atención inmediata", expanded=True):
            if len(_alerta_critica) > 0:
                st.markdown(_alerta_html("🔴","#ff4444","RIESGO CRÍTICO · ACWR + FATIGA",
                    len(_alerta_critica), ", ".join(_alerta_critica["player_id"].tolist()),
                    "ACWR >1.5 y fatiga ≥7. Reducción de carga inmediata recomendada."),
                    unsafe_allow_html=True)
            if len(_alerta_acwr) > 0:
                st.markdown(_alerta_html("⚡","#ffaa00","SOBRECARGA AGUDA · ACWR ELEVADO",
                    len(_alerta_acwr), ", ".join(_alerta_acwr["player_id"].tolist()),
                    "ACWR >1.5. Monitorizar carga en próximas 48h."),
                    unsafe_allow_html=True)
            if len(_alerta_fatiga) > 0:
                st.markdown(_alerta_html("😓","#ffee00","FATIGA SUBJETIVA ALTA",
                    len(_alerta_fatiga), ", ".join(_alerta_fatiga["player_id"].tolist()),
                    "Fatiga ≥8/10. Evaluar estado físico y mental."),
                    unsafe_allow_html=True)
            if len(_alerta_sueno) > 0:
                st.markdown(_alerta_html("😴","#7b8cff","CALIDAD DE SUEÑO DEFICIENTE",
                    len(_alerta_sueno), ", ".join(_alerta_sueno["player_id"].tolist()),
                    "Sueño ≤3/10. Derivar a protocolo de recuperación."),
                    unsafe_allow_html=True)
            if len(_alerta_stress) > 0:
                st.markdown(_alerta_html("🧠","#cc66ff","ESTRÉS ELEVADO",
                    len(_alerta_stress), ", ".join(_alerta_stress["player_id"].tolist()),
                    "Estrés ≥8/10. Considerar apoyo psicológico o descanso."),
                    unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#011a05,#021f06);
             border:1px solid #22ff44;border-left:5px solid #22ff44;
             border-radius:12px;padding:14px 20px;margin-bottom:4px;">
            <div style="display:flex;align-items:center;gap:10px;">
                <div style="font-size:1.3rem;">✅</div>
                <div style="font-family:Rajdhani,sans-serif;font-size:0.95rem;color:#c8d6e5;">
                    <strong style="color:#22ff44;">Sin alertas activas</strong>
                    — Todos los jugadores están dentro de los umbrales de seguridad.
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='glowing-divider'></div>", unsafe_allow_html=True)

    # ── KPIs ──
    n_alto  = int((df_last["estado"] == "🔴 ALTO").sum())
    n_medio = int((df_last["estado"] == "🟡 MEDIO").sum())
    n_bajo  = int((df_last["estado"] == "🟢 BAJO").sum())
    acwr_prom   = round(df_last["acwr"].mean(), 2)
    dist_prom   = int(df_last["distance_m"].mean())
    fatiga_prom = round(df_last["fatigue_score"].mean(), 1)

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    kpis = [
        ("🔴", str(n_alto),        "RIESGO ALTO",   "Requieren atención"),
        ("🟡", str(n_medio),       "RIESGO MEDIO",  "Monitorizar"),
        ("🟢", str(n_bajo),        "RIESGO BAJO",   "Estado óptimo"),
        ("📐", str(acwr_prom),     "ACWR MEDIO",    "Ratio aguda/crónica"),
        ("🏃", f"{dist_prom:,}m",  "DIST. MEDIA",   "Última sesión"),
        ("😴", str(fatiga_prom),   "FATIGA MEDIA",  "Escala 1-10"),
    ]
    for col, (icon, val, label, delta) in zip([k1, k2, k3, k4, k5, k6], kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-icon">{icon}</div>
                <div class="card-value" style="font-size:1.45rem;">{val}</div>
                <div class="card-label">{label}</div>
                <div class="card-delta">{delta}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='glowing-divider'></div>", unsafe_allow_html=True)

    # ── Gráficos fila superior ──
    col_g1, col_g2 = st.columns([1.2, 1], gap="large")

    with col_g1:
        st.markdown("<div style='font-family:Orbitron,sans-serif;font-size:0.75rem;color:#8899aa;letter-spacing:2px;margin-bottom:12px;'>RIESGO COMPUESTO POR JUGADOR</div>", unsafe_allow_html=True)
        colores_bar = ["#ff4444" if e == "🔴 ALTO" else "#ffaa00" if e == "🟡 MEDIO" else "#22ff44"
                       for e in df_last["estado"]]
        fig_bar = go.Figure(go.Bar(
            x=df_last["player_id"],
            y=df_last["risk_score"],
            marker_color=colores_bar,
            marker_line_width=0,
            hovertemplate="<b>%{x}</b><br>Score: %{y:.2f}<extra></extra>",
        ))
        fig_bar.add_hline(y=0.55, line_dash="dash", line_color="rgba(255,68,68,0.5)",
                          annotation_text="Umbral alto", annotation_font_color="#ff4444")
        fig_bar.add_hline(y=0.30, line_dash="dash", line_color="rgba(255,170,0,0.5)",
                          annotation_text="Umbral medio", annotation_font_color="#ffaa00")
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c8d6e5", font_family="Rajdhani",
            yaxis=dict(title="Score riesgo", gridcolor="rgba(255,255,255,0.06)", range=[0, 1]),
            xaxis=dict(tickangle=-35, gridcolor="rgba(255,255,255,0.04)"),
            margin=dict(l=10, r=10, t=10, b=80), height=290,
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    with col_g2:
        st.markdown("<div style='font-family:Orbitron,sans-serif;font-size:0.75rem;color:#8899aa;letter-spacing:2px;margin-bottom:12px;'>ACWR vs FATIGA SUBJETIVA</div>", unsafe_allow_html=True)
        colores_sc = ["#ff4444" if e == "🔴 ALTO" else "#ffaa00" if e == "🟡 MEDIO" else "#22ff44"
                      for e in df_last["estado"]]
        fig_sc = go.Figure(go.Scatter(
            x=df_last["acwr"],
            y=df_last["fatigue_score"],
            mode="markers",
            marker=dict(color=colores_sc, size=13, opacity=0.85,
                        line=dict(width=1, color="rgba(0,242,255,0.35)")),
            text=df_last["player_id"],
            hovertemplate="<b>%{text}</b><br>ACWR: %{x:.2f}<br>Fatiga: %{y}<extra></extra>"
        ))
        fig_sc.add_vline(x=1.5, line_dash="dash", line_color="rgba(255,68,68,0.4)",
                         annotation_text="ACWR crítico", annotation_font_color="#ff4444")
        fig_sc.add_hline(y=7, line_dash="dash", line_color="rgba(255,170,0,0.4)",
                         annotation_text="Fatiga alta", annotation_font_color="#ffaa00")
        fig_sc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c8d6e5", font_family="Rajdhani",
            xaxis=dict(title="ACWR", gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="Fatiga (1-10)", gridcolor="rgba(255,255,255,0.06)", range=[0, 11]),
            margin=dict(l=10, r=10, t=10, b=30), height=290,
        )
        st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div class='glowing-divider'></div>", unsafe_allow_html=True)

    # ── Fila inferior: Donut + Histograma ACWR + Tabla ──
    col_d1, col_d2, col_d3 = st.columns([0.8, 1, 1.6], gap="large")

    with col_d1:
        st.markdown("<div style='font-family:Orbitron,sans-serif;font-size:0.75rem;color:#8899aa;letter-spacing:2px;margin-bottom:12px;'>ESTADOS DEL PLANTEL</div>", unsafe_allow_html=True)
        fig_donut = go.Figure(go.Pie(
            values=[n_alto, n_medio, n_bajo],
            labels=["Alto", "Medio", "Bajo"],
            hole=0.65,
            marker_colors=["#ff4444", "#ffaa00", "#22ff44"],
            textinfo="label+percent",
            textfont=dict(color="#c8d6e5", family="Rajdhani", size=12),
        ))
        fig_donut.add_annotation(
            text=f"{len(df_last)}<br>jugadores",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#00f2ff", family="Orbitron")
        )
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font_color="#c8d6e5",
            showlegend=False, margin=dict(l=0, r=0, t=10, b=0), height=230
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

    with col_d2:
        st.markdown("<div style='font-family:Orbitron,sans-serif;font-size:0.75rem;color:#8899aa;letter-spacing:2px;margin-bottom:12px;'>DISTRIBUCIÓN DE ACWR (EQUIPO)</div>", unsafe_allow_html=True)
        fig_hist = go.Figure(go.Histogram(
            x=df_full["acwr"],
            nbinsx=30,
            marker_color="#00f2ff",
            marker_line_width=0,
            opacity=0.75,
        ))
        fig_hist.add_vline(x=0.8,  line_dash="dot",  line_color="#ffaa00", annotation_text="Min óptimo", annotation_font_color="#ffaa00")
        fig_hist.add_vline(x=1.3,  line_dash="dot",  line_color="#22ff44", annotation_text="Max óptimo", annotation_font_color="#22ff44")
        fig_hist.add_vline(x=1.5,  line_dash="dash", line_color="#ff4444", annotation_text="Zona riesgo", annotation_font_color="#ff4444")
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c8d6e5", font_family="Rajdhani",
            xaxis=dict(title="ACWR", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Frecuencia", gridcolor="rgba(255,255,255,0.05)"),
            margin=dict(l=10, r=10, t=10, b=30), height=230, bargap=0.05,
        )
        st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

    with col_d3:
        st.markdown("<div style='font-family:Orbitron,sans-serif;font-size:0.75rem;color:#8899aa;letter-spacing:2px;margin-bottom:12px;'>ESTADO DETALLADO DEL PLANTEL</div>", unsafe_allow_html=True)
        df_tabla = df_last[[
            "player_id", "distance_m", "hsr_distance_m", "acwr",
            "fatigue_score", "sleep_quality", "stress", "muscle_soreness", "estado"
        ]].copy()
        df_tabla["acwr"] = df_tabla["acwr"].round(2)
        st.dataframe(
            df_tabla.rename(columns={
                "player_id":       "Jugador",
                "distance_m":      "Dist. (m)",
                "hsr_distance_m":  "HSR (m)",
                "acwr":            "ACWR",
                "fatigue_score":   "Fatiga",
                "sleep_quality":   "Sueño",
                "stress":          "Estrés",
                "muscle_soreness": "Dolor musc.",
                "estado":          "Estado",
            }),
            use_container_width=True,
            height=250,
            hide_index=True,
        )


# ─────────────────────────────────────────────────────────────
# PANTALLA 3: ANALIZADOR HÍBRIDO
# ─────────────────────────────────────────────────────────────
elif opcion == "🤖  Analizador Híbrido":
    st.markdown("<div class='page-title'>ANALIZADOR HÍBRIDO</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Diagnóstico inteligente de riesgo · ML + IA Generativa</div>", unsafe_allow_html=True)

    col_form, col_resultado = st.columns([1.1, 1], gap="large")

    with col_form:
        # ── Bloque GPS & Carga ──
        with st.container():
            st.markdown("""<div style="background:#0d1526;border:1px solid rgba(0,242,255,0.15);
                border-radius:14px;padding:22px;margin-bottom:14px;">
                <div style="font-family:Orbitron,sans-serif;font-size:0.78rem;color:#00f2ff;letter-spacing:2px;margin-bottom:16px;">
                🛰️ GPS & CARGA EXTERNA</div>""", unsafe_allow_html=True)

            g1, g2 = st.columns(2)
            with g1:
                dist = st.number_input("Distancia Total (m)", value=8000, step=100, min_value=0, max_value=20000)
                dist_hd = st.number_input("Distancia HD >21 km/h (m)", value=900, step=50, min_value=0)
                sprints = st.number_input("Nº Sprints (>25 km/h)", value=18, step=1, min_value=0)
            with g2:
                acwr_val = st.slider("ACWR", 0.5, 2.5, 1.1, 0.01,
                                     help="Ratio Carga Aguda/Crónica. Zona óptima: 0.8–1.3")
                acelerac = st.number_input("Aceleraciones Explosivas", value=32, step=1, min_value=0)
                tiempo_juego = st.number_input("Minutos jugados (última semana)", value=270, step=10, min_value=0)

            st.markdown("</div>", unsafe_allow_html=True)

        # ── Bloque Biométrico ──
        with st.container():
            st.markdown("""<div style="background:#0d1526;border:1px solid rgba(0,242,255,0.15);
                border-radius:14px;padding:22px;margin-bottom:14px;">
                <div style="font-family:Orbitron,sans-serif;font-size:0.78rem;color:#00f2ff;letter-spacing:2px;margin-bottom:16px;">
                💓 DATOS BIOMÉTRICOS</div>""", unsafe_allow_html=True)

            b1, b2, b3 = st.columns(3)
            with b1:
                fc_max = st.number_input("FC Máx Sesión (%)", value=88, step=1, min_value=60, max_value=100)
            with b2:
                horas_sueno = st.number_input("Horas de Sueño", value=7.5, step=0.5, min_value=3.0, max_value=12.0)
            with b3:
                fatiga_subj = st.slider("Fatiga Subjetiva (1-10)", 1, 10, 5)

            st.markdown("</div>", unsafe_allow_html=True)

        # ── Bloque Posición ──
        with st.container():
            st.markdown("""<div style="background:#0d1526;border:1px solid rgba(0,242,255,0.15);
                border-radius:14px;padding:16px 22px;margin-bottom:14px;">
                <div style="font-family:Orbitron,sans-serif;font-size:0.78rem;color:#00f2ff;letter-spacing:2px;margin-bottom:12px;">
                🧍 PERFIL DEL JUGADOR</div>""", unsafe_allow_html=True)

            p1, p2 = st.columns([1.2, 1])
            with p1:
                nombre_jugador = st.text_input(
                    "Nombre / ID del jugador",
                    value="",
                    placeholder="Ej: Jugador 7 / J. García",
                    label_visibility="collapsed",
                    key="nombre_jugador"
                )
            with p2:
                posicion_jugador = st.selectbox(
                    "Posición en el campo",
                    ["Defensa", "Centrocampista", "Delantero", "Portero"],
                    label_visibility="collapsed"
                )
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Bloque Reporte Fisio + PDF ──
        with st.container():
            st.markdown("""<div style="background:#0d1526;border:1px solid rgba(0,242,255,0.15);
                border-radius:14px;padding:22px;margin-bottom:18px;">
                <div style="font-family:Orbitron,sans-serif;font-size:0.78rem;color:#00f2ff;letter-spacing:2px;margin-bottom:14px;">
                💬 REPORTE DEL FISIOTERAPEUTA</div>""", unsafe_allow_html=True)

            # Tab: texto manual vs PDF
            tab_texto, tab_pdf = st.tabs(["✏️  Texto manual", "📄  Subir PDF"])

            with tab_texto:
                prompt_manual = st.text_area(
                    "Observaciones clínicas:",
                    placeholder="Ej: Jugador refiere molestias en sóleo derecho post-entrenamiento. Leve tensión muscular. Sin inflamación visible...",
                    height=120,
                    label_visibility="collapsed",
                    key="prompt_manual"
                )

            with tab_pdf:
                st.markdown("""
                <div style="font-family:Rajdhani,sans-serif;font-size:0.88rem;color:#8899aa;margin-bottom:10px;">
                Sube el informe médico en PDF. El sistema extraerá el texto automáticamente
                y lo usará como reporte clínico.
                </div>""", unsafe_allow_html=True)

                pdf_file = st.file_uploader(
                    "Arrastra o selecciona el PDF",
                    type=["pdf"],
                    label_visibility="collapsed",
                    key="pdf_uploader"
                )

                prompt_pdf = ""
                if pdf_file is not None:
                    try:
                        import io, pdfplumber
                        pdf_bytes = pdf_file.read()
                        pages_text = []
                        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf_obj:
                            n_pages = len(pdf_obj.pages)
                            for page in pdf_obj.pages:
                                t = page.extract_text()
                                if t:
                                    pages_text.append(t.strip())
                        prompt_pdf = "\n\n".join(pages_text)

                        if prompt_pdf.strip():
                            n_chars = len(prompt_pdf)
                            st.markdown(f"""
                            <div style="background:rgba(0,242,255,0.06);border:1px solid rgba(0,242,255,0.25);
                                 border-radius:10px;padding:14px 18px;margin-top:8px;">
                                <div style="font-family:Orbitron,sans-serif;font-size:0.7rem;color:#00f2ff;letter-spacing:2px;margin-bottom:6px;">
                                ✅ PDF CARGADO CORRECTAMENTE</div>
                                <div style="font-family:Rajdhani,sans-serif;font-size:0.85rem;color:#c8d6e5;">
                                📄 {pdf_file.name} &nbsp;·&nbsp; {n_pages} página{"s" if n_pages>1 else ""} &nbsp;·&nbsp; {n_chars:,} caracteres extraídos
                                </div>
                            </div>""", unsafe_allow_html=True)

                            with st.expander("👁️  Vista previa del texto extraído"):
                                st.text_area("", value=prompt_pdf[:3000] + ("…" if len(prompt_pdf) > 3000 else ""),
                                             height=160, disabled=True, label_visibility="collapsed")
                        else:
                            st.warning("⚠️ No se pudo extraer texto. El PDF puede estar escaneado (imagen). "
                                       "Prueba a copiar el texto manualmente en la pestaña de texto.")
                    except Exception as e:
                        st.error(f"Error al leer el PDF: {e}")

            # Combinar fuentes: PDF tiene prioridad si está cargado
            prompt = prompt_pdf.strip() if prompt_pdf.strip() else prompt_manual.strip()

            # Indicador de fuente activa
            if prompt_pdf.strip():
                st.markdown("""<div style="font-family:Rajdhani,sans-serif;font-size:0.8rem;
                    color:#00f2ff;margin-top:6px;">📄 Fuente activa: PDF</div>""", unsafe_allow_html=True)
            elif prompt_manual.strip():
                st.markdown("""<div style="font-family:Rajdhani,sans-serif;font-size:0.8rem;
                    color:#8899aa;margin-top:6px;">✏️ Fuente activa: texto manual</div>""", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        analizar = st.button("⚡  ANALIZAR AHORA", use_container_width=True)

    # ─── COLUMNA RESULTADO ───
    with col_resultado:
        st.markdown("""
        <div style="background:#0d1526;border:1px solid rgba(0,242,255,0.12);
             border-radius:16px;padding:24px;text-align:center;margin-bottom:14px;">
            <div style="font-family:Orbitron,sans-serif;font-size:0.75rem;color:#8899aa;letter-spacing:3px;margin-bottom:4px;">
                DIAGNÓSTICO EN TIEMPO REAL
            </div>
            <div style="font-family:Rajdhani,sans-serif;font-size:0.9rem;color:#c8d6e5;opacity:0.7;">
                Introduce datos y pulsa ANALIZAR
            </div>
        </div>
        """, unsafe_allow_html=True)

        resultado_placeholder     = st.empty()
        gauge_placeholder         = st.empty()
        reason_placeholder        = st.empty()
        tiempo_placeholder        = st.empty()
        radar_placeholder         = st.empty()
        pdf_diagnosis_placeholder = st.empty()

        # Semáforo inicial (apagado)
        resultado_placeholder.markdown("""
        <div class="semaforo-wrapper">
            <div class="semaforo-carcasa">
                <div class="luz luz-roja"></div>
                <div class="luz luz-ambar"></div>
                <div class="luz luz-verde"></div>
            </div>
            <div style="font-family:Rajdhani,sans-serif;font-size:0.85rem;color:#8899aa;letter-spacing:2px;margin-top:8px;">
                EN ESPERA
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ─── LÓGICA DE ANÁLISIS ───
    if analizar:
        with st.spinner("Sincronizando módulos de análisis..."):

            # 1. Construir fallback de features con las claves EXACTAS que espera decision_arbitrada
            features_fallback = {
                "zona_lesion":                "desconocida",
                "severidad_lesion":           fatiga_subj,          # proxy
                "tendencia_lesion":           "estable",
                "fatiga_cualitativa":         float(fatiga_subj),
                "calidad_suenio_cualitativa": round((horas_sueno - 3) / 6 * 9 + 1, 1),  # mapeo 3-9h → 1-10
                "estres_cualitativo":         5.0,
                "dolor_muscular_cualitativo": 3.0,
                "confianza_extraccion":       0.3,
                "grado_lesion":               "leve",
            }

            # 2. Llamada real a Mistral si hay reporte; fallback si no
            if prompt.strip():
                try:
                    features_llm = extraer_caracteristicas_cualitativas(prompt, use_llm=True)
                except Exception as e:
                    st.warning(f"Mistral no disponible, usando parser heurístico. ({e})")
                    features_llm = features_fallback
            else:
                features_llm = features_fallback

            # 3. Score ML compuesto (sustituir por modelo .pkl cuando esté disponible)
            prob_ml = min(1.0, max(0.0,
                (acwr_val - 0.8) / 1.7 * 0.35 +
                (features_llm.get("fatiga_cualitativa", fatiga_subj) / 10) * 0.25 +
                ((fc_max - 60) / 40) * 0.15 +
                ((10 - features_llm.get("calidad_suenio_cualitativa", horas_sueno)) / 9) * 0.15 +
                (features_llm.get("dolor_muscular_cualitativo", 3) / 10) * 0.10
            ))

            # 4. Decisión arbitrada con firma COMPLETA
            try:
                res = decision_arbitrada(
                    ml_prob=prob_ml,
                    features=features_llm,
                    reporte=prompt,
                    acwr_real=acwr_val,
                    use_llm=bool(prompt.strip()),
                    posicion_jugador=posicion_jugador,
                )
            except Exception as e:
                st.error(f"Error en decision_arbitrada: {e}")
                st.stop()

            score        = res["risk_score"]
            decision_txt = res["decision"]          # "ROJO" | "ÁMBAR" | "VERDE"
            razon        = res["razonamiento"]
            tiempo_rec   = res.get("tiempo_recuperacion", "No estimado")
            fuente       = res.get("fuente", "—")

            # 5. Mapear decisión a colores/semáforo
            # decision_arbitrada devuelve "ROJO", "ÁMBAR" o "VERDE" (sin emojis)
            d_upper = decision_txt.upper()
            if "ROJO" in d_upper:
                estado_color = "rojo";  luz_activa = "roja";  etiqueta = "RIESGO ALTO";  gauge_color = "#ff4444"
            elif "MBAR" in d_upper or "AMBER" in d_upper:   # cubre ÁMBAR y AMBAR
                estado_color = "ambar"; luz_activa = "ambar"; etiqueta = "RIESGO MEDIO"; gauge_color = "#ffaa00"
            else:
                estado_color = "verde"; luz_activa = "verde"; etiqueta = "RIESGO BAJO";  gauge_color = "#22ff44"

            # ── Semáforo ──
            resultado_placeholder.markdown(f"""
            <div class="semaforo-wrapper">
                <div class="semaforo-carcasa">
                    <div class="luz luz-roja {'activa' if luz_activa == 'roja' else ''}"></div>
                    <div class="luz luz-ambar {'activa' if luz_activa == 'ambar' else ''}"></div>
                    <div class="luz luz-verde {'activa' if luz_activa == 'verde' else ''}"></div>
                </div>
                <div class="semaforo-label {estado_color}">{etiqueta}</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Gauge ──
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(score * 100, 1),
                number={"suffix": "%", "font": {"color": gauge_color, "family": "Orbitron", "size": 28}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#8899aa",
                             "tickfont": {"color": "#8899aa", "family": "Rajdhani"}},
                    "bar": {"color": gauge_color, "thickness": 0.22},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0,  40],  "color": "rgba(34,255,68,0.07)"},
                        {"range": [40, 70],  "color": "rgba(255,170,0,0.07)"},
                        {"range": [70, 100], "color": "rgba(255,68,68,0.07)"},
                    ],
                    "threshold": {"line": {"color": gauge_color, "width": 3},
                                  "thickness": 0.8, "value": score * 100}
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font_color="#c8d6e5",
                margin=dict(l=20, r=20, t=10, b=0), height=200,
            )
            gauge_placeholder.plotly_chart(fig_gauge, use_container_width=True,
                                           config={"displayModeBar": False})

            # ── Razonamiento + tiempo de baja ──
            reason_placeholder.markdown(f"""
            <div class="resultado-card">
                <h4>🧠 RAZONAMIENTO IA <span style="float:right;color:#00f2ff;font-size:0.75rem;">Fuente: {fuente}</span></h4>
                <p>{razon}</p>
            </div>
            """, unsafe_allow_html=True)

            tiempo_placeholder.markdown(f"""
            <div style="background:#0d1526;border:1px solid rgba(0,242,255,0.15);border-radius:12px;
                 padding:16px 20px;margin-top:10px;display:flex;align-items:center;gap:14px;">
                <div style="font-size:1.8rem;">🗓️</div>
                <div>
                    <div style="font-family:Orbitron,sans-serif;font-size:0.7rem;color:#8899aa;letter-spacing:2px;">TIEMPO ESTIMADO DE BAJA</div>
                    <div style="font-family:Orbitron,sans-serif;font-size:1.1rem;color:{gauge_color};margin-top:4px;">{tiempo_rec}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Radar multidimensional ──
            cats_radar = ["ACWR", "Fatiga", "Sueño", "FC Máx", "Dolor musc.", "Estrés"]
            vals_radar = [
                min(1.0, acwr_val / 2.5),
                features_llm.get("fatiga_cualitativa", fatiga_subj) / 10,
                max(0.0, 1 - (features_llm.get("calidad_suenio_cualitativa", 5) - 1) / 9),
                (fc_max - 60) / 40,
                features_llm.get("dolor_muscular_cualitativo", 3) / 10,
                features_llm.get("estres_cualitativo", 5) / 10,
            ]
            fill_rgb = "255,68,68" if estado_color == "rojo" else "255,170,0" if estado_color == "ambar" else "34,255,68"
            fig_radar = go.Figure(go.Scatterpolar(
                r=vals_radar + [vals_radar[0]],
                theta=cats_radar + [cats_radar[0]],
                fill="toself",
                line_color=gauge_color,
                fillcolor=f"rgba({fill_rgb},0.12)",
                name="Perfil de riesgo"
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 1], color="#8899aa",
                                   gridcolor="rgba(255,255,255,0.08)"),
                    angularaxis=dict(color="#c8d6e5", gridcolor="rgba(255,255,255,0.08)",
                                     tickfont=dict(family="Rajdhani", size=11))
                ),
                paper_bgcolor="rgba(0,0,0,0)", font_color="#c8d6e5",
                margin=dict(l=20, r=20, t=20, b=10), height=260, showlegend=False
            )
            st.markdown("<div style='font-family:Orbitron,sans-serif;font-size:0.72rem;color:#8899aa;letter-spacing:2px;margin-top:8px;'>PERFIL MULTIDIMENSIONAL DE RIESGO</div>", unsafe_allow_html=True)
            radar_placeholder.plotly_chart(fig_radar, use_container_width=True,
                                           config={"displayModeBar": False})

            # ── Diagnóstico narrativo del PDF (solo si la fuente es PDF) ──
            if prompt_pdf.strip():
                pdf_diagnosis_placeholder.markdown("""
                <div style="background:#0d1526;border:1px solid rgba(0,242,255,0.2);
                     border-radius:14px;padding:20px 22px;margin-top:14px;">
                    <div style="font-family:Orbitron,sans-serif;font-size:0.72rem;color:#00f2ff;
                         letter-spacing:2px;margin-bottom:10px;">📄 ANÁLISIS NARRATIVO DEL PDF</div>
                    <div style="font-family:Rajdhani,sans-serif;color:#8899aa;font-size:0.9rem;">
                        Generando diagnóstico completo…
                    </div>
                </div>
                """, unsafe_allow_html=True)

                try:
                    # Llamada a Mistral para diagnóstico narrativo completo del PDF
                    import requests as _req, json as _json

                    _mistral_url = "https://api.mistral.ai/v1/chat/completions"
                    _mistral_key = st.secrets.get("MISTRAL_API_KEY", None)
                    _headers = {
                        "Authorization": f"Bearer {_mistral_key}",
                        "Content-Type": "application/json"
                    }
                    _texto_truncado = prompt_pdf[:6000]  # límite seguro de tokens
                    _payload = {
                        "model": "mistral-small-latest",
                        "max_tokens": 600,
                        "temperature": 0.3,
                        "messages": [{
                            "role": "user",
                            "content": f"""Eres un médico deportivo experto. Analiza el siguiente informe médico de un futbolista profesional y redacta:

1. **Diagnóstico principal**: qué lesión o condición presenta.
2. **Zonas afectadas**: anatomía implicada.
3. **Gravedad y tendencia**: leve / moderada / grave, y si mejora o empeora.
4. **Recomendaciones inmediatas**: protocolo de actuación para el cuerpo técnico.
5. **Tiempo estimado de baja**: según la literatura médico-deportiva.

Sé conciso, claro y usa terminología médico-deportiva. Responde en español.

INFORME MÉDICO:
{_texto_truncado}"""
                        }]
                    }

                    _resp = _req.post(_mistral_url, headers=_headers, json=_payload, timeout=40)
                    if _resp.status_code == 200:
                        _narrativo = _resp.json()["choices"][0]["message"]["content"]
                    else:
                        _narrativo = f"(Mistral no disponible — HTTP {_resp.status_code})"
                except Exception as _e:
                    _narrativo = f"(Error al conectar con Mistral: {_e})"

                # Convertir markdown básico a HTML para st.markdown
                import re as _re
                _html_narr = _narrativo
                _html_narr = _re.sub(r'\*\*(.+?)\*\*', r'<strong style="color:#00f2ff;">\1</strong>', _html_narr)
                _html_narr = _re.sub(r'\*(.+?)\*',     r'<em>\1</em>', _html_narr)
                _html_narr = _re.sub(r'\n\n', '<br><br>', _html_narr)
                _html_narr = _re.sub(r'\n',   '<br>',    _html_narr)

                pdf_diagnosis_placeholder.markdown(f"""
                <div style="background:#0d1526;border:1px solid rgba(0,242,255,0.2);
                     border-radius:14px;padding:20px 22px;margin-top:14px;">
                    <div style="font-family:Orbitron,sans-serif;font-size:0.72rem;color:#00f2ff;
                         letter-spacing:2px;margin-bottom:14px;">📄 ANÁLISIS NARRATIVO DEL PDF</div>
                    <div style="font-family:Rajdhani,sans-serif;color:#c8d6e5;font-size:0.97rem;line-height:1.7;">
                        {_html_narr}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Guardar diagnóstico narrativo en session_state para el informe
            st.session_state["_diag_pdf"] = _narrativo if prompt_pdf.strip() else ""

        # ── Guardar resultado en session_state para botón descarga ──────────────
        _fecha_analisis = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.session_state["_informe"] = dict(
            nombre_jugador      = nombre_jugador or "Sin nombre",
            posicion            = posicion_jugador,
            fecha               = datetime.now().strftime("%d/%m/%Y %H:%M"),
            acwr                = acwr_val,
            distancia_m         = dist,
            dist_hd             = dist_hd,
            sprints             = sprints,
            acelerac            = acelerac,
            tiempo_juego        = tiempo_juego,
            fc_max              = fc_max,
            horas_sueno         = horas_sueno,
            fatiga_subj         = fatiga_subj,
            decision            = decision_txt,
            estado_color        = estado_color,
            risk_score          = score,
            razonamiento        = razon,
            tiempo_recuperacion = tiempo_rec,
            fuente              = fuente,
            reporte_fisio       = prompt,
            diagnostico_pdf     = st.session_state.get("_diag_pdf", ""),
        )

        # ── Guardar en historial CSV ──────────────────────────────────────────
        guardar_historial({
            "fecha":               _fecha_analisis,
            "jugador":             nombre_jugador or "Sin nombre",
            "posicion":            posicion_jugador,
            "decision":            decision_txt,
            "risk_score":          round(score, 4),
            "acwr":                acwr_val,
            "distancia_m":         dist,
            "fatiga_subj":         fatiga_subj,
            "horas_sueno":         horas_sueno,
            "fc_max":              fc_max,
            "sprints":             sprints,
            "tiempo_recuperacion": tiempo_rec,
            "fuente":              fuente,
        })

    # ── Botón descarga PDF (fuera del spinner, siempre visible tras análisis) ──
    if "informe" in "".join(st.session_state.keys()) and "_informe" in st.session_state:
        inf = st.session_state["_informe"]
        if generar_informe_pdf:
            try:
                pdf_bytes = generar_informe_pdf(**inf)
                nombre_archivo = (
                    f"Greenlight_Informe_{inf['nombre_jugador'].replace(' ','_')}_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                )
                st.markdown("<div class='glowing-divider'></div>", unsafe_allow_html=True)
                st.download_button(
                    label="📥  DESCARGAR INFORME PDF",
                    data=pdf_bytes,
                    file_name=nombre_archivo,
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as _e:
                st.error(f"Error generando el PDF: {_e}")
        else:
            st.warning("Módulo generar_informe.py no encontrado.")


# ─────────────────────────────────────────────────────────────
# PANTALLA 4: HISTORIAL DE ANÁLISIS
# ─────────────────────────────────────────────────────────────
elif opcion == "📋  Historial":

    st.markdown("<div class='page-title'>HISTORIAL DE ANÁLISIS</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Registro de diagnósticos realizados · Evolución temporal del riesgo</div>", unsafe_allow_html=True)

    df_hist = cargar_historial()

    if df_hist.empty:
        st.markdown("""
        <div style="background:#0d1526;border:1px solid rgba(0,242,255,0.15);border-radius:16px;
             padding:48px;text-align:center;margin-top:20px;">
            <div style="font-size:3rem;margin-bottom:16px;">📋</div>
            <div style="font-family:Orbitron,sans-serif;font-size:0.9rem;color:#00f2ff;letter-spacing:2px;margin-bottom:10px;">
                SIN ANÁLISIS REGISTRADOS
            </div>
            <div style="font-family:Rajdhani,sans-serif;font-size:1rem;color:#8899aa;">
                Realiza tu primer análisis en el <strong style="color:#00f2ff;">Analizador Híbrido</strong>
                para que aparezca aquí automáticamente.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Asegurar tipos correctos ──────────────────────────────────────────
        df_hist["fecha"]      = pd.to_datetime(df_hist["fecha"], errors="coerce")
        df_hist["risk_score"] = pd.to_numeric(df_hist["risk_score"], errors="coerce")
        df_hist["acwr"]       = pd.to_numeric(df_hist["acwr"],       errors="coerce")

        # Color de estado
        def _estado_color(d):
            d = str(d).upper()
            if "ROJO"  in d: return "🔴"
            if "MBAR"  in d: return "🟡"
            return "🟢"
        df_hist["🚦"] = df_hist["decision"].apply(_estado_color)

        # ── KPIs resumen ──────────────────────────────────────────────────────
        total      = len(df_hist)
        n_rojos    = int(df_hist["decision"].str.upper().str.contains("ROJO").sum())
        n_ambar    = int(df_hist["decision"].str.upper().str.contains("MBAR").sum())
        n_verde    = total - n_rojos - n_ambar
        jugadores  = df_hist["jugador"].nunique()
        score_med  = round(df_hist["risk_score"].mean() * 100, 1)

        k1, k2, k3, k4, k5 = st.columns(5)
        for col, (icon, val, label) in zip(
            [k1, k2, k3, k4, k5],
            [("📋", str(total),         "ANÁLISIS TOTALES"),
             ("👤", str(jugadores),      "JUGADORES"),
             ("🔴", str(n_rojos),        "DECISIONES ROJO"),
             ("🟡", str(n_ambar),        "DECISIONES ÁMBAR"),
             ("📊", f"{score_med}%",     "SCORE MEDIO")]
        ):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="card-icon">{icon}</div>
                    <div class="card-value" style="font-size:1.5rem;">{val}</div>
                    <div class="card-label">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<div class='glowing-divider'></div>", unsafe_allow_html=True)

        # ── Filtro por jugador ────────────────────────────────────────────────
        jugadores_lista = ["Todos"] + sorted(df_hist["jugador"].dropna().unique().tolist())
        col_f1, col_f2 = st.columns([1, 3])
        with col_f1:
            filtro_jugador = st.selectbox("Filtrar por jugador", jugadores_lista,
                                          label_visibility="collapsed")
        with col_f2:
            st.markdown("""<div style="font-family:Rajdhani,sans-serif;font-size:0.85rem;
                color:#8899aa;padding:10px 0;">Selecciona un jugador para ver su evolución individual</div>""",
                unsafe_allow_html=True)

        df_filtrado = df_hist if filtro_jugador == "Todos" else df_hist[df_hist["jugador"] == filtro_jugador]
        df_filtrado = df_filtrado.sort_values("fecha", ascending=False)

        st.markdown("<div class='glowing-divider'></div>", unsafe_allow_html=True)

        # ── Gráficos ──────────────────────────────────────────────────────────
        col_g1, col_g2 = st.columns([1.4, 1], gap="large")

        with col_g1:
            st.markdown("<div style='font-family:Orbitron,sans-serif;font-size:0.72rem;color:#8899aa;"
                        "letter-spacing:2px;margin-bottom:10px;'>EVOLUCIÓN DEL SCORE DE RIESGO</div>",
                        unsafe_allow_html=True)

            df_plot = df_filtrado.dropna(subset=["fecha","risk_score"]).sort_values("fecha")

            if len(df_plot) >= 2:
                colores_line = ["#ff4444" if d > 0.65 else "#ffaa00" if d > 0.35 else "#22ff44"
                                for d in df_plot["risk_score"]]

                fig_line = go.Figure()
                # Banda de zona óptima
                fig_line.add_hrect(y0=0, y1=0.35, fillcolor="rgba(34,255,68,0.04)",
                                   line_width=0, annotation_text="Zona segura",
                                   annotation_font_color="#22ff44", annotation_font_size=9)
                fig_line.add_hrect(y0=0.35, y1=0.65, fillcolor="rgba(255,170,0,0.04)",
                                   line_width=0, annotation_text="Zona media",
                                   annotation_font_color="#ffaa00", annotation_font_size=9)
                fig_line.add_hrect(y0=0.65, y1=1.0, fillcolor="rgba(255,68,68,0.04)",
                                   line_width=0, annotation_text="Zona riesgo",
                                   annotation_font_color="#ff4444", annotation_font_size=9)
                # Línea de evolución
                fig_line.add_trace(go.Scatter(
                    x=df_plot["fecha"],
                    y=df_plot["risk_score"],
                    mode="lines+markers",
                    line=dict(color="#00f2ff", width=2),
                    marker=dict(color=colores_line, size=10,
                                line=dict(width=1.5, color="#060b14")),
                    text=df_plot["jugador"],
                    hovertemplate="<b>%{text}</b><br>%{x|%d/%m %H:%M}<br>Score: %{y:.2f}<extra></extra>",
                    fill="tozeroy",
                    fillcolor="rgba(0,242,255,0.05)",
                ))
                fig_line.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#c8d6e5", font_family="Rajdhani",
                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", showgrid=True),
                    yaxis=dict(title="Score riesgo", range=[0, 1],
                               gridcolor="rgba(255,255,255,0.05)"),
                    margin=dict(l=10, r=10, t=10, b=10), height=280,
                    showlegend=False,
                )
                st.plotly_chart(fig_line, use_container_width=True, config={"displayModeBar": False})
            else:
                st.markdown("""<div style="background:#0d1526;border:1px solid rgba(0,242,255,0.1);
                    border-radius:10px;padding:20px;text-align:center;height:180px;display:flex;
                    align-items:center;justify-content:center;">
                    <span style="font-family:Rajdhani,sans-serif;color:#8899aa;">
                    Se necesitan al menos 2 análisis para mostrar la evolución.</span></div>""",
                    unsafe_allow_html=True)

        with col_g2:
            st.markdown("<div style='font-family:Orbitron,sans-serif;font-size:0.72rem;color:#8899aa;"
                        "letter-spacing:2px;margin-bottom:10px;'>DISTRIBUCIÓN DE DECISIONES</div>",
                        unsafe_allow_html=True)

            conteo = df_filtrado["decision"].str.upper().apply(
                lambda d: "ROJO" if "ROJO" in d else "ÁMBAR" if "MBAR" in d else "VERDE"
            ).value_counts()

            fig_pie = go.Figure(go.Pie(
                values=conteo.values,
                labels=conteo.index,
                hole=0.6,
                marker_colors=["#ff4444" if l=="ROJO" else "#ffaa00" if l=="ÁMBAR" else "#22ff44"
                               for l in conteo.index],
                textinfo="label+percent",
                textfont=dict(color="#c8d6e5", family="Rajdhani", size=11),
            ))
            fig_pie.add_annotation(
                text=f"{len(df_filtrado)}<br>análisis",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="#00f2ff", family="Orbitron")
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font_color="#c8d6e5",
                showlegend=False, margin=dict(l=0, r=0, t=10, b=0), height=280,
            )
            st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

        st.markdown("<div class='glowing-divider'></div>", unsafe_allow_html=True)

        # ── Tabla de registros ────────────────────────────────────────────────
        st.markdown("<div style='font-family:Orbitron,sans-serif;font-size:0.72rem;color:#8899aa;"
                    "letter-spacing:2px;margin-bottom:10px;'>REGISTRO COMPLETO</div>",
                    unsafe_allow_html=True)

        df_tabla = df_filtrado[[
            "🚦", "fecha", "jugador", "posicion", "decision",
            "risk_score", "acwr", "fatiga_subj", "tiempo_recuperacion"
        ]].copy()
        df_tabla["fecha"]      = df_tabla["fecha"].dt.strftime("%d/%m/%Y %H:%M")
        df_tabla["risk_score"] = (df_tabla["risk_score"] * 100).round(1).astype(str) + "%"

        st.dataframe(
            df_tabla.rename(columns={
                "fecha":               "Fecha",
                "jugador":             "Jugador",
                "posicion":            "Posición",
                "decision":            "Decisión",
                "risk_score":          "Score",
                "acwr":                "ACWR",
                "fatiga_subj":         "Fatiga",
                "tiempo_recuperacion": "Tiempo baja",
            }),
            use_container_width=True,
            height=320,
            hide_index=True,
        )

        # ── Botón exportar historial ──────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        col_exp1, col_exp2, col_exp3 = st.columns([1, 1, 1])
        with col_exp2:
            csv_bytes = df_hist.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥  EXPORTAR HISTORIAL CSV",
                data=csv_bytes,
                file_name=f"Greenlight_Historial_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
            )