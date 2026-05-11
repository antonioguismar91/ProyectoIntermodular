# 🟢 GreenLight AI — Prevención de Lesiones en el Fútbol Profesional

> **Sistema de Soporte a la Decisión (DSS) híbrido** que combina Machine Learning, Procesamiento de Lenguaje Natural y reglas clínicas deterministas para predecir el riesgo de lesión de un futbolista y emitir una recomendación de acción visual: 🔴 ROJO · 🟡 ÁMBAR · 🟢 VERDE.

**Proyecto Fin de Máster — Máster en Inteligencia Artificial y Big Data**  
Centro FP Superior · Campus Cámara Comercio Sevilla · Mayo 2026  
**Autor:** Adrián Pavón Alcón y Antonio Guisado Martín

---

## ¿Por qué existe este proyecto?

Las cinco grandes ligas europeas acumularon **22.596 lesiones** entre 2020 y 2025, con un coste salarial de **3.450 millones de euros** (Howden, 2025). Los equipos de élite disponen de plataformas de telemetría propietarias como Catapult o STATSports. El resto, no.

GreenLight AI cierra esa brecha: un sistema de predicción accesible, **sin dependencia de datos GPS privados**, construido íntegramente con herramientas de código abierto y datos sintéticos calibrados con la literatura científica.

---

## Cómo funciona

El sistema opera en cuatro pilares en cascada:

```
Reporte del fisio (texto libre)
        ↓
  [Pilar 3 — NLP]  Mistral extrae 9 variables cualitativas (zona, severidad, fatiga…)
        ↓
  [Pilar 1 — ML]   Stacking Ensemble (XGB + LGB + CB + RF) → P(lesión)
        ↓
  [Pilar 4 — DSS]  Sistema experto: criticidad por posición × zona × grado clínico
        ↓
   🔴 ROJO / 🟡 ÁMBAR / 🟢 VERDE  +  Tiempo estimado de baja
```

**Métricas del modelo:**

| Modelo | AUC | Recall | F1 |
|--------|-----|--------|-----|
| Baseline (LogReg) | 0.7051 | 0.6625 | 0.3954 |
| XGBoost | 0.8059 | 0.6375 | 0.4889 |
| LightGBM | 0.8113 | 0.6548 | 0.4873 |
| CatBoost | 0.8154 | 0.6913 | 0.4856 |
| Random Forest | 0.8037 | 0.5279 | 0.4933 |
| **Stacking ★** | **0.8156** | **0.7154** | **0.4860** |

---

## Inicio rápido

### Requisitos previos
- Python 3.10 instalado con el Python Launcher (`py`)
- API key de Mistral AI gratuita → [console.mistral.ai](https://console.mistral.ai) (plan Experiment)

### Instalación y arranque

```bash
# 1. Clonar el repositorio
git clone https://github.com/apavalc3108/ProyectoIntermodular.git
cd ProyectoIntermodular

# 2. Instalar entorno (solo la primera vez)
setup_env.bat       # Crea .venv, instala dependencias y configura la API key

# 3. Arrancar la aplicación
run_app.bat         # Lanza Streamlit en http://localhost:8501
```

> ⚠️ La API key de Mistral se guarda en `.streamlit/secrets.toml`, excluido del repositorio por `.gitignore`. Sin ella el sistema funciona en modo heurístico.

---

## Estructura del repositorio

```
ProyectoIntermodular/
│
├── 📂 .streamlit/
│   └── config.toml              # Tema oscuro de la app (automático)
│
├── 📂 catboost_info/            # Logs generados automáticamente por CatBoost
│
├── 📂 data/
│   ├── dataset_sintetico_soccermon.csv   # Dataset sintético (30.000 registros)
│   └── historial_analisis.csv           # Historial de análisis realizados
│
├── 📂 docs/
│   ├── 01_Memoria_GreenLight_AI.docx    # Memoria del Proyecto Intermodular
│   ├── 02_Evaluacion_Tecnica.docx       # Evaluación técnica y métricas
│   ├── 03_Manual_Tecnico.docx           # Instalación, arquitectura y despliegue
│   ├── 04_Manual_Usuario.docx           # Guía para el cuerpo técnico
│   └── readme_docs.md                   # Índice de la documentación
│
├── 📂 models/
│   ├── modelo_stacking_ensemble.pkl     # ★ Modelo principal
│   ├── modelo_xgboost.pkl
│   ├── modelo_lightgbm.pkl
│   ├── modelo_catboost.pkl
│   ├── modelo_random_forest.pkl
│   ├── modelo_regresion_logistica.pkl
│   └── scaler.pkl                       # RobustScaler del pipeline
│
├── 📂 reports/
│   ├── shap_importance.png              # Importancia de variables SHAP
│   ├── shap_summary.png                 # SHAP summary plot
│   ├── curvas_roc.png                   # Curvas ROC comparativas
│   ├── confusion_matrix_opt.png         # Matriz de confusión
│   ├── correlaciones.png                # Heatmap de correlaciones
│   ├── distribuciones_marginales.png    # Distribuciones del dataset
│   ├── boxplots_por_lesion.png          # Boxplots por clase
│   ├── pairplot_top5.png                # Pairplot top 5 variables
│   └── logo_greenlight.jpg              # Logo oficial del proyecto
│
├── 📂 src/
│   ├── dataset_final.py                 # Pipeline ML: generación del dataset y entrenamiento
│   └── mistral_definitivo.py            # DSS híbrido: NLP, sistema experto, reglas clínicas
│
├── 📂 ui/
│   ├── app.py                           # Interfaz Streamlit (4 pantallas)
│   └── generar_informe.py               # Generador de informes PDF (ReportLab)
│
├── .gitignore
├── Arquitectura_definitiva.png          # Diagrama oficial de la arquitectura
├── requirements.txt                     # Dependencias del proyecto
├── setup_env.bat                        # ① Instalar entorno (ejecutar una vez)
└── run_app.bat                          # ② Arrancar la aplicación
```

---

## La app en 4 pantallas

| Pantalla | Qué hace |
|----------|----------|
| 🏠 **Bienvenida** | Presentación del sistema, métricas clave y descripción de los tres módulos |
| 📊 **Dashboard** | Alertas automáticas del plantel, gráficos de ACWR vs fatiga, distribución de estados |
| 🤖 **Analizador Híbrido** | Formulario GPS + biométrico + reporte del fisio → semáforo + radar + informe PDF |
| 📋 **Historial** | Evolución temporal del riesgo por jugador, exportación a CSV |

---

## Tecnologías

| Categoría | Herramientas |
|-----------|-------------|
| Lenguaje | Python 3.10 |
| Machine Learning | Scikit-learn · XGBoost · LightGBM · CatBoost · SHAP |
| NLP | Mistral API (`mistral-small-latest`) |
| Interfaz | Streamlit · Plotly |
| PDF | ReportLab · pdfplumber |
| Datos sintéticos | SciPy (cópula gaussiana) |
| Serialización | Joblib |
| Entorno | Google Colab · VS Code · Git |

---

## Documentación

Toda la documentación está en la carpeta [`docs/`](./docs/):

- **[Memoria del PI](./docs/01_Memoria_GreenLight_AI.docx)** — Contexto, objetivos, metodología, resultados y conclusiones
- **[Evaluación Técnica](./docs/02_Evaluacion_Tecnica.docx)** — Métricas detalladas, casos de prueba y comparativas
- **[Manual Técnico](./docs/03_Manual_Tecnico.docx)** — Instalación, arquitectura del código y despliegue
- **[Manual de Usuario](./docs/04_Manual_Usuario.docx)** — Guía completa para el cuerpo técnico

---

> *GreenLight AI no sustituye el criterio médico. Es una herramienta de apoyo a la decisión clínica.*
