# 📁 docs — Documentación de GreenLight AI

Esta carpeta contiene la documentación completa del proyecto **GreenLight AI**, un Sistema de Soporte a la Decisión (DSS) híbrido para la prevención de lesiones en el fútbol profesional, desarrollado como Proyecto Intermodular del Master de Inteligencia Artificial y Big Data.

---

## 📄 Documentos incluidos

### 01 · Memoria GreenLight AI
**`01_Memoria_GreenLight_AI.pdf`**

Documento principal del Proyecto Intermodular, redactado siguiendo la estructura oficial del centro. Recoge el ciclo completo del proyecto: desde el problema de partida hasta las conclusiones finales.

| Capítulo | Contenido |
|----------|-----------|
| 1. Introducción | Idea general, contexto y justificación, evolución del proyecto, definición del problema, hipótesis, delimitación y objetivos (1 OG + 6 OE) |
| 2. Marco Teórico | Tecnologías utilizadas, conceptos clave (cópula gaussiana, ensemble learning, NLP, reglas deterministas), estado del arte, análisis DAFO y análisis CAME |
| 3. Marco Metodológico | Metodología SCRUM, 6 sprints, herramientas, proceso de desarrollo en 5 fases, diagrama de Gantt y presupuesto (4.500 €) |
| 4. Resultados | Resultados por objetivo: dataset sintético, modelo ML (AUC 0.8156, Recall 71.5%), módulo NLP, sistema experto, evaluación y app Streamlit. Incluye errores y correcciones. |
| 5. Conclusiones | Valoración personal, aprendizajes, dificultades encontradas y trabajo futuro |
| 6. Referencias | 10 referencias en formato IEEE |
| Anexo I | Ética, privacidad y legalidad (RGPD, AI Act, uso responsable de IA) |
| Anexo II | Código fuente principal comentado (dataset, DSS, app) y tabla del repositorio |
| Anexo III | 24 capturas de pantalla del proceso organizadas en 4 secciones: dataset, ML, NLP y evolución de la identidad visual |

---

### 02 · Evaluación Técnica
**`02_Evaluacion_Tecnica.pdf`**

Documento complementario centrado en el análisis de rendimiento del sistema. Orientado a un lector con conocimientos técnicos que quiera verificar los resultados de forma detallada.

| Sección | Contenido |
|---------|-----------|
| 1. Evaluación ML | Configuración del experimento, tabla comparativa de métricas (AUC, Recall, F1, Precisión) de los 6 modelos, análisis de resultados y explicabilidad SHAP |
| 2. Evaluación NLP | 7 casos de prueba (A–G) con zona, grado, posición, decisión, fuente y tiempo de baja estimado |
| 3. Evaluación DSS | Comparativa ML puro vs DSS: cuándo y cómo el sistema experto corrige al modelo |
| 4. Rendimiento global | Tabla resumen de métricas del sistema completo |
| 5. Limitaciones | Limitaciones identificadas con propuestas de solución para cada una |

**Métricas clave del Stacking Ensemble:**

```
AUC-ROC:  0.8156   ✅ > 0.80
Recall:   0.7154   ✅ > 0.70
Umbral:   0.4809
```

---

### 03 · Manual Técnico
**`03_Manual_Tecnico.pdf`**

Guía completa para desarrolladores. Cubre instalación, configuración, arquitectura interna y despliegue del sistema.

| Sección | Contenido |
|---------|-----------|
| 1. Estructura del repositorio | Tabla con las 19 rutas del proyecto y su responsabilidad |
| 2. Instalación y configuración | Requisitos previos, clonado, instalación de dependencias y configuración de la API key de Mistral |
| 3. Dependencias | 14 librerías con versión mínima y uso específico en el proyecto |
| 4. Arquitectura del código | Descripción de las funciones principales de `mistral_definitivo.py`, `dataset_final.py`, `app.py` y `generar_informe.py` |
| 5. Integrar el modelo .pkl | Código exacto para conectar `modelo_stacking_ensemble.pkl` en la UI (pendiente técnico) |
| 6. Despliegue Streamlit Cloud | Pasos para publicar la aplicación con gestión segura de secretos |
| 7. Scripts .bat | Documentación de `setup_env.bat` y `run_app.bat`: pasos, lógica de la API key y flujo de uso completo |

**Inicio rápido:**

```bash
# 1. Clonar
git clone https://github.com/[usuario]/ProyectoIntermodular.git
cd ProyectoIntermodular

# 2. Instalar (solo una vez)
setup_env.bat

# 3. Arrancar
run_app.bat
```

---

### 04 · Manual de Usuario
**`04_Manual_Usuario.pdf`**

Guía para el cuerpo técnico (preparadores físicos, fisioterapeutas, médicos). No requiere conocimientos de programación.

| Sección | Contenido |
|---------|-----------|
| El semáforo de riesgo | Significado de ROJO, ÁMBAR y VERDE con acción recomendada para cada estado |
| Navegación | Descripción de las 4 pantallas: Bienvenida, Dashboard, Analizador Híbrido e Historial |
| Dashboard | Alertas automáticas (ACWR, fatiga, sueño, estrés) y cómo leer los gráficos del equipo |
| Analizador Híbrido | Guía paso a paso: datos GPS → datos biométricos → perfil del jugador → reporte del fisio → resultado |
| Resultado del análisis | Cómo leer el semáforo, el gauge de riesgo, el razonamiento IA, el tiempo de baja y el radar |
| Informe PDF | Cómo descargar y compartir el informe clínico generado automáticamente |
| Historial | Cómo consultar la evolución temporal del riesgo y exportar a CSV |
| Preguntas frecuentes | 5 preguntas habituales sobre el sistema (ROJO conservador, fuentes de decisión, tiempos de baja, Mistral no disponible, privacidad) |
| Glosario | 10 términos técnicos explicados en lenguaje accesible (ACWR, CTL, ATL, HSR, criticidad anatómica, LLM, etc.) |

**El semáforo de un vistazo:**

| Señal | Significado | Acción |
|-------|-------------|--------|
| 🔴 ROJO | Riesgo alto detectado | Reposo. Valoración médica. |
| 🟡 ÁMBAR | Riesgo moderado | Entrenamiento reducido. Vigilar. |
| 🟢 VERDE | Sin señales de riesgo | Puede entrenar con normalidad. |

---

## 🗂️ Guía de lectura recomendada

```
¿Quieres entender el proyecto?      → 01_Memoria_GreenLight_AI.docx
¿Quieres verificar los resultados?  → 02_Evaluacion_Tecnica.docx
¿Quieres instalar o desarrollar?    → 03_Manual_Tecnico.docx
¿Quieres usar la aplicación?        → 04_Manual_Usuario.docx
```

---

## ✍️ Autores

**Adrián Pavón Alcón**  
**Antonio Guisado Marín**  
Master en Inteligencia Artificial y Big Data
Centro FP Superior · Campus Cámara Comercio Sevilla · Mayo 2026
