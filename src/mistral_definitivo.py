# ================================================================
# PIPELINE FINAL– DSS HÍBRIDO ROBUSTO
# ================================================================

import streamlit as st

MISTRAL_API_KEY = st.secrets.get('MISTRAL_API_KEY', None)

import requests, json, logging, time, re, zipfile, os, joblib
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from jsonschema import validate, ValidationError

# ----------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL = "mistral-small-latest"
HEADERS = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FEATURE_SCHEMA = {
    "type": "object",
    "properties": {
        "zona_lesion": {"type": "string"},
        "severidad_lesion": {"type": "number"},
        "tendencia_lesion": {"type": "string"},
        "fatiga_cualitativa": {"type": "number"},
        "calidad_suenio_cualitativa": {"type": "number"},
        "estres_cualitativo": {"type": "number"},
        "dolor_muscular_cualitativo": {"type": "number"},
        "confianza_extraccion": {"type": "number"}
    },
    "required": ["severidad_lesion"]
}

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string"},
        "risk_score": {"type": "number"},
        "razonamiento": {"type": "string"}
    },
    "required": ["decision", "risk_score"]
}




# ----------------------------------------------------------------
# DICCIONARIO DE CRITICIDAD POR ZONA Y POSICIÓN
# ----------------------------------------------------------------
CRITICIDAD = {
    "Portero": {
        "mano": "ALTA", "muñeca": "ALTA", "muñeca rota": "EXTREMA",
        "dedos": "ALTA", "dedos rotos": "EXTREMA",
        "hombro": "ALTA", "rodilla": "ALTA", "tobillo": "MEDIA",
        "isquiotibiales": "BAJA", "cuádriceps": "BAJA"
    },
    "Defensa": {
        "mano": "BAJA", "muñeca": "BAJA", "muñeca rota": "EXTREMA",
        "dedos": "BAJA", "dedos rotos": "EXTREMA",
        "rodilla": "ALTA", "tobillo": "ALTA", "isquiotibiales": "ALTA",
        "cuádriceps": "ALTA", "hombro": "MEDIA"
    },
    "Centrocampista": {
        "mano": "BAJA", "muñeca": "BAJA", "muñeca rota": "EXTREMA",
        "dedos": "BAJA", "dedos rotos": "EXTREMA",
        "rodilla": "ALTA", "tobillo": "ALTA", "isquiotibiales": "ALTA",
        "cuádriceps": "ALTA", "hombro": "MEDIA"
    },
    "Delantero": {
        "mano": "BAJA", "muñeca": "BAJA", "muñeca rota": "EXTREMA",
        "dedos": "BAJA", "dedos rotos": "EXTREMA",
        "rodilla": "ALTA", "tobillo": "ALTA", "isquiotibiales": "ALTA",
        "cuádriceps": "ALTA", "hombro": "MEDIA"
    }
}
# ----------------------------------------------------------------
# MATRIZ DE COMBINACIÓN: CRITICIDAD BASE + GRADO DE LESIÓN
# ----------------------------------------------------------------
COMBINACION_GRADO = {
    "EXTREMA": {"leve": "EXTREMA", "moderado": "EXTREMA", "grave": "EXTREMA"},
    "ALTA":    {"leve": "ALTA",    "moderado": "EXTREMA", "grave": "EXTREMA"},
    "MEDIA":   {"leve": "MEDIA",   "moderado": "ALTA",    "grave": "EXTREMA"},
    "BAJA":    {"leve": "BAJA",    "moderado": "MEDIA",   "grave": "ALTA"},
}
# ================================================================
# DICCIONARIO DE TIEMPOS DE RECUPERACIÓN POR ZONA Y GRADO
# (Valores basados en Ekstrand et al., 2023; Werner et al., 2009;
#  Hägglund et al., 2013; Orchard Sports Injury Classification)
# ================================================================
TIEMPOS_RECUPERACION = {
    # --- MUSLO (ISQUIOTIBIALES) ---
    ("isquiotibiales", "leve"):     "2-3 semanas",
    ("isquiotibiales", "grado 1"):  "2-3 semanas",
    ("isquiotibiales", "moderado"): "6-8 semanas",
    ("isquiotibiales", "grado 2"):  "6-8 semanas",
    ("isquiotibiales", "grave"):    "12-16 semanas",
    ("isquiotibiales", "grado 3"):  "12-16 semanas",

    # --- MUSLO (CUÁDRICEPS) ---
    ("cuádriceps", "leve"):         "2-3 semanas",
    ("cuádriceps", "grado 1"):      "2-3 semanas",
    ("cuádriceps", "moderado"):     "6-8 semanas",
    ("cuádriceps", "grado 2"):      "6-8 semanas",
    ("cuádriceps", "grave"):        "12-16 semanas",
    ("cuádriceps", "grado 3"):      "12-16 semanas",

    # --- ADUCTORES / INGLE ---
    ("aductor", "leve"):            "1-2 semanas",
    ("aductor", "grado 1"):         "1-2 semanas",
    ("aductor", "moderado"):        "4-6 semanas",
    ("aductor", "grado 2"):         "4-6 semanas",
    ("aductor", "grave"):           "8-12 semanas",
    ("aductor", "grado 3"):         "8-12 semanas",
    ("ingle", "leve"):              "1-2 semanas",
    ("ingle", "grado 1"):           "1-2 semanas",
    ("ingle", "moderado"):          "4-6 semanas",
    ("ingle", "grado 2"):           "4-6 semanas",
    ("ingle", "grave"):             "8-12 semanas",
    ("ingle", "grado 3"):           "8-12 semanas",

    # --- GEMELOS / SÓLEO ---
    ("gemelo", "leve"):             "2-3 semanas",
    ("gemelo", "grado 1"):          "2-3 semanas",
    ("gemelo", "moderado"):         "6-8 semanas",
    ("gemelo", "grado 2"):          "6-8 semanas",
    ("gemelo", "grave"):            "10-14 semanas",
    ("gemelo", "grado 3"):          "10-14 semanas",
    ("sóleo", "leve"):              "2-3 semanas",
    ("sóleo", "grado 1"):           "2-3 semanas",
    ("sóleo", "moderado"):          "6-8 semanas",
    ("sóleo", "grado 2"):           "6-8 semanas",
    ("sóleo", "grave"):             "10-14 semanas",
    ("sóleo", "grado 3"):           "10-14 semanas",

    # --- TOBILLO (ESGUINCE) ---
    ("tobillo", "leve"):            "1-2 semanas",
    ("tobillo", "grado 1"):         "1-2 semanas",
    ("tobillo", "moderado"):        "3-6 semanas",
    ("tobillo", "grado 2"):         "3-6 semanas",
    ("tobillo", "grave"):           "8-16 semanas",
    ("tobillo", "grado 3"):         "8-16 semanas",
    ("tobillo izquierdo", "leve"):  "1-2 semanas",
    ("tobillo izquierdo", "grado 1"): "1-2 semanas",
    ("tobillo izquierdo", "moderado"): "3-6 semanas",
    ("tobillo izquierdo", "grado 2"): "3-6 semanas",
    ("tobillo izquierdo", "grave"): "8-16 semanas",
    ("tobillo izquierdo", "grado 3"): "8-16 semanas",
    ("tobillo derecho", "leve"):    "1-2 semanas",
    ("tobillo derecho", "grado 1"): "1-2 semanas",
    ("tobillo derecho", "moderado"): "3-6 semanas",
    ("tobillo derecho", "grado 2"): "3-6 semanas",
    ("tobillo derecho", "grave"):   "8-16 semanas",
    ("tobillo derecho", "grado 3"): "8-16 semanas",

    # --- RODILLA (LIGAMENTOS / MENISCO) ---
    ("rodilla", "leve"):            "2-4 semanas",
    ("rodilla", "grado 1"):         "2-4 semanas",
    ("rodilla", "moderado"):        "6-12 semanas",
    ("rodilla", "grado 2"):         "6-12 semanas",
    ("rodilla", "grave"):           "16-36 semanas",
    ("rodilla", "grado 3"):         "16-36 semanas",
    ("rodilla izquierda", "leve"):  "2-4 semanas",
    ("rodilla izquierda", "grado 1"): "2-4 semanas",
    ("rodilla izquierda", "moderado"): "6-12 semanas",
    ("rodilla izquierda", "grado 2"): "6-12 semanas",
    ("rodilla izquierda", "grave"): "16-36 semanas",
    ("rodilla izquierda", "grado 3"): "16-36 semanas",
    ("rodilla derecha", "leve"):    "2-4 semanas",
    ("rodilla derecha", "grado 1"): "2-4 semanas",
    ("rodilla derecha", "moderado"): "6-12 semanas",
    ("rodilla derecha", "grado 2"): "6-12 semanas",
    ("rodilla derecha", "grave"):   "16-36 semanas",
    ("rodilla derecha", "grado 3"): "16-36 semanas",

    # --- CADERA ---
    ("cadera", "leve"):             "1-2 semanas",
    ("cadera", "grado 1"):          "1-2 semanas",
    ("cadera", "moderado"):         "4-8 semanas",
    ("cadera", "grado 2"):          "4-8 semanas",
    ("cadera", "grave"):            "10-16 semanas",
    ("cadera", "grado 3"):          "10-16 semanas",

    # --- ZONA LUMBAR / ESPALDA BAJA ---
    ("espalda baja", "leve"):       "1-2 semanas",
    ("espalda baja", "grado 1"):    "1-2 semanas",
    ("espalda baja", "moderado"):   "3-6 semanas",
    ("espalda baja", "grado 2"):    "3-6 semanas",
    ("espalda baja", "grave"):      "8-12 semanas",
    ("espalda baja", "grado 3"):    "8-12 semanas",

    # --- PUBIS / OSTEÍTIS PÚBICA ---
    ("pubis", "leve"):              "1-2 semanas",
    ("pubis", "grado 1"):           "1-2 semanas",
    ("pubis", "moderado"):          "4-8 semanas",
    ("pubis", "grado 2"):           "4-8 semanas",
    ("pubis", "grave"):             "10-16 semanas",
    ("pubis", "grado 3"):           "10-16 semanas",

    # --- HOMBRO ---
    ("hombro", "leve"):             "1-2 semanas",
    ("hombro", "grado 1"):          "1-2 semanas",
    ("hombro", "moderado"):         "4-6 semanas",
    ("hombro", "grado 2"):          "4-6 semanas",
    ("hombro", "grave"):            "10-16 semanas",
    ("hombro", "grado 3"):          "10-16 semanas",
    ("hombro izquierdo", "leve"):   "1-2 semanas",
    ("hombro izquierdo", "grado 1"): "1-2 semanas",
    ("hombro izquierdo", "moderado"): "4-6 semanas",
    ("hombro izquierdo", "grado 2"): "4-6 semanas",
    ("hombro izquierdo", "grave"):  "10-16 semanas",
    ("hombro izquierdo", "grado 3"): "10-16 semanas",
    ("hombro derecho", "leve"):     "1-2 semanas",
    ("hombro derecho", "grado 1"):  "1-2 semanas",
    ("hombro derecho", "moderado"): "4-6 semanas",
    ("hombro derecho", "grado 2"):  "4-6 semanas",
    ("hombro derecho", "grave"):    "10-16 semanas",
    ("hombro derecho", "grado 3"):  "10-16 semanas",

    # --- MUÑECA / MANO ---
    ("muñeca", "leve"):             "3-5 días",
    ("muñeca", "grado 1"):          "3-5 días",
    ("muñeca", "moderado"):         "2-4 semanas",
    ("muñeca", "grado 2"):          "2-4 semanas",
    ("muñeca", "grave"):            "6-8 semanas",
    ("muñeca", "grado 3"):          "6-8 semanas",
    ("muñeca izquierda", "leve"):   "3-5 días",
    ("muñeca izquierda", "grado 1"): "3-5 días",
    ("muñeca izquierda", "moderado"): "2-4 semanas",
    ("muñeca izquierda", "grado 2"): "2-4 semanas",
    ("muñeca izquierda", "grave"):  "6-8 semanas",
    ("muñeca izquierda", "grado 3"): "6-8 semanas",
    ("muñeca derecha", "leve"):     "3-5 días",
    ("muñeca derecha", "grado 1"):  "3-5 días",
    ("muñeca derecha", "moderado"): "2-4 semanas",
    ("muñeca derecha", "grado 2"):  "2-4 semanas",
    ("muñeca derecha", "grave"):    "6-8 semanas",
    ("muñeca derecha", "grado 3"):  "6-8 semanas",
    ("mano", "leve"):               "3-5 días",
    ("mano", "grado 1"):            "3-5 días",
    ("mano", "moderado"):           "2-4 semanas",
    ("mano", "grado 2"):            "2-4 semanas",
    ("mano", "grave"):              "6-8 semanas",
    ("mano", "grado 3"):            "6-8 semanas",
    ("dedos", "leve"):              "2-3 días",
    ("dedos", "grado 1"):           "2-3 días",
    ("dedos", "moderado"):          "1-2 semanas",
    ("dedos", "grado 2"):           "1-2 semanas",
    ("dedos", "grave"):             "3-6 semanas",
    ("dedos", "grado 3"):           "3-6 semanas",

    # --- PIE ---
    ("pie", "leve"):                "3-7 días",
    ("pie", "grado 1"):             "3-7 días",
    ("pie", "moderado"):            "2-4 semanas",
    ("pie", "grado 2"):             "2-4 semanas",
    ("pie", "grave"):               "6-10 semanas",
    ("pie", "grado 3"):             "6-10 semanas",

    # --- FRACTURAS ESPECÍFICAS ---
    ("peroné", "grave"):            "12-16 semanas",
    ("peroné", "grado 3"):          "12-16 semanas",
    ("tibia", "grave"):             "16-24 semanas",
    ("tibia", "grado 3"):           "16-24 semanas",
    ("metatarso", "grave"):         "8-12 semanas",
    ("metatarso", "grado 3"):       "8-12 semanas",
    ("clavícula", "grave"):         "8-10 semanas",
    ("clavícula", "grado 3"):       "8-10 semanas",

    # --- CONMOCIÓN CEREBRAL ---
    ("cabeza", "leve"):             "6-7 días (protocolo conmoción)",
    ("cabeza", "grado 1"):          "6-7 días (protocolo conmoción)",
    ("cabeza", "moderado"):         "2-3 semanas",
    ("cabeza", "grado 2"):          "2-3 semanas",
    ("cabeza", "grave"):            "4-6 semanas",
    ("cabeza", "grado 3"):          "4-6 semanas",

    # --- COSTILLAS ---
    ("costillas", "leve"):          "1-2 semanas",
    ("costillas", "grado 1"):       "1-2 semanas",
    ("costillas", "moderado"):      "3-5 semanas",
    ("costillas", "grado 2"):       "3-5 semanas",
    ("costillas", "grave"):         "6-8 semanas",
    ("costillas", "grado 3"):       "6-8 semanas",

    # --- GENÉRICOS / NO ESPECIFICADOS ---
    ("desconocida", "leve"):        "Consultar al médico",
    ("desconocida", "grado 1"):     "Consultar al médico",
    ("desconocida", "moderado"):    "Consultar al médico",
    ("desconocida", "grado 2"):     "Consultar al médico",
    ("desconocida", "grave"):       "Consultar al médico",
    ("desconocida", "grado 3"):     "Consultar al médico",
}

def obtener_tiempo_recuperacion_determinista(criticidad: str, severidad: float) -> str:
    """Calcula el tiempo de baja según criticidad y severidad numérica."""
    if severidad <= 3:
        rango = (1, 3)
    elif severidad <= 6:
        rango = (4, 6)
    else:
        rango = (7, 10)

    tabla = {
        "EXTREMA": {(1, 3): "4-6 semanas", (4, 6): "8-16 semanas", (7, 10): "16-36 semanas"},
        "ALTA":    {(1, 3): "2-4 semanas", (4, 6): "6-12 semanas", (7, 10): "12-24 semanas"},
        "MEDIA":   {(1, 3): "1-2 semanas", (4, 6): "3-6 semanas", (7, 10): "6-12 semanas"},
        "BAJA":    {(1, 3): "3-7 días", (4, 6): "2-4 semanas", (7, 10): "4-8 semanas"},
    }

    if criticidad in tabla and rango in tabla[criticidad]:
        return tabla[criticidad][rango]
    return "No estimado"

def normalizar_zona(zona: str) -> str:
    """Normaliza el nombre de la zona de lesión para que coincida con las claves del diccionario."""
    zona_lower = zona.lower()
    if "isquiotibiales" in zona_lower:
        return "isquiotibiales"
    if "cuádriceps" in zona_lower or "cuadriceps" in zona_lower:
        return "cuádriceps"
    if "aductor" in zona_lower or "ingle" in zona_lower:
        # Both 'aductor' and 'ingle' are keys in TIEMPOS_RECUPERACION, prefer the specific one if present
        if "ingle" in zona_lower: return "ingle"
        return "aductor"
    if "gemelo" in zona_lower or "sóleo" in zona_lower or "soleo" in zona_lower:
        # Both 'gemelo' and 'sóleo' are keys, prefer the specific one if present
        if "sóleo" in zona_lower or "soleo" in zona_lower: return "sóleo"
        return "gemelo"
    if "tobillo" in zona_lower:
        # Specific 'tobillo izquierdo'/'tobillo derecho' exist, try to match those first
        if "tobillo izquierdo" in zona_lower: return "tobillo izquierdo"
        if "tobillo derecho" in zona_lower: return "tobillo derecho"
        return "tobillo"
    if "rodilla" in zona_lower:
        # Specific 'rodilla izquierda'/'rodilla derecha' exist, try to match those first
        if "rodilla izquierda" in zona_lower: return "rodilla izquierda"
        if "rodilla derecha" in zona_lower: return "rodilla derecha"
        return "rodilla"
    if "cadera" in zona_lower:
        return "cadera"
    if "espalda baja" in zona_lower or "lumbar" in zona_lower:
        return "espalda baja"
    if "pubis" in zona_lower:
        return "pubis"
    if "hombro" in zona_lower:
        # Specific 'hombro izquierdo'/'hombro derecho' exist, try to match those first
        if "hombro izquierdo" in zona_lower: return "hombro izquierdo"
        if "hombro derecho" in zona_lower: return "hombro derecho"
        return "hombro"
    if "muñeca" in zona_lower or "muneca" in zona_lower:
        # Specific 'muñeca izquierda'/'muñeca derecha' exist, try to match those first
        if "muñeca izquierda" in zona_lower or "muneca izquierda" in zona_lower: return "muñeca izquierda"
        if "muñeca derecha" in zona_lower or "muneca derecha" in zona_lower: return "muñeca derecha"
        return "muñeca"
    if "mano" in zona_lower:
        return "mano"
    if "dedos" in zona_lower:
        return "dedos"
    if "pie" in zona_lower:
        return "pie"
    if "peroné" in zona_lower or "perone" in zona_lower:
        return "peroné"
    if "tibia" in zona_lower:
        return "tibia"
    if "metatarso" in zona_lower:
        return "metatarso"
    if "clavícula" in zona_lower or "clavicula" in zona_lower:
        return "clavícula"
    if "cabeza" in zona_lower or "conmoción" in zona_lower or "conmocion" in zona_lower:
        return "cabeza"
    if "costillas" in zona_lower:
        return "costillas"
    return "desconocida"

# ----------------------------------------------------------------
# LLAMADA HTTP ROBUSTA
# ----------------------------------------------------------------
def llamar_modelo(prompt: str, max_tokens=300, temperature=0.2, retries=2) -> Optional[Dict]:
    payload = {"model": MISTRAL_MODEL, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": temperature}
    for intento in range(retries + 1):
        try:
            start = time.time()
            resp = requests.post(MISTRAL_URL, headers=HEADERS, json=payload, timeout=30)
            latency = time.time() - start
            if resp.status_code == 429:
                logging.warning(f"Rate limit (intento {intento+1}). Esperando 5s...")
                time.sleep(5); continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            logging.info(f"Latencia Mistral: {latency:.2f}s")
            return {"text": content, "latency": latency}
        except requests.exceptions.RequestException as e:
            logging.error(f"Error HTTP (intento {intento+1}): {e}")
            if intento < retries: time.sleep(3)
        except Exception as e:
            logging.error(f"Error inesperado: {e}"); break
    return None

# ----------------------------------------------------------------
# EXTRACCIÓN DE JSON ROBUSTA + REPARACIÓN
# ----------------------------------------------------------------
def extraer_json_seguro(texto: str) -> Optional[Dict]:
    try:
        match = re.search(r'\{.*\}', texto, re.DOTALL)
        if match: return json.loads(match.group().replace("\n", " ").strip())
    except: pass
    try:
        start, end = texto.find("{"), texto.rfind("}") + 1
        if start != -1 and end > start: return json.loads(texto[start:end])
    except: pass
    return None

def call_llm_json(prompt: str, schema: Dict, retries: int = 2) -> Optional[Dict]:
    for _ in range(retries + 1):
        raw = llamar_modelo(prompt)
        if not raw: continue
        parsed = extraer_json_seguro(raw["text"])
        if parsed:
            try:
                validate(instance=parsed, schema=schema); return parsed
            except ValidationError: pass
        fix_prompt = f"""Corrige este JSON para que cumpla EXACTAMENTE este schema:
{json.dumps(schema, indent=2)}
JSON a corregir:
{raw['text']}
Devuelve SOLO JSON válido."""
        raw_fix = llamar_modelo(fix_prompt)
        if raw_fix:
            parsed = extraer_json_seguro(raw_fix["text"])
            if parsed:
                try:
                    validate(instance=parsed, schema=schema); return parsed
                except: continue
    return None

# ----------------------------------------------------------------
# VALIDACIÓN DE RANGOS
# ----------------------------------------------------------------
def validar_features(f: Dict) -> Dict:
    def clamp(v, a, b, d):
        try: return max(a, min(b, float(v)))
        except: return d
    return {
        "zona_lesion": str(f.get("zona_lesion", "desconocida")),
        "severidad_lesion": clamp(f.get("severidad_lesion"), 1, 10, 5),
        "tendencia_lesion": str(f.get("tendencia_lesion", "estable")),
        "fatiga_cualitativa": clamp(f.get("fatiga_cualitativa"), 1, 10, 5),
        "calidad_suenio_cualitativa": clamp(f.get("calidad_suenio_cualitativa"), 1, 10, 5),
        "estres_cualitativo": clamp(f.get("estres_cualitativo"), 1, 10, 5),
        "dolor_muscular_cualitativo": clamp(f.get("dolor_muscular_cualitativo"), 1, 10, 5),
        "confianza_extraccion": clamp(f.get("confianza_extraccion"), 0, 1, 0.5),
        "grado_lesion": str(f.get("grado_lesion", "leve")).lower()   # <-- NUEVO
    }
def validar_decision(d: Dict) -> Optional[Dict]:
    try:
        if d.get("decision") not in ["ROJO", "ÁMBAR", "VERDE"]: return None
        risk = float(d.get("risk_score", 0))
        if not (0 <= risk <= 1): return None
        return {"decision": d["decision"], "risk_score": round(risk, 2), "razonamiento": str(d.get("razonamiento", ""))}
    except: return None

class HeuristicParser:
    def parse(self, text):
        t = text.lower()
        zonas = ["tobillo izquierdo","tobillo derecho","rodilla izquierda","rodilla derecha",
                 "cadera","espalda baja","isquiotibiales","cuádriceps","ingle"]
        zona = "desconocida"
        for z in zonas:
            if z in t: zona = z.replace(" ","_"); break
        severidad = 8 if any(w in t for w in ["severo","agudo","inflamación","muy dolor"]) else \
                    6 if any(w in t for w in ["dolor","molestia","preocupante"]) else \
                    4 if any(w in t for w in ["ligera","leve","tensión"]) else 2
        tendencia = "mejorando" if any(w in t for w in ["mejorando","recuperando","mejor"]) else \
                    "empeorando" if any(w in t for w in ["empeorando","peor","crítico"]) else "estable"
        fatigue = 9 if any(w in t for w in ["fatiga máxima","muy cansado","exhausto"]) else \
                  5 if any(w in t for w in ["fatiga moderada","algo cansado","cansado"]) else \
                  3 if any(w in t for w in ["fatiga normal","fatiga baja","descansado"]) else 5
        sleep = 2 if any(w in t for w in ["sin dormir","dormir mal","muy mal","pobre"]) else \
                5 if any(w in t for w in ["regular","dormir regular"]) else \
                8 if any(w in t for w in ["bien","duerme bien","excelente","muy bien"]) else 5
        stress = 8 if any(w in t for w in ["estrés alto","muy estresado","preocupado"]) else \
                 5 if any(w in t for w in ["estrés moderado","algo estresado"]) else \
                 2 if any(w in t for w in ["estrés bajo","relajado"]) else 5
        soreness = 8 if any(w in t for w in ["dolor muscular generalizado","muy dolorido"]) else \
                   5 if any(w in t for w in ["dolor muscular","molestia"]) else \
                   2 if any(w in t for w in ["sin dolor","dolor mínimo"]) else 3

        # --- NUEVO: Detección heurística del grado de lesión ---
        grado = "leve"
        if any(w in t for w in ["grave","grado 3","rotura total","fractura","ligamento cruzado"]):
            grado = "grave"
        elif any(w in t for w in ["moderado","grado 2","rotura parcial"]):
            grado = "moderado"

        return {"zona_lesion":zona,"severidad_lesion":severidad,"tendencia_lesion":tendencia,
                "fatiga_cualitativa":fatigue,"calidad_suenio_cualitativa":sleep,
                "estres_cualitativo":stress,"dolor_muscular_cualitativo":soreness,
                "confianza_extraccion":0.3,
                "grado_lesion": grado}   # <-- AÑADIDO

# ----------------------------------------------------------------
# PARSING NUMÉRICO PREVIO (MEJORA: extraer números del reporte)
# ----------------------------------------------------------------
def parsear_numeros_del_reporte(reporte: str) -> Dict:
    nums = {}
    for campo, patron in [("fatiga", r'Fatiga:\s*(\d+)/10'),
                           ("sueno", r'Sueño:\s*(\d+)/10'),
                           ("estres", r'Estrés:\s*(\d+)/10'),
                           ("dolor", r'Dolor:\s*(\d+)/10'),
                           ("acwr", r'ACWR:\s*([\d.]+)')]:
        m = re.search(patron, reporte)
        if m: nums[campo] = float(m.group(1))
    return nums

# ----------------------------------------------------------------
# EXTRACCIÓN CUALITATIVA (MEJORADA: prompt con números ya parseados)
# ----------------------------------------------------------------
def extraer_caracteristicas_cualitativas(reporte: str, use_llm: bool = True) -> Dict:
    if use_llm:
        numeros = parsear_numeros_del_reporte(reporte)
        prompt = f"""Eres un asistente médico-deportivo. Extrae del siguiente reporte los campos en formato JSON.
Responde SOLO con el JSON, sin explicaciones adicionales.

USA EXACTAMENTE estos valores numéricos (no los interpretes ni los cambies):
{json.dumps(numeros, ensure_ascii=False)}

Formato:
{{
  "zona_lesion": "nombre de la zona en español",
  "severidad_lesion": número del 1 al 10,
  "tendencia_lesion": "mejorando" | "estable" | "empeorando",
  "fatiga_cualitativa": número del 1 al 10,
  "calidad_suenio_cualitativa": número del 1 al 10,
  "estres_cualitativo": número del 1 al 10,
  "dolor_muscular_cualitativo": número del 1 al 10,
  "confianza_extraccion": número del 0 al 1
}}

Reporte: "{reporte}" """
        parsed = call_llm_json(prompt, FEATURE_SCHEMA)
        if parsed:
            features = validar_features(parsed)
            if features["confianza_extraccion"] < 0.4:
                logging.warning("Confianza de extracción baja (<0.4). Derivando a heurístico.")
                return HeuristicParser().parse(reporte)
            logging.info("Extracción realizada con Mistral")
            return features
        logging.warning("Mistral no disponible o respuesta inválida. Usando parser heurístico.")
    return HeuristicParser().parse(reporte)


def decision_arbitrada(ml_prob: float, features: Dict, reporte: str,
                       acwr_real: float = 0.0, use_llm: bool = True,
                       posicion_jugador: str = "Defensa") -> Dict:
    # Circuit breaker por baja confianza
    if features.get("confianza_extraccion", 1.0) < 0.4:
        use_llm = False
        logging.info("Decisión forzada a heurístico por baja confianza en extracción.")

    # --- CAPA 0: CRITICIDAD POR POSICIÓN, ZONA Y GRADO ---
    zona_lesion = features.get("zona_lesion", "desconocida")
    severidad = features.get("severidad_lesion", 0)
    grado_lesion = features.get("grado_lesion", "leve").lower()
    confianza = features.get("confianza_extraccion", 1.0)

    # 1. Criticidad base según posición y zona
    criticidad = CRITICIDAD.get(posicion_jugador, {}).get(zona_lesion, "MEDIA")

    # 2. Ajuste por grado de lesión (matriz de combinación)
    criticidad = COMBINACION_GRADO.get(criticidad, {}).get(grado_lesion, criticidad)

    # --- Cálculo del tiempo de recuperación (con doble fallback) ---
    if confianza >= 0.4:
        # Intentar primero el diccionario por zona y grado (más específico)
        zona_norm = normalizar_zona(zona_lesion)
        tiempo_rec = TIEMPOS_RECUPERACION.get((zona_norm, grado_lesion))
        if tiempo_rec is None:
            tiempo_rec = obtener_tiempo_recuperacion_determinista(criticidad, severidad)
    else:
        # Con baja confianza, usar directamente el diccionario por criticidad y severidad
        tiempo_rec = obtener_tiempo_recuperacion_determinista(criticidad, severidad)

    # REGLA 1: Criticidad EXTREMA → ROJO
    if criticidad == "EXTREMA":
        decision, risk_score = "ROJO", 0.95
        razon = (f"Lesión grave ({zona_lesion}, grado {grado_lesion}) detectada. "
                 f"Es incapacitante independientemente de la posición. REPOSO INMEDIATO.")
        logging.info(json.dumps({"fuente":"REGLA_POSICION_EXTREMA","decision":decision,"risk":risk_score}))
        return {
            "decision": decision,
            "risk_score": round(risk_score, 2),
            "razonamiento": razon + f" | Tiempo estimado de baja: {tiempo_rec}.",
            "fuente": "REGLA_POSICION_EXTREMA",
            "tiempo_recuperacion": tiempo_rec
        }

    # REGLA 2: Criticidad ALTA + severidad relevante → ROJO
    if criticidad == "ALTA" and severidad > 3:
        decision, risk_score = "ROJO", max(ml_prob, 0.80)
        razon = (f"Regla por posición: zona crítica ({zona_lesion}, grado {grado_lesion}) "
                 f"con severidad ({severidad}/10) para un {posicion_jugador}. REPOSO OBLIGATORIO.")
        logging.info(json.dumps({"fuente":"REGLA_POSICION_ALTA","decision":decision,"risk":risk_score}))
        return {
            "decision": decision,
            "risk_score": round(risk_score, 2),
            "razonamiento": razon + f" | Tiempo estimado de baja: {tiempo_rec}.",
            "fuente": "REGLA_POSICION_ALTA",
            "tiempo_recuperacion": tiempo_rec
        }

    # REGLA 3: Severidad muy alta (>=7) → ROJO
    if severidad >= 7:
        decision, risk_score = "ROJO", 0.90
        razon = (f"Regla por severidad extrema ({severidad}/10) en {zona_lesion}. "
                 f"Dolor insoportable. REPOSO OBLIGATORIO.")
        logging.info(json.dumps({"fuente":"REGLA_SEVERIDAD_EXTREMA","decision":decision,"risk":risk_score}))
        return {
            "decision": decision,
            "risk_score": round(risk_score, 2),
            "razonamiento": razon + f" | Tiempo estimado de baja: {tiempo_rec}.",
            "fuente": "REGLA_SEVERIDAD_EXTREMA",
            "tiempo_recuperacion": tiempo_rec
        }

    # REGLA 4: Criticidad BAJA + severidad baja + ML tranquilo → VERDE
    if criticidad == "BAJA" and severidad < 5 and ml_prob < 0.5:
        decision, risk_score = "VERDE", ml_prob * 0.5
        razon = (f"Regla por posición: zona no crítica ({zona_lesion}, grado {grado_lesion}) "
                 f"con severidad baja ({severidad}/10) para un {posicion_jugador}. "
                 f"Puede continuar con normalidad.")
        logging.info(json.dumps({"fuente":"REGLA_POSICION_BAJA","decision":decision,"risk":risk_score}))
        return {
            "decision": decision,
            "risk_score": round(risk_score, 2),
            "razonamiento": razon + f" | Tiempo estimado de baja: {tiempo_rec}.",
            "fuente": "REGLA_POSICION_BAJA",
            "tiempo_recuperacion": tiempo_rec
        }

    # REGLA 5: Criticidad BAJA + severidad moderada + ML elevado → ÁMBAR
    if criticidad == "BAJA" and 5 <= severidad < 7 and ml_prob >= 0.5:
        decision, risk_score = "ÁMBAR", max(ml_prob, 0.40)
        razon = (f"Regla por posición: zona no crítica ({zona_lesion}, grado {grado_lesion}) "
                 f"con severidad moderada ({severidad}/10), pero el modelo de carga detecta riesgo "
                 f"(ml_prob={ml_prob:.2f}). Se recomienda monitoreo.")
        logging.info(json.dumps({"fuente":"REGLA_POSICION_BAJA_AMBAR","decision":decision,"risk":risk_score}))
        return {
            "decision": decision,
            "risk_score": round(risk_score, 2),
            "razonamiento": razon + f" | Tiempo estimado de baja: {tiempo_rec}.",
            "fuente": "REGLA_POSICION_BAJA_AMBAR",
            "tiempo_recuperacion": tiempo_rec
        }

    # --- CAPA 1: REGLAS DETERMINISTAS DE CARGA ---
    if ml_prob > 0.7 or acwr_real > 1.5:
        decision, risk_score = "ROJO", min(1.0, ml_prob + 0.15)
        if ml_prob > 0.7 and acwr_real > 1.5:
            razon = f"Regla clínica: alta probabilidad de lesión ({ml_prob:.2f}) y ACWR elevado ({acwr_real:.2f}). REPOSO OBLIGATORIO."
        elif ml_prob > 0.7:
            razon = f"Regla clínica: alta probabilidad de lesión ({ml_prob:.2f}). REPOSO OBLIGATORIO."
        else:
            razon = f"Regla clínica: ACWR muy elevado ({acwr_real:.2f}). REPOSO OBLIGATORIO."
        logging.info(json.dumps({"fuente":"REGLA","decision":decision,"risk":risk_score}))
        return {
            "decision": decision,
            "risk_score": round(risk_score, 2),
            "razonamiento": razon + f" | Tiempo estimado de baja: {tiempo_rec}.",
            "fuente": "REGLA",
            "tiempo_recuperacion": tiempo_rec
        }

    # --- CAPA 2: REGLA SUAVE ---
    if (ml_prob < 0.35 and
        features["fatiga_cualitativa"] < 6 and
        features["dolor_muscular_cualitativo"] < 5 and
        features["calidad_suenio_cualitativa"] > 4 and
        features["tendencia_lesion"] != "empeorando"):
        return {
            "decision": "VERDE",
            "risk_score": round(ml_prob, 2),
            "razonamiento": "Bajo riesgo global consistente. Sin signos de alarma." + f" | Tiempo estimado de baja: {tiempo_rec}.",
            "fuente": "REGLA_SUAVE",
            "tiempo_recuperacion": tiempo_rec
        }

    # --- CAPA 3: LLM ---
    if use_llm:
        numeros = parsear_numeros_del_reporte(reporte)
        prompt = f"""Eres un médico deportivo experto en prevención de lesiones.

DATOS PROPORCIONADOS (NO INVENTES VALORES ADICIONALES):
- Probabilidad de lesión según IA: {ml_prob}
- Características cualitativas extraídas: {json.dumps(features, ensure_ascii=False)}
- Valores numéricos conocidos: {json.dumps(numeros, ensure_ascii=False)}
- Reporte original: "{reporte}"

REGLAS DE APOYO (Gabbett 2016):
- Si fatiga > 7 → ÁMBAR.
- Si sueño < 4 → riesgo.
- Si dolor muscular > 6 → evitar VERDE.
- En ausencia de factores de riesgo, considera VERDE.

ESTIMACIÓN DE TIEMPO DE BAJA:
- Basándote en la zona, la severidad, el grado (si se menciona) y el contexto del reporte,
  estima el tiempo de baja probable.
- Indícalo en el campo opcional "tiempo_estimado" (ej: "2-3 semanas", "6-8 semanas").
- Si no tienes suficiente información, indica "No estimado".

IMPORTANTE:
- NO inventes valores.
- Usa SOLO los datos proporcionados.

Devuelve SOLO JSON con tu decisión:
{{
  "decision": "ÁMBAR" | "VERDE",
  "risk_score": número entre 0 y 1,
  "razonamiento": "explica brevemente",
  "tiempo_estimado": "estimación opcional del tiempo de baja"
}}"""
        raw = call_llm_json(prompt, DECISION_SCHEMA)
        if raw:
            validado = validar_decision(raw)
            if validado:
                if features["fatiga_cualitativa"] > 8:
                    validado["decision"] = "ÁMBAR"; validado["razonamiento"] += " (Ajuste por fatiga extrema)"
                if features["dolor_muscular_cualitativo"] > 7:
                    validado["decision"] = "ÁMBAR"; validado["razonamiento"] += " (Ajuste por dolor severo)"
                if features["calidad_suenio_cualitativa"] < 3:
                    validado["decision"] = "ÁMBAR"; validado["razonamiento"] += " (Ajuste por sueño muy pobre)"

                # --- Estimación del tiempo de baja (contextual o determinista) ---
                if "tiempo_estimado" in raw and raw["tiempo_estimado"] != "No estimado":
                    tiempo_rec = raw["tiempo_estimado"]
                    validado["razonamiento"] += f" | Tiempo estimado (contexto): {tiempo_rec}."
                else:
                    validado["razonamiento"] += f" | Tiempo estimado (determinista): {tiempo_rec}."

                logging.info(json.dumps({"fuente":"LLM","decision":validado["decision"],"risk":validado["risk_score"]}))
                validado["fuente"] = "LLM"
                validado["tiempo_recuperacion"] = tiempo_rec
                return validado
        logging.warning("Mistral no disponible o decisión inválida. Aplicando fallback heurístico.")

    # --- CAPA 4: FALLBACK HEURÍSTICO ---
    f, d, s = features["fatiga_cualitativa"]/10, features["dolor_muscular_cualitativo"]/10, 1-features["calidad_suenio_cualitativa"]/10
    risk = ml_prob*0.5 + f*0.15 + d*0.15 + s*0.1 + (f*d)*0.1
    if features["tendencia_lesion"] == "empeorando": risk += 0.15; razon = "Tendencia empeora. "
    elif features["tendencia_lesion"] == "mejorando": risk -= 0.10; razon = "Tendencia mejora. "
    else: razon = ""
    conf = features.get("confianza_extraccion", 1.0)
    if conf < 0.7: risk += (0.7 - conf) * 0.2
    risk = max(0.0, min(risk, 1.0))
    decision = "ÁMBAR" if risk > 0.4 else "VERDE"
    razon += "Riesgo combinado moderado." if decision == "ÁMBAR" else "Riesgo combinado bajo."
    logging.info(json.dumps({"fuente":"HEURISTICO","decision":decision,"risk":round(risk,2)}))
    return {
        "decision": decision,
        "risk_score": round(risk, 2),
        "razonamiento": razon + f" | Tiempo estimado de baja: {tiempo_rec}.",
        "fuente": "HEURISTICO",
        "debug": {
            "ml_prob": round(ml_prob, 3),
            "acwr": round(acwr_real, 2),
            "fatiga": features["fatiga_cualitativa"],
            "sueno": features["calidad_suenio_cualitativa"],
            "dolor": features["dolor_muscular_cualitativo"],
            "confianza": features["confianza_extraccion"]
        },
        "tiempo_recuperacion": tiempo_rec
    }

import sys
# !{sys.executable} -m pip install catboost

# ================================================================
# CARGA DE MODELOS Y DATASET (DESDE ZIP)
# ================================================================
ZIP_NAME = 'resultados_tfg.zip'
if not os.path.exists(ZIP_NAME):
    print("El archivo resultados_tfg.zip no se encuentra. Súbelo a Colab.")
else:
    with zipfile.ZipFile(ZIP_NAME, 'r') as z:
        for f in ['content/scaler.pkl','content/modelo_stacking_ensemble.pkl','content/dataset_sintetico_soccermon.csv']:
            if f in z.namelist():
                info = z.getinfo(f); info.filename = os.path.basename(f); z.extract(info); print(f"✓ {info.filename}")
    scaler = joblib.load('scaler.pkl')
    stack = joblib.load('modelo_stacking_ensemble.pkl')
    df = pd.read_csv('dataset_sintetico_soccermon.csv')

    feature_cols = ['distance_m','hsr_distance_m','acwr','ctl','atl',
                    'fatigue_score','sleep_quality','stress','muscle_soreness',
                    'inter_acwr_fatigue','th_acwr_fatigue','th_stress_sleep','inter_sleep_sore']

    ejemplos = df.sample(4, random_state=42)
    X_muestra = ejemplos[feature_cols].copy()
    X_scaled = scaler.transform(pd.DataFrame(X_muestra, columns=feature_cols))
    probas = stack.predict_proba(X_scaled)[:, 1]
    acwr_reales = X_muestra['acwr'].values

    def generar_reporte(fila):
        zona = "muslo" if fila['muscle_soreness']>5 else "sin zona definida"
        return (f"Zona: {zona}. Fatiga: {int(fila['fatigue_score'])}/10. "
                f"Sueño: {int(fila['sleep_quality'])}/10. Estrés: {int(fila['stress'])}/10. "
                f"Dolor: {int(fila['muscle_soreness'])}/10. ACWR: {fila['acwr']:.2f}")

    ejemplos['reporte'] = X_muestra.apply(generar_reporte, axis=1)
    casos = [{"reporte": ejemplos['reporte'].iloc[i], "ml_prob": float(probas[i])} for i in range(len(ejemplos))]

    print("="*80)
    print("DEMOSTRACIÓN FINAL – DSS HÍBRIDO CON TRAZABILIDAD (v3.1)")
    print("="*80)
    for i, caso in enumerate(casos, 1):
        print(f"\n--- Caso {i} ---")
        print(f"Reporte: {caso['reporte']}")
        print(f"P(lesión): {caso['ml_prob']:.3f}")
        features = extraer_caracteristicas_cualitativas(caso["reporte"], use_llm=True)
        decision = decision_arbitrada(caso["ml_prob"], features, caso["reporte"],
                                      acwr_real=acwr_reales[i-1], use_llm=True)
        print(f"Decisión: {decision['decision']} | Risk: {decision['risk_score']} | Fuente: {decision['fuente']}")
        print(f"Razonamiento: {decision['razonamiento']}")
        if "debug" in decision:
            print(f"Debug: {json.dumps(decision['debug'], ensure_ascii=False)}")

# ================================================================
# DEMOSTRACIÓN DE CRITICIDAD POR POSICIÓN, ZONA Y GRADO
# ================================================================

scaler = None
stack = None

if scaler is not None and stack is not None:
    print("\n" + "="*80)
    print("DEMOSTRACIÓN DE CRITICIDAD POR POSICIÓN, ZONA Y GRADO")
    print("="*80)

    # Caso A: Defensa con muñeca rota (grado grave) -> ROJO por EXTREMA
    print("\n--- Caso A: Defensa con muñeca rota (grave) ---")
    features_a = {
        "zona_lesion": "muñeca",
        "severidad_lesion": 9,
        "tendencia_lesion": "estable",
        "fatiga_cualitativa": 3,
        "calidad_suenio_cualitativa": 7,
        "estres_cualitativo": 4,
        "dolor_muscular_cualitativo": 2,
        "confianza_extraccion": 0.8,
        "grado_lesion": "grave"
    }
    dec_a = decision_arbitrada(0.25, features_a, "Muñeca rota tras caída", acwr_real=1.1, use_llm=False, posicion_jugador="Defensa")
    print(f"Decisión: {dec_a['decision']} | Fuente: {dec_a['fuente']}")
    print(f"Razonamiento: {dec_a['razonamiento']}")

    # Caso B: Portero con dolor leve en muñeca (grado leve) -> ROJO por ALTA
    print("\n--- Caso B: Portero con dolor leve en muñeca (leve) ---")
    features_b = {
        "zona_lesion": "muñeca",
        "severidad_lesion": 4,
        "tendencia_lesion": "estable",
        "fatiga_cualitativa": 3,
        "calidad_suenio_cualitativa": 7,
        "estres_cualitativo": 4,
        "dolor_muscular_cualitativo": 2,
        "confianza_extraccion": 0.8,
        "grado_lesion": "leve"
    }
    dec_b = decision_arbitrada(0.25, features_b, "Dolor leve en muñeca", acwr_real=1.1, use_llm=False, posicion_jugador="Portero")
    print(f"Decisión: {dec_b['decision']} | Fuente: {dec_b['fuente']}")
    print(f"Razonamiento: {dec_b['razonamiento']}")

    # Caso C: Delantero con dolor leve en muñeca (grado leve) -> VERDE por BAJA
    print("\n--- Caso C: Delantero con dolor leve en muñeca (leve) ---")
    features_c = {
        "zona_lesion": "muñeca",
        "severidad_lesion": 4,
        "tendencia_lesion": "estable",
        "fatiga_cualitativa": 3,
        "calidad_suenio_cualitativa": 7,
        "estres_cualitativo": 4,
        "dolor_muscular_cualitativo": 2,
        "confianza_extraccion": 0.8,
        "grado_lesion": "leve"
    }
    dec_c = decision_arbitrada(0.25, features_c, "Dolor leve en muñeca", acwr_real=1.1, use_llm=False, posicion_jugador="Delantero")
    print(f"Decisión: {dec_c['decision']} | Fuente: {dec_c['fuente']}")
    print(f"Razonamiento: {dec_c['razonamiento']}")

    # Caso D: Delantero con fractura de muñeca (grado grave) -> ALTA -> ROJO
    print("\n--- Caso D: Delantero con fractura de muñeca (grave) ---")
    features_d = {
        "zona_lesion": "muñeca",
        "severidad_lesion": 8,
        "tendencia_lesion": "estable",
        "fatiga_cualitativa": 3,
        "calidad_suenio_cualitativa": 7,
        "estres_cualitativo": 4,
        "dolor_muscular_cualitativo": 2,
        "confianza_extraccion": 0.8,
        "grado_lesion": "grave"
    }
    dec_d = decision_arbitrada(0.25, features_d, "Fractura de muñeca confirmada", acwr_real=1.1, use_llm=False, posicion_jugador="Delantero")
    print(f"Decisión: {dec_d['decision']} | Fuente: {dec_d['fuente']}")
    print(f"Razonamiento: {dec_d['razonamiento']}")

    # Caso E: Defensa con rotura fibrilar grado 1 en isquiotibiales (leve) -> ALTA -> ROJO
    print("\n--- Caso E: Defensa con rotura fibrilar grado 1 en isquiotibiales (leve) ---")
    features_e = {
        "zona_lesion": "isquiotibiales",
        "severidad_lesion": 4,
        "tendencia_lesion": "estable",
        "fatiga_cualitativa": 3,
        "calidad_suenio_cualitativa": 7,
        "estres_cualitativo": 4,
        "dolor_muscular_cualitativo": 2,
        "confianza_extraccion": 0.8,
        "grado_lesion": "leve"
    }
    dec_e = decision_arbitrada(0.25, features_e, "Rotura fibrilar grado 1 en isquiotibiales", acwr_real=1.1, use_llm=False, posicion_jugador="Defensa")
    print(f"Decisión: {dec_e['decision']} | Fuente: {dec_e['fuente']}")
    print(f"Razonamiento: {dec_e['razonamiento']}")

# ================================================================
# EJEMPLO PRACTICO COMPLETO - SISTEMA EXPERTO CON GRADOS Y TIEMPOS
# ================================================================
import json

print("=" * 80)
print("EJEMPLO PRACTICO COMPLETO DEL SISTEMA EXPERTO")
print("=" * 80)

# --- Caso: Delantero con rotura fibrilar grado 2 en isquiotibiales ---
print("\n" + "=" * 80)
print("CASO PRACTICO: DELANTERO CON ROTURA FIBRILAR GRADO 2 EN ISQUIOTIBIALES")
print("=" * 80)

# 1. Datos de entrada (simulando lo que pondria un fisioterapeuta en la app)
reporte = (
    "Rotura fibrilar grado 2 en isquiotibiales izquierdos. "
    "Dolor intenso al sprintar. Fatiga: 4/10. Sueno: 7/10. "
    "Estres: 5/10. Dolor: 8/10. ACWR: 1.15"
)

posicion = "Delantero"
ml_prob = 0.42    # Probabilidad que daria el modelo de stacking
acwr_real = 1.15

print(f"\nReporte del fisioterapeuta:\n   \"{reporte}\"")
print(f"\nPosicion del jugador: {posicion}")
print(f"Probabilidad de lesion (ML): {ml_prob:.2f}")
print(f"ACWR real: {acwr_real:.2f}")

# 2. Extraccion de caracteristicas cualitativas (con Mistral si esta disponible)
print("\n" + "-" * 60)
print("PASO 1: Extraccion de caracteristicas cualitativas")
print("-" * 60)

features = extraer_caracteristicas_cualitativas(reporte, use_llm=True)
print(json.dumps(features, indent=2, ensure_ascii=False))

# 3. Decision hibrida (con todas las capas: criticidad, grado, tiempos)
print("\n" + "-" * 60)
print("PASO 2: Decision del DSS hibrido")
print("-" * 60)

decision = decision_arbitrada(
    ml_prob,
    features,
    reporte,
    acwr_real=acwr_real,
    use_llm=True,
    posicion_jugador=posicion
)

# 4. Mostrar resultados
print("\n" + "=" * 60)
print("RESULTADOS DEL DSS")
print("=" * 60)
print(f"Decision:              {decision['decision']}")
print(f"Risk Score:            {decision['risk_score']:.2f}")
print(f"Fuente de decision:    {decision['fuente']}")
print(f"Tiempo de recuperacion: {decision.get('tiempo_recuperacion', 'No estimado')}")
print(f"\nRazonamiento:          {decision['razonamiento']}")

if "debug" in decision:
    print(f"\nDebug:               {json.dumps(decision['debug'], ensure_ascii=False)}")

# 5. Explicacion de como se aplicaron las reglas
print("\n" + "=" * 60)
print("TRAZA DE APLICACION DE REGLAS")
print("=" * 60)

zona = features.get("zona_lesion", "desconocida")
grado = features.get("grado_lesion", "leve")
criticidad_base = CRITICIDAD.get(posicion, {}).get(zona, "MEDIA")
criticidad_ajustada = COMBINACION_GRADO.get(criticidad_base, {}).get(grado, criticidad_base)
tiempo = TIEMPOS_RECUPERACION.get((zona, grado), "No estimado")

print(f"1. Zona extraida:       '{zona}'")
print(f"2. Posicion:            '{posicion}'")
print(f"3. Criticidad base:     {criticidad_base}")
print(f"4. Grado de lesion:     '{grado}'")
print(f"5. Criticidad ajustada: {criticidad_ajustada}")
print(f"6. Tiempo recuperacion: {tiempo}")
print(f"7. ml_prob > 0.7?:     {'SI' if ml_prob > 0.7 else 'NO'}")
print(f"8. acwr_real > 1.5?:   {'SI' if acwr_real > 1.5 else 'NO'}")
print(f"9. Criticidad ALTA y severidad > 3?: {'SI' if criticidad_ajustada == 'ALTA' and features.get('severidad_lesion', 0) > 3 else 'NO'}")
print(f"10. Decision final:     {decision['decision']}")

# ================================================================
# EJEMPLO PRACTICO COMPLETO - SISTEMA EXPERTO CON GRADOS Y TIEMPOS
# (Caso con tiempo de recuperacion estimado)
# ================================================================
import json

print("=" * 80)
print("EJEMPLO PRACTICO COMPLETO DEL SISTEMA EXPERTO")
print("=" * 80)

# --- Caso: Defensa con rotura fibrilar grado 2 en isquiotibiales ---
print("\n" + "=" * 80)
print("CASO PRACTICO: DEFENSA CON ROTURA FIBRILAR GRADO 2 EN ISQUIOTIBIALES")
print("=" * 80)

# 1. Datos de entrada (simulando lo que pondria un fisioterapeuta en la app)
reporte = (
    "Rotura fibrilar grado 2 en isquiotibiales. "
    "Dolor moderado al sprintar. Fatiga: 3/10. Sueno: 6/10. "
    "Estres: 4/10. Dolor: 5/10. ACWR: 1.20"
)

posicion = "Defensa"
ml_prob = 0.38    # Probabilidad que daria el modelo de stacking
acwr_real = 1.20

print(f"\nReporte del fisioterapeuta:\n   \"{reporte}\"")
print(f"\nPosicion del jugador: {posicion}")
print(f"Probabilidad de lesion (ML): {ml_prob:.2f}")
print(f"ACWR real: {acwr_real:.2f}")

# 2. Extraccion de caracteristicas cualitativas (con Mistral si esta disponible)
print("\n" + "-" * 60)
print("PASO 1: Extraccion de caracteristicas cualitativas")
print("-" * 60)

features = extraer_caracteristicas_cualitativas(reporte, use_llm=True)
print(json.dumps(features, indent=2, ensure_ascii=False))

# 3. Decision hibrida (con todas las capas: criticidad, grado, tiempos)
print("\n" + "-" * 60)
print("PASO 2: Decision del DSS hibrido")
print("-" * 60)

decision = decision_arbitrada(
    ml_prob,
    features,
    reporte,
    acwr_real=acwr_real,
    use_llm=True,
    posicion_jugador=posicion
)

# 4. Mostrar resultados
print("\n" + "=" * 60)
print("RESULTADOS DEL DSS")
print("=" * 60)
print(f"Decision:              {decision['decision']}")
print(f"Risk Score:            {decision['risk_score']:.2f}")
print(f"Fuente de decision:    {decision['fuente']}")
print(f"Tiempo de recuperacion: {decision.get('tiempo_recuperacion', 'No estimado')}")
print(f"\nRazonamiento:          {decision['razonamiento']}")

if "debug" in decision:
    print(f"\nDebug:               {json.dumps(decision['debug'], ensure_ascii=False)}")

# 5. Explicacion de como se aplicaron las reglas
print("\n" + "=" * 60)
print("TRAZA DE APLICACION DE REGLAS")
print("=" * 60)

zona = features.get("zona_lesion", "desconocida")
grado = features.get("grado_lesion", "leve")
criticidad_base = CRITICIDAD.get(posicion, {}).get(zona, "MEDIA")
criticidad_ajustada = COMBINACION_GRADO.get(criticidad_base, {}).get(grado, criticidad_base)
tiempo = TIEMPOS_RECUPERACION.get((zona, grado), "No estimado")

print(f"1. Zona extraida:       '{zona}'")
print(f"2. Posicion:            '{posicion}'")
print(f"3. Criticidad base:     {criticidad_base}")
print(f"4. Grado de lesion:     '{grado}'")
print(f"5. Criticidad ajustada: {criticidad_ajustada}")
print(f"6. Tiempo recuperacion: {tiempo}")
print(f"7. ml_prob > 0.7?:     {'SI' if ml_prob > 0.7 else 'NO'}")
print(f"8. acwr_real > 1.5?:   {'SI' if acwr_real > 1.5 else 'NO'}")
print(f"9. Criticidad ALTA y severidad > 3?: {'SI' if criticidad_ajustada == 'ALTA' and features.get('severidad_lesion', 0) > 3 else 'NO'}")
print(f"10. Decision final:     {decision['decision']}")

# ================================================================
# DEMOSTRACIÓN COMPLETA DEL SISTEMA EXPERTO CON TIEMPOS DE BAJA
# ================================================================
if scaler is not None and stack is not None:
    print("\n" + "=" * 80)
    print("DEMOSTRACIÓN DE CRITICIDAD, GRADOS Y TIEMPOS DE RECUPERACIÓN")
    print("=" * 80)

    # --- Caso A: Defensa con muñeca rota (grave) - usa diccionario específico ---
    print("\n--- Caso A: Defensa con muñeca rota (grave) ---")
    features_a = {
        "zona_lesion": "muñeca",
        "severidad_lesion": 9,
        "tendencia_lesion": "estable",
        "fatiga_cualitativa": 3,
        "calidad_suenio_cualitativa": 7,
        "estres_cualitativo": 4,
        "dolor_muscular_cualitativo": 2,
        "confianza_extraccion": 0.85,
        "grado_lesion": "grave"
    }
    dec_a = decision_arbitrada(0.25, features_a, "Muñeca rota tras caída",
                               acwr_real=1.1, use_llm=False, posicion_jugador="Defensa")
    print(f"Decisión: {dec_a['decision']} | Risk: {dec_a['risk_score']} | Fuente: {dec_a['fuente']}")
    print(f"Tiempo de recuperación: {dec_a.get('tiempo_recuperacion', 'No estimado')}")
    print(f"Razonamiento: {dec_a['razonamiento']}")

    # --- Caso B: Portero con dolor leve en muñeca (leve) - usa diccionario específico ---
    print("\n--- Caso B: Portero con dolor leve en muñeca (leve) ---")
    features_b = {
        "zona_lesion": "muñeca",
        "severidad_lesion": 4,
        "tendencia_lesion": "estable",
        "fatiga_cualitativa": 3,
        "calidad_suenio_cualitativa": 7,
        "estres_cualitativo": 4,
        "dolor_muscular_cualitativo": 2,
        "confianza_extraccion": 0.8,
        "grado_lesion": "leve"
    }
    dec_b = decision_arbitrada(0.25, features_b, "Dolor leve en muñeca",
                               acwr_real=1.1, use_llm=False, posicion_jugador="Portero")
    print(f"Decisión: {dec_b['decision']} | Risk: {dec_b['risk_score']} | Fuente: {dec_b['fuente']}")
    print(f"Tiempo de recuperación: {dec_b.get('tiempo_recuperacion', 'No estimado')}")
    print(f"Razonamiento: {dec_b['razonamiento']}")

    # --- Caso C: Delantero con dolor leve en muñeca (leve) - usa diccionario específico ---
    print("\n--- Caso C: Delantero con dolor leve en muñeca (leve) ---")
    features_c = {
        "zona_lesion": "muñeca",
        "severidad_lesion": 4,
        "tendencia_lesion": "estable",
        "fatiga_cualitativa": 3,
        "calidad_suenio_cualitativa": 7,
        "estres_cualitativo": 4,
        "dolor_muscular_cualitativo": 2,
        "confianza_extraccion": 0.8,
        "grado_lesion": "leve"
    }
    dec_c = decision_arbitrada(0.25, features_c, "Dolor leve en muñeca",
                               acwr_real=1.1, use_llm=False, posicion_jugador="Delantero")
    print(f"Decisión: {dec_c['decision']} | Risk: {dec_c['risk_score']} | Fuente: {dec_c['fuente']}")
    print(f"Tiempo de recuperación: {dec_c.get('tiempo_recuperacion', 'No estimado')}")
    print(f"Razonamiento: {dec_c['razonamiento']}")

    # --- Caso D: Delantero con fractura de muñeca (grave) - usa diccionario específico ---
    print("\n--- Caso D: Delantero con fractura de muñeca (grave) ---")
    features_d = {
        "zona_lesion": "muñeca",
        "severidad_lesion": 8,
        "tendencia_lesion": "estable",
        "fatiga_cualitativa": 3,
        "calidad_suenio_cualitativa": 7,
        "estres_cualitativo": 4,
        "dolor_muscular_cualitativo": 2,
        "confianza_extraccion": 0.8,
        "grado_lesion": "grave"
    }
    dec_d = decision_arbitrada(0.25, features_d, "Fractura de muñeca confirmada",
                               acwr_real=1.1, use_llm=False, posicion_jugador="Delantero")
    print(f"Decisión: {dec_d['decision']} | Risk: {dec_d['risk_score']} | Fuente: {dec_d['fuente']}")
    print(f"Tiempo de recuperación: {dec_d.get('tiempo_recuperacion', 'No estimado')}")
    print(f"Razonamiento: {dec_d['razonamiento']}")

    # --- Caso E: Defensa con rotura fibrilar grado 1 en isquiotibiales (leve) - usa diccionario específico ---
    print("\n--- Caso E: Defensa con rotura fibrilar grado 1 en isquiotibiales (leve) ---")
    features_e = {
        "zona_lesion": "isquiotibiales",
        "severidad_lesion": 4,
        "tendencia_lesion": "estable",
        "fatiga_cualitativa": 3,
        "calidad_suenio_cualitativa": 7,
        "estres_cualitativo": 4,
        "dolor_muscular_cualitativo": 2,
        "confianza_extraccion": 0.8,
        "grado_lesion": "leve"
    }
    dec_e = decision_arbitrada(0.25, features_e, "Rotura fibrilar grado 1 en isquiotibiales",
                               acwr_real=1.1, use_llm=False, posicion_jugador="Defensa")
    print(f"Decisión: {dec_e['decision']} | Risk: {dec_e['risk_score']} | Fuente: {dec_e['fuente']}")
    print(f"Tiempo de recuperación: {dec_e.get('tiempo_recuperacion', 'No estimado')}")
    print(f"Razonamiento: {dec_e['razonamiento']}")

    # --- Caso F: Delantero con lesión en espalda baja (leve) - usa diccionario específico ---
    print("\n--- Caso F: Delantero con dolor lumbar leve ---")
    features_f = {
        "zona_lesion": "espalda baja",
        "severidad_lesion": 3,
        "tendencia_lesion": "estable",
        "fatiga_cualitativa": 3,
        "calidad_suenio_cualitativa": 7,
        "estres_cualitativo": 4,
        "dolor_muscular_cualitativo": 2,
        "confianza_extraccion": 0.75,
        "grado_lesion": "leve"
    }
    dec_f = decision_arbitrada(0.20, features_f, "Dolor leve en la zona lumbar",
                               acwr_real=1.0, use_llm=False, posicion_jugador="Delantero")
    print(f"Decisión: {dec_f['decision']} | Risk: {dec_f['risk_score']} | Fuente: {dec_f['fuente']}")
    print(f"Tiempo de recuperación: {dec_f.get('tiempo_recuperacion', 'No estimado')}")
    print(f"Razonamiento: {dec_f['razonamiento']}")

    # --- Caso G: Baja confianza del NLP (usa fallback determinista por criticidad/severidad) ---
    print("\n--- Caso G: Defensa con zona desconocida (confianza baja) ---")
    features_g = {
        "zona_lesion": "parte trasera del muslo",  # no está en el diccionario
        "severidad_lesion": 5,
        "tendencia_lesion": "estable",
        "fatiga_cualitativa": 3,
        "calidad_suenio_cualitativa": 7,
        "estres_cualitativo": 4,
        "dolor_muscular_cualitativo": 2,
        "confianza_extraccion": 0.3,          # confianza baja
        "grado_lesion": "moderado"
    }
    dec_g = decision_arbitrada(0.35, features_g,
                               "Molestias en la parte trasera del muslo",
                               acwr_real=1.1, use_llm=False, posicion_jugador="Defensa")
    print(f"Decisión: {dec_g['decision']} | Risk: {dec_g['risk_score']} | Fuente: {dec_g['fuente']}")
    print(f"Tiempo de recuperación: {dec_g.get('tiempo_recuperacion', 'No estimado')}")
    print(f"Razonamiento: {dec_g['razonamiento']}")
