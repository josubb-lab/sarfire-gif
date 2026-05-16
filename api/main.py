from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date

BASE = Path(__file__).parent.parent
MODEL = joblib.load(BASE / "models" / "xgb_optimized.joblib")
PREPROCESSOR = joblib.load(BASE / "models" / "preprocessor_baseline.joblib")

PROVINCIA_REGION: dict[int, str] = {
    1: "Norte", 2: "Centro", 3: "Levante", 4: "Sur", 5: "Centro",
    6: "Centro", 7: "Baleares", 8: "Nordeste", 9: "Centro", 10: "Centro",
    11: "Sur", 12: "Levante", 13: "Centro", 14: "Sur", 15: "Noroeste",
    16: "Centro", 17: "Nordeste", 18: "Sur", 19: "Centro", 20: "Norte",
    21: "Sur", 22: "Nordeste", 23: "Sur", 24: "Centro", 25: "Nordeste",
    26: "Norte", 27: "Noroeste", 28: "Centro", 29: "Sur", 30: "Levante",
    31: "Norte", 32: "Noroeste", 33: "Noroeste", 34: "Centro", 35: "Canarias",
    36: "Noroeste", 37: "Centro", 38: "Canarias", 39: "Norte", 40: "Centro",
    41: "Sur", 42: "Centro", 43: "Nordeste", 44: "Nordeste", 45: "Centro",
    46: "Levante", 47: "Centro", 48: "Norte", 49: "Centro", 50: "Nordeste",
    51: "Sur", 52: "Sur",
}

PROVINCIA_NOMBRE: dict[int, str] = {
    1: "Álava", 2: "Albacete", 3: "Alicante", 4: "Almería", 5: "Ávila",
    6: "Badajoz", 7: "Balears (Illes)", 8: "Barcelona", 9: "Burgos", 10: "Cáceres",
    11: "Cádiz", 12: "Castellón", 13: "Ciudad Real", 14: "Córdoba", 15: "Coruña (A)",
    16: "Cuenca", 17: "Girona", 18: "Granada", 19: "Guadalajara", 20: "Gipuzkoa",
    21: "Huelva", 22: "Huesca", 23: "Jaén", 24: "León", 25: "Lleida",
    26: "Rioja (La)", 27: "Lugo", 28: "Madrid", 29: "Málaga", 30: "Murcia",
    31: "Navarra", 32: "Ourense", 33: "Asturias", 34: "Palencia", 35: "Palmas (Las)",
    36: "Pontevedra", 37: "Salamanca", 38: "Santa Cruz de Tenerife", 39: "Cantabria",
    40: "Segovia", 41: "Sevilla", 42: "Soria", 43: "Tarragona", 44: "Teruel",
    45: "Toledo", 46: "Valencia", 47: "Valladolid", 48: "Bizkaia", 49: "Zamora",
    50: "Zaragoza", 51: "Ceuta", 52: "Melilla",
}


def fwi_to_cat(fwi: float) -> str:
    if fwi < 5.2:
        return "bajo"
    if fwi < 11.2:
        return "moderado"
    if fwi < 21.3:
        return "alto"
    if fwi < 38.0:
        return "muy_alto"
    return "extremo"


def mes_to_estacion(mes: int) -> str:
    if mes in (12, 1, 2):
        return "invierno"
    if mes in (3, 4, 5):
        return "primavera"
    if mes in (6, 7, 8):
        return "verano"
    return "otoño"


def riesgo_label(prob: float) -> str:
    if prob < 0.3:
        return "bajo"
    if prob < 0.5:
        return "moderado"
    if prob < 0.7:
        return "alto"
    return "critico"


app = FastAPI(
    title="SARFIRE-GIF API",
    description="Predicción de Grandes Incendios Forestales (>500 ha) en España",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    fecha: date = Field(..., description="Fecha del incendio (YYYY-MM-DD)")
    provincia_id: int = Field(..., ge=1, le=52, description="Código INE de provincia (1-52)")
    fwi_mean: float = Field(..., ge=0, description="FWI promedio provincial diario")
    fwi_max: float = Field(..., ge=0, description="FWI máximo provincial")
    fwi_p90: float = Field(..., ge=0, description="FWI percentil 90 provincial")


class PredictResponse(BaseModel):
    probabilidad_gif: float
    riesgo: str
    es_gif: bool
    provincia: str
    region: str
    features: dict


@app.get("/")
def root():
    return {"status": "ok", "modelo": "XGBoost optimizado", "recall": "70.8%"}


@app.get("/provincias")
def provincias():
    return [{"id": k, "nombre": v, "region": PROVINCIA_REGION[k]} for k, v in PROVINCIA_NOMBRE.items()]


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if req.provincia_id not in PROVINCIA_REGION:
        raise HTTPException(status_code=400, detail=f"Provincia {req.provincia_id} no reconocida")

    mes = req.fecha.month
    dia_semana = req.fecha.weekday()
    año = req.fecha.year
    estacion = mes_to_estacion(mes)
    fwi_cat = fwi_to_cat(req.fwi_max)
    region = PROVINCIA_REGION[req.provincia_id]

    features = {
        "fwi_mean": req.fwi_mean,
        "fwi_max": req.fwi_max,
        "fwi_p90": req.fwi_p90,
        "año": año,
        "fwi_cat": fwi_cat,
        "estacion": estacion,
        "region": region,
        "dia_semana": dia_semana,
        "mes": mes,
    }

    X = pd.DataFrame([features])
    X_pre = PREPROCESSOR.transform(X)
    prob = float(MODEL.predict_proba(X_pre)[0][1])

    return PredictResponse(
        probabilidad_gif=round(prob, 4),
        riesgo=riesgo_label(prob),
        es_gif=prob >= 0.5,
        provincia=PROVINCIA_NOMBRE[req.provincia_id],
        region=region,
        features=features,
    )
