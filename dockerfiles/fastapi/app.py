"""API de predicción de precios de autos usados (CarDekho, India).

Sirve el modelo `car_price_gb@champion` del registry de MLflow. Documentación interactiva
en /docs.
"""
import logging
from contextlib import asynccontextmanager
from typing import Annotated, NamedTuple

import mlflow
import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion

from car_price.config import ALIAS_CHAMPION, NOMBRE_MODELO
from schemas import EJEMPLO_AUTO, Auto, InfoModelo, MetricasTest, Prediccion, Recarga, Salud

log = logging.getLogger("uvicorn.error")


class Champion(NamedTuple):
    """Modelo y versión juntos: se reemplazan de una sola vez al recargar."""
    modelo: object
    version: ModelVersion


def cargar_champion():
    """Carga el modelo con alias champion. Si no hay o MLflow no responde, lanza excepción.

    Primero resuelve a qué versión apunta el alias y después carga esa versión, así lo que
    se informa es lo que se cargó aunque el alias se mueva en el medio.
    """
    version = MlflowClient().get_model_version_by_alias(NOMBRE_MODELO, ALIAS_CHAMPION)
    modelo = mlflow.sklearn.load_model(f"models:/{NOMBRE_MODELO}/{version.version}")
    log.info("Modelo cargado: %s versión %s (@%s)", NOMBRE_MODELO, version.version, ALIAS_CHAMPION)
    return Champion(modelo, version)


def sin_modelo():
    """503 cuando no hay champion cargado, con el motivo y qué hacer."""
    return HTTPException(
        status_code=503,
        detail=f"No hay modelo {NOMBRE_MODELO}@{ALIAS_CHAMPION} cargado ({app.state.error}). "
               f"Corré el DAG train_car_price y después POST /modelo/recargar.",
    )


@asynccontextmanager
async def lifespan(app):
    # Si no hay champion la API arranca igual: /predict responde 503 hasta que se recargue
    try:
        app.state.champion, app.state.error = cargar_champion(), None
    except Exception as e:
        app.state.champion, app.state.error = None, str(e).rstrip(".")
        log.warning("Sin modelo %s@%s: %s. Corré el DAG train_car_price.",
                    NOMBRE_MODELO, ALIAS_CHAMPION, app.state.error)
    yield


DESCRIPCION = """
Estima el precio de un auto usado en India a partir de los datos de su publicación.

**Modelo.** Gradient Boosting entrenado con el dataset de
[CarDekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)
(8.128 publicaciones, 6.691 después de la limpieza). Es un Pipeline de scikit-learn que
incluye la limpieza: recibe los campos tal como vienen en el CSV (`"23.4 kmpl"`,
`"1248 CC"`) y predice el logaritmo del precio. La API devuelve `exp(predicción)`, en rupias.

**Desempeño de referencia** en test (1.339 autos que el modelo no vio): R² ≈ 0.92 y
RMSE ≈ 0.21 en escala log, es decir, un error típico de ±20 % sobre el precio. El error
absoluto medio ronda las 72.000 INR. Las métricas exactas del modelo cargado están en
`GET /modelo`.

**De dónde sale el modelo.** El DAG `train_car_price` de Airflow entrena con búsqueda de
hiperparámetros, registra cada versión en MLflow como `car_price_gb` y le pasa el alias
`champion` solo si mejora el RMSE en test. La API carga el champion al arrancar; después
de una corrida del DAG, `POST /modelo/recargar` toma la versión nueva sin reiniciar.
"""

TAGS = [
    {"name": "Predicción", "description": "Precio estimado de un auto."},
    {"name": "Modelo", "description": "Qué modelo se está sirviendo y cómo actualizarlo."},
    {"name": "Servicio", "description": "Estado de la API."},
]

EJEMPLO_503 = {
    "description": "No hay un modelo champion cargado.",
    "content": {"application/json": {"example": {"detail": (
        f"No hay modelo {NOMBRE_MODELO}@{ALIAS_CHAMPION} cargado (INVALID_PARAMETER_VALUE: "
        f"Registered model alias champion not found). Corré el DAG train_car_price y "
        f"después POST /modelo/recargar.")}}},
}

EJEMPLOS_PREDICT = {
    "completo": {
        "summary": "Publicación completa",
        "description": "Una fila del CSV tal cual (el primer auto del dataset, vendido a 450.000 INR).",
        "value": EJEMPLO_AUTO,
    },
    "sin_ficha_tecnica": {
        "summary": "Sin ficha técnica",
        "description": "Solo los campos obligatorios: el modelo imputa rendimiento, motor, "
                       "potencia y asientos.",
        "value": {k: EJEMPLO_AUTO[k] for k in
                  ("name", "year", "km_driven", "fuel", "seller_type", "transmission", "owner")},
    },
    "gnc_automatico": {
        "summary": "GNC, segundo dueño",
        "description": "Rendimiento en km/kg, como en los autos a gas.",
        "value": {"name": "Maruti Wagon R LXI CNG", "year": 2017, "km_driven": 40000,
                  "fuel": "CNG", "seller_type": "Dealer", "transmission": "Manual",
                  "owner": "Second Owner", "mileage": "26.6 km/kg", "engine": "998 CC",
                  "max_power": "58.16 bhp", "torque": "77Nm@ 3500rpm", "seats": 5},
    },
}

app = FastAPI(
    title="Precio de autos usados · CarDekho",
    description=DESCRIPCION,
    version="1.0.0",
    openapi_tags=TAGS,
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def raiz():
    return RedirectResponse("/docs")


@app.get("/health", response_model=Salud, tags=["Servicio"], summary="Estado del servicio")
def health():
    """Responde `ok` siempre que la API esté arriba, haya o no modelo cargado (así el
    healthcheck de Docker no reinicia el contenedor antes de que corra el DAG).
    `modelo_cargado` dice si `/predict` va a poder responder."""
    return Salud(status="ok", modelo_cargado=app.state.champion is not None)


@app.post("/predict", response_model=Prediccion, tags=["Predicción"],
          summary="Estimar el precio de un auto",
          responses={503: EJEMPLO_503})
def predict(auto: Annotated[Auto, Body(openapi_examples=EJEMPLOS_PREDICT)]):
    """Recibe una publicación con los campos del CSV de CarDekho y devuelve el precio
    estimado en rupias, junto con la versión del modelo que lo calculó.

    Los campos con unidades van como texto, igual que en el CSV (`"23.4 kmpl"`,
    `"1248 CC"`, `"74 bhp"`); la limpieza la hace el modelo. Los de la ficha técnica
    (`mileage`, `engine`, `max_power`, `torque`, `seats`) son opcionales.

    - **422** si falta un campo obligatorio o un valor está fuera de lo que vio el modelo
      (por ejemplo, `owner: "Test Drive Car"` o un año posterior a 2020).
    - **503** si todavía no hay modelo champion.
    """
    champion = app.state.champion
    if champion is None:
        raise sin_modelo()

    # El modelo predice log(precio) y hace la limpieza adentro: recibe la fila cruda
    X = pd.DataFrame([auto.model_dump()])
    log_precio = champion.modelo.predict(X)[0]
    return Prediccion(
        precio_inr=round(float(np.exp(log_precio))),
        version_modelo=champion.version.version,
        alias=ALIAS_CHAMPION,
    )


@app.get("/modelo", response_model=InfoModelo, tags=["Modelo"],
         summary="Modelo que se está sirviendo", responses={503: EJEMPLO_503})
def info_modelo():
    """Versión de `car_price_gb` cargada, el run de MLflow que la generó y sus métricas
    en test (`rmse_log` y `r2_log` en escala log; `mae_inr` en rupias)."""
    champion = app.state.champion
    if champion is None:
        raise sin_modelo()

    v = champion.version
    tags = v.tags or {}
    metricas = (MetricasTest(**{k: float(tags[k]) for k in MetricasTest.model_fields})
                if all(k in tags for k in MetricasTest.model_fields) else None)
    return InfoModelo(nombre=NOMBRE_MODELO, version=v.version, alias=ALIAS_CHAMPION,
                      run_id=v.run_id, metricas_test=metricas)


@app.post("/modelo/recargar", response_model=Recarga, tags=["Modelo"],
          summary="Tomar el champion actual sin reiniciar",
          responses={503: {"description": "No se pudo recargar; sigue el modelo anterior.",
                           "content": {"application/json": {"example": {"detail": (
                               "No se pudo recargar (INVALID_PARAMETER_VALUE: Registered "
                               "model alias champion not found); sigue la versión 2.")}}}}})
def recargar():
    """Vuelve a leer el alias champion y carga esa versión, sin reiniciar el contenedor.

    Si la versión no cambió, no vuelve a cargar nada. Si falla (no hay champion o MLflow
    no responde), sigue sirviendo el modelo anterior y responde 503.
    """
    anterior = app.state.champion
    v_anterior = anterior.version.version if anterior else None
    try:
        v_actual = MlflowClient().get_model_version_by_alias(NOMBRE_MODELO, ALIAS_CHAMPION).version
        if v_actual != v_anterior:
            app.state.champion, app.state.error = cargar_champion(), None
    except Exception as e:
        sigue = f"sigue la versión {v_anterior}" if anterior else "sigue sin modelo"
        raise HTTPException(status_code=503,
                            detail=f"No se pudo recargar ({str(e).rstrip('.')}); {sigue}.")

    return Recarga(version_anterior=v_anterior, version_actual=v_actual,
                   cambio=v_actual != v_anterior)
