import logging
from contextlib import asynccontextmanager

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient

from car_price.config import ALIAS_CHAMPION, NOMBRE_MODELO
from schemas import Auto, Prediccion

log = logging.getLogger("uvicorn.error")


def cargar_champion(app):
    """Carga el modelo con alias champion y guarda en `app.state` el modelo y su versión.

    Primero resuelve a qué versión apunta el alias y después carga esa versión, así lo que
    se informa es lo que se cargó aunque el alias se mueva en el medio. Si no hay champion
    (o MLflow no responde), deja el modelo en None y guarda el motivo: la API arranca igual.
    """
    try:
        version = MlflowClient().get_model_version_by_alias(NOMBRE_MODELO, ALIAS_CHAMPION)
        modelo = mlflow.sklearn.load_model(f"models:/{NOMBRE_MODELO}/{version.version}")
    except Exception as e:
        app.state.modelo, app.state.version, app.state.error = None, None, str(e).rstrip(".")
        log.warning("Sin modelo %s@%s: %s. Corré el DAG train_car_price.",
                    NOMBRE_MODELO, ALIAS_CHAMPION, app.state.error)
        return

    app.state.modelo, app.state.version, app.state.error = modelo, version, None
    log.info("Modelo cargado: %s versión %s (@%s)", NOMBRE_MODELO, version.version, ALIAS_CHAMPION)


@asynccontextmanager
async def lifespan(app):
    cargar_champion(app)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"message": "Welcome to the Model Service"}


@app.post("/predict", response_model=Prediccion)
def predict(auto: Auto):
    """Precio estimado en rupias para una publicación con el formato del CSV."""
    if app.state.modelo is None:
        raise HTTPException(
            status_code=503,
            detail=f"No hay modelo {NOMBRE_MODELO}@{ALIAS_CHAMPION} cargado "
                   f"({app.state.error}). Corré el DAG train_car_price y reiniciá la API.",
        )

    # El modelo predice log(precio) y hace la limpieza adentro: recibe la fila cruda
    X = pd.DataFrame([auto.model_dump()])
    log_precio = app.state.modelo.predict(X)[0]
    return Prediccion(
        precio_inr=round(float(np.exp(log_precio))),
        version_modelo=app.state.version.version,
        alias=ALIAS_CHAMPION,
    )
