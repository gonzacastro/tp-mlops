import logging
from contextlib import asynccontextmanager
from typing import NamedTuple

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion

from car_price.config import ALIAS_CHAMPION, NOMBRE_MODELO
from schemas import Auto, InfoModelo, MetricasTest, Prediccion, Recarga, Salud

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


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"message": "Welcome to the Model Service"}


@app.get("/health", response_model=Salud)
def health():
    """Para el healthcheck de Docker: responde aunque no haya modelo cargado."""
    return Salud(status="ok", modelo_cargado=app.state.champion is not None)


@app.post("/predict", response_model=Prediccion)
def predict(auto: Auto):
    """Precio estimado en rupias para una publicación con el formato del CSV."""
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


@app.get("/modelo", response_model=InfoModelo)
def info_modelo():
    """Versión, run y métricas en test del modelo que se está sirviendo."""
    champion = app.state.champion
    if champion is None:
        raise sin_modelo()

    v = champion.version
    tags = v.tags or {}
    metricas = (MetricasTest(**{k: float(tags[k]) for k in MetricasTest.model_fields})
                if all(k in tags for k in MetricasTest.model_fields) else None)
    return InfoModelo(nombre=NOMBRE_MODELO, version=v.version, alias=ALIAS_CHAMPION,
                      run_id=v.run_id, metricas_test=metricas)


@app.post("/modelo/recargar", response_model=Recarga)
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
