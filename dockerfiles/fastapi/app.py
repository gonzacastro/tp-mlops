import logging
from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI
from mlflow import MlflowClient

from car_price.config import ALIAS_CHAMPION, NOMBRE_MODELO

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
