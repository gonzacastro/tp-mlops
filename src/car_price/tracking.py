"""Logueo del modelo en MLflow.

El pipeline se guarda con skops (el formato por defecto de mlflow, que no ejecuta código
al cargar). Los tipos que skops no conoce se declaran como confiables al loguear y quedan
en el MLmodel, así que para cargar el modelo alcanza con tener `car_price` en el PYTHONPATH.
"""
import mlflow
from mlflow.models import infer_signature

from car_price.config import EXPERIMENTO

NOMBRE_ARTEFACTO = 'modelo'

# Tipos del pipeline que skops no trae como confiables (salen de skops.io.get_untrusted_types)
TIPOS_CONFIABLES = [
    'car_price.transformers.LimpiezaColumnas',
    'car_price.transformers.agrupar_asientos',
    'numpy.dtype',
    'sklearn.model_selection._split.KFold',
    'sklearn.tree._tree.Tree',
]


def loguear_modelo(pipe, X_ejemplo, metricas, params):
    """Guarda un pipeline entrenado en MLflow, con sus métricas, parámetros y firma.

    `X_ejemplo` son filas crudas del CSV (sin `selling_price`): con ellas se infiere la
    firma y la primera queda como ejemplo de entrada. Loguea en el run activo; si no hay
    ninguno, abre uno en el experimento `EXPERIMENTO`.

    Devuelve el `ModelInfo` de MLflow (`run_id`, `model_uri`, etc.).
    """
    if mlflow.active_run() is None:
        mlflow.set_experiment(EXPERIMENTO)
        with mlflow.start_run(run_name='modelo'):
            return _loguear(pipe, X_ejemplo, metricas, params)
    return _loguear(pipe, X_ejemplo, metricas, params)


def _loguear(pipe, X_ejemplo, metricas, params):
    mlflow.log_params(params)
    mlflow.log_metrics(metricas)
    firma = infer_signature(X_ejemplo, pipe.predict(X_ejemplo))
    return mlflow.sklearn.log_model(
        pipe,
        name=NOMBRE_ARTEFACTO,
        signature=firma,
        input_example=X_ejemplo.head(1),
        skops_trusted_types=TIPOS_CONFIABLES,
    )
