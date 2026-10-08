"""Logueo y registro del modelo en MLflow.

El pipeline se guarda con skops (el formato por defecto de mlflow, que no ejecuta código
al cargar). Los tipos que skops no conoce se declaran como confiables al loguear y quedan
en el MLmodel, así que para cargar el modelo alcanza con tener `car_price` en el PYTHONPATH.
"""
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature

from car_price.config import ALIAS_CHAMPION, EXPERIMENTO, NOMBRE_MODELO

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


def registrar_si_mejora(model_uri, metricas, nombre=NOMBRE_MODELO):
    """Registra un modelo logueado como versión nueva y la hace champion si mejora.

    `model_uri` es el de `loguear_modelo(...).model_uri` (`models:/m-...`). `metricas` son
    las de test (la salida de `evaluar`) y quedan como tags de la versión.

    El alias `champion` pasa a la versión nueva si no había champion o si su RMSE-log en
    test es menor que el del champion actual; si no, la versión queda registrada sin alias.
    Todas las versiones se evalúan sobre el mismo test (split con semilla fija), así que
    los RMSE se pueden comparar.

    Devuelve un dict con `version`, `champion` (si quedó como champion) y
    `champion_anterior` (la versión que tenía el alias antes, o None).
    """
    client = MlflowClient()
    version = mlflow.register_model(model_uri, nombre).version
    for k, v in metricas.items():
        client.set_model_version_tag(nombre, version, k, v)

    anterior = client.get_registered_model(nombre).aliases.get(ALIAS_CHAMPION)
    if anterior is None:
        mejora = True
    else:
        rmse_champion = float(client.get_model_version(nombre, anterior).tags['rmse_log'])
        mejora = metricas['rmse_log'] < rmse_champion

    if mejora:
        client.set_registered_model_alias(nombre, ALIAS_CHAMPION, version)

    return {'version': int(version), 'champion': mejora,
            'champion_anterior': int(anterior) if anterior is not None else None}
