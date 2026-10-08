"""El modelo completo: de una fila cruda del CSV a log(precio)."""
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline

from car_price.config import MEJORES_PARAMS_GB, RANDOM_STATE
from car_price.features import construir_preprocesador
from car_price.transformers import LimpiezaColumnas


def construir_pipeline(**params):
    """Pipeline sin fittear: limpieza de columnas, preprocesamiento y Gradient Boosting.

    Por defecto usa `MEJORES_PARAMS_GB`; los `params` (sin prefijo, p. ej.
    `max_depth=4`) pisan esos valores. La búsqueda de hiperparámetros, en cambio,
    recorre `ESPACIO_GB` sobre el Pipeline con el prefijo `modelo__`.

    Recibe las columnas crudas del CSV (sin `selling_price`) y predice log(precio).
    """
    params_gb = {**MEJORES_PARAMS_GB, **params}
    return Pipeline([
        ('limpieza', LimpiezaColumnas()),
        ('prep', construir_preprocesador()),
        ('modelo', GradientBoostingRegressor(random_state=RANDOM_STATE, **params_gb)),
    ])
