"""Transformaciones propias que viajan adentro del Pipeline serializado.

Todo está a nivel de módulo (nada de lambdas ni funciones locales) para que el modelo
se pueda deserializar en FastAPI con solo tener `car_price` en el PYTHONPATH.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from car_price.config import RARE_SEATS
from car_price.data import limpiar_columnas


class LimpiezaColumnas(BaseEstimator, TransformerMixin):
    """Primer paso del Pipeline: de columnas crudas del CSV a columnas numéricas.

    Aplica `limpiar_columnas` (parseo de unidades, `brand`, `age`). No aprende nada de
    los datos, así que `fit` no hace nada. No necesita `selling_price`: la API no lo manda.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return limpiar_columnas(X)


def agrupar_asientos(X_in):
    """Colapsa las plazas raras (RARE_SEATS) en 'other'; el resto queda como texto."""
    arr = np.asarray(X_in).ravel()
    return np.where(np.isin(arr, RARE_SEATS), 'other',
                    arr.astype(object).astype(str)).reshape(-1, 1)
