"""Métricas del modelo: las de AdM I en escala log, más el error en rupias."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluar(pipe, X, y):
    """Evalúa un pipeline ya entrenado sobre (X, y), con y en log(precio).

    - `rmse_log` y `r2_log`: en la escala en que entrena el modelo.
    - `mae_inr`: error absoluto medio en rupias, volviendo con exp(). Es el que se
      entiende sin saber de logaritmos.
    """
    pred = pipe.predict(X)
    return {
        'rmse_log': float(np.sqrt(mean_squared_error(y, pred))),
        'r2_log': float(r2_score(y, pred)),
        'mae_inr': float(mean_absolute_error(np.exp(y), np.exp(pred))),
    }
