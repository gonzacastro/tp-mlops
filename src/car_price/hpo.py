"""Búsqueda de hiperparámetros del Gradient Boosting, logueada en MLflow.

Un run padre `hpo-gb` con la mejor combinación y, adentro, un run hijo por cada
combinación probada con su RMSE-log de validación cruzada. Así se comparan en la UI.
"""
import mlflow
from sklearn.model_selection import KFold, RandomizedSearchCV

from car_price.config import ESPACIO_GB, EXPERIMENTO, N_FOLDS, RANDOM_STATE
from car_price.model import construir_pipeline

PREFIJO = 'modelo__'


def sin_prefijo(params):
    """{'modelo__max_depth': 5} -> {'max_depth': 5}, el formato de `construir_pipeline`."""
    return {k.removeprefix(PREFIJO): v for k, v in params.items()}


def buscar_hiperparametros(X_train, y_train, n_iter):
    """RandomizedSearchCV sobre el Pipeline completo, con CV de `N_FOLDS` folds.

    No reentrena el mejor (`refit=False`): de eso se encarga `entrenar_y_registrar` con
    todo el train. Loguea en el run activo como hijo si lo hay; si no, en `EXPERIMENTO`.

    Devuelve un dict con `run_id` (del run padre), `mejores_params` (sin prefijo, listos
    para `construir_pipeline`) y la media y el desvío del RMSE-log de CV de esa combinación.
    """
    busqueda = RandomizedSearchCV(
        construir_pipeline(),
        ESPACIO_GB,
        n_iter=n_iter,
        scoring='neg_root_mean_squared_error',
        cv=KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        refit=False,
        n_jobs=2,                     # más procesos ahogan al worker de Docker
        random_state=RANDOM_STATE,
    )

    anidado = mlflow.active_run() is not None
    if not anidado:
        mlflow.set_experiment(EXPERIMENTO)

    with mlflow.start_run(run_name='hpo-gb', nested=anidado) as padre:
        busqueda.fit(X_train, y_train)
        res = busqueda.cv_results_

        # El scorer devuelve -RMSE: se da vuelta el signo para loguear el RMSE-log
        for i, params in enumerate(res['params']):
            with mlflow.start_run(run_name=f'gb-{i:02d}', nested=True):
                mlflow.log_params(sin_prefijo(params))
                mlflow.log_metrics({
                    'rmse_log_cv_media': -res['mean_test_score'][i],
                    'rmse_log_cv_desvio': res['std_test_score'][i],
                    'tiempo_fit_s': res['mean_fit_time'][i],
                })

        mejor = busqueda.best_index_
        resultado = {
            'run_id': padre.info.run_id,
            'mejores_params': sin_prefijo(busqueda.best_params_),
            'rmse_log_cv_media': float(-res['mean_test_score'][mejor]),
            'rmse_log_cv_desvio': float(res['std_test_score'][mejor]),
        }
        mlflow.log_params({'n_iter': n_iter, 'n_folds': N_FOLDS})
        mlflow.log_params({f'mejor_{k}': v for k, v in resultado['mejores_params'].items()})
        mlflow.log_metrics({'mejor_rmse_log_cv_media': resultado['rmse_log_cv_media'],
                            'mejor_rmse_log_cv_desvio': resultado['rmse_log_cv_desvio']})

    return resultado
