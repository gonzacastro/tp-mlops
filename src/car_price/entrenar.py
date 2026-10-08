"""Entrenamiento con registro en MLflow: el punto de entrada que usa el DAG.

También se puede correr desde la máquina, sin Airflow, con el stack levantado y las
variables de entorno del README:

    PYTHONPATH=src python -m car_price.entrenar --n-iter 10
"""
import argparse

import mlflow
import numpy as np
from sklearn.model_selection import train_test_split

from car_price.config import EXPERIMENTO, RANDOM_STATE, TEST_SIZE
from car_price.data import cargar_crudo, filas_validas
from car_price.hpo import buscar_hiperparametros
from car_price.metricas import evaluar
from car_price.model import construir_pipeline
from car_price.tracking import loguear_modelo, registrar_si_mejora


def separar_target(df):
    """Columnas crudas y target en escala log, como en `split`."""
    return df.drop(columns=['selling_price']), np.log(df['selling_price'])


def entrenar_y_registrar(train_df, test_df, n_iter):
    """Búsqueda de hiperparámetros, reentrenamiento, evaluación y registro.

    `train_df` y `test_df` tienen las columnas crudas del CSV más `selling_price`. Busca
    hiperparámetros con CV sobre el train, reentrena la mejor combinación con todo el
    train, la evalúa en test, la loguea en un run `modelo-final` y la registra en
    `car_price_gb` (con alias `champion` solo si mejora al actual).

    Devuelve un dict serializable (para XCom) con la versión registrada, si quedó como
    champion, las métricas de test y de CV, los mejores parámetros y los ids de los runs.
    """
    X_train, y_train = separar_target(train_df)
    X_test, y_test = separar_target(test_df)

    hpo = buscar_hiperparametros(X_train, y_train, n_iter)

    pipe = construir_pipeline(**hpo['mejores_params']).fit(X_train, y_train)
    metricas = evaluar(pipe, X_test, y_test)

    mlflow.set_experiment(EXPERIMENTO)
    with mlflow.start_run(run_name='modelo-final') as run:
        mlflow.set_tag('hpo_run_id', hpo['run_id'])
        mlflow.log_metrics({'rmse_log_cv_media': hpo['rmse_log_cv_media'],
                            'rmse_log_cv_desvio': hpo['rmse_log_cv_desvio']})
        info = loguear_modelo(pipe, X_train, metricas, hpo['mejores_params'])

    registro = registrar_si_mejora(info.model_uri, metricas)

    return {
        **registro,
        'metricas_test': metricas,
        'rmse_log_cv_media': hpo['rmse_log_cv_media'],
        'mejores_params': hpo['mejores_params'],
        'run_id': run.info.run_id,
        'hpo_run_id': hpo['run_id'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--n-iter', type=int, default=10,
                        help='combinaciones a probar en la búsqueda (por defecto 10)')
    args = parser.parse_args()

    # Mismas filas de train y test que `split` (misma semilla y proporción)
    train_df, test_df = train_test_split(filas_validas(cargar_crudo()),
                                         test_size=TEST_SIZE, random_state=RANDOM_STATE)
    r = entrenar_y_registrar(train_df, test_df, args.n_iter)

    m = r['metricas_test']
    print(f"\nVersión {r['version']} de car_price_gb "
          f"({'nuevo champion' if r['champion'] else 'no mejora al champion'}; "
          f"champion anterior: {r['champion_anterior']})")
    print(f"Test: RMSE-log={m['rmse_log']:.4f}  R²-log={m['r2_log']:.4f}  "
          f"MAE={m['mae_inr']:,.0f} INR  |  CV: RMSE-log={r['rmse_log_cv_media']:.4f}")
    print(f"Mejores parámetros: {r['mejores_params']}")


if __name__ == '__main__':
    main()
