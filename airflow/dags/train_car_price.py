"""Entrenamiento del modelo de precios de autos usados (CarDekho).

Flujo: `subir_crudo` copia el CSV a MinIO, `preparar` limpia y separa train/test en
MinIO, y `entrenar` hace la búsqueda de hiperparámetros, evalúa en test y registra el
modelo en MLflow (alias `champion` solo si mejora al actual).

Toda la lógica del modelo vive en `car_price`; este DAG solo orquesta.
"""
from datetime import datetime

from airflow.sdk import dag, task


def _s3():
    """Cliente de MinIO. Credenciales y endpoint salen del entorno de Airflow."""
    import boto3
    return boto3.client("s3")


def _partir_uri(uri):
    """'s3://data/raw/x.csv' -> ('data', 'raw/x.csv')"""
    bucket, _, clave = uri.removeprefix("s3://").partition("/")
    return bucket, clave


def _leer_csv(uri):
    import pandas as pd
    bucket, clave = _partir_uri(uri)
    return pd.read_csv(_s3().get_object(Bucket=bucket, Key=clave)["Body"])


def _escribir_csv(df, bucket, clave):
    _s3().put_object(Bucket=bucket, Key=clave, Body=df.to_csv(index=False).encode())
    return f"s3://{bucket}/{clave}"


@dag(
    dag_id="train_car_price",
    schedule=None,
    start_date=datetime(2026, 10, 1),
    catchup=False,
    tags=["car_price", "entrenamiento"],
    doc_md=__doc__,
)
def train_car_price():

    @task
    def subir_crudo():
        """Sube el CSV crudo montado en /opt/airflow/data a s3://data/raw/.

        Devuelve la URI por XCom.
        """
        from car_price.config import BUCKET_DATOS, CLAVE_CRUDO, DATA_PATH

        _s3().upload_file(str(DATA_PATH), BUCKET_DATOS, CLAVE_CRUDO)
        uri = f"s3://{BUCKET_DATOS}/{CLAVE_CRUDO}"
        print(f"{DATA_PATH} -> {uri}")
        return uri

    @task
    def preparar(uri_crudo):
        """Limpia el crudo y guarda train y test en s3://data/processed/.

        Los CSV conservan las columnas crudas más `selling_price`: la limpieza de
        columnas va adentro del Pipeline. Devuelve las dos URIs por XCom.
        """
        from car_price.config import BUCKET_DATOS, CLAVE_TEST, CLAVE_TRAIN
        from car_price.data import filas_validas, limpiar, separar_train_test

        crudo = _leer_csv(uri_crudo)
        _, traza = limpiar(crudo)
        print("Traza de limpieza:\n" + traza.to_string(index=False))

        train_df, test_df = separar_train_test(filas_validas(crudo))
        print(f"{len(crudo):,} crudas -> {len(train_df):,} train / {len(test_df):,} test")

        return {
            "train": _escribir_csv(train_df, BUCKET_DATOS, CLAVE_TRAIN),
            "test": _escribir_csv(test_df, BUCKET_DATOS, CLAVE_TEST),
        }

    @task
    def entrenar(rutas):
        """Búsqueda de hiperparámetros, evaluación en test y registro en MLflow.

        `n_iter` sale de la Variable `hpo_n_iter` (airflow/secrets/variables.yaml).
        El modelo queda como versión nueva de `car_price_gb`, con alias `champion`
        solo si mejora al actual. Devuelve el resumen por XCom.
        """
        from airflow.sdk import Variable

        from car_price.entrenar import entrenar_y_registrar

        n_iter = int(Variable.get("hpo_n_iter", default=10))
        print(f"Búsqueda de hiperparámetros con n_iter={n_iter}")

        r = entrenar_y_registrar(_leer_csv(rutas["train"]), _leer_csv(rutas["test"]), n_iter)

        m = r["metricas_test"]
        print(f"Versión {r['version']} de car_price_gb: "
              f"{'nuevo champion' if r['champion'] else 'no mejora al champion'} "
              f"(champion anterior: {r['champion_anterior']})")
        print(f"Test: RMSE-log={m['rmse_log']:.4f}  R²-log={m['r2_log']:.4f}  "
              f"MAE={m['mae_inr']:,.0f} INR  |  CV: RMSE-log={r['rmse_log_cv_media']:.4f}")
        print(f"Mejores parámetros: {r['mejores_params']}")
        return r

    entrenar(preparar(subir_crudo()))


train_car_price()
