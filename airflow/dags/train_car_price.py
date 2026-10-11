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
    def preparar():
        """Limpia y separa train/test en s3://data/processed/ (D3)."""
        print("preparar: pendiente de implementar")

    @task
    def entrenar():
        """Búsqueda de hiperparámetros y registro en MLflow (D4)."""
        print("entrenar: pendiente de implementar")

    subir_crudo() >> preparar() >> entrenar()


train_car_price()
