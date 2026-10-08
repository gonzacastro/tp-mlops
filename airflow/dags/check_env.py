"""DAG de humo: verifica que el worker de Airflow ve el código y los datos del proyecto.

Importa `car_price`, imprime las versiones de las librerías que usa el modelo y lee
las primeras filas del CSV montado en /opt/airflow/data. Si este DAG corre en verde,
el resto de los DAGs pueden usar `car_price` sin sorpresas.
"""
from datetime import datetime

from airflow.sdk import dag, task


@dag(
    dag_id="check_env",
    schedule=None,
    start_date=datetime(2026, 10, 1),
    catchup=False,
    tags=["infra", "smoke"],
    doc_md=__doc__,
)
def check_env():

    @task
    def verificar_entorno():
        """Importa car_price, muestra versiones y el head() del dataset."""
        import mlflow
        import pandas as pd
        import sklearn

        import car_price
        from car_price.config import DATA_PATH
        from car_price.data import cargar_crudo

        print(f"car_price desde: {car_price.__file__}")
        print(f"scikit-learn {sklearn.__version__} | pandas {pd.__version__} | mlflow {mlflow.__version__}")
        print(f"Dataset: {DATA_PATH}")

        df = cargar_crudo()
        print(f"Forma: {df.shape}")
        print(df.head().to_string())

        return {"filas": len(df), "columnas": df.shape[1]}

    verificar_entorno()


check_env()
