"""Script de humo: verifica que MLflow y MinIO se pueden usar desde la máquina.

Loguea un parámetro, una métrica y un archivo de texto como artefacto en el experimento
`smoke`. Si el run aparece en la UI de MLflow y el artefacto en el bucket `mlflow` de
MinIO, el tracking y el almacenamiento de artefactos funcionan.

Con el stack levantado (`docker compose --profile all up`):

    python scripts/smoke_mlflow.py

Las variables de entorno pisan los valores por defecto, que son los del `.env`.
"""
import os
import tempfile
from pathlib import Path

# Valores por defecto para correr desde la máquina contra el stack de Docker
os.environ.setdefault('MLFLOW_TRACKING_URI', 'http://localhost:5001')
os.environ.setdefault('MLFLOW_S3_ENDPOINT_URL', 'http://localhost:9000')
os.environ.setdefault('AWS_ACCESS_KEY_ID', 'minio')
os.environ.setdefault('AWS_SECRET_ACCESS_KEY', 'minio123')

import mlflow  # noqa: E402  (después de fijar el entorno)

EXPERIMENTO_SMOKE = 'smoke'


def main():
    mlflow.set_experiment(EXPERIMENTO_SMOKE)

    with mlflow.start_run(run_name='smoke-mlflow') as run:
        mlflow.log_param('origen', 'scripts/smoke_mlflow.py')
        mlflow.log_metric('valor_prueba', 1.0)

        with tempfile.TemporaryDirectory() as tmp:
            archivo = Path(tmp) / 'hola.txt'
            archivo.write_text('Si ves este archivo en MinIO, los artefactos funcionan.\n')
            mlflow.log_artifact(str(archivo))

    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Run {run.info.run_id} en el experimento '{EXPERIMENTO_SMOKE}'")
    print(f"Artefactos en: {run.info.artifact_uri}")


if __name__ == '__main__':
    main()
