# Changelog

Cambios entre entregas del TP. La entrega preliminar es el estado del repo al 27/09/2026
(commit `a66ebd1`).

## Entrega final

### Infraestructura
- Sincronizado con la versión actualizada del template de la cátedra (imagen de MinIO,
  `psycopg[binary]`, mlflow 3.16.1).
- Versiones fijadas en Airflow, FastAPI y `requirements.txt`: scikit-learn 1.9.1,
  pandas 3.0.6, numpy 2.5.3.
- `src/` y `data/` montados en Airflow y FastAPI, con `PYTHONPATH` apuntando a `src/`: el
  mismo paquete `car_price` se usa para entrenar y para servir.
- `MLFLOW_TRACKING_URI` definido en Airflow y FastAPI.
- DAGs de ejemplo de Airflow desactivados.
- DAG `check_env` para verificar que el worker ve el código, las versiones y el dataset.
- `scripts/smoke_mlflow.py` para verificar tracking y artefactos desde la máquina.

### Modelo (`src/car_price/`)
- El preprocesamiento y el Gradient Boosting del TP de AdM I pasaron del notebook a código.
- El modelo es un único Pipeline de scikit-learn que incluye la limpieza de columnas
  (`LimpiezaColumnas`): recibe las filas tal como vienen en el CSV.
- `filas_validas` devuelve las filas crudas que sobreviven a la limpieza; `separar_train_test`
  hace el mismo split que `split` sin separar el target.
- `TargetEncoder` adaptado a scikit-learn 1.9 (la semilla va en el `KFold` del parámetro `cv`).
- `python -m car_price.train` entrena en local y reproduce las métricas de AdM I en test
  (RMSE-log 0.2066, R² 0.9211, MAE 74.104 INR).
- Tests con pytest para la limpieza, el split y el pipeline (`tests/`, `requirements-dev.txt`).

### MLflow
- Búsqueda de hiperparámetros con `RandomizedSearchCV` y CV de 5 folds, logueada como un run
  `hpo-gb` con un run hijo por combinación.
- El mejor modelo se reentrena con todo el train, se evalúa en test y se loguea en un run
  `modelo-final` con firma y ejemplo de entrada (serializado con skops).
- Cada entrenamiento registra una versión de `car_price_gb`. El alias `champion` pasa a la
  versión nueva solo si baja el RMSE-log en test.

### Airflow
- DAG `train_car_price` con tres tareas: `subir_crudo` (CSV a `s3://data/raw/`), `preparar`
  (train y test a `s3://data/processed/`) y `entrenar` (búsqueda, evaluación y registro).
- Las tareas solo orquestan: llaman a funciones de `car_price` y se pasan rutas por XCom.
- Cantidad de combinaciones de la búsqueda configurable con la Variable `hpo_n_iter`
  (`airflow/secrets/variables.yaml`, por defecto 10).

### API (FastAPI)
- La API carga el champion de MLflow al arrancar. Si no hay, arranca igual y `/predict`
  responde 503 con qué hacer.
- `POST /predict`: recibe una publicación con los campos del CSV y devuelve el precio en
  rupias y la versión del modelo. Valida los campos con Pydantic (422 si algo no corresponde).
- `GET /health`, `GET /modelo` (versión, run y métricas en test) y `POST /modelo/recargar`
  (toma un champion nuevo sin reiniciar; si falla, sigue con el modelo anterior).
- Documentación OpenAPI acorde al modelo, con ejemplos en `/docs`.

### Documentación
- README con el diagrama de arquitectura, el orden de ejecución de punta a punta, qué
  verificar en Airflow, MinIO, MLflow y la API, y ejemplos de request y response.
- Este CHANGELOG.

## Entrega preliminar (27/09/2026)

- Entorno productivo en Docker a partir del template de la cátedra (Airflow, MLflow, MinIO,
  Postgres y FastAPI).
- Dataset de CarDekho en `data/` y la limpieza de AdM I en `src/car_price/data.py`.
- README con integrantes, servicios, cómo levantarlo y próximos pasos.
