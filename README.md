# TP Final - Operaciones de Aprendizaje Automático I (MLOps)

**Integrantes:** Martín Birman, Gonzalo Castro

Puesta en producción del modelo de predicción de precios de autos usados que armamos en
Aprendizaje de Máquina I. Es un Gradient Boosting sobre el dataset de CarDekho que entrena
por un DAG de Airflow, se trackea en MLflow y se sirve con una API en FastAPI. Todo corre
en Docker.

La infraestructura base es la de la cátedra
([amq2-service-ml](https://github.com/facundolucianna/amq2-service-ml)); sobre eso agregamos
el modelo, el DAG y la API.

## Datos

El dataset es el de [CarDekho en Kaggle](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)
y está en `data/Car details v3.csv`. Son 8.128 publicaciones de autos usados en India, con el
precio en rupias.

En `src/car_price/data.py` está la misma limpieza que hicimos en AdM I, pasada del notebook a
código. Después de limpiar quedan 6.691 filas, que se separan 80/20 en train y test. El target
es el logaritmo del precio.

Para correrlo en local:

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m car_price.data
```

## Servicios

| Servicio | URL | Usuario / contraseña |
|---|---|---|
| Airflow | http://localhost:8080 | `airflow` / `airflow` |
| MLflow  | http://localhost:5001 | — |
| MinIO   | http://localhost:9001 | `minio` / `minio123` |
| API     | http://localhost:8800/docs | — |

MLflow está en el 5001 y no en el 5000 porque en macOS el 5000 lo ocupa AirPlay (se cambia
con `MLFLOW_PORT` en `.env`). Adentro de Docker sigue siendo `mlflow:5000`.

## Cómo levantarlo

Necesitás Docker. En Linux/Mac poné tu UID en `.env` (`AIRFLOW_UID`, lo sacás con `id -u`)
antes de arrancar, si no Airflow deja sus carpetas como root.

```bash
docker compose --profile all up
```

Para bajarlo:

```bash
docker compose --profile all down
```

## Usar MLflow desde la máquina

Adentro de Docker los servicios se ven por nombre (`mlflow:5000`, `s3:9000`), pero desde la
máquina hay que usar los puertos publicados. MLflow guarda los artefactos directo en MinIO,
así que además del tracking hacen falta el endpoint y las credenciales de S3:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5001
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minio
export AWS_SECRET_ACCESS_KEY=minio123
```

Para comprobar que funciona, con al menos `docker compose --profile mlflow up` corriendo:

```bash
python scripts/smoke_mlflow.py
```

El script usa esos mismos valores por defecto y loguea un parámetro, una métrica y un archivo
de texto en el experimento `smoke`. Si anda, el run aparece en la UI de MLflow y el archivo en
el bucket `mlflow` de MinIO.

## Orden de ejecución

De punta a punta, desde el repo recién clonado:

1. **Levantar el stack.** La primera vez construye las imágenes y tarda varios minutos.

   ```bash
   docker compose --profile all up -d
   ```

   Está listo cuando `docker compose ps` muestra todos los servicios `healthy`. La API
   arranca aunque todavía no haya modelo: `GET /health` responde `modelo_cargado: false`.

2. **Activar y correr el DAG.** En Airflow, entrar a `train_car_price`, activarlo con el
   interruptor de al lado del nombre (los DAGs arrancan pausados) y apretar **Trigger**.
   También se puede desde la terminal:

   ```bash
   docker compose exec airflow-scheduler airflow dags unpause train_car_price
   docker compose exec airflow-scheduler airflow dags trigger train_car_price
   ```

   Tarda alrededor de un minuto. Tiene tres tareas que corren en orden:

   | Tarea | Qué hace |
   |---|---|
   | `subir_crudo` | Copia `data/Car details v3.csv` a `s3://data/raw/car_details_v3.csv` |
   | `preparar` | Limpia y guarda `s3://data/processed/train.csv` (5.352 filas) y `test.csv` (1.339) |
   | `entrenar` | Búsqueda de hiperparámetros, evaluación en test y registro en MLflow |

   Cuántas combinaciones prueba la búsqueda sale de `hpo_n_iter` en
   `airflow/secrets/variables.yaml` (por defecto 10).

3. **Verificar** que las tres tareas terminaron en verde y que los datos y el modelo quedaron
   donde corresponde, en MinIO y en MLflow.

4. **Hacer que la API tome el modelo.** Si la API se levantó antes de que existiera un
   champion, hay que pedirle que lo cargue:

   ```bash
   curl -X POST http://localhost:8800/modelo/recargar
   ```

   Lo mismo después de cualquier corrida del DAG que cambie el champion.

5. **Probar la API.** En http://localhost:8800/docs, `POST /predict` → **Try it out** →
   **Execute** con el ejemplo cargado. Tiene que devolver un precio en rupias y la versión del
   modelo que lo calculó.

### Volver a entrenar

Para entrenar con datos nuevos se reemplaza `data/Car details v3.csv` (mismo nombre) y se
repiten los pasos 2 a 4. Cada corrida registra una versión nueva de `car_price_gb`; el alias
`champion` solo se mueve si la versión nueva tiene menor RMSE en test.

### Tests

```bash
pip install -r requirements-dev.txt
PYTHONPATH=src pytest
```
