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
   donde corresponde (ver [Qué verificar](#qué-verificar) más abajo).

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

## Qué verificar

Después de una corrida de `train_car_price`, esto es lo que tiene que aparecer en cada lugar.

### Airflow (http://localhost:8080)

- En `train_car_price`, la corrida nueva con `subir_crudo`, `preparar` y `entrenar` en verde.
- **Log de `preparar`**: la traza de limpieza paso por paso (8.128 → 6.907 → 6.902 → 6.694 →
  6.691 filas) y el split `5,352 train / 1,339 test`.
- **Log de `entrenar`**: la versión registrada, si quedó como champion y las métricas en test.
  Las líneas que Airflow marca como error son mensajes `INFO` de MLflow que salen por stderr.
- **Pestaña XCom** de cada tarea: lo que se pasan entre sí. `subir_crudo` devuelve la ruta del
  crudo, `preparar` las de train y test, y `entrenar` el resumen (versión, métricas,
  mejores hiperparámetros y los ids de los runs de MLflow).

### MinIO (http://localhost:9001)

| Bucket | Qué tiene que haber |
|---|---|
| `data` | `raw/car_details_v3.csv` (el CSV original, ~1 MB) y `processed/train.csv` y `processed/test.csv`, con la fecha de la última corrida. Los dos de `processed` tienen las columnas crudas más `selling_price`. |
| `mlflow` | Una carpeta por modelo logueado en `<id del experimento>/models/<id del modelo>/artifacts/`, con `model.skops` (el pipeline serializado), `MLmodel` (firma y metadatos), `input_example.json` y los `requirements`. |

### MLflow (http://localhost:5001)

MLflow 3 abre en modo *GenAI*: hay que cambiar a **Model training** arriba a la izquierda.

- **Experimento `car_price`**: cada corrida del DAG deja dos runs.
  - `hpo-gb`, la búsqueda de hiperparámetros. Al desplegarlo aparecen `hpo_n_iter` runs
    hijos (`gb-00`, `gb-01`, ...), uno por combinación, con sus parámetros y el RMSE-log de
    validación cruzada. Seleccionándolos y apretando **Compare** se ve qué hiperparámetros
    funcionan mejor. En el padre quedan `n_iter` y la mejor combinación (`mejor_*`).
  - `modelo-final`, el mejor modelo reentrenado con todo el train: sus parámetros, las
    métricas en test (`rmse_log`, `r2_log`, `mae_inr`) y, en **Artifacts**, el modelo con su
    firma y un ejemplo de entrada.
- **Models → `car_price_gb`**: una versión por corrida, cada una con sus métricas de test como
  tags. Una sola tiene el alias **`champion`**, la de menor `rmse_log`. Es la que carga la API.
  Como el split es siempre el mismo, volver a correr el DAG sin cambiar los datos registra una
  versión con las mismas métricas y el champion no se mueve.

Con los datos actuales y `hpo_n_iter: 10`, el champion queda con RMSE-log ≈ 0.205, R² ≈ 0.922
y MAE ≈ 71.800 INR en test.

### API (http://localhost:8800/docs)

- `GET /health` → `modelo_cargado: true`.
- `GET /modelo` → la misma versión que tiene el alias `champion` en MLflow, con sus métricas.

## Usar la API

La documentación interactiva está en http://localhost:8800/docs: ahí se pueden probar todos
los endpoints con ejemplos ya cargados. Desde la terminal:

| Endpoint | Para qué |
|---|---|
| `POST /predict` | Precio estimado de un auto, en rupias |
| `GET /modelo` | Qué versión del modelo se está sirviendo y sus métricas en test |
| `POST /modelo/recargar` | Tomar el champion actual de MLflow sin reiniciar la API |
| `GET /health` | Estado del servicio (lo usa el healthcheck de Docker) |

### Predecir un precio

Se manda una publicación con los mismos campos que el CSV, con las unidades adentro del texto:

```bash
curl -X POST http://localhost:8800/predict \
  -H 'Content-Type: application/json' \
  -d '{"name": "Maruti Swift Dzire VDI", "year": 2014, "km_driven": 145500, "fuel": "Diesel", "seller_type": "Individual", "transmission": "Manual", "owner": "First Owner", "mileage": "23.4 kmpl", "engine": "1248 CC", "max_power": "74 bhp", "torque": "190Nm@ 2000rpm", "seats": 5}'
```

```json
{
  "precio_inr": 457722,
  "version_modelo": "2",
  "alias": "champion"
}
```

Es el primer auto del dataset, que se publicó a 450.000 INR. El entrenamiento es
determinístico (split y semillas fijas), así que con estos datos el precio da lo mismo en
cualquier máquina; lo que cambia es `version_modelo`, que depende de cuántas veces se corrió
el DAG.

Los campos de la ficha técnica (`mileage`, `engine`, `max_power`, `torque`, `seats`) son
opcionales; si faltan, el modelo los imputa:

```bash
curl -X POST http://localhost:8800/predict \
  -H 'Content-Type: application/json' \
  -d '{"name": "Hyundai i20 Sportz 1.2", "year": 2017, "km_driven": 35000, "fuel": "Petrol", "seller_type": "Dealer", "transmission": "Manual", "owner": "First Owner"}'
```

```json
{
  "precio_inr": 693403,
  "version_modelo": "2",
  "alias": "champion"
}
```

### Errores

**422: datos inválidos.** Falta un campo obligatorio o un valor está fuera de lo que vio el
modelo. La respuesta lista cada problema con el campo (`loc`) y el motivo (`msg`). Acá falta
`year` y `owner` es `Test Drive Car`, una categoría que se filtró en el entrenamiento:

```bash
curl -X POST http://localhost:8800/predict \
  -H 'Content-Type: application/json' \
  -d '{"name": "Maruti Swift Dzire VDI", "km_driven": 145500, "fuel": "Diesel", "seller_type": "Individual", "transmission": "Manual", "owner": "Test Drive Car"}'
```

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "year"],
      "msg": "Field required",
      "input": {"name": "Maruti Swift Dzire VDI", "km_driven": 145500, "fuel": "Diesel", "seller_type": "Individual", "transmission": "Manual", "owner": "Test Drive Car"}
    },
    {
      "type": "literal_error",
      "loc": ["body", "owner"],
      "msg": "Input should be 'First Owner', 'Second Owner', 'Third Owner' or 'Fourth & Above Owner'",
      "input": "Test Drive Car",
      "ctx": {"expected": "'First Owner', 'Second Owner', 'Third Owner' or 'Fourth & Above Owner'"}
    }
  ]
}
```

**503: no hay modelo.** Pasa si la API se levantó antes de la primera corrida del DAG. La API
arranca igual (`GET /health` da `"modelo_cargado": false`), pero no puede predecir:

```json
{
  "detail": "No hay modelo car_price_gb@champion cargado (RESOURCE_DOES_NOT_EXIST: Registered Model with name=car_price_gb not found). Corré el DAG train_car_price y después POST /modelo/recargar."
}
```

Se arregla corriendo el DAG y recargando:

```bash
curl -X POST http://localhost:8800/modelo/recargar
```

```json
{
  "version_anterior": null,
  "version_actual": "2",
  "cambio": true
}
```

Si se recarga antes de que el DAG termine, también da 503 y la API sigue como estaba:

```json
{
  "detail": "No se pudo recargar (RESOURCE_DOES_NOT_EXIST: Registered Model with name=car_price_gb not found); sigue sin modelo."
}
```

### Información del modelo

```bash
curl http://localhost:8800/modelo
```

```json
{
  "nombre": "car_price_gb",
  "version": "2",
  "alias": "champion",
  "run_id": "6b8bd02e87b14f2291c3b625a4668e6c",
  "metricas_test": {
    "rmse_log": 0.20492928605504668,
    "r2_log": 0.9224190865693929,
    "mae_inr": 71844.00766983234
  }
}
```

`rmse_log` y `r2_log` están en la escala en que entrena el modelo (logaritmo del precio);
`mae_inr` es el error absoluto medio en rupias. El `run_id` es el del run `modelo-final` en
MLflow.

Después de una corrida del DAG que no cambió el champion, `recargar` no vuelve a cargar nada:

```json
{
  "version_anterior": "2",
  "version_actual": "2",
  "cambio": false
}
```
