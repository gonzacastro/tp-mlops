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

| Servicio | URL |
|---|---|
| Airflow | http://localhost:8080 |
| MLflow  | http://localhost:5001 |
| MinIO   | http://localhost:9001 |
| API     | http://localhost:8800/docs |

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

## Próximos pasos

Por ahora está resuelta la parte de datos. Lo que falta:

1. Pasar a `src/car_price/` el preprocesamiento y el Gradient Boosting del notebook, así el
   modelo se entrena desde código.
2. Un DAG de Airflow que suba el CSV a MinIO, lo limpie y entrene el modelo.
3. Loguear en MLflow la búsqueda de hiperparámetros y registrar el mejor modelo como champion.
4. Que la API cargue el champion y tenga un endpoint `/predict` que devuelva el precio en
   rupias.
5. Completar la documentación con ejemplos de request y response.

La idea es serializar el pipeline completo, con la limpieza incluida, para que la API reciba los datos en el mismo formato que el CSV y no haya que repetir código.
