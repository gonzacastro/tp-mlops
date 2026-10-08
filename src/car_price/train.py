"""Entrenamiento local de punta a punta, sin MLflow: `python -m car_price.train`.

Limpia, separa train/test, entrena con los mejores hiperparámetros de AdM I y muestra
las métricas en test. Sirve para chequear el modelo antes de pasar por Airflow.
"""
import time

from car_price.data import cargar_crudo, filas_validas, split
from car_price.metricas import evaluar
from car_price.model import construir_pipeline


def main():
    crudo = cargar_crudo()
    df = filas_validas(crudo)
    X_train, X_test, y_train, y_test = split(df)
    print(f'Filas: {len(crudo):,} crudas -> {len(df):,} válidas '
          f'({len(X_train):,} train / {len(X_test):,} test)')

    t0 = time.time()
    pipe = construir_pipeline().fit(X_train, y_train)
    print(f'Entrenado en {time.time() - t0:.1f}s\n')

    for nombre, X, y in [('train', X_train, y_train), ('test', X_test, y_test)]:
        m = evaluar(pipe, X, y)
        print(f'{nombre:5s}  RMSE-log={m["rmse_log"]:.4f}  R²-log={m["r2_log"]:.4f}  '
              f'MAE={m["mae_inr"]:,.0f} INR')


if __name__ == '__main__':
    main()
