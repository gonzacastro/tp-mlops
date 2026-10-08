import pytest

from car_price.data import cargar_crudo, filas_validas, split
from car_price.model import construir_pipeline


@pytest.fixture(scope='session')
def crudo():
    return cargar_crudo()


@pytest.fixture(scope='session')
def particion(crudo):
    """(X_train, X_test, y_train, y_test) con columnas crudas."""
    return split(filas_validas(crudo))


@pytest.fixture(scope='session')
def pipe(particion):
    """Pipeline entrenado con los mejores hiperparámetros (~2 s)."""
    X_train, _, y_train, _ = particion
    return construir_pipeline().fit(X_train, y_train)
