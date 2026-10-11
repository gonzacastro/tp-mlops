import numpy as np
import pandas as pd

from car_price.data import filas_validas, separar_train_test


def test_filas_validas_conserva_columnas_crudas(crudo):
    df = filas_validas(crudo)
    assert len(df) == 6691
    assert list(df.columns) == list(crudo.columns)


def test_split_tamanios_y_sin_target_en_X(particion):
    X_train, X_test, y_train, y_test = particion
    assert (len(X_train), len(X_test)) == (5352, 1339)
    assert len(y_train) == len(X_train) and len(y_test) == len(X_test)
    assert 'selling_price' not in X_train.columns


def test_separar_train_test_mismas_filas_que_split(crudo, particion):
    X_train, X_test, y_train, y_test = particion
    train_df, test_df = separar_train_test(filas_validas(crudo))
    pd.testing.assert_frame_equal(train_df.drop(columns=['selling_price']), X_train)
    pd.testing.assert_frame_equal(test_df.drop(columns=['selling_price']), X_test)
    pd.testing.assert_series_equal(np.log(test_df['selling_price']), y_test)
