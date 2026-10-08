from car_price.data import filas_validas


def test_filas_validas_conserva_columnas_crudas(crudo):
    df = filas_validas(crudo)
    assert len(df) == 6691
    assert list(df.columns) == list(crudo.columns)


def test_split_tamanios_y_sin_target_en_X(particion):
    X_train, X_test, y_train, y_test = particion
    assert (len(X_train), len(X_test)) == (5352, 1339)
    assert len(y_train) == len(X_train) and len(y_test) == len(X_test)
    assert 'selling_price' not in X_train.columns
