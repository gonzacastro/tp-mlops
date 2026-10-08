import pickle

import numpy as np

from car_price.metricas import evaluar


def test_predice_fila_cruda_sin_selling_price(pipe, crudo):
    fila = crudo.drop(columns=['selling_price']).head(1)
    pred = pipe.predict(fila)
    assert pred.shape == (1,)
    assert np.isfinite(pred).all()


def test_pickle_ida_y_vuelta(pipe, particion):
    _, X_test, _, _ = particion
    copia = pickle.loads(pickle.dumps(pipe))
    np.testing.assert_allclose(copia.predict(X_test.head(20)), pipe.predict(X_test.head(20)))


def test_metricas_en_test_como_adm1(pipe, particion):
    _, X_test, _, y_test = particion
    m = evaluar(pipe, X_test, y_test)
    assert m['r2_log'] > 0.91
    assert m['rmse_log'] < 0.22
