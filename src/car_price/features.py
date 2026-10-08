"""Preprocesador del modelo, portado del TP de AdM I.

Recibe la salida de `LimpiezaColumnas` (11 columnas) y devuelve 19 features numéricas.
Sin escalado: el Gradient Boosting es invariante a transformaciones monótonas.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder, OrdinalEncoder,
                                   TargetEncoder)

from car_price.config import N_FOLDS, NOMINALES, NUM_IMPUTE, OWNER_ORDER, RANDOM_STATE
from car_price.transformers import agrupar_asientos


def construir_preprocesador():
    """Devuelve el ColumnTransformer sin fittear.

    - `km_driven`: log1p (cola larga a la derecha).
    - Técnicas y `age`: imputación por mediana de los NaN aislados que quedan.
    - `owner`: ordinal, de primer dueño a cuarto o más.
    - `seats`: plazas raras agrupadas en 'other' y one-hot.
    - `fuel`, `seller_type`, `transmission`: one-hot sin la primera categoría.
    - `brand`: target encoding con cross-fitting (31 marcas, varias con pocos autos).
    """
    # En sklearn 1.9 la semilla del cross-fitting se pasa con el KFold (antes, random_state)
    cv_brand = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    return ColumnTransformer(
        transformers=[
            ('km_log', FunctionTransformer(np.log1p, feature_names_out='one-to-one'), ['km_driven']),
            ('num', SimpleImputer(strategy='median'), NUM_IMPUTE),
            ('owner', OrdinalEncoder(categories=[OWNER_ORDER]), ['owner']),
            ('seats', Pipeline([
                ('agrupar', FunctionTransformer(agrupar_asientos, feature_names_out='one-to-one')),
                ('ohe', OneHotEncoder(handle_unknown='ignore')),
            ]), ['seats']),
            ('nom', OneHotEncoder(drop='first', handle_unknown='ignore'), NOMINALES),
            ('brand', TargetEncoder(target_type='continuous', smooth=10.0, cv=cv_brand), ['brand']),
        ],
        remainder='drop',
        verbose_feature_names_out=False,
    )
