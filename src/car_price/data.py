import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from car_price.config import (ANIO_REFERENCIA, DATA_PATH, KM_MAX, RANDOM_STATE,
                              TECNICAS, TEST_SIZE)


def cargar_crudo(path=DATA_PATH):
    """Lee el CSV crudo de CarDekho (8.128 filas x 13 columnas)."""
    return pd.read_csv(path)


def extraer_primer_numero(s):
    """'23.4 kmpl' -> 23.4 ; '1248 CC' -> 1248.0 ; NaN -> NaN"""
    if pd.isna(s):
        return np.nan
    m = re.search(r'(\d+\.?\d*)', str(s))
    return float(m.group(1)) if m else np.nan


def limpiar_columnas(df):
    """Parsea las columnas con unidades, deriva `brand` y `age` y descarta las que no se usan.

    Los ceros de `mileage`, `engine` y `max_power` pasan a NaN: un auto no rinde 0 kmpl ni
    tiene 0 CC, son faltantes codificados como cero.
    """
    df = df.copy()
    for col in ['mileage', 'engine', 'max_power']:
        df[col] = df[col].apply(extraer_primer_numero)
        df.loc[df[col] == 0, col] = np.nan

    df['brand'] = df['name'].str.split(n=1).str[0]    # primera palabra = fabricante
    df['age'] = ANIO_REFERENCIA - df['year']
    return df.drop(columns=['torque', 'name', 'year'])


def filtrar_filas(df):
    """Descarta duplicados, registros inválidos, faltantes estructurales y errores de carga.

    Devuelve el DataFrame filtrado y la traza con cuántas filas eliminó cada paso.
    """
    trazas = []

    def registrar(paso, n_antes, detalle):
        trazas.append({'paso': paso, 'filas_eliminadas': n_antes - len(df),
                       'filas_restantes': len(df), 'motivo': detalle})

    n = len(df); df = df.drop_duplicates().reset_index(drop=True)
    registrar('Duplicados exactos', n, 'Publicaciones repetidas del scraping (riesgo de leakage train/test)')

    n = len(df); df = df[df['owner'] != 'Test Drive Car'].reset_index(drop=True)
    registrar('owner = Test Drive Car', n, 'No encaja en la escala ordinal de owner')

    n = len(df); df = df[~df[TECNICAS].isna().all(axis=1)].reset_index(drop=True)
    registrar('Bloque sin ficha técnica', n, 'Los 4 campos técnicos faltan juntos; mecanismo MAR')

    n = len(df); df = df[df['km_driven'] <= KM_MAX].reset_index(drop=True)
    registrar(f'km_driven > {KM_MAX:,}', n, 'Errores de carga (valores físicamente imposibles)')

    return df, pd.DataFrame(trazas)


def limpiar(df_crudo):
    """Limpieza completa para entrenamiento: columnas y después filas."""
    return filtrar_filas(limpiar_columnas(df_crudo))


def split(df):
    """Separa target en escala log y predictores, y parte en train/test."""
    y = np.log(df['selling_price'])
    X = df.drop(columns=['selling_price'])
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


if __name__ == '__main__':
    crudo = cargar_crudo()
    df, traza = limpiar(crudo)

    print('TRAZA DE LIMPIEZA')
    print(traza.to_string(index=False))
    print(f'\nFilas: {len(crudo):,} → {len(df):,}  (retención {len(df)/len(crudo)*100:.1f}%)')
    na = df.isna().sum()
    print('NaN aislados que quedan:')
    print(na[na > 0].to_string() if na.sum() else '  (ninguno)')

    X_train, X_test, y_train, y_test = split(df)
    print(f'\nX_train: {X_train.shape}   y_train: {y_train.shape}')
    print(f'X_test : {X_test.shape}   y_test : {y_test.shape}')
    print(f'y (log-precio) — train: media={y_train.mean():.3f}  sd={y_train.std():.3f}')
    print(f'y (log-precio) — test : media={y_test.mean():.3f}  sd={y_test.std():.3f}')
