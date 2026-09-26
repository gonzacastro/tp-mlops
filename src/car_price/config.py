from pathlib import Path

RAIZ_REPO = Path(__file__).resolve().parents[2]
DATA_PATH = RAIZ_REPO / 'data' / 'Car details v3.csv'

RANDOM_STATE = 42
TEST_SIZE = 0.2
ANIO_REFERENCIA = 2020
KM_MAX = 500_000

TECNICAS = ['mileage', 'engine', 'max_power', 'seats']
