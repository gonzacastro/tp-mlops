from pathlib import Path

RAIZ_REPO = Path(__file__).resolve().parents[2]
DATA_PATH = RAIZ_REPO / 'data' / 'Car details v3.csv'

RANDOM_STATE = 42
TEST_SIZE = 0.2
ANIO_REFERENCIA = 2020
KM_MAX = 500_000

TECNICAS = ['mileage', 'engine', 'max_power', 'seats']

# --- Preprocesamiento (igual que en el TP de AdM I) ---
OWNER_ORDER = ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner']
RARE_SEATS = [6, 9, 10, 14]                      # plazas raras que se agrupan en 'other'
NUM_IMPUTE = ['mileage', 'engine', 'max_power', 'age']
NOMINALES = ['fuel', 'seller_type', 'transmission']

# --- Gradient Boosting ---
N_FOLDS = 5

# Mejor combinación del RandomizedSearch de AdM I (RMSE-log CV 0.2115 ± 0.0040)
MEJORES_PARAMS_GB = {
    'n_estimators': 800,
    'learning_rate': 0.02,
    'max_depth': 5,
    'min_samples_leaf': 10,
    'subsample': 1.0,
    'max_features': 0.6,
}

# Espacio de búsqueda sobre el Pipeline completo (prefijo del paso 'modelo')
ESPACIO_GB = {
    'modelo__n_estimators': [200, 400, 600, 800, 1000],
    'modelo__learning_rate': [0.01, 0.02, 0.05, 0.1],
    'modelo__max_depth': [2, 3, 4, 5, 6],
    'modelo__min_samples_leaf': [1, 5, 10, 20],
    'modelo__subsample': [0.7, 0.85, 1.0],      # <1 = stochastic gradient boosting
    'modelo__max_features': [None, 'sqrt', 0.6],
}

# --- MLflow ---
EXPERIMENTO = 'car_price'
NOMBRE_MODELO = 'car_price_gb'
ALIAS_CHAMPION = 'champion'
