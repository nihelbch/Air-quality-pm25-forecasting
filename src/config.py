"""Paths, study protocol and column names for PM2.5 Part 1.

The three downloaded CSV files remain in data/raw/ and are NEVER edited.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results'
FIGURES_DIR = PROJECT_ROOT / 'figures'
FOLDS_DIR = PROJECT_ROOT / 'folds'

DATA_FILE = RAW_DIR / 'combined_gee_pm25.csv'
CLEAN_FILE = PROCESSED_DIR / 'pm25_cleaned.csv'
DEV_FILE = PROCESSED_DIR / 'pm25_forecast_dev.csv'
TEST_FILE = PROCESSED_DIR / 'pm25_forecast_future_test.csv'

SEED = 42
TEST_START = '2025-01-01'
MIN_OBS_PER_STATION = 100
MIN_RUN_DAYS = 20
FORECAST_HORIZON_DAYS = 1
MODEL_ID = 'eda_pm25'
MODEL_GROUP = 'G18'  # From the original notebook; VERIFY with your assigned group.

TARGET = 'pm25_target_next_day'
PREDICTORS = ['AOD', 'NO2', 'CO', 'O3', 'temperature_celsius',
              'pressure_mb', 'wind_u', 'wind_v']
GASES = ['NO2', 'CO', 'O3']
FEATURES = (PREDICTORS + ['pm25_t0', 'pm25_lag1', 'AOD_lag1',
                        'AOD_lag2', 'AOD_lag3', 'pm25_roll7_mean',
                        'doy_sin', 'doy_cos', 'NO2_missing', 'CO_missing',
                        'O3_missing'])

# Use unverified labels for units absent from the original CSV metadata.
UNITS = {
    'pm25': 'µg/m³', 'AOD': 'dimensionless',
    'NO2': 'unverified (possible mol/m²)',
    'CO': 'unverified (possible mol/m²)',
    'O3': 'unverified (possible mol/m²)',
    'temperature_celsius': '°C', 'pressure_mb': 'mb',
    'wind_u': 'm/s', 'wind_v': 'm/s',
    'latitude': 'degrees', 'longitude': 'degrees',
}


def ensure_directories() -> None:
    for path in (PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR,
                 FIGURES_DIR / 'eda', FIGURES_DIR / 'timeseries',
                 FOLDS_DIR):
        path.mkdir(parents=True, exist_ok=True)
