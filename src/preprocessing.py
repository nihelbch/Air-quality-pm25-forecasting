"""Raw-data loading, immutable target cleaning, and fold-only imputation/scaling."""
from __future__ import annotations
import inspect
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler

from .config import DATA_FILE, TEST_START, MIN_OBS_PER_STATION

REQUIRED = ('date', 'id', 'pm25', 'sensor_name', 'latitude', 'longitude',
            'AOD', 'NO2', 'CO', 'O3', 'temperature_celsius', 'pressure_mb',
            'wind_u', 'wind_v')


def load_raw(path: Path = DATA_FILE) -> pd.DataFrame:
    """Load the downloaded CSV without overwriting it; fail on corrupt keys."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f'Raw data not found: {path}\nPut combined_gee_pm25.csv in data/raw/.')
    raw = pd.read_csv(path)
    absent = sorted(set(REQUIRED) - set(raw.columns))
    if absent:
        raise ValueError(f'Required columns missing from raw CSV: {absent}')
    raw['date'] = pd.to_datetime(raw['date'], errors='raise').dt.normalize()
    if raw['date'].isna().any() or raw['id'].isna().any():
        raise ValueError('Null station identifier or date: cannot construct station timelines.')
    return raw


def clean_target(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Mandatory cleaning: remove null, <=0 and >=500 PM2.5 ONLY.

    Keep genuine high-but-plausible measurements and all predictor NaNs.
    Duplicate station/date keys cannot be safely disambiguated; fail instead
    of silently averaging or dropping them.
    """
    y = pd.to_numeric(raw['pm25'], errors='coerce')
    mask = y.notna() & y.gt(0) & y.lt(500)
    clean = raw.loc[mask].copy()
    clean['pm25'] = y.loc[mask]
    duplicate_keys = int(clean.duplicated(['date', 'id']).sum())
    if duplicate_keys:
        raise ValueError(f'{duplicate_keys} duplicate (station ID, date) records. '
                         'Audit before constructing a daily timeline.')
    clean = clean.drop(columns=['datetime_utc'], errors='ignore')
    clean = clean.sort_values(['id', 'date']).reset_index(drop=True)
    report = {
        'rows_raw': int(len(raw)),
        'n_missing_pm25': int(y.isna().sum()),
        'n_invalid_le_zero': int(y.le(0).sum()),
        'n_invalid_ge_500': int(y.ge(500).sum()),
        'rows_removed': int((~mask).sum()),
        'rows_clean': int(len(clean)),
        'duplicate_exact_raw': int(raw.duplicated().sum()),
        'duplicate_station_date_raw': int(raw.duplicated(['date', 'id']).sum()),
        'duplicate_date_lat_lon_id_raw': int(raw.duplicated(['date', 'latitude', 'longitude', 'id']).sum()),
        'station_count_raw': int(raw['id'].nunique()),
        'station_count_clean': int(clean['id'].nunique()),
        'multiple_ids_per_sensor_name': int(raw.groupby('sensor_name')['id'].nunique().gt(1).sum()),
    }
    return clean, report


def eligible_stations(clean: pd.DataFrame, *, test_start: str = TEST_START,
                      min_observations: int = MIN_OBS_PER_STATION) -> pd.Index:
    """Select station eligibility on development dates, never using future test labels."""
    development = clean.loc[clean['date'] < pd.Timestamp(test_start)]
    counts = development.groupby('id', observed=True).size()
    return counts[counts >= min_observations].index


def build_numeric_preprocessor(*, scale: str | None = 'standard') -> Pipeline:
    """UNFITTED steps for use INSIDE a fresh estimator Pipeline for each CV fold.

    Gas-missingness flags are explicit input features from src.features. Do not
    fit this transformer to all development rows before cross-validation.
    """
    kwargs = {'strategy': 'median'}
    if 'keep_empty_features' in inspect.signature(SimpleImputer).parameters:
        kwargs['keep_empty_features'] = True
    steps = [('imputer', SimpleImputer(**kwargs))]
    if scale == 'standard':
        steps.append(('scaler', StandardScaler()))
    elif scale == 'robust':
        steps.append(('scaler', RobustScaler()))
    elif scale is not None:
        raise ValueError('scale must be None, standard or robust')
    return Pipeline(steps)
