"""Gap-aware, past-only station features for explicit t -> t+1 forecasting."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import PREDICTORS, GASES, FEATURES, TARGET, TEST_START
from .preprocessing import eligible_stations


def station_daily(g: pd.DataFrame) -> pd.DataFrame:
    """Calendar daily grid. Missing days become NaN; NEVER shift across gaps."""
    if g.duplicated('date').any():
        raise ValueError('Repeated date in same station; audit duplicates first')
    return g.sort_values('date').set_index('date').asfreq('D')


def build_next_day_frame(clean: pd.DataFrame,
                         *, horizon: int = 1) -> pd.DataFrame:
    """Forecast PM2.5 at t+h using measurements known at origin date t.

    Critical operational assumption: same-day atmospheric observations and
    PM2.5 are available at forecast issue time (end of calendar day t).
    Lag shifts refer to calendar days, not consecutive observed rows.
    A full 7-observed-day rolling window is mandatory; short gaps yield NaN.
    No station IDs or coordinates enter FEATURES.
    """
    if horizon != 1:
        raise ValueError('Only the reviewed one-day-ahead protocol is implemented.')
    blocks = []
    for sid, g in clean.groupby('id', sort=True):
        day = station_daily(g)
        day['id'] = sid
        day['origin_date'] = day.index
        day['target_date'] = day.index + pd.Timedelta(days=horizon)
        day[TARGET] = day['pm25'].shift(-horizon)
        day['pm25_t0'] = day['pm25']
        day['pm25_lag1'] = day['pm25'].shift(1)
        for k in (1, 2, 3):
            day[f'AOD_lag{k}'] = day['AOD'].shift(k)
        # The previous seven days up to and including the forecast-origin day.
        # min_periods=7 ensures there are no unobserved days in the window.
        day['pm25_roll7_mean'] = day['pm25'].rolling(7, min_periods=7).mean()
        doy = day['target_date'].dt.dayofyear
        day['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
        day['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)
        for gas in GASES:
            day[f'{gas}_missing'] = day[gas].isna().astype('int8')
        # Origin must be an actual observed day with known current PM2.5;
        # the labelled next calendar day must also be observed.
        valid = day['pm25_t0'].notna() & day[TARGET].notna()
        blocks.append(day.loc[valid, ['id', 'origin_date', 'target_date',
                                       'sensor_name', TARGET] + FEATURES])
    if not blocks:
        raise ValueError('No stations available to build the forecasting set')
    data = pd.concat(blocks, ignore_index=True)
    data = data.sort_values(['target_date', 'id']).reset_index(drop=True)
    assert data['target_date'].eq(data['origin_date'] + pd.Timedelta(days=1)).all()
    assert data[TARGET].notna().all()
    return data


def make_development_and_test(clean: pd.DataFrame, *, test_start: str = TEST_START,
                              min_observations: int = 100) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    eligible = eligible_stations(clean, test_start=test_start,
                                 min_observations=min_observations)
    # Eligibility is decided on pretest rows only; still build features from
    # full chronology so test-origin predictors can use earlier observed days.
    all_rows = build_next_day_frame(clean.loc[clean['id'].isin(eligible)])
    cutoff = pd.Timestamp(test_start)
    dev = all_rows.loc[all_rows['target_date'] < cutoff].reset_index(drop=True)
    test = all_rows.loc[all_rows['target_date'] >= cutoff].reset_index(drop=True)
    if dev.empty or test.empty:
        raise ValueError('Development or future test is empty; review TEST_START.')
    assert dev['target_date'].max() < test['target_date'].min()
    train_station_ids = set(clean.loc[clean['date'] < cutoff, 'id'])
    future_station_ids = set(clean.loc[clean['date'] >= cutoff, 'id'])
    stats = {
        'forecast_horizon_days': 1,
        'forecast_issue_definition': 'end-of-day t; t observations assumed available',
        'future_test_start_target_date': str(cutoff.date()),
        'eligible_stations_based_on_development_only': int(len(eligible)),
        'development_station_count': int(dev['id'].nunique()),
        'future_test_station_count': int(test['id'].nunique()),
        'development_forecast_pairs': int(len(dev)),
        'future_test_forecast_pairs': int(len(test)),
        'future_only_station_count_not_in_model': int(len(future_station_ids - train_station_ids)),
        'excluded_station_count': int(clean['id'].nunique() - len(eligible)),
        'feature_count': len(FEATURES),
        'feature_names': list(FEATURES),
        'note': ('Consecutive observed day t and t+1 required: sample may be biased '
                 'towards better-sampled stations/days. Future test is a rolling '
                 'one-day-ahead evaluation with actual origin-day observations, '
                 'not recursive multi-day forecasting.'),
    }
    return dev, test, stats
