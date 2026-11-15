"""Daily/station temporal diagnostics, descriptive seasonality and lead-lag EDA."""
from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import pacf, adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from .config import (PREDICTORS, GASES, MIN_RUN_DAYS, MIN_OBS_PER_STATION,
                     MODEL_ID, MODEL_GROUP, FIGURES_DIR)
from .features import station_daily

TEMPORAL_COLUMNS = [
    'model_id', 'model_group', 'variable', 'node_id', 'n_records', 't_min', 't_max',
    'sampling_interval_median_sec', 'sampling_interval_std_sec',
    'pct_missing_intervals', 'acf_lag1', 'acf_lag7', 'acf_lag24',
    'pacf_lag1', 'ljung_box_p', 'adf_p', 'kpss_p', 'trend_slope_per_day',
    'trend_pval', 'diurnal_amplitude', 'seasonal_amplitude',
    'target_drift_stat', 'target_drift_pval', 'n_days_above_25',
    'n_days_above_50', 'pct_days_above_25', 'pct_days_above_50',
    'regime_breaks', 'cv_strategy_recommended',
    'gap_between_train_test_sec', 'n_cycles_observed',
    'no2_pct_missing_by_month', 'co_pct_missing_by_month',
    'o3_pct_missing_by_month', 'notes'
]


def longest_observed_run(series: pd.Series) -> pd.Series:
    """Return the longest contiguous series with a nonmissing observation daily."""
    observed = series.notna()
    if not observed.any():
        return series.iloc[0:0]
    group_ids = observed.ne(observed.shift()).cumsum()
    return max((part for _, part in series.loc[observed].groupby(group_ids[observed])),
               key=len)


def _safe_stationarity(run: pd.Series, *, min_days: int = MIN_RUN_DAYS):
    result = dict(adf_p=np.nan, kpss_p=np.nan, ljung_box_p=np.nan,
                  pacf_lag1=np.nan, n_lags_recommended=np.nan)
    if len(run) < min_days or run.std() <= 0:
        return result
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            result['adf_p'] = float(adfuller(run.values, autolag='AIC')[1])
        except (ValueError, np.linalg.LinAlgError):
            pass
        try:
            result['kpss_p'] = float(kpss(run.values, regression='c', nlags='auto')[1])
        except (ValueError, np.linalg.LinAlgError):
            pass
        try:
            result['ljung_box_p'] = float(acorr_ljungbox(run.values, lags=[7])['lb_pvalue'].iloc[0])
        except (ValueError, np.linalg.LinAlgError):
            pass
        try:
            pp = pacf(run.values, nlags=min(14, len(run)//2-1), method='ywm')
            result['pacf_lag1'] = float(pp[1])
            n = 0
            for value in pp[1:]:
                if abs(value) > 1.96 / np.sqrt(len(run)):
                    n += 1
                else:
                    break
            result['n_lags_recommended'] = int(n)
        except (ValueError, np.linalg.LinAlgError):
            pass
    return result


def _monthly_missing_json(g: pd.DataFrame, column: str) -> str:
    groups = g['date'].dt.to_period('M').astype(str)
    d = g[column].isna().groupby(groups).mean().mul(100).round(1).to_dict()
    return json.dumps(d, ensure_ascii=False)


def _annual_cycles(g: pd.DataFrame) -> int:
    """Conservative proxy: complete calendar year with >=10 measurements each month.

    This does not establish a physically observed annual period; see notes.
    """
    counts = g.groupby([g['date'].dt.year, g['date'].dt.month]).size()
    per_year = counts.groupby(level=0).apply(lambda s: int(len(s) == 12 and s.min() >= 10))
    return int(per_year.sum())


def temporal_station_table(clean: pd.DataFrame) -> pd.DataFrame:
    """One row per station; never interpolate PM2.5 across missing dates."""
    out = []
    name_map = clean.drop_duplicates('id').set_index('id')['sensor_name']
    for sid, g in clean.groupby('id', sort=True):
        g = g.sort_values('date').copy()
        timeline = station_daily(g)['pm25']
        run = longest_observed_run(timeline)
        steps = g['date'].diff().dt.total_seconds().dropna()
        span = (g['date'].max()-g['date'].min()).days+1
        vals = g['pm25'].to_numpy()
        # Pairwise correlation at calendar-day offsets, not compressed-row offsets.
        acfs = {k: timeline.corr(timeline.shift(k), min_periods=30)
                for k in (1, 7, 24)}
        st = _safe_stationarity(run)
        days_from_start = (g['date']-g['date'].min()).dt.days.to_numpy()
        trend = stats.linregress(days_from_start, vals) if len(g) >= 3 else None
        month_means = g.groupby(g['date'].dt.month)['pm25'].mean()
        mid = len(g) // 2
        drift = stats.ks_2samp(vals[:mid], vals[mid:]) if len(g) >= 20 else None
        name = str(name_map[sid])
        notes = [name,
                 'eligible based on pretest count, see splits report',
                 'ACF lag24 means 24 DAYS (not diurnal)',
                 'monthly gas missingness among recorded station rows',
                 'n_cycles_observed: years with >=10 records in all 12 months']
        if len(run) < MIN_RUN_DAYS:
            notes.append(f'no contiguous-run statistical tests (longest run < {MIN_RUN_DAYS} days)')
        out.append({
            'model_id': MODEL_ID, 'model_group': MODEL_GROUP,
            'variable': 'pm25_ugm3', 'node_id': str(sid),
            'n_records': int(len(g)),
            't_min': g['date'].min().date().isoformat(),
            't_max': g['date'].max().date().isoformat(),
            'sampling_interval_median_sec': steps.median() if len(steps) else np.nan,
            'sampling_interval_std_sec': steps.std() if len(steps) > 1 else np.nan,
            'pct_missing_intervals': 100 * (span-len(g))/span,
            'acf_lag1': acfs[1], 'acf_lag7': acfs[7], 'acf_lag24': acfs[24],
            'pacf_lag1': st['pacf_lag1'],
            'ljung_box_p': st['ljung_box_p'],
            'adf_p': st['adf_p'], 'kpss_p': st['kpss_p'],
            'trend_slope_per_day': trend.slope if trend else np.nan,
            'trend_pval': trend.pvalue if trend else np.nan,
            'diurnal_amplitude': np.nan,  # Cannot calculate from DAILY observations.
            'seasonal_amplitude': (month_means.max()-month_means.min()) if len(month_means)>=6 else np.nan,
            'target_drift_stat': drift.statistic if drift else np.nan,
            'target_drift_pval': drift.pvalue if drift else np.nan,
            'n_days_above_25': int((vals > 25).sum()),
            'n_days_above_50': int((vals > 50).sum()),
            'pct_days_above_25': 100 * (vals > 25).mean(),
            'pct_days_above_50': 100 * (vals > 50).mean(),
            'regime_breaks': np.nan, # Not computed; optional method PELT not silently faked.
            'cv_strategy_recommended': 'temporal',
            'gap_between_train_test_sec': 86400.0,
            'n_cycles_observed': _annual_cycles(g),
            'no2_pct_missing_by_month': _monthly_missing_json(g, 'NO2'),
            'co_pct_missing_by_month': _monthly_missing_json(g, 'CO'),
            'o3_pct_missing_by_month': _monthly_missing_json(g, 'O3'),
            'notes': '; '.join(notes),
            '_longest_run_days': int(len(run)),
            '_n_lags_recommended': st['n_lags_recommended'],
        })
    return pd.DataFrame(out).sort_values('node_id').reset_index(drop=True)


def lead_lag_correlations(clean: pd.DataFrame, *, max_lag: int = 7) -> pd.DataFrame:
    """Descriptive lag correlations, including future leads for EDA only.

    Lag >0 is predictor(t-lag) vs PM2.5(t); lag <0 uses future predictors
    and is NEVER an acceptable forecasting feature.
    """
    vars_to_check = ['AOD', 'NO2', 'CO', 'O3', 'temperature_celsius', 'wind_u', 'wind_v']
    timelines = [station_daily(g)[['pm25'] + vars_to_check].assign(id=sid)
                 for sid, g in clean.groupby('id', sort=True)]
    all_days = pd.concat(timelines, ignore_index=True)
    blocks = []
    for variable in vars_to_check:
        for lag in range(-max_lag, max_lag + 1):
            shifted = all_days.groupby('id')[variable].shift(lag)
            pairs = pd.DataFrame({'pm25': all_days['pm25'], 'var': shifted}).dropna()
            rho = pairs['pm25'].corr(pairs['var'], method='spearman') if len(pairs)>=30 else np.nan
            blocks.append({'variable': variable, 'lag_days': lag,
                           'spearman_rho': rho, 'n_pairs': len(pairs),
                           'future_information_not_for_model': bool(lag < 0)})
    return pd.DataFrame(blocks)


def temporal_figures(clean: pd.DataFrame, diagnostics: pd.DataFrame,
                     lead_lag: pd.DataFrame, output: Path = FIGURES_DIR / 'timeseries') -> None:
    """Generate reusable, script-driven station/time charts."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    output.mkdir(parents=True, exist_ok=True)
    month = clean.groupby(clean['date'].dt.to_period('M'))['pm25'].agg(
        mean='mean', median='median', q1=lambda x:x.quantile(.25),
        q3=lambda x:x.quantile(.75), pct_above_25=lambda x:100*(x>25).mean())
    month.index = month.index.to_timestamp()
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(month.index, month['median'], label='Monthly median')
    axes[0].fill_between(month.index, month['q1'], month['q3'], alpha=.25, label='IQR')
    axes[0].set(ylabel='PM2.5 (µg/m³)', title='Monthly target distribution')
    axes[0].legend()
    axes[1].plot(month.index, month['pct_above_25'])
    axes[1].set(ylabel='Percent above 25 µg/m³', xlabel='Month',
                title='Frequency of high concentrations')
    fig.tight_layout();fig.savefig(output/'monthly_target_drift.png',dpi=160);plt.close(fig)

    monthly_predictors = clean.groupby(clean['date'].dt.month)[PREDICTORS + ['pm25']].mean()
    norm = monthly_predictors.sub(monthly_predictors.mean()).div(monthly_predictors.std().replace(0,np.nan))
    fig, ax = plt.subplots(figsize=(12, 5))
    norm.plot(marker='o', ax=ax)
    ax.set(xlabel='Calendar month', ylabel='Standardized monthly mean',
           title='Seasonal patterns (descriptive; pooled station coverage changes)')
    fig.tight_layout();fig.savefig(output/'monthly_predictor_patterns.png',dpi=160);plt.close(fig)

    p = lead_lag.pivot(index='variable', columns='lag_days', values='spearman_rho')
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.heatmap(p, cmap='coolwarm', center=0, annot=True, fmt='.2f', ax=ax)
    ax.set(title='Station-aware lead/lag Spearman correlation (exploratory ONLY)',
           xlabel='Predictor lag (days; negative = FUTURE input)', ylabel='Predictor')
    fig.tight_layout();fig.savefig(output/'lead_lag_heatmap.png',dpi=160);plt.close(fig)

    counts = diagnostics.set_index('node_id')['_longest_run_days']
    if counts.empty: return
    station = str(counts.idxmax())
    source = clean.loc[clean['id'].astype(str) == station]
    run = longest_observed_run(station_daily(source)['pm25'])
    if len(run) >= MIN_RUN_DAYS:
        fig, axes = plt.subplots(1,2,figsize=(12,4))
        ml = min(28,len(run)//2-1)
        plot_acf(run, lags=ml, ax=axes[0]);plot_pacf(run,lags=ml,method='ywm',ax=axes[1])
        fig.suptitle(f'ACF and PACF: station {station}, longest contiguous run ({len(run)} days)')
        fig.tight_layout();fig.savefig(output/'acf_pacf_longest_run.png',dpi=160);plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,4))
    sns.histplot(diagnostics['pct_missing_intervals'],bins=25,ax=ax,color='steelblue')
    ax.set(xlabel='Missing calendar days per station (%)',title='Station timeline incompleteness')
    fig.tight_layout();fig.savefig(output/'missing_calendar_intervals.png',dpi=160);plt.close(fig)
