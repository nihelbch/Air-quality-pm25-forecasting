"""Reproducible descriptive EDA; calculations do NOT change the raw input."""
from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from .config import (MODEL_ID, MODEL_GROUP, PREDICTORS, GASES,
                     UNITS, SEED, FIGURES_DIR)

EDA_COLUMNS = [
    'model_id', 'model_group', 'variable', 'family', 'type', 'unit', 'n_total',
    'n_missing', 'pct_missing', 'n_unique', 'n_duplicates_key', 'min', 'q1',
    'median', 'q3', 'max', 'mean', 'std', 'skewness', 'kurtosis', 'iqr',
    'n_out_iqr', 'pct_out_iqr', 'n_out_z', 'pct_out_z', 'n_out_if', 'pct_out_if',
    'n_out_lof', 'pct_out_lof', 'n_sentinel_low', 'n_sentinel_high',
    'n_removed_total', 'max_abs_corr', 'corr_partner', 'vif', 'modalities',
    'target_assoc', 'target_pval', 'acf_lag1', 'acf_lag24', 'pacf_lag1',
    'n_lags_recommended', 'adf_p', 'is_leakage_suspect', 'notes',
]

FAMILIES = {'pm25':'Target','AOD':'AOD','NO2':'NO2','CO':'CO','O3':'O3',
            'temperature_celsius':'Meteorological','pressure_mb':'Meteorological',
            'wind_u':'Meteorological','wind_v':'Meteorological',
            'latitude':'Spatial','longitude':'Spatial','date':'Calendar',
            'id':'ID','sensor_name':'ID'}


def _univariate_outliers(s: pd.Series) -> dict:
    s = s.dropna()
    if len(s) < 3:
        return dict(n_out_iqr=np.nan, pct_out_iqr=np.nan,
                    n_out_z=np.nan, pct_out_z=np.nan)
    q1, q3 = s.quantile([.25,.75])
    iqr = q3-q1
    is_iqr = (s < q1-1.5*iqr) | (s > q3+1.5*iqr)
    z = np.abs((s-s.mean()) / s.std()) if s.std() > 0 else pd.Series(0.,index=s.index)
    return dict(n_out_iqr=int(is_iqr.sum()), pct_out_iqr=float(100*is_iqr.mean()),
                n_out_z=int((z>3).sum()), pct_out_z=float(100*(z>3).mean()))


def _multivariate_outliers(clean: pd.DataFrame) -> dict:
    """Exploratory flags only; no rows are dropped. Complete-case denominator documented."""
    sample = clean[['pm25']+PREDICTORS].dropna()
    if len(sample) < 50:
        return dict(n_out_if=np.nan,pct_out_if=np.nan,n_out_lof=np.nan,
                    pct_out_lof=np.nan,n_complete=len(sample))
    from sklearn.preprocessing import StandardScaler
    x = StandardScaler().fit_transform(sample)
    if_model = IsolationForest(n_estimators=100, contamination=.01, random_state=SEED, n_jobs=-1)
    is_if = if_model.fit_predict(x) == -1
    # LOF can be expensive on large datasets; deterministic <= 15,000-row subset.
    # Counts and percentages for LOF refer ONLY to this subset, unlike IF.
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(len(x),size=min(15000,len(x)),replace=False))
    subset = x[idx]
    lof = LocalOutlierFactor(n_neighbors=20, contamination=.01, n_jobs=-1)
    is_lof = lof.fit_predict(subset) == -1
    return dict(n_out_if=int(is_if.sum()),pct_out_if=100*float(is_if.mean()),
                n_out_lof=int(is_lof.sum()),pct_out_lof=100*float(is_lof.mean()),
                n_complete=int(len(sample)),n_lof_subset=int(len(subset)))


def audit_tables(raw: pd.DataFrame, clean: pd.DataFrame, report: dict) -> dict:
    ids_per_name = raw.groupby('sensor_name')['id'].nunique()
    try:
        timestamps = pd.to_datetime(raw['datetime_utc'], errors='coerce')
        same_timestamp_day = float(timestamps.dt.normalize().eq(raw['date']).mean())
    except (KeyError, ValueError, TypeError):
        same_timestamp_day = float('nan')
    coordinate_spread = raw.groupby('id')[['latitude','longitude']].nunique()
    audit = {
        **report,
        'fraction_datetime_utc_matching_date': same_timestamp_day,
        'station_ids_with_multiple_coordinate_pairs': int(coordinate_spread.gt(1).any(axis=1).sum()),
        'date_min': str(raw['date'].min().date()),
        'date_max': str(raw['date'].max().date()),
        'unique_station_names': int(raw['sensor_name'].nunique()),
        'stations_with_shared_names': int(ids_per_name.gt(1).sum()),
        'latitude_min': float(raw['latitude'].min()),
        'latitude_max': float(raw['latitude'].max()),
        'longitude_min': float(raw['longitude'].min()),
        'longitude_max': float(raw['longitude'].max()),
        'original_columns': list(raw.columns),
        'dataset_license': 'UNVERIFIED: verify with original Kaggle dataset',
        'raw_file_is_downloaded': True,
        'gas_units': 'UNVERIFIED: check Earth Engine source metadata',
        'station_gee_matching': 'NOT VERIFIABLE from downloaded combined CSV alone',
    }
    return audit


def exploratory_data_table(raw: pd.DataFrame, clean: pd.DataFrame,
                           cleaning: dict, temporal: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    outliers = _multivariate_outliers(clean)
    fields = ['pm25']+PREDICTORS+['latitude','longitude','date','id','sensor_name']
    numeric = ['pm25']+PREDICTORS+['latitude','longitude']
    pearson = clean[numeric].corr('pearson')
    keys_dups = int(clean.duplicated(['date','latitude','longitude','id']).sum())
    # VIF is descriptive, computed on complete rows. Do not use this on folds to
    # select model features: that requires fold-specific learning.
    x = clean[PREDICTORS].dropna()
    vifs = {}
    if len(x)>len(PREDICTORS)+2:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mat = sm.add_constant(x)
            for j,c in enumerate(mat.columns):
                if c!='const':
                    try: vifs[c] = float(variance_inflation_factor(mat.to_numpy(),j))
                    except (ValueError,np.linalg.LinAlgError): vifs[c]=np.nan
    rows=[]
    for c in fields:
        s=clean[c]
        kind = 'id' if c in ('id','sensor_name') else ('datetime' if c=='date' else 'numeric')
        row = dict(model_id=MODEL_ID,model_group=MODEL_GROUP,
                   variable='pm25_ugm3' if c=='pm25' else c,
                   family=FAMILIES[c],type=kind,unit=UNITS.get(c,''),
                   n_total=len(clean),n_missing=int(s.isna().sum()),
                   pct_missing=float(s.isna().mean()*100),n_unique=int(s.nunique()),
                   n_duplicates_key=keys_dups,n_removed_total=cleaning['rows_removed'],
                   is_leakage_suspect=False,notes='')
        if kind=='id': row['modalities']=int(s.nunique())
        if kind=='numeric':
            x=s.dropna()
            q1,q2,q3=x.quantile([.25,.50,.75])
            row.update(min=float(x.min()),q1=float(q1),median=float(q2),q3=float(q3),
                       max=float(x.max()),mean=float(x.mean()),std=float(x.std()),
                       skewness=float(x.skew()),kurtosis=float(x.kurtosis()),iqr=float(q3-q1))
            row.update(_univariate_outliers(s))
            other=pearson[c].drop(c).abs().dropna()
            if len(other):
                match=other.idxmax()
                row.update(corr_partner=match,max_abs_corr=float(other[match]))
            if c in vifs: row['vif']=vifs[c]
            if c!='pm25':
                pairs=clean[[c,'pm25']].dropna()
                if len(pairs)>2 and pairs[c].nunique()>1:
                    p_pear,p_val=stats.pearsonr(pairs[c],pairs['pm25'])
                    p_spear=stats.spearmanr(pairs[c],pairs['pm25']).statistic
                    row['target_assoc']=f'pearson_r={p_pear:.3f}; spearman={p_spear:.3f}'
                    row['target_pval']=float(p_val)
                    # CORRELATION ALONE IS NOT A RELIABLE LEAKAGE DETECTOR.
                    if abs(p_pear)>.95:
                        row['is_leakage_suspect']=True
                        row['notes']='High correlation: inspect feature provenance; NOT proof of leakage'
        if c=='pm25':
            row.update(n_sentinel_low=cleaning['n_invalid_le_zero'],
                       n_sentinel_high=cleaning['n_invalid_ge_500'],
                       n_out_if=outliers['n_out_if'],pct_out_if=outliers['pct_out_if'],
                       n_out_lof=outliers['n_out_lof'],pct_out_lof=outliers['pct_out_lof'],
                       acf_lag1=temporal['acf_lag1'].median(),
                       acf_lag24=temporal['acf_lag24'].median(),
                       pacf_lag1=temporal['pacf_lag1'].median(),
                       adf_p=temporal['adf_p'].median(),
                       n_lags_recommended=np.nan,
                       notes=(f'PM2.5 target; IF complete rows n={outliers["n_complete"]}; '
                              f'LOF deterministic subset n={outliers.get("n_lof_subset",0)}; '
                              'ACF lag24 = 24 DAYS; station median metrics; '
                              'plausible extremes retained; feature provenance needs audit'))
        if c in GASES:
            row['notes']=(row.get('notes','')+'; ' if row.get('notes') else '') + 'Original gas unit unverified; NO forward fill'
        if c in ('id','sensor_name','latitude','longitude'):
            row['notes']='Metadata only, excluded from forecasting features'
        rows.append(row)
    frame = pd.DataFrame(rows).reindex(columns=EDA_COLUMNS)
    integer_fields = [x for x in frame.columns if x.startswith('n_') or x == 'modalities']
    for c in integer_fields:
        frame[c]=pd.to_numeric(frame[c],errors='coerce').astype('Int64')
    return frame,outliers


def eda_figures(clean: pd.DataFrame, output: Path = FIGURES_DIR / 'eda') -> None:
    """Save high-value Part 1 charts. Figures are reproducible from the same cleaned CSV."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    output.mkdir(parents=True,exist_ok=True)
    use=['pm25']+PREDICTORS
    fig,axes=plt.subplots(3,3,figsize=(15,11))
    for ax,c in zip(axes.flat,use):
        sns.histplot(clean[c].dropna(),bins=40,ax=ax,color="steelblue")
        ax.set_title(c)
    fig.tight_layout();fig.savefig(output/'eda_histograms.png',dpi=145);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(15,11))
    for ax,c in zip(axes.flat,use):
        sns.boxplot(x=clean[c].dropna(),ax=ax)
        ax.set_title(c)
    fig.tight_layout();fig.savefig(output/'eda_boxplots.png',dpi=145);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(15,6))
    for ax,method in zip(axes,('pearson','spearman')):
        corr=clean[use].corr(method=method)
        sns.heatmap(corr,cmap='coolwarm',center=0,vmin=-1,vmax=1,annot=True,fmt='.2f',ax=ax)
        ax.set_title(method.title())
    fig.tight_layout();fig.savefig(output/'correlation_matrix.png',dpi=155);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4))
    missing=(clean[PREDICTORS].isna().mean()*100).sort_values(ascending=False)
    missing.plot.bar(ax=ax)
    ax.set(ylabel='Missing values (%)',title='Missing predictors after target cleaning')
    fig.tight_layout();fig.savefig(output/'missing_predictors.png',dpi=155);plt.close(fig)
    group=clean.groupby(clean['date'].dt.month)['pm25'].agg(['median','mean'])
    fig,ax=plt.subplots(figsize=(10,4))
    group.plot(ax=ax,marker='o');ax.set(xlabel='Month',ylabel='PM2.5 (µg/m³)',title='Seasonal median and mean')
    fig.tight_layout();fig.savefig(output/'seasonal_pm25.png',dpi=155);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    clean.groupby(clean['date'].dt.year).size().plot.bar(ax=axes[0]);axes[0].set(title='Observations by year',xlabel='Year')
    clean.groupby('id').size().hist(bins=30,ax=axes[1]);axes[1].set(title='Records per station',xlabel='Records')
    fig.tight_layout();fig.savefig(output/'coverage_by_year_station.png',dpi=155);plt.close(fig)

    # PM2.5/AOD relationship by season, descriptive only (not predictive causation).
    samples=clean[['date','AOD','pm25','wind_u','wind_v','temperature_celsius']].dropna().copy()
    if not samples.empty:
        month=samples['date'].dt.month
        seasons=np.select([month.isin([12,1,2]),month.isin([3,4,5]),
                           month.isin([6,7,8])],['winter','spring','summer'],default='autumn')
        samples['season']=seasons
        if len(samples)>20000:
            samples=samples.sample(n=20000,random_state=SEED)
        fig,axes=plt.subplots(2,2,figsize=(12,9),sharex=True,sharey=True)
        for ax,season in zip(axes.flat,['winter','spring','summer','autumn']):
            sub=samples.loc[samples.season.eq(season)]
            ax.scatter(sub['AOD'],sub['pm25'],alpha=.18,s=8)
            ax.set(title=season.title(),xlabel='AOD',ylabel='PM2.5 (µg/m³)')
        fig.suptitle('PM2.5 vs AOD across seasons (sampling descriptive, not causal)')
        fig.tight_layout();fig.savefig(output/'pm25_vs_aod_by_season.png',dpi=160);plt.close(fig)
        samples['wind_speed']=np.hypot(samples.wind_u,samples.wind_v)
        samples['wind_bin']=pd.qcut(samples.wind_speed,q=4,duplicates='drop')
        fig,ax=plt.subplots(figsize=(11,5))
        sns.boxplot(data=samples,x='wind_bin',y='pm25',ax=ax,color='steelblue',showfliers=False)
        ax.set(xlabel='Wind-speed quartile (m/s)',ylabel='PM2.5 (µg/m³)',
               title='PM2.5 by wind-speed group (outliers hidden for visualization only)')
        ax.tick_params(axis='x',labelrotation=12)
        fig.tight_layout();fig.savefig(output/'pm25_by_wind_quartile.png',dpi=160);plt.close(fig)
    # Variation across monitoring stations (top 15 by observation count).
    top=clean.groupby('id').size().nlargest(15).index
    sub=clean.loc[clean.id.isin(top),['id','pm25']]
    fig,ax=plt.subplots(figsize=(12,5))
    sns.boxplot(data=sub,x='id',y='pm25',ax=ax,color='steelblue',showfliers=False)
    ax.set(xlabel='Station ID (top 15 by coverage)',ylabel='PM2.5 (µg/m³)',
           title='Station variation (outliers hidden ONLY in figure)')
    fig.tight_layout();fig.savefig(output/'pm25_by_station.png',dpi=160);plt.close(fig)
