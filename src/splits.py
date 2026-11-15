"""Immutable, reusable CV definitions on DEVELOPMENT data ONLY."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit
from .config import FOLDS_DIR, SEED, TARGET


def _serializable(train: np.ndarray, val: np.ndarray, name: str, number: int, dev: pd.DataFrame) -> dict:
    if not len(train) or not len(val):
        raise AssertionError(f'Empty {name} fold')
    if np.intersect1d(train,val).size:
        raise AssertionError('Training and validation rows overlap')
    return {'fold':number, 'train_idx':train.tolist(), 'val_idx':val.tolist(),
            'n_train':len(train),'n_validation':len(val),
            'train_target_date_max':str(dev.iloc[train]['target_date'].max().date()),
            'validation_target_date_min':str(dev.iloc[val]['target_date'].min().date())}


def create_folds(dev: pd.DataFrame, *, output_dir: Path = FOLDS_DIR, seed: int = SEED) -> dict:
    """Each index refers to canonical dev order (target_date, id), NOT source CSV row IDs.

    The one-day temporal gap is ONE FULL CALENDAR DAY between the last
    training TARGET DATE and first validation TARGET DATE, not one pooled row.
    """
    output_dir.mkdir(parents=True,exist_ok=True)
    if dev.empty or dev[TARGET].isna().any():
        raise ValueError('Provide nonempty, labelled development data')
    if not dev.index.equals(pd.RangeIndex(len(dev))):
        raise ValueError('Pass reset_index(drop=True) development data')
    if not dev[['target_date','id']].equals(dev[['target_date','id']].sort_values(['target_date','id']).reset_index(drop=True)):
        raise ValueError('Development rows must be sorted by target_date,id')
    if dev['target_date'].isna().any():
        raise ValueError('Missing target dates')
    unique_dates = pd.date_range(dev['target_date'].min(), dev['target_date'].max(), freq='D')
    # Gap=1 excludes one *calendar day* between train/validation target dates.
    cv=TimeSeriesSplit(n_splits=5,gap=1)
    temporal=[]
    for k,(td,vd) in enumerate(cv.split(unique_dates)):
        train=np.flatnonzero(dev['target_date'].isin(unique_dates[td]).to_numpy())
        val=np.flatnonzero(dev['target_date'].isin(unique_dates[vd]).to_numpy())
        f=_serializable(train,val,'temporal',k,dev)
        dt=(dev.iloc[val]['target_date'].min()-dev.iloc[train]['target_date'].max()).days
        if dt<2: raise AssertionError(f'Temporal gap must exclude at least one calendar day, found {dt=}')
        f['calendar_days_between_last_train_and_first_validation']=dt
        temporal.append(f)
    if dev['id'].nunique()<5:
        raise ValueError('Need at least five distinct stations for GroupKFold(5)')
    spatial=[_serializable(tr,va,'spatial',k,dev) for k,(tr,va) in enumerate(
        GroupKFold(n_splits=5).split(dev,groups=dev['id']))]
    for fold in spatial:
        station_tr=set(dev.iloc[fold['train_idx']]['id'])
        station_val=set(dev.iloc[fold['val_idx']]['id'])
        if station_tr & station_val: raise AssertionError('Station leakage in spatial fold')
    random_folds=[_serializable(tr,va,'random_diagnostic',k,dev) for k,(tr,va) in enumerate(
        KFold(n_splits=5,shuffle=True,random_state=seed).split(dev))]
    # Hash station/target-date ordering to detect inadvertent row reordering.
    identity=dev[['target_date','id']].astype(str).to_csv(index=False).encode('utf-8')
    fingerprint=hashlib.sha256(identity).hexdigest()
    outputs={}
    for fname,kind,folds in (
        ('folds_temporal.json','temporal',temporal),
        ('folds_spatial.json','spatial',spatial),
        ('folds_standard.json','random_diagnostic',random_folds)):
        payload={'scheme':kind,'n_folds':5,'n_development_rows':len(dev),
                 'data_order_sha256':fingerprint,'index_base':0,
                 'train_and_validation_only':True,'seed':seed,
                 'temporal_gap_calendar_days':1 if kind=='temporal' else None,
                 'notes':('Diagnostic only, NOT for selecting best model' if kind=='random_diagnostic' else
                          'Spatial CV uses station IDs, but overlapping physical sites require separate checking' if kind=='spatial' else
                          'TimeSeriesSplit on complete calendar TARGET dates; some missing dates have no records'),
                 'folds':folds}
        path=output_dir/fname
        path.write_text(json.dumps(payload,indent=2),encoding='utf-8')
        outputs[kind]=path
    return outputs
