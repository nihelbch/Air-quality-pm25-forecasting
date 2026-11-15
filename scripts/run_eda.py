"""One-command Part 1 pipeline: python -m scripts.run_eda."""
from __future__ import annotations
import json
import sys
from pathlib import Path

# Supports both: python -m scripts.run_eda and PyCharm 'Run run_eda.py'.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.config import (DATA_FILE, CLEAN_FILE, DEV_FILE, TEST_FILE,
                        FIGURES_DIR, RESULTS_DIR, MODEL_GROUP, ensure_directories)
from src.preprocessing import load_raw, clean_target
from src.features import make_development_and_test
from src.timeseries import TEMPORAL_COLUMNS, temporal_station_table, lead_lag_correlations, temporal_figures
from src.eda import eda_figures, exploratory_data_table, audit_tables
from src.splits import create_folds


def main() -> None:
    ensure_directories()
    print(f'Loading original dataset (never overwritten): {DATA_FILE}',flush=True)
    raw=load_raw()
    clean, cleaning=clean_target(raw)
    assert len(raw)==len(clean)+cleaning['rows_removed']
    print(f'Cleaned PM2.5: {len(raw):,} -> {len(clean):,} ({cleaning["rows_removed"]:,} removed)',flush=True)
    clean.to_csv(CLEAN_FILE,index=False)
    print('Computing station diagnostics...',flush=True)
    temporal=temporal_station_table(clean)
    temporal.reindex(columns=TEMPORAL_COLUMNS).to_csv(RESULTS_DIR/'Exploratory_temporal.csv',index=False)
    print('Computing EDA CSV and plots...',flush=True)
    eda,outlier_info=exploratory_data_table(raw,clean,cleaning,temporal)
    eda.to_csv(RESULTS_DIR/'Exploratory_Data.csv',index=False)
    eda_figures(clean)
    leadlag=lead_lag_correlations(clean,max_lag=7)
    temporal_figures(clean,temporal,leadlag)
    leadlag.to_csv(RESULTS_DIR/'lead_lag_diagnostic.csv',index=False)
    print('Constructing NEXT-DAY forecast examples and folds...',flush=True)
    dev,test,forecast_report=make_development_and_test(clean)
    dev.to_csv(DEV_FILE,index=False)
    test.to_csv(TEST_FILE,index=False)
    saved=create_folds(dev)
    audit=audit_tables(raw,clean,cleaning)
    audit.update({'forecast':forecast_report,
                  'outlier_method_note': 'IF on all complete rows; LOF on deterministic <=15000 subset; EDA only',
                  'holdout_remains_untouched': True,
                  'cv_fold_files': {k:str(v.relative_to(DATA_FILE.parents[2])) for k,v in saved.items()},
                  'model_group': MODEL_GROUP,
                  'explicit_checks_still_needed': [
                      'Verify Kaggle dataset license and scientific units for NO2/CO/O3',
                      'Inspect original station-to-GEE matching, which is not reconstructible from combined CSV alone',
                      'Confirm assigned ML model group G18 with professor',
                      'Review multiple GEE IDs for the same monitoring station name',
                      'Operational availability of end-of-day PM2.5/AOD/gas predictors',
                      'Assess selection bias from restricting to observed consecutive-day forecast pairs',
                  ]})
    (RESULTS_DIR/'data_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\nOutputs:',flush=True)
    for x in [CLEAN_FILE,DEV_FILE,TEST_FILE,RESULTS_DIR/'Exploratory_Data.csv',
              RESULTS_DIR/'Exploratory_temporal.csv',RESULTS_DIR/'data_audit.json',*saved.values()]:
        print('  ',x,flush=True)
    print('Forecast rows: development',len(dev),'future test',len(test),flush=True)
    print('Finished. NO model has been trained; future test remains untouched.',flush=True)


if __name__=='__main__': main()
