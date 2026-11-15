"""Re-run time-series EDA without regenerating model training results."""
from __future__ import annotations
import sys
from pathlib import Path
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.config import RESULTS_DIR, ensure_directories
from src.preprocessing import load_raw, clean_target
from src.timeseries import temporal_station_table, TEMPORAL_COLUMNS, lead_lag_correlations, temporal_figures

def main():
    ensure_directories()
    clean,_=clean_target(load_raw())
    diagnostics=temporal_station_table(clean)
    diagnostics.reindex(columns=TEMPORAL_COLUMNS).to_csv(RESULTS_DIR/'Exploratory_temporal.csv',index=False)
    lag=lead_lag_correlations(clean)
    lag.to_csv(RESULTS_DIR/'lead_lag_diagnostic.csv',index=False)
    temporal_figures(clean,diagnostics,lag)
    print('Time-series diagnostic outputs regenerated.')

if __name__=='__main__': main()
