# PM2.5 Forecasting – Part 1

**Master 1 – Data Analysis | 2026–2027**

**Group members:** Nihel Beghachi g1, Aouane Chaima g1, Baha Chaimaa g2, Akliouat Dounia g1

## About the project

This is our mini-project about daily PM2.5 air pollution in the Mediterranean Basin. In Part 1, we explored the dataset, cleaned invalid values, studied missing data and correlations, and looked at seasonal and time-series patterns. We also prepared the data for future forecasting experiments.

## Dataset

We use the *Air Quality – Mediterranean Basin (2016–2026)* dataset from [Kaggle](https://www.kaggle.com/datasets/sarahameziane/air-quality-mediterranean-basin-20162026/).

- Original rows: **41,718**
- Rows after cleaning: **40,983**
- Invalid PM2.5 values removed: **735**

The original CSV files are **not uploaded to GitHub**. Download them from the dataset source and put them in `data/raw/` (especially `combined_gee_pm25.csv`). The cleaned files are created automatically in `data/processed/`.

## Project folders

- `data/` – raw and processed datasets (kept locally)
- `notebooks/` – EDA and time-series notebooks
- `src/` – Python functions for cleaning, analysis, features and splits
- `scripts/` – scripts to run the analyses
- `figures/` – generated graphs
- `results/` – EDA and time-series results in CSV format
- `folds/` – saved validation splits
- `reports/` – our Part 1 report
- `presentation/` – our slides

## How to run

We worked with Python in PyCharm. Install the libraries and run the project from the main folder:

```bash
python -m pip install -r requirements.txt
python -m scripts.run_eda
python -m scripts.run_timeseries
```

You can also open `notebooks/01_eda.ipynb` and `notebooks/02_timeseries.ipynb` in PyCharm.

## What we found

PM2.5 values are mostly low, but some days have much higher pollution. NO₂, CO and O₃ have missing values, and the station time series have many missing days. Pollution levels also vary by season and station. These findings will guide the forecasting stage.

**Current status:** Part 1 (EDA and preprocessing). Model training and performance evaluation are not included yet.