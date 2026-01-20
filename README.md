# GPU-Accelerated Multi-Asset Portfolio Optimization with Macro Regime Detection

🚧 *This repo is under construction*

## Overview
A practice repo covering portfolio optimization system combining GPU-accelerated machine learning (cuML), Bayesian regime detection, and stochastic programming for multi-asset allocation under market uncertainty.

## Key Features
- **GPU-Accelerated Feature Engineering**: 10x faster processing using RAPIDS cuDF/cuML
- **Market Regime Detection**: Hidden Markov Models + Bayesian changepoint detection
- **Multi-Regime Forecasting**: Separate ML models per market state
- **Stochastic Optimization**: Regime-aware mean-variance optimization with transaction costs
- **Comprehensive Backtesting**: Walk-forward validation with realistic assumptions

## Technologies (TBD)
- **GPU Computing**: RAPIDS cuDF, cuML, cuGraph
- **ML/AI**: Scikit-learn, PyTorch, HMMLearn
- **Optimization**: CVXPY, SciPy
- **Bayesian**: PyMC
- **Visualization**: Plotly, Streamlit

## Major Modules (TBD)

### 1. Data Ingestion

Complete data pipeline for downloading and preprocessing financial market data for portfolio optimization.

Basic usage:

```bash
cd /src/data
python download_data.py
```

This downloads:

- ~50 assets across equities, bonds, commodities, currencies
- ~20 macro indicators (GDP, inflation, unemployment, etc.)
- Sentiment data (VIX, market stress indicators)

Unit tests are provided in [tests/data/test_data_download.py](tests/data/test_data_download.py).

```bash
pytest tests/data/test_data_download.py -v
```


### 2. Feature Engineering

TODO

### 3. Regime Detection

TODO

### 4. Multi-Asset Forecasting

TODO

### 5. Stochastic Portfolio Optimization

TODO

### 6. Backtesting Engine

TODO

### 7. Visualizations

TODO