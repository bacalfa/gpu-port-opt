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

GPU-accelerated technical indicator computation and feature engineering pipeline for portfolio optimization.

**Prerequisites:** Make sure you've completed the data ingestion module first:

```bash
# You should have these files from data ingestion
data/raw/asset_prices.csv
data/raw/macro_data.csv (optional)
data/raw/sentiment_data.csv (optional)
```

Basic usage:

```bash
cd /src/feature_engineering
python technical_indicators.py
```

Features computed:

#### Technical Indicators (30+)

##### **Returns & Log Returns**
- Daily returns (1-day)
- Multi-period returns (5d, 21d, 63d)
- Log returns (better for statistical modeling)

##### **Trend Indicators**
- **SMA (Simple Moving Average)**: 20, 50, 200-day windows
- **EMA (Exponential Moving Average)**: 12, 26, 50-day spans

##### **Momentum Indicators**
- **RSI (Relative Strength Index)**: 14-day window
  - Range: 0-100
  - >70 = Overbought, <30 = Oversold
- **MACD (Moving Average Convergence Divergence)**:
  - MACD Line (EMA12 - EMA26)
  - Signal Line (EMA9 of MACD)
  - Histogram (MACD - Signal)
- **Momentum**: 10, 20, 50-day price momentum

##### **Volatility Indicators**
- **Rolling Volatility**: 20, 60, 252-day windows (annualized)
- **Bollinger Bands**:
  - Upper/Lower bands (±2σ)
  - Bandwidth
  - %B (position within bands)
- **Average True Range (ATR)**: only if columns `High`, `Low` and `Adj Close` are present

##### **Correlation Features**
- Rolling correlation with benchmark (SPY)
- 60-day window

#### Engineered Features

##### **Lag Features**
- Lagged returns: 1, 5, 21-day lags
- Captures momentum and mean reversion

##### **Cross-Sectional Features**
- **Rank**: Percentile rank within universe
- **Z-score**: Standardized returns
- **Deviation from mean**: Relative performance

#### **Interaction Features**
- Pair-wise interactions/products (e.g., `sma_20` and `ema_12`)

#### External Features (if provided)

##### **Macroeconomic** (from data ingestion)
- Interest rates, inflation, employment
- GDP growth, consumer sentiment
- Credit spreads, money supply

##### **Market Sentiment**
- VIX, SKEW, volatility indices

Unit tests are provided in [tests/feature_engineering/test_feature_engineering.py](tests/feature_engineering/test_feature_engineering.py).

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

## Installation

1. Clone the repository

```shell
git git@github.com:bacalfa/gpu-port-opt.git
cd gpu-port-opt
```

2. Create virtual or a conda environment (example using `uv`)

```shell
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies (example using `uv`)

```shell
uv sync
```

4. Create file `.env` in top folder and containing the following text (replace API keys with yours)

```
# FRED API Key from https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```