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

Advanced market regime detection using multiple methodologies: volatility analysis, clustering, Hidden Markov Models (HMM), and Bayesian changepoint detection.

*TODO: Use features engineered in module 2 as inputs to at least one regime detector*

Regime detectors:

#### Volatility-Based Detection (Fastest)

Simple but effective method based on rolling volatility quantiles.

How it works:

- Calculates rolling volatility across all assets
- Divides into regimes based on quantile thresholds
- Identifies: Low Vol, Normal, High Vol, Crisis

```
Vol_t = σ(r_{t-w:t})
Regime_t = Quantile_bin(Vol_t)
```

#### Clustering-Based Detection (GPU-Accelerated)

Uses K-Means clustering on market features to identify regimes.

Features used:

- Market returns (equal-weighted)
- Rolling volatility
- Average correlation
- Return dispersion (cross-sectional std)

How it works:

- Compute rolling market features
- Standardize features
- Optional: PCA for dimensionality reduction
- K-Means clustering (GPU-accelerated with cuML)
- Assign regime names based on characteristics

```
min Σ ||x_i - μ_{c(i)}||^2
subject to: c(i) ∈ {1,...,K}
```

#### Hidden Markov Model (HMM)

Statistical model that assumes market states are "hidden" and inferred from observed data.

How it works:

- Assumes market evolves through hidden states
- Observes returns and volatility
- Uses Expectation-Maximization (EM) to learn:
  - Transition probabilities between states
  - Emission distributions (what each state looks like)
- Viterbi algorithm finds most likely state sequence


```
P(s_t | s_{t-1}) = Transition matrix
P(x_t | s_t) = Emission distribution
```


#### Bayesian Changepoint Detection (Most Advanced)

Uses Bayesian inference to detect structural breaks and regime changes.

How it works:

- Places priors on changepoint locations
- Places priors on regime means and variances
- Uses MCMC (Markov Chain Monte Carlo) sampling
- Posterior distribution gives uncertainty estimates

```
τ ~ Uniform(T_min, T_max)
μ_k ~ Normal(0, σ_μ)
σ_k ~ HalfNormal(σ_σ)
```

Unit tests are provided in [tests/regime_detection/test_regime_detection.py](tests/regime_detection/test_regime_detection.py).

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