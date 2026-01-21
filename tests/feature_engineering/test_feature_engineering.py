"""
Pytest test suite for the feature engineering module.

Run with: pytest tests/feature_engineering/test_feature_engineering.py -v
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.technical_indicators import (
    FeatureEngineer,
    GPUAccelerator,
    TechnicalIndicators,
)


@pytest.fixture(autouse=True)
def _set_seed():
    np.random.seed(42)


def create_sample_data(n_days: int = 252, n_assets: int = 5) -> pd.DataFrame:
    """Create deterministic random-walk prices for tests."""
    dates = pd.date_range(start="2020-01-01", periods=n_days, freq="D")
    rets = np.random.randn(n_days, n_assets) * 0.01
    prices = 100 * np.exp(rets.cumsum(axis=0))
    return pd.DataFrame(prices, index=dates, columns=[f"ASSET_{i}" for i in range(n_assets)])


def test_gpu_accelerator_roundtrip_preserves_data():
    gpu = GPUAccelerator()
    df = create_sample_data()

    df_gpu = gpu.to_gpu(df)
    df_cpu = gpu.to_cpu(df_gpu)

    assert isinstance(df_cpu, pd.DataFrame)
    pd.testing.assert_frame_equal(df, df_cpu)


def test_returns_matches_pandas_pct_change():
    prices = create_sample_data()
    tech = TechnicalIndicators(use_gpu=True)  # GPU flag should safely fall back when unavailable

    out = tech.compute_returns(prices, periods=[1])

    expected = prices.pct_change()
    expected.columns = [f"{c}_return_daily" for c in expected.columns]

    assert list(out.columns) == list(expected.columns)
    pd.testing.assert_frame_equal(out, expected)


def test_technical_indicators_shapes_and_ranges():
    prices = create_sample_data(n_days=260)  # enough for 200-day windows, but still fast
    tech = TechnicalIndicators(use_gpu=True)

    sma = tech.compute_sma(prices, windows=[20, 50])
    assert sma.shape == (prices.shape[0], prices.shape[1] * 2)

    ema = tech.compute_ema(prices, spans=[12, 26])
    assert ema.shape == (prices.shape[0], prices.shape[1] * 2)

    rsi = tech.compute_rsi(prices, window=14)
    assert rsi.shape == (prices.shape[0], prices.shape[1])
    assert rsi.min().min() >= 0
    assert rsi.max().max() <= 100

    macd = tech.compute_macd(prices)
    assert macd.shape == (prices.shape[0], prices.shape[1] * 3)

    bb = tech.compute_bollinger_bands(prices)
    assert bb.shape == (prices.shape[0], prices.shape[1] * 4)

    mom = tech.compute_momentum(prices, windows=[10, 20])
    assert mom.shape == (prices.shape[0], prices.shape[1] * 2)

    returns_1d = prices.pct_change()
    vol = tech.compute_volatility(returns_1d, windows=[20, 60])
    assert vol.shape == (prices.shape[0], prices.shape[1] * 2)


def test_feature_engineer_build_complete_features_smoke():
    prices = create_sample_data(n_days=260, n_assets=6)
    engineer = FeatureEngineer(use_gpu=True)

    features = engineer.build_complete_features(
        prices=prices,
        include_lags=True,
        include_cross_sectional=True,
        include_interaction=True,
    )

    assert isinstance(features, pd.DataFrame)
    assert features.index.equals(prices.index)
    assert features.shape[0] == prices.shape[0]
    assert features.shape[1] > prices.shape[1]  # should add features beyond raw prices
    assert not features.columns.duplicated().any()

    # basic sanity: we should have some of the common feature families present
    cols = features.columns.astype(str)
    assert any("return" in c for c in cols)
    assert any("_sma_" in c for c in cols)
    assert any("_ema_" in c for c in cols)
    assert any("_rsi_" in c for c in cols)
    assert any("_vol_" in c for c in cols)


@pytest.mark.slow
def test_gpu_vs_cpu_outputs_are_compatible_when_gpu_available():
    cudf = pytest.importorskip("cudf")

    prices = create_sample_data(n_days=260, n_assets=8)
    try:
        _ = cudf.from_pandas(prices.head(1))
    except Exception as e:
        pytest.skip(f"cuDF present but unusable in this environment: {e}")

    tech_cpu = TechnicalIndicators(use_gpu=False)
    tech_gpu = TechnicalIndicators(use_gpu=True)

    features_cpu = tech_cpu.compute_all_indicators(
        prices, compute_macd_flag=False, compute_bb_flag=False
    )
    features_gpu = tech_gpu.compute_all_indicators(
        prices, compute_macd_flag=False, compute_bb_flag=False
    )

    assert features_cpu.shape == features_gpu.shape
    assert list(features_cpu.columns) == list(features_gpu.columns)

    # allow small numeric differences across backends
    pd.testing.assert_frame_equal(
        features_cpu, features_gpu, check_dtype=False, atol=1e-8, rtol=1e-6
    )


def test_lag_features_create_expected_columns_and_values():
    prices = create_sample_data(n_days=300, n_assets=5)
    returns = prices.pct_change()
    returns.columns = [f"{col}_return" for col in prices.columns]

    engineer = FeatureEngineer(use_gpu=True)

    base_cols = list(returns.columns[:3])
    lags = [1, 5, 21]
    lag_features = engineer.create_lag_features(returns, columns=base_cols, lags=lags)

    assert lag_features.shape[1] == len(base_cols) * len(lags)
    for c in base_cols:
        for lag in lags:
            assert f"{c}_lag_{lag}" in lag_features.columns

    # verify a specific lag relationship: x[t-1] == lag_1[t]
    test_col = base_cols[0]
    lag_col = f"{test_col}_lag_1"
    assert np.isclose(returns[test_col].iloc[5], lag_features[lag_col].iloc[6], atol=1e-12)


def test_cross_sectional_features_rank_is_consistent_for_dominant_asset():
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")

    # Make ASSET_0 strictly best and ASSET_4 strictly worst on every row.
    returns_data = np.random.randn(100, 5) * 1e-6
    returns_data[:, 0] = 0.05
    returns_data[:, 4] = -0.05
    returns = pd.DataFrame(returns_data, index=dates, columns=[f"ASSET_{i}" for i in range(5)])

    engineer = FeatureEngineer(use_gpu=True)
    cross_features = engineer.create_cross_sectional_features(returns)

    assert cross_features.shape[1] == len(returns.columns) * 3  # rank, zscore, dev_mean
    assert "ASSET_0_rank" in cross_features.columns

    # With 5 assets, top percentile rank should be 1.0 each day for ASSET_0.
    assert cross_features["ASSET_0_rank"].mean() > 0.9


def test_save_load_parquet_roundtrip(tmp_path: Path):
    prices = create_sample_data(n_days=260, n_assets=4)
    engineer = FeatureEngineer(use_gpu=True)

    features = engineer.build_complete_features(
        prices=prices,
        include_lags=False,
        include_cross_sectional=False,
        include_interaction=False,
    )

    out_path = tmp_path / "test_features.parquet"
    engineer.save_features(features, out_path, format="parquet")

    loaded = pd.read_parquet(out_path)
    assert loaded.shape == features.shape
    assert list(loaded.columns) == list(features.columns)
    pd.testing.assert_index_equal(loaded.index, features.index)
