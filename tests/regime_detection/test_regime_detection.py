"""
Test suite for regime detection module

Run this to verify all regime detection methods work correctly.
"""

import numpy as np
import pandas as pd
import pytest

# Optional dependencies
HMM_AVAILABLE = False
PYMC_AVAILABLE = False
try:
    from hmmlearn import hmm

    HMM_AVAILABLE = True
except ImportError:
    pass
try:
    import pymc as pm

    PYMC_AVAILABLE = True
except ImportError:
    pass

from src.regime_detection.regime_detection import (
    BayesianRegimeDetector,
    ClusteringRegimeDetector,
    HMMRegimeDetector,
    RegimeAnalyzer,
    VolatilityRegimeDetector,
)


# Fixture for deterministic tests
@pytest.fixture(autouse=True)
def _set_seed():
    np.random.seed(42)


def create_sample_returns(n_days=500, n_assets=10):
    """Create sample return data with regime switches"""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=n_days, freq="D")

    # Create returns with regime switches
    returns = np.zeros((n_days, n_assets))

    # Regime 1: Low vol, positive returns (first 150 days)
    returns[:150, :] = np.random.randn(150, n_assets) * 0.005 + 0.001

    # Regime 2: High vol, negative returns (next 150 days)
    returns[150:300, :] = np.random.randn(150, n_assets) * 0.025 - 0.002

    # Regime 3: Low vol, positive returns (next 150 days)
    returns[300:450, :] = np.random.randn(150, n_assets) * 0.008 + 0.0015

    # Regime 4: High vol, neutral returns (last 50 days)
    returns[450:, :] = np.random.randn(50, n_assets) * 0.030

    df = pd.DataFrame(returns, index=dates, columns=[f"ASSET_{i}" for i in range(n_assets)])

    return df


def test_volatility_detector():
    """Test volatility-based regime detection"""
    returns = create_sample_returns()
    detector = VolatilityRegimeDetector(use_gpu=True)
    regimes = detector.detect_regimes(returns, window=21)

    required_cols = ["regime", "volatility", "regime_name"]
    for col in required_cols:
        assert col in regimes.columns, f"Missing column: {col}"

    prob_cols = [c for c in regimes.columns if c.startswith("prob_regime_")]
    if prob_cols:
        prob_sums = regimes[prob_cols].sum(axis=1)
        assert np.allclose(prob_sums, 1.0), "Probabilities don't sum to 1"


def test_clustering_detector():
    """Test clustering-based regime detection"""
    returns = create_sample_returns()
    detector = ClusteringRegimeDetector(n_regimes=4, use_gpu=True)
    regimes = detector.detect_regimes(returns, use_pca=True, n_components=3)

    assert regimes["regime"].nunique() == 4, (
        f"Expected 4 regimes, got {regimes['regime'].nunique()}"
    )
    prob_cols = [c for c in regimes.columns if c.startswith("prob_regime_")]
    assert len(prob_cols) == 4, f"Expected 4 probability columns, got {len(prob_cols)}"
    prob_sums = regimes[prob_cols].sum(axis=1)
    assert np.allclose(prob_sums, 1.0, atol=0.01), "Probabilities don't sum to 1"
    assert "volatility" in regimes.columns, "Volatility feature not in output"


def test_hmm_detector():
    """Test HMM-based regime detection"""
    if not HMM_AVAILABLE:
        pytest.skip("hmmlearn not available")
    returns = create_sample_returns()
    detector = HMMRegimeDetector(n_regimes=4, n_iter=50)
    regimes = detector.detect_regimes(returns)
    assert "regime" in regimes.columns, "Missing regime column"
    prob_cols = [c for c in regimes.columns if c.startswith("prob_regime_")]
    if prob_cols:
        prob_sums = regimes[prob_cols].sum(axis=1)
        assert np.allclose(prob_sums, 1.0, atol=0.01), "Probabilities don't sum to 1"


def test_bayesian_detector():
    """Test Bayesian changepoint detection"""
    if not PYMC_AVAILABLE:
        pytest.skip("PyMC not available")
    returns = create_sample_returns(n_assets=5)
    market_returns = returns.mean(axis=1)
    detector = BayesianRegimeDetector(n_regimes=3)
    regimes = detector.detect_changepoints(
        market_returns,
        n_samples=500,
        n_chains=2,
    )
    assert "regime" in regimes.columns


def test_regime_analyzer():
    """Test regime analysis functions"""
    returns = create_sample_returns()
    detector = ClusteringRegimeDetector(n_regimes=4, use_gpu=True)
    regimes = detector.detect_regimes(returns)
    analyzer = RegimeAnalyzer()
    characteristics = analyzer.analyze_regime_characteristics(regimes, returns)
    required_cols = ["regime_id", "regime_name", "n_days", "mean_return", "volatility"]
    for col in required_cols:
        assert col in characteristics.columns, f"Missing column: {col}"
    transitions = analyzer.calculate_regime_transitions(regimes)
    row_sums = transitions.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=0.01), "Transition matrix rows don't sum to 1"


def test_regime_persistence():
    """Test that regimes have reasonable persistence"""
    returns = create_sample_returns()
    detector = ClusteringRegimeDetector(n_regimes=4, use_gpu=True)
    regimes = detector.detect_regimes(returns)
    regime_series = regimes["regime"]
    regime_changes = regime_series != regime_series.shift(1)
    regime_runs = regime_changes.cumsum()
    run_lengths = regime_runs.value_counts()
    avg_run_length = run_lengths.mean()
    assert avg_run_length > 2, "Regimes change too frequently (possible overfitting)"


def test_gpu_vs_cpu():
    """Test GPU vs CPU consistency"""
    try:
        import cudf
    except ImportError:
        pytest.skip("GPU not available")
    returns = create_sample_returns(n_days=200, n_assets=10)
    detector_cpu = ClusteringRegimeDetector(n_regimes=4, use_gpu=False)
    regimes_cpu = detector_cpu.detect_regimes(returns)
    detector_gpu = ClusteringRegimeDetector(n_regimes=4, use_gpu=True)
    regimes_gpu = detector_gpu.detect_regimes(returns)
    cpu_dist = regimes_cpu["regime"].value_counts().sort_index()
    gpu_dist = regimes_gpu["regime"].value_counts().sort_index()
    # Results may not be identical due to randomness, but should be similar in distribution
    assert set(cpu_dist.index) == set(gpu_dist.index), "Regime indices differ between CPU and GPU"
