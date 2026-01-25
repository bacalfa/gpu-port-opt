"""
Pytest test suite for the portfolio optimization module.

Run with: pytest tests/optimization/test_portfolio_optimization.py -v
"""

import numpy as np
import pandas as pd
import pytest

# Optional dependencies
PYOMO_AVAILABLE = False
try:
    import pyomo.environ as pyo

    PYOMO_AVAILABLE = True
except ImportError:
    pass

from src.optimization.portfolio_optimization import (
    GPUPortfolioAccelerator,
    PortfolioBacktester,
    PortfolioOptimizer,
    RegimeAwareOptimizer,
)


# Fixture for deterministic tests
@pytest.fixture(autouse=True)
def _set_seed():
    np.random.seed(42)


def create_sample_data(n_days=252, n_assets=10):
    """Create sample return data"""
    dates = pd.date_range(start="2020-01-01", periods=n_days, freq="D")
    returns = np.random.randn(n_days, n_assets) * 0.01
    factor = np.random.randn(n_days, 1) * 0.015
    returns += factor
    return pd.DataFrame(returns, index=dates, columns=[f"ASSET_{i}" for i in range(n_assets)])


def test_gpu_accelerator():
    """Test GPU accelerator"""
    returns = create_sample_data()
    gpu = GPUPortfolioAccelerator()
    cov_cpu = returns.cov()
    cov_gpu = gpu.compute_covariance(returns, use_gpu=True)
    assert cov_gpu.shape == cov_cpu.shape


def test_mean_variance():
    """Test mean-variance optimization"""
    if not PYOMO_AVAILABLE:
        pytest.skip("Pyomo not available")
    returns = create_sample_data()
    expected_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    optimizer = PortfolioOptimizer(solver="ipopt", use_gpu=True)
    result = optimizer.mean_variance_optimization(
        expected_returns=expected_returns, covariance_matrix=cov_matrix, risk_aversion=2.0
    )
    weights = result["weights"]
    assert abs(weights.sum() - 1.0) < 1e-4, "Weights don't sum to 1"
    assert (weights >= -1e-6).all(), "Some weights negative"
    calc_return = (weights * expected_returns).sum()
    assert abs(calc_return - result["expected_return"]) < 1e-4


def test_minimum_variance():
    """Test minimum variance optimization"""
    if not PYOMO_AVAILABLE:
        pytest.skip("Pyomo not available")
    returns = create_sample_data()
    cov_matrix = returns.cov() * 252
    optimizer = PortfolioOptimizer(solver="ipopt")
    result = optimizer.minimum_variance_portfolio(covariance_matrix=cov_matrix)
    n_random = 100
    random_vols = []
    for _ in range(n_random):
        random_weights = np.random.dirichlet(np.ones(len(cov_matrix)))
        random_vol = np.sqrt(random_weights @ cov_matrix.values @ random_weights)
        random_vols.append(random_vol)
    min_random_vol = min(random_vols)
    assert result["volatility"] <= min_random_vol + 1e-3


def test_maximum_sharpe():
    """Test maximum Sharpe ratio optimization"""
    if not PYOMO_AVAILABLE:
        pytest.skip("Pyomo not available")
    returns = create_sample_data()
    expected_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    optimizer = PortfolioOptimizer(solver="ipopt")
    result = optimizer.maximum_sharpe_ratio(
        expected_returns=expected_returns, covariance_matrix=cov_matrix, risk_free_rate=0.02
    )
    calc_sharpe = (result["expected_return"] - result["risk_free_rate"]) / result["volatility"]
    assert abs(calc_sharpe - result["sharpe_ratio"]) < 1e-3


def test_risk_parity():
    """Test risk parity portfolio"""
    if not PYOMO_AVAILABLE:
        pytest.skip("Pyomo not available")
    returns = create_sample_data()
    cov_matrix = returns.cov() * 252
    optimizer = PortfolioOptimizer(solver="ipopt")
    result = optimizer.risk_parity_portfolio(covariance_matrix=cov_matrix)
    risk_contrib = result["risk_contributions"]
    max_contrib = risk_contrib.max()
    min_contrib = risk_contrib.min()
    ratio = max_contrib / min_contrib if min_contrib > 0 else float("inf")
    assert ratio < 3.0, "Risk contributions vary too widely"


def test_regime_aware():
    """Test regime-aware optimization"""
    if not PYOMO_AVAILABLE:
        pytest.skip("Pyomo not available")
    returns_regime_0 = create_sample_data(n_days=126)
    returns_regime_0 += 0.001
    returns_regime_1 = create_sample_data(n_days=126)
    returns_regime_1 -= 0.001
    returns_by_regime = {0: returns_regime_0, 1: returns_regime_1}
    regime_probabilities = pd.Series({0: 0.6, 1: 0.4})
    current_weights = pd.Series(1.0 / returns_regime_0.shape[1], index=returns_regime_0.columns)
    optimizer = RegimeAwareOptimizer(solver="ipopt")
    result = optimizer.optimize_multi_regime(
        returns_by_regime=returns_by_regime,
        regime_probabilities=regime_probabilities,
        risk_aversion=2.0,
        current_weights=current_weights,
    )
    assert "expected_return" in result
    assert "volatility" in result


def test_backtester():
    """Test portfolio backtesting"""
    returns = create_sample_data(n_days=252)
    weights = pd.DataFrame(1.0 / len(returns.columns), index=returns.index, columns=returns.columns)
    backtester = PortfolioBacktester()
    results = backtester.backtest_strategy(
        weights=weights, returns=returns, transaction_cost=0.001, rebalance_frequency=21
    )
    assert len(results) > 0, "No backtest results"
    assert "portfolio_value" in results.columns
    metrics = backtester.calculate_performance_metrics(results)
    assert "annualized_return" in metrics
    assert "sharpe_ratio" in metrics


def test_weight_constraints():
    """Test weight constraints"""
    if not PYOMO_AVAILABLE:
        pytest.skip("Pyomo not available")
    returns = create_sample_data(n_assets=5)
    expected_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    optimizer = PortfolioOptimizer(solver="ipopt")
    result = optimizer.mean_variance_optimization(
        expected_returns=expected_returns,
        covariance_matrix=cov_matrix,
        risk_aversion=2.0,
        min_weight=0.1,
        max_weight=0.3,
    )
    weights = result["weights"]
    assert (weights >= 0.1 - 1e-4).all(), "Min weight constraint violated"
    assert (weights <= 0.3 + 1e-4).all(), "Max weight constraint violated"
