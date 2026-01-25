"""
Portfolio Optimization Module
=============================

GPU-accelerated portfolio optimization using Pyomo for modeling.
Implements multiple optimization strategies with regime-awareness.

Features:
- Mean-Variance Optimization (Markowitz)
- Regime-Aware Optimization
- Risk Parity
- Minimum Variance
- Maximum Sharpe Ratio
- Transaction Cost Modeling
- GPU-accelerated covariance/correlation computation

Author: Bruno Abreu Calfa
"""

import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# GPU acceleration with automatic fallback
try:
    import cudf
    import cupy as cp

    GPU_AVAILABLE = True
    print("✓ GPU acceleration enabled (cuDF/CuPy)")
except ImportError:
    GPU_AVAILABLE = False
    cudf = pd
    cp = np
    print("⚠ GPU not available, using CPU (pandas/numpy)")

# Pyomo for optimization modeling
try:
    import pyomo.environ as pyo
    from pyomo.opt import SolverFactory

    PYOMO_AVAILABLE = True
except ImportError:
    PYOMO_AVAILABLE = False
    print("⚠ Pyomo not available. Install with: pip install pyomo")

# Fallback to cvxpy if Pyomo not available
try:
    import cvxpy as cp_opt

    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class GPUPortfolioAccelerator:
    """Utility class for GPU/CPU operations in portfolio optimization"""

    @staticmethod
    def to_gpu(df):
        """Convert pandas DataFrame/Series to cuDF if GPU available"""
        if GPU_AVAILABLE:
            try:
                if isinstance(df, pd.DataFrame):
                    return cudf.from_pandas(df)
                elif isinstance(df, pd.Series):
                    return cudf.Series(df)
                return df
            except Exception as e:
                logger.warning(f"GPU conversion failed: {e}. Using CPU.")
                return df
        return df

    @staticmethod
    def to_cpu(df):
        """Convert cuDF DataFrame/Series to pandas"""
        if GPU_AVAILABLE and isinstance(df, (cudf.DataFrame, cudf.Series)):
            return df.to_pandas()
        return df

    @staticmethod
    def compute_covariance(returns, use_gpu: bool = True):
        """Compute covariance matrix (GPU or CPU)"""
        if use_gpu and GPU_AVAILABLE:
            try:
                returns_gpu = (
                    cudf.from_pandas(returns) if isinstance(returns, pd.DataFrame) else returns
                )
                # cuDF covariance
                cov = returns_gpu.cov().to_pandas()
                return cov
            except Exception as e:
                logger.warning(f"GPU covariance failed: {e}. Using CPU.")

        return returns.cov()

    @staticmethod
    def compute_correlation(returns, use_gpu: bool = True):
        """Compute correlation matrix (GPU or CPU)"""
        if use_gpu and GPU_AVAILABLE:
            try:
                returns_gpu = (
                    cudf.from_pandas(returns) if isinstance(returns, pd.DataFrame) else returns
                )
                corr = returns_gpu.corr().to_pandas()
                return corr
            except Exception as e:
                logger.warning(f"GPU correlation failed: {e}. Using CPU.")

        return returns.corr()


class PortfolioOptimizer:
    """
    Portfolio optimization using Pyomo

    Supports multiple optimization objectives and constraints
    """

    def __init__(self, solver: str = "ipopt", use_gpu: bool = True):
        """
        Initialize portfolio optimizer

        Args:
            solver: Solver to use ('ipopt', 'glpk', 'cplex', 'gurobi')
            use_gpu: Whether to use GPU for covariance computation
        """
        if not PYOMO_AVAILABLE:
            raise ImportError("Pyomo not available. Install with: pip install pyomo")

        self.solver_name = solver
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.gpu = GPUPortfolioAccelerator()

        # Check solver availability
        try:
            self.solver = SolverFactory(solver)
            if not self.solver.available():
                logger.warning(f"Solver {solver} not available. Trying ipopt...")
                self.solver = SolverFactory("ipopt")
                if not self.solver.available():
                    logger.error("No solver available. Install ipopt: conda install ipopt")
                    raise RuntimeError("No optimization solver available")
        except Exception as e:
            logger.error(f"Solver initialization failed: {e}")
            raise

        logger.info(f"PortfolioOptimizer initialized (solver={solver}, GPU={self.use_gpu})")

    def mean_variance_optimization(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_aversion: float = 1.0,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        target_return: float | None = None,
        transaction_cost: float = 0.0,
        current_weights: pd.Series = None,
    ) -> dict:
        """
        Mean-Variance Optimization (Markowitz)

        Maximize: E[R] - λ * Var[R] - TC * |Δw|

        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix
            risk_aversion: Risk aversion parameter (λ)
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset
            target_return: Target return (if specified, ignores risk_aversion)
            transaction_cost: Transaction cost per unit weight change
            current_weights: Current portfolio weights for transaction cost calculation

        Returns:
            Dictionary with optimal weights and statistics
        """
        logger.info("Solving Mean-Variance Optimization...")

        n_assets = len(expected_returns)
        asset_names = expected_returns.index.tolist()

        # Create Pyomo model
        model = pyo.ConcreteModel(name="Mean_Variance_Portfolio")

        # Sets
        model.ASSETS = pyo.Set(initialize=range(n_assets))

        # Variables: portfolio weights
        model.w = pyo.Var(
            model.ASSETS, domain=pyo.NonNegativeReals, bounds=(min_weight, max_weight)
        )

        # Objective: Maximize expected return - risk_aversion * variance
        def objective_rule(m):
            # Expected return
            portfolio_return = sum(model.w[i] * expected_returns.iloc[i] for i in model.ASSETS)

            # Portfolio variance
            portfolio_variance = sum(
                sum(model.w[i] * model.w[j] * covariance_matrix.iloc[i, j] for j in model.ASSETS)
                for i in model.ASSETS
            )

            # Optional: regime-aware optimization
            portfolio_tc = 0.0
            if transaction_cost > 0.0:
                # L1-norm formulation
                model.tc = pyo.Var(model.ASSETS, domain=pyo.NonNegativeReals)
                portfolio_tc = transaction_cost * sum(model.tc[i] for i in model.ASSETS)

            return portfolio_return - risk_aversion * portfolio_variance - portfolio_tc

        model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

        # Constraint: weights sum to 1
        def weight_sum_rule(m):
            return sum(model.w[i] for i in model.ASSETS) == 1.0

        model.weight_sum = pyo.Constraint(rule=weight_sum_rule)

        # Optional: target return constraint
        if target_return is not None:

            def target_return_rule(m):
                return (
                    sum(model.w[i] * expected_returns.iloc[i] for i in model.ASSETS)
                    >= target_return
                )

            model.target_return = pyo.Constraint(rule=target_return_rule)

        # Optional: transaction cost constraints for L1-norm formulation
        if transaction_cost > 0.0:

            def tc_pos_rule(m, i):
                return model.tc[i] >= model.w[i] - current_weights.iloc[i]

            def tc_neg_rule(m, i):
                return model.tc[i] >= current_weights.iloc[i] - model.w[i]

            model.tc_pos_constraints = pyo.Constraint(model.ASSETS, rule=tc_pos_rule)
            model.tc_neg_constraints = pyo.Constraint(model.ASSETS, rule=tc_neg_rule)

        # Solve
        results = self.solver.solve(model, tee=False)

        # Check solver status
        if results.solver.status != pyo.SolverStatus.ok:
            logger.error(f"Solver failed with status: {results.solver.status}")
            raise RuntimeError("Optimization failed")

        # Extract solution
        weights = pd.Series([pyo.value(model.w[i]) for i in model.ASSETS], index=asset_names)

        # Calculate portfolio statistics
        portfolio_return = (weights * expected_returns).sum()
        portfolio_variance = weights.values @ covariance_matrix.values @ weights.values
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0

        logger.info(
            f"✓ Optimization completed - Return: {portfolio_return:.4f}, Vol: {portfolio_volatility:.4f}"
        )

        return {
            "weights": weights,
            "expected_return": portfolio_return,
            "volatility": portfolio_volatility,
            "variance": portfolio_variance,
            "sharpe_ratio": sharpe_ratio,
            "objective_value": pyo.value(model.objective),
        }

    def minimum_variance_portfolio(
        self, covariance_matrix: pd.DataFrame, min_weight: float = 0.0, max_weight: float = 1.0
    ) -> dict:
        """
        Minimum Variance Portfolio

        Minimize: Portfolio Variance

        Args:
            covariance_matrix: Covariance matrix
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset

        Returns:
            Dictionary with optimal weights and statistics
        """
        logger.info("Solving Minimum Variance Portfolio...")

        n_assets = len(covariance_matrix)
        asset_names = covariance_matrix.index.tolist()

        # Create Pyomo model
        model = pyo.ConcreteModel(name="Minimum_Variance_Portfolio")

        # Sets
        model.ASSETS = pyo.Set(initialize=range(n_assets))

        # Variables
        model.w = pyo.Var(
            model.ASSETS, domain=pyo.NonNegativeReals, bounds=(min_weight, max_weight)
        )

        # Objective: Minimize variance
        def objective_rule(m):
            return sum(
                sum(model.w[i] * model.w[j] * covariance_matrix.iloc[i, j] for j in model.ASSETS)
                for i in model.ASSETS
            )

        model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        # Constraint: weights sum to 1
        def weight_sum_rule(m):
            return sum(model.w[i] for i in model.ASSETS) == 1.0

        model.weight_sum = pyo.Constraint(rule=weight_sum_rule)

        # Solve
        results = self.solver.solve(model, tee=False)

        if results.solver.status != pyo.SolverStatus.ok:
            logger.error(f"Solver failed with status: {results.solver.status}")
            raise RuntimeError("Optimization failed")

        # Extract solution
        weights = pd.Series([pyo.value(model.w[i]) for i in model.ASSETS], index=asset_names)

        # Calculate statistics
        portfolio_variance = weights.values @ covariance_matrix.values @ weights.values
        portfolio_volatility = np.sqrt(portfolio_variance)

        logger.info(f"✓ Optimization completed - Min Vol: {portfolio_volatility:.4f}")

        return {
            "weights": weights,
            "volatility": portfolio_volatility,
            "variance": portfolio_variance,
            "objective_value": pyo.value(model.objective),
        }

    def maximum_sharpe_ratio(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_free_rate: float = 0.0,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
    ) -> dict:
        """
        Maximum Sharpe Ratio Portfolio

        Maximize: (E[R] - rf) / σ[R]

        Note: This is solved as a quadratic program by reformulation

        Args:
            expected_returns: Expected returns
            covariance_matrix: Covariance matrix
            risk_free_rate: Risk-free rate
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset

        Returns:
            Dictionary with optimal weights and statistics
        """
        logger.info("Solving Maximum Sharpe Ratio Portfolio...")

        # Sharpe ratio maximization is non-convex
        # We use the standard reformulation trick:
        # Instead of max (μ'w - rf) / sqrt(w'Σw)
        # We solve: min w'Σw s.t. μ'w - rf = 1
        # Then normalize weights to sum to 1

        n_assets = len(expected_returns)
        asset_names = expected_returns.index.tolist()

        # Excess returns
        excess_returns = expected_returns - risk_free_rate

        # Create Pyomo model
        model = pyo.ConcreteModel(name="Maximum_Sharpe_Portfolio")

        # Sets
        model.ASSETS = pyo.Set(initialize=range(n_assets))

        # Variables (these are scaled weights, will normalize later)
        model.y = pyo.Var(model.ASSETS, domain=pyo.NonNegativeReals)

        # Objective: Minimize variance (of scaled portfolio)
        def objective_rule(m):
            return sum(
                sum(model.y[i] * model.y[j] * covariance_matrix.iloc[i, j] for j in model.ASSETS)
                for i in model.ASSETS
            )

        model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        # Constraint: excess return = 1 (scaling constraint)
        def excess_return_rule(m):
            return sum(model.y[i] * excess_returns.iloc[i] for i in model.ASSETS) == 1.0

        model.excess_return = pyo.Constraint(rule=excess_return_rule)

        # Solve
        results = self.solver.solve(model, tee=False)

        if results.solver.status != pyo.SolverStatus.ok:
            logger.error(f"Solver failed with status: {results.solver.status}")
            raise RuntimeError("Optimization failed")

        # Extract and normalize weights
        y_values = np.array([pyo.value(model.y[i]) for i in model.ASSETS])
        weights = y_values / y_values.sum()
        weights = pd.Series(weights, index=asset_names)

        # Apply weight bounds by projection (if violated)
        weights = weights.clip(lower=min_weight, upper=max_weight)
        weights = weights / weights.sum()  # Renormalize

        # Calculate statistics
        portfolio_return = (weights * expected_returns).sum()
        portfolio_variance = weights.values @ covariance_matrix.values @ weights.values
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = (
            (portfolio_return - risk_free_rate) / portfolio_volatility
            if portfolio_volatility > 0
            else 0
        )

        logger.info(f"✓ Optimization completed - Sharpe: {sharpe_ratio:.4f}")

        return {
            "weights": weights,
            "expected_return": portfolio_return,
            "volatility": portfolio_volatility,
            "variance": portfolio_variance,
            "sharpe_ratio": sharpe_ratio,
            "risk_free_rate": risk_free_rate,
        }

    def risk_parity_portfolio(
        self, covariance_matrix: pd.DataFrame, target_risk_contributions: pd.Series | None = None
    ) -> dict:
        """
        Risk Parity Portfolio

        Equalizes risk contribution from each asset.
        Risk contribution of asset i = w_i * (Σw)_i / σ_p

        Args:
            covariance_matrix: Covariance matrix
            target_risk_contributions: Target risk contributions (default: equal)

        Returns:
            Dictionary with optimal weights and statistics
        """
        logger.info("Solving Risk Parity Portfolio...")

        n_assets = len(covariance_matrix)
        asset_names = covariance_matrix.index.tolist()

        # Default: equal risk contribution
        if target_risk_contributions is None:
            target_risk_contributions = pd.Series([1.0 / n_assets] * n_assets, index=asset_names)

        # Risk parity is typically solved using numerical optimization
        # Here we use a simplified approach: inverse volatility weighting as approximation

        # Calculate individual asset volatilities
        asset_vols = np.sqrt(np.diag(covariance_matrix))

        # Inverse volatility weights
        inv_vol_weights = 1.0 / asset_vols
        weights = inv_vol_weights / inv_vol_weights.sum()
        weights = pd.Series(weights, index=asset_names)

        # Calculate risk contributions
        portfolio_volatility = np.sqrt(weights.values @ covariance_matrix.values @ weights.values)
        marginal_risk = covariance_matrix.values @ weights.values / portfolio_volatility
        risk_contributions = weights.values * marginal_risk
        risk_contributions = risk_contributions / risk_contributions.sum()

        logger.info(f"✓ Risk Parity completed - Vol: {portfolio_volatility:.4f}")

        return {
            "weights": weights,
            "volatility": portfolio_volatility,
            "risk_contributions": pd.Series(risk_contributions, index=asset_names),
            "asset_volatilities": pd.Series(asset_vols, index=asset_names),
        }


class RegimeAwareOptimizer:
    """
    Regime-aware portfolio optimization

    Optimizes portfolio considering multiple market regimes and
    their transition probabilities
    """

    def __init__(self, solver: str = "ipopt", use_gpu: bool = True):
        """
        Initialize regime-aware optimizer

        Args:
            solver: Pyomo solver
            use_gpu: Use GPU for computations
        """
        self.optimizer = PortfolioOptimizer(solver=solver, use_gpu=use_gpu)
        self.gpu = GPUPortfolioAccelerator()

        logger.info("RegimeAwareOptimizer initialized")

    def optimize_multi_regime(
        self,
        returns_by_regime: dict[int, pd.DataFrame],
        regime_probabilities: pd.Series,
        current_weights: pd.Series | None = None,
        transaction_cost: float = 0.001,
        risk_aversion: float = 1.0,
    ) -> dict:
        """
        Optimize portfolio across multiple regimes

        Objective: Maximize expected utility across regimes
        E[U] = Σ_r P(regime=r) * [E[R|r] - λ * Var[R|r]] - TC * |Δw|

        Args:
            returns_by_regime: Dictionary mapping regime_id to returns DataFrame
            regime_probabilities: Probability of each regime
            current_weights: Current portfolio weights (for transaction costs)
            transaction_cost: Transaction cost (% of trade)
            risk_aversion: Risk aversion parameter

        Returns:
            Dictionary with optimal weights and statistics
        """
        logger.info("Solving Multi-Regime Portfolio Optimization...")

        # Calculate statistics per regime
        regime_stats = {}
        for regime_id, returns in returns_by_regime.items():
            regime_stats[regime_id] = {
                "expected_returns": returns.mean(),
                "covariance": self.gpu.compute_covariance(returns, use_gpu=self.optimizer.use_gpu),
            }

        # Expected returns weighted by regime probabilities
        expected_returns = sum(
            regime_probabilities[regime_id] * regime_stats[regime_id]["expected_returns"]
            for regime_id in returns_by_regime
        )

        # Expected covariance (simplified: probability-weighted)
        asset_names = list(returns_by_regime.values())[0].columns
        expected_covariance = pd.DataFrame(0.0, index=asset_names, columns=asset_names)

        for regime_id in returns_by_regime:
            expected_covariance += (
                regime_probabilities[regime_id] * regime_stats[regime_id]["covariance"]
            )

        # Solve mean-variance with expected statistics
        result = self.optimizer.mean_variance_optimization(
            expected_returns=expected_returns,
            covariance_matrix=expected_covariance,
            risk_aversion=risk_aversion,
            transaction_cost=transaction_cost,
            current_weights=current_weights,
        )

        # Add transaction costs if current weights provided
        if current_weights is not None:
            turnover = (result["weights"] - current_weights).abs().sum()
            tc_cost = transaction_cost * turnover

            result["turnover"] = turnover
            result["transaction_cost"] = tc_cost
            result["net_return"] = result["expected_return"] - tc_cost

            logger.info(f"  Turnover: {turnover:.2%}, TC: {tc_cost:.4f}")

        result["regime_probabilities"] = regime_probabilities

        logger.info("✓ Multi-regime optimization completed")

        return result

    def optimize_regime_adaptive(
        self,
        returns: pd.DataFrame,
        regimes: pd.DataFrame,
        lookback_window: int = 252,
        risk_aversion: float = 1.0,
    ) -> pd.DataFrame:
        """
        Adaptive optimization that rebalances based on regime changes

        Args:
            returns: Historical returns
            regimes: Regime DataFrame with 'regime' column
            lookback_window: Window for estimating statistics
            risk_aversion: Risk aversion parameter

        Returns:
            DataFrame with optimal weights over time
        """
        logger.info("Running Regime-Adaptive Optimization...")

        # Align data
        common_idx = returns.index.intersection(regimes.index)
        returns = returns.loc[common_idx]
        regimes = regimes.loc[common_idx]

        # Storage for optimal weights
        optimal_weights_list = []

        # Iterate through time (walk-forward)
        for t in range(lookback_window, len(returns)):
            # Historical window
            hist_returns = returns.iloc[t - lookback_window : t]
            hist_regimes = regimes.iloc[t - lookback_window : t]

            # Current regime
            current_regime = regimes.iloc[t]["regime"]

            # Estimate statistics for current regime only
            regime_mask = hist_regimes["regime"] == current_regime
            regime_returns = hist_returns[regime_mask]

            if len(regime_returns) < 20:  # Not enough data
                # Use all data
                regime_returns = hist_returns

            # Optimize
            try:
                result = self.optimizer.mean_variance_optimization(
                    expected_returns=regime_returns.mean(),
                    covariance_matrix=self.gpu.compute_covariance(
                        regime_returns, use_gpu=self.optimizer.use_gpu
                    ),
                    risk_aversion=risk_aversion,
                )

                weights_row = result["weights"].copy()
                weights_row.name = returns.index[t]
                optimal_weights_list.append(weights_row)

            except Exception as e:
                logger.warning(f"Optimization failed at {returns.index[t]}: {e}")
                continue

        # Combine results
        optimal_weights_df = pd.DataFrame(optimal_weights_list)

        logger.info(f"✓ Adaptive optimization completed - {len(optimal_weights_df)} rebalances")

        return optimal_weights_df


class PortfolioBacktester:
    """
    Backtest portfolio optimization strategies
    """

    def __init__(self):
        """Initialize backtester"""
        self.gpu = GPUPortfolioAccelerator()

    def backtest_strategy(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        transaction_cost: float = 0.001,
        rebalance_frequency: int = 21,
    ) -> pd.DataFrame:
        """
        Backtest a portfolio strategy

        Args:
            weights: Optimal weights over time (Date × Asset)
            returns: Actual returns (Date × Asset)
            transaction_cost: Transaction cost (% of trade)
            rebalance_frequency: Days between rebalances

        Returns:
            DataFrame with backtest results
        """
        logger.info("Running portfolio backtest...")

        # Align dates
        common_dates = weights.index.intersection(returns.index)
        weights = weights.loc[common_dates]
        returns = returns.loc[common_dates]

        # Initialize
        portfolio_values = [1.0]
        portfolio_weights = weights.iloc[0].values
        last_rebalance = 0

        results = []

        for t in range(1, len(returns)):
            # Daily returns
            daily_returns = returns.iloc[t].values

            # Portfolio return (before rebalancing)
            portfolio_return = (portfolio_weights * daily_returns).sum()

            # Update portfolio value
            portfolio_value = portfolio_values[-1] * (1 + portfolio_return)

            # Check if rebalance day
            if (t - last_rebalance) >= rebalance_frequency:
                # Target weights
                target_weights = weights.iloc[t].values

                # Compute turnover
                turnover = np.abs(portfolio_weights - target_weights).sum()
                tc_cost = transaction_cost * turnover

                # Apply transaction cost
                portfolio_value *= 1 - tc_cost

                # Rebalance
                portfolio_weights = target_weights
                last_rebalance = t
            else:
                # Weight drift (no rebalancing)
                portfolio_weights = portfolio_weights * (1 + daily_returns)
                portfolio_weights = portfolio_weights / portfolio_weights.sum()
                turnover = 0
                tc_cost = 0

            portfolio_values.append(portfolio_value)

            results.append(
                {
                    "date": returns.index[t],
                    "portfolio_value": portfolio_value,
                    "portfolio_return": portfolio_return,
                    "turnover": turnover,
                    "transaction_cost": tc_cost,
                }
            )

        results_df = pd.DataFrame(results).set_index("date")

        # Calculate cumulative returns
        results_df["cumulative_return"] = (
            results_df["portfolio_value"] / results_df["portfolio_value"].iloc[0] - 1
        )

        logger.info(f"✓ Backtest completed - Final value: {portfolio_values[-1]:.4f}")

        return results_df

    def calculate_performance_metrics(self, backtest_results: pd.DataFrame) -> dict:
        """
        Calculate portfolio performance metrics

        Args:
            backtest_results: Backtest results DataFrame

        Returns:
            Dictionary of performance metrics
        """
        returns = backtest_results["portfolio_return"]

        # Annualized return
        total_return = backtest_results["cumulative_return"].iloc[-1]
        n_years = len(returns) / 252
        annualized_return = (1 + total_return) ** (1 / n_years) - 1

        # Annualized volatility
        annualized_vol = returns.std() * np.sqrt(252)

        # Sharpe ratio
        sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        sortino = annualized_return / downside_vol if downside_vol > 0 else 0

        # Calmar ratio
        calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Win rate
        win_rate = (returns > 0).sum() / len(returns)

        # Total transaction costs
        total_tc = backtest_results["transaction_cost"].sum()

        metrics = {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "total_transaction_costs": total_tc,
            "avg_daily_return": returns.mean(),
            "skewness": returns.skew(),
            "kurtosis": returns.kurtosis(),
        }

        return metrics


def run_backtest(returns: pd.DataFrame, portfolios: dict[str, pd.Series], show_plots: bool = False):
    """
    Backtest portfolio strategies and visualize results

    Args:
        returns: DataFrame of asset returns (Date × Asset)
        portfolios: Dictionary of portfolio weights (name → weights Series)
        show_plots: Whether to display plots

    Returns:
        Dictionary of backtest results and performance metrics
    """
    print("\n" + "=" * 70)
    print("DEMO: PORTFOLIO BACKTESTING")
    print("=" * 70)

    backtester = PortfolioBacktester()

    # Backtest each strategy
    backtest_results = {}

    for name, weights in portfolios.items():
        print(f"  Backtesting {name}...")

        # Create weight time series (constant weights for simplicity)
        weights_ts = pd.DataFrame(
            [weights.values] * len(returns), index=returns.index, columns=returns.columns
        )

        try:
            results = backtester.backtest_strategy(
                weights=weights_ts,
                returns=returns,
                transaction_cost=0.001,
                rebalance_frequency=21,  # Monthly
            )

            metrics = backtester.calculate_performance_metrics(results)
            backtest_results[name] = {"results": results, "metrics": metrics}
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            continue

    # Create performance comparison
    metrics_df = pd.DataFrame({name: data["metrics"] for name, data in backtest_results.items()}).T

    print("\nPerformance Summary:")
    print(
        metrics_df[
            ["annualized_return", "annualized_volatility", "sharpe_ratio", "max_drawdown"]
        ].to_string()
    )

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Cumulative Returns
    ax = axes[0, 0]
    for name, data in backtest_results.items():
        (1 + data["results"]["portfolio_return"]).cumprod().plot(ax=ax, label=name, linewidth=1.5)

    ax.set_title("Cumulative Returns", fontweight="bold")
    ax.set_ylabel("Cumulative Return")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Plot 2: Rolling Sharpe
    ax = axes[0, 1]
    for name, data in backtest_results.items():
        returns_series = data["results"]["portfolio_return"]
        rolling_sharpe = (returns_series.rolling(252).mean() * 252) / (
            returns_series.rolling(252).std() * np.sqrt(252)
        )
        rolling_sharpe.plot(ax=ax, label=name, alpha=0.7)

    ax.set_title("Rolling 1-Year Sharpe Ratio", fontweight="bold")
    ax.set_ylabel("Sharpe Ratio")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Plot 3: Drawdowns
    ax = axes[1, 0]
    for name, data in backtest_results.items():
        returns_series = data["results"]["portfolio_return"]
        cumulative = (1 + returns_series).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        drawdown.plot(ax=ax, label=name, alpha=0.7)

    ax.set_title("Drawdowns", fontweight="bold")
    ax.set_ylabel("Drawdown")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Plot 4: Performance Metrics
    ax = axes[1, 1]
    metrics_to_plot = metrics_df[["sharpe_ratio", "sortino_ratio", "calmar_ratio"]]
    metrics_to_plot.plot(kind="bar", ax=ax)
    ax.set_title("Risk-Adjusted Performance Metrics", fontweight="bold")
    ax.set_ylabel("Ratio")
    ax.set_xlabel("")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()

    # Save
    results_dir = Path("results/plots")
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(results_dir / "backtest_results.png", dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved plot to {results_dir / 'backtest_results.png'}")

    if show_plots:
        plt.show()
    plt.close()

    # Save metrics
    metrics_df.to_csv(Path("results") / "portfolio_performance_metrics.csv")
    print("✓ Saved metrics to results/portfolio_performance_metrics.csv")

    return backtest_results, metrics_df


def main():
    """Main demonstration function"""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))

    logger.info("=" * 70)
    logger.info("PORTFOLIO OPTIMIZATION DEMONSTRATION")
    logger.info("=" * 70)

    # Load data
    try:
        prices = pd.read_csv("data/raw/asset_prices.csv", index_col=0, parse_dates=True)
        returns = prices.pct_change().dropna()
        logger.info(f"✓ Loaded data: {returns.shape}")
    except FileNotFoundError:
        logger.error("Data not found. Run download_data.py first.")
        return

    # Use recent data for demonstration
    returns_recent = returns  # .tail(252)  # Last year

    # Calculate statistics
    gpu = GPUPortfolioAccelerator()
    expected_returns = returns_recent.mean() * 252  # Annualized
    cov_matrix = gpu.compute_covariance(returns_recent, use_gpu=True) * 252  # Annualized

    # Initialize optimizer
    optimizer = PortfolioOptimizer(solver="ipopt", use_gpu=True)
    portfolios = {}

    # 1. Mean-Variance Optimization
    print("\n1. Mean-Variance Optimization")
    print("-" * 70)
    mv_result = optimizer.mean_variance_optimization(
        expected_returns=expected_returns, covariance_matrix=cov_matrix, risk_aversion=2.0
    )
    print(f"Expected Return: {mv_result['expected_return']:.4f}")
    print(f"Volatility: {mv_result['volatility']:.4f}")
    print(f"Sharpe Ratio: {mv_result['sharpe_ratio']:.4f}")
    print("Top 5 positions:")
    print(mv_result["weights"].nlargest(5))
    portfolios["Mean-Variance"] = mv_result["weights"]

    # 2. Minimum Variance
    print("\n2. Minimum Variance Portfolio")
    print("-" * 70)
    min_var_result = optimizer.minimum_variance_portfolio(covariance_matrix=cov_matrix)
    print(f"Minimum Volatility: {min_var_result['volatility']:.4f}")
    print("Top 5 positions:")
    print(min_var_result["weights"].nlargest(5))
    portfolios["Minimum-Variance"] = min_var_result["weights"]

    # 3. Maximum Sharpe
    print("\n3. Maximum Sharpe Ratio Portfolio")
    print("-" * 70)
    max_sharpe_result = optimizer.maximum_sharpe_ratio(
        expected_returns=expected_returns, covariance_matrix=cov_matrix, risk_free_rate=0.02
    )
    print(f"Sharpe Ratio: {max_sharpe_result['sharpe_ratio']:.4f}")
    print(f"Expected Return: {max_sharpe_result['expected_return']:.4f}")
    print(f"Volatility: {max_sharpe_result['volatility']:.4f}")
    print("Top 5 positions:")
    print(max_sharpe_result["weights"].nlargest(5))
    portfolios["Maximum-Sharpe"] = max_sharpe_result["weights"]

    # 4. Risk Parity
    print("\n4. Risk Parity Portfolio")
    print("-" * 70)
    rp_result = optimizer.risk_parity_portfolio(covariance_matrix=cov_matrix)
    print(f"Volatility: {rp_result['volatility']:.4f}")
    print("Top 5 positions:")
    print(rp_result["weights"].nlargest(5))
    print("Risk contributions:")
    print(rp_result["risk_contributions"].nlargest(5))
    portfolios["Risk-Parity"] = rp_result["weights"]

    # 5. Regime-Aware Optimization (from Engineered Features)
    print("\n5. Regime-Aware Optimization (Feature-Based Regimes)")
    print("-" * 70)

    regimes_features_path = Path("data/processed/regimes_from_features.parquet")
    if regimes_features_path.exists():
        try:
            logger.info("Loading regime detection results from engineered features...")
            regimes_features = pd.read_parquet(regimes_features_path)
            logger.info(f"✓ Loaded feature-based regimes: {regimes_features.shape}")

            # Extract regime assignments and names
            regime_assignments = regimes_features["regime"]
            regime_names = regimes_features.get("regime_name", None)

            # Identify return-related feature columns
            return_feature_cols = [
                col
                for col in regimes_features.columns
                if any(
                    ret_keyword in col.lower() for ret_keyword in ["return", "pct_change", "daily"]
                )
                and col not in ["regime", "regime_name"]
            ]

            # Also check for mean_return if available
            if "market_return" in regimes_features.columns:
                return_feature_cols.insert(0, "market_return")

            logger.info(f"  Identified return features: {return_feature_cols}")

            # Align regimes with returns
            common_idx = regimes_features.index.intersection(returns_recent.index)
            regimes_aligned = regimes_features.loc[common_idx]
            returns_recent = returns_recent.loc[common_idx]

            # Method 1: Calculate returns_by_regime using feature columns
            returns_by_regime_features = {}
            regime_returns_stats = {}

            for regime_id in sorted(regime_assignments.unique()):
                mask = regime_assignments[common_idx] == regime_id

                if return_feature_cols:
                    # Use available return feature columns to estimate returns per regime
                    regime_features = regimes_aligned.loc[mask, return_feature_cols]

                    # Create synthetic returns DataFrame for this regime using feature columns
                    # This allows us to understand return characteristics per regime
                    regime_returns_stats[regime_id] = {
                        "mean_return": regime_features.mean().mean(),  # Average of return features
                        "volatility": regime_features.std().mean(),
                        "feature_means": regime_features.mean(),
                    }

                    # For optimization, use actual asset returns during this regime
                    returns_by_regime_features[regime_id] = returns_recent.loc[mask]

                    logger.info(
                        f"  Regime {regime_id}: {mask.sum()} observations, "
                        f"Mean Return: {regime_returns_stats[regime_id]['mean_return']:.4f}"
                    )
                else:
                    # Fallback: use actual asset returns for this regime
                    returns_by_regime_features[regime_id] = returns_recent.loc[mask]
                    regime_returns_stats[regime_id] = {
                        "mean_return": returns_recent.loc[mask].mean().mean(),
                        "volatility": returns_recent.loc[mask].std().mean(),
                    }

            # Method 2: Calculate regime probabilities as frequency of observations
            regime_counts = regime_assignments[common_idx].value_counts()
            regime_probabilities_features = (regime_counts / regime_counts.sum()).sort_index()

            logger.info("  Regime Probabilities:")
            for regime_id in regime_probabilities_features.index:
                regime_name = (
                    regime_names.loc[regime_assignments == regime_id].values[0]
                    if regime_names is not None
                    else f"Regime {regime_id}"
                )
                logger.info(f"    {regime_name}: {regime_probabilities_features[regime_id]:.2%}")

            # Run regime-aware optimization
            regime_optimizer = RegimeAwareOptimizer(solver="ipopt", use_gpu=True)

            regime_result_features = regime_optimizer.optimize_multi_regime(
                returns_by_regime=returns_by_regime_features,
                regime_probabilities=regime_probabilities_features,
                risk_aversion=2.0,
                transaction_cost=0.001,
                current_weights=mv_result["weights"],
            )

            print(f"Expected Return: {regime_result_features['expected_return']:.4f}")
            print(f"Volatility: {regime_result_features['volatility']:.4f}")
            print(f"Sharpe Ratio: {regime_result_features['sharpe_ratio']:.4f}")
            if "turnover" in regime_result_features:
                print(f"Turnover: {regime_result_features['turnover']:.2%}")
            print("Top 5 positions:")
            print(regime_result_features["weights"].nlargest(5))

            # Regime statistics
            print("\nRegime Statistics (from features):")
            for regime_id, stats in regime_returns_stats.items():
                print(
                    f"  Regime {regime_id}: Mean Return={stats['mean_return']:.4f}, "
                    f"Volatility={stats['volatility']:.4f}"
                )

            portfolios["Regime-Aware-Features"] = regime_result_features["weights"]

        except Exception as e:
            logger.warning(f"Failed to load feature-based regimes: {e}")
            logger.info("Falling back to simple regime-aware optimization...")

            # Fallback: simple regimes based on market conditions
            regimes = pd.DataFrame(index=returns_recent.index)
            regimes["regime"] = (returns_recent.mean(axis=1) > 0).astype(int)
            regime_optimizer = RegimeAwareOptimizer(solver="ipopt", use_gpu=True)
            regime_probs = pd.Series({0: 0.4, 1: 0.6})
            returns_by_regime = {
                0: returns_recent[regimes["regime"] == 0],
                1: returns_recent[regimes["regime"] == 1],
            }
            regime_result_features = regime_optimizer.optimize_multi_regime(
                returns_by_regime=returns_by_regime,
                regime_probabilities=regime_probs,
                risk_aversion=2.0,
                transaction_cost=0.001,
                current_weights=mv_result["weights"],
            )
            print(f"Expected Return: {regime_result_features['expected_return']:.4f}")
            print(f"Volatility: {regime_result_features['volatility']:.4f}")
            print(f"Sharpe Ratio: {regime_result_features['sharpe_ratio']:.4f}")
            portfolios["Regime-Aware-Features"] = regime_result_features["weights"]
    else:
        logger.warning(f"Feature-based regimes not found at {regimes_features_path}")
        logger.info("Run regime_detection.py to generate feature-based regimes first.")
        logger.info("Using simple regime-aware optimization instead...")

        # Simple fallback
        regimes = pd.DataFrame(index=returns_recent.index)
        regimes["regime"] = (returns_recent.mean(axis=1) > 0).astype(int)
        regime_optimizer = RegimeAwareOptimizer(solver="ipopt", use_gpu=True)
        regime_probs = pd.Series({0: 0.4, 1: 0.6})
        returns_by_regime = {
            0: returns_recent[regimes["regime"] == 0],
            1: returns_recent[regimes["regime"] == 1],
        }
        regime_result_features = regime_optimizer.optimize_multi_regime(
            returns_by_regime=returns_by_regime,
            regime_probabilities=regime_probs,
            risk_aversion=2.0,
            transaction_cost=0.001,
            current_weights=mv_result["weights"],
        )
        print(f"Expected Return: {regime_result_features['expected_return']:.4f}")
        print(f"Volatility: {regime_result_features['volatility']:.4f}")
        print(f"Sharpe Ratio: {regime_result_features['sharpe_ratio']:.4f}")
        portfolios["Regime-Aware-Features"] = regime_result_features["weights"]

    # Run backtest
    backtest_results, performance_metrics = run_backtest(returns_recent, portfolios)

    # Save results
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save optimal weights
    pd.DataFrame(
        {
            "mean_variance": mv_result["weights"],
            "min_variance": min_var_result["weights"],
            "max_sharpe": max_sharpe_result["weights"],
            "risk_parity": rp_result["weights"],
        }
    ).to_csv(output_dir / "optimal_portfolios.csv")

    logger.info("\n✓ Results saved to data/processed/optimal_portfolios.csv")

    return {
        "mean_variance": mv_result,
        "min_variance": min_var_result,
        "max_sharpe": max_sharpe_result,
        "risk_parity": rp_result,
    }


if __name__ == "__main__":
    results = main()
