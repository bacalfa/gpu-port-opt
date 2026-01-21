"""
Technical Indicators Feature Engineering Module
===============================================

GPU-accelerated computation of technical indicators and engineered features
for portfolio optimization and regime detection.

Features:
- GPU acceleration with cuDF when available, automatic CPU fallback
- 30+ technical indicators (momentum, volatility, trend)
- Rolling statistics and cross-sectional features
- Lag features and temporal transformations
- Handles missing data gracefully
- Memory-efficient batch processing

Author: Bruno Abreu Calfa
"""

import logging
import warnings
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

# GPU acceleration with automatic fallback
try:
    import cudf
    import cupy as cp

    GPU_AVAILABLE = True
    print("✓ GPU acceleration enabled (cuDF/CuPy)")
    DataFrame = Union[pd.DataFrame, cudf.DataFrame]
    Series = Union[pd.Series, cudf.Series]
except ImportError:
    GPU_AVAILABLE = False
    print("⚠ GPU not available, using CPU (pandas/numpy)")
    cudf = pd
    cp = np
    DataFrame = pd.DataFrame
    Series = pd.Series

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class GPUAccelerator:
    """Utility class for GPU/CPU operations with automatic fallback"""

    @staticmethod
    def to_gpu(df: pd.DataFrame) -> DataFrame:
        """Convert pandas DataFrame to cuDF if GPU available"""
        if GPU_AVAILABLE:
            try:
                return cudf.from_pandas(df)
            except Exception as e:
                logger.warning(f"GPU conversion failed: {e}. Using CPU.")
                return df
        return df

    @staticmethod
    def to_cpu(df: DataFrame) -> pd.DataFrame:
        """Convert cuDF DataFrame to pandas"""
        if GPU_AVAILABLE and isinstance(df, cudf.DataFrame):
            return df.to_pandas()
        return df

    @staticmethod
    def sqrt(x):
        """Square root (GPU or CPU)"""
        return cp.sqrt(x) if GPU_AVAILABLE else np.sqrt(x)

    @staticmethod
    def abs(x):
        """Absolute value (GPU or CPU)"""
        return cp.abs(x) if GPU_AVAILABLE else np.abs(x)

    @staticmethod
    def exp(x):
        """Exponential (GPU or CPU)"""
        return cp.exp(x) if GPU_AVAILABLE else np.exp(x)

    @staticmethod
    def log(x):
        """Natural log (GPU or CPU)"""
        return cp.log(x) if GPU_AVAILABLE else np.log(x)


class TechnicalIndicators:
    """
    Compute technical indicators with GPU acceleration

    All methods support both pandas and cuDF DataFrames automatically.
    """

    def __init__(self, use_gpu: bool = True):
        """
        Initialize technical indicators calculator

        Args:
            use_gpu: Whether to use GPU if available (default: True)
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.gpu = GPUAccelerator()

        if self.use_gpu:
            logger.info("TechnicalIndicators initialized with GPU acceleration")
        else:
            logger.info("TechnicalIndicators initialized with CPU")

    def compute_returns(self, prices: DataFrame, periods: list[int] = [1, 5, 21, 63]) -> DataFrame:
        """
        Compute returns over multiple periods

        Args:
            prices: Price data (Date × Asset)
            periods: List of return periods in days

        Returns:
            DataFrame with return columns for each period
        """
        logger.info(f"Computing returns for periods: {periods}")

        # Convert to GPU if needed
        if self.use_gpu:
            prices = self.gpu.to_gpu(prices)

        returns_dict = {}

        for period in periods:
            if period == 1:
                # Daily returns
                returns = prices.pct_change()
                suffix = "daily"
            else:
                # Multi-period returns
                returns = prices.pct_change(periods=period)
                suffix = f"{period}d"

            # Rename columns
            returns.columns = [f"{col}_return_{suffix}" for col in prices.columns]
            returns_dict[suffix] = returns

        # Combine all returns
        result = pd.concat([self.gpu.to_cpu(r) for r in returns_dict.values()], axis=1)

        logger.info(f"✓ Computed {len(result.columns)} return features")
        return result

    def compute_log_returns(self, prices: DataFrame) -> DataFrame:
        """
        Compute log returns (more suitable for statistical models)

        Args:
            prices: Price data

        Returns:
            DataFrame with log returns
        """
        if self.use_gpu:
            prices = self.gpu.to_gpu(prices)

        # Log returns: ln(P_t / P_{t-1})
        log_returns = self.gpu.log(prices / prices.shift(1))
        log_returns.columns = [f"{col}_log_return" for col in prices.columns]

        return self.gpu.to_cpu(log_returns)

    def compute_sma(self, prices: DataFrame, windows: list[int] = [20, 50, 200]) -> DataFrame:
        """
        Simple Moving Average (SMA)

        Args:
            prices: Price data
            windows: List of SMA window sizes

        Returns:
            DataFrame with SMA features
        """
        logger.info(f"Computing SMA for windows: {windows}")

        if self.use_gpu:
            prices = self.gpu.to_gpu(prices)

        sma_dict = {}

        for window in windows:
            sma = prices.rolling(window=window).mean()
            sma.columns = [f"{col}_sma_{window}" for col in prices.columns]
            sma_dict[window] = sma

        result = pd.concat([self.gpu.to_cpu(s) for s in sma_dict.values()], axis=1)
        logger.info(f"✓ Computed {len(result.columns)} SMA features")

        return result

    def compute_ema(self, prices: DataFrame, spans: list[int] = [12, 26, 50]) -> DataFrame:
        """
        Exponential Moving Average (EMA)

        Args:
            prices: Price data
            spans: List of EMA span sizes

        Returns:
            DataFrame with EMA features
        """
        logger.info(f"Computing EMA for spans: {spans}")

        if self.use_gpu:
            prices = self.gpu.to_gpu(prices)

        ema_dict = {}

        for span in spans:
            ema = prices.ewm(span=span, adjust=False).mean()
            ema.columns = [f"{col}_ema_{span}" for col in prices.columns]
            ema_dict[span] = ema

        result = pd.concat([self.gpu.to_cpu(e) for e in ema_dict.values()], axis=1)
        logger.info(f"✓ Computed {len(result.columns)} EMA features")

        return result

    def compute_rsi(self, prices: DataFrame, window: int = 14) -> DataFrame:
        """
        Relative Strength Index (RSI)

        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss

        Args:
            prices: Price data
            window: RSI period (default: 14)

        Returns:
            DataFrame with RSI values (0-100)
        """
        logger.info(f"Computing RSI with window={window}")

        if self.use_gpu:
            prices = self.gpu.to_gpu(prices)

        # Calculate price changes
        delta = prices.diff()

        # Separate gains and losses
        gains = delta.copy()
        losses = delta.copy()

        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = self.gpu.abs(losses)

        # Calculate average gains and losses
        avg_gains = gains.rolling(window=window).mean()
        avg_losses = losses.rolling(window=window).mean()

        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        rsi.columns = [f"{col}_rsi_{window}" for col in prices.columns]

        return self.gpu.to_cpu(rsi)

    def compute_macd(
        self, prices: DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> DataFrame:
        """
        Moving Average Convergence Divergence (MACD)

        MACD Line = EMA(fast) - EMA(slow)
        Signal Line = EMA(MACD, signal)
        MACD Histogram = MACD Line - Signal Line

        Args:
            prices: Price data
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line period (default: 9)

        Returns:
            DataFrame with MACD, signal, and histogram
        """
        logger.info(f"Computing MACD ({fast}, {slow}, {signal})")

        if self.use_gpu:
            prices = self.gpu.to_gpu(prices)

        # Calculate MACD line
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow

        # Calculate signal line
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()

        # Calculate histogram
        histogram = macd_line - signal_line

        # Rename columns
        macd_line.columns = [f"{col}_macd" for col in prices.columns]
        signal_line.columns = [f"{col}_macd_signal" for col in prices.columns]
        histogram.columns = [f"{col}_macd_hist" for col in prices.columns]

        result = pd.concat(
            [self.gpu.to_cpu(macd_line), self.gpu.to_cpu(signal_line), self.gpu.to_cpu(histogram)],
            axis=1,
        )

        logger.info(f"✓ Computed {len(result.columns)} MACD features")

        return result

    def compute_bollinger_bands(
        self, prices: DataFrame, window: int = 20, num_std: float = 2.0
    ) -> DataFrame:
        """
        Bollinger Bands

        Middle Band = SMA(window)
        Upper Band = Middle Band + (std * num_std)
        Lower Band = Middle Band - (std * num_std)
        Bandwidth = (Upper - Lower) / Middle
        %B = (Price - Lower) / (Upper - Lower)

        Args:
            prices: Price data
            window: Rolling window (default: 20)
            num_std: Number of standard deviations (default: 2.0)

        Returns:
            DataFrame with Bollinger Bands features
        """
        logger.info(f"Computing Bollinger Bands (window={window}, std={num_std})")

        if self.use_gpu:
            prices = self.gpu.to_gpu(prices)

        # Middle band (SMA)
        middle = prices.rolling(window=window).mean()

        # Standard deviation
        std = prices.rolling(window=window).std()

        # Upper and lower bands
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)

        # Bandwidth
        bandwidth = (upper - lower) / middle

        # %B (position within bands)
        percent_b = (prices - lower) / (upper - lower)

        # Rename columns
        upper.columns = [f"{col}_bb_upper" for col in prices.columns]
        lower.columns = [f"{col}_bb_lower" for col in prices.columns]
        bandwidth.columns = [f"{col}_bb_width" for col in prices.columns]
        percent_b.columns = [f"{col}_bb_pct" for col in prices.columns]

        result = pd.concat(
            [
                self.gpu.to_cpu(upper),
                self.gpu.to_cpu(lower),
                self.gpu.to_cpu(bandwidth),
                self.gpu.to_cpu(percent_b),
            ],
            axis=1,
        )

        logger.info(f"✓ Computed {len(result.columns)} Bollinger Band features")

        return result

    def compute_atr(
        self, high: DataFrame, low: DataFrame, close: DataFrame, window: int = 14
    ) -> DataFrame:
        """
        Average True Range (ATR) - volatility indicator

        Note: Requires high, low, close data. For price-only data,
        use rolling standard deviation instead.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: ATR period (default: 14)

        Returns:
            DataFrame with ATR values
        """
        logger.info(f"Computing ATR with window={window}")

        if self.use_gpu:
            high = self.gpu.to_gpu(high)
            low = self.gpu.to_gpu(low)
            close = self.gpu.to_gpu(close)

        # True Range components
        tr1 = high - low
        tr2 = self.gpu.abs(high - close.shift(1))
        tr3 = self.gpu.abs(low - close.shift(1))

        # True Range = max(tr1, tr2, tr3)
        # For GPU/CPU compatibility, use component-wise max
        tr = tr1.copy()
        tr = tr.where(tr >= tr2, tr2)
        tr = tr.where(tr >= tr3, tr3)

        # ATR = EMA of True Range
        atr = tr.ewm(span=window, adjust=False).mean()

        atr.columns = [f"{col}_atr_{window}" for col in close.columns]

        return self.gpu.to_cpu(atr)

    def compute_momentum(self, prices: DataFrame, windows: list[int] = [10, 20, 50]) -> DataFrame:
        """
        Price Momentum

        Momentum = (Price_t / Price_{t-n}) - 1

        Args:
            prices: Price data
            windows: List of momentum periods

        Returns:
            DataFrame with momentum features
        """
        logger.info(f"Computing momentum for windows: {windows}")

        if self.use_gpu:
            prices = self.gpu.to_gpu(prices)

        momentum_dict = {}

        for window in windows:
            mom = (prices / prices.shift(window)) - 1
            mom.columns = [f"{col}_momentum_{window}" for col in prices.columns]
            momentum_dict[window] = mom

        result = pd.concat([self.gpu.to_cpu(m) for m in momentum_dict.values()], axis=1)
        logger.info(f"✓ Computed {len(result.columns)} momentum features")

        return result

    def compute_volatility(
        self, returns: DataFrame, windows: list[int] = [20, 60, 252]
    ) -> DataFrame:
        """
        Rolling volatility (annualized)

        Vol = std(returns) * sqrt(252)

        Args:
            returns: Return data
            windows: List of volatility windows

        Returns:
            DataFrame with annualized volatility
        """
        logger.info(f"Computing volatility for windows: {windows}")

        if self.use_gpu:
            returns = self.gpu.to_gpu(returns)

        vol_dict = {}

        for window in windows:
            vol = returns.rolling(window=window).std() * self.gpu.sqrt(252)
            vol.columns = [f"{col}_vol_{window}" for col in returns.columns]
            vol_dict[window] = vol

        result = pd.concat([self.gpu.to_cpu(v) for v in vol_dict.values()], axis=1)
        logger.info(f"✓ Computed {len(result.columns)} volatility features")

        return result

    def compute_rolling_correlation(
        self, returns: DataFrame, benchmark: str = "SPY", window: int = 60
    ) -> DataFrame:
        """
        Rolling correlation with benchmark

        Args:
            returns: Return data
            benchmark: Benchmark ticker (default: 'SPY')
            window: Correlation window

        Returns:
            DataFrame with rolling correlations
        """
        logger.info(f"Computing rolling correlation with {benchmark} (window={window})")

        # Find benchmark column
        benchmark_col = None
        for col in returns.columns:
            if benchmark in col:
                benchmark_col = col
                break

        if benchmark_col is None:
            logger.warning(f"Benchmark {benchmark} not found in returns")
            return pd.DataFrame()

        if self.use_gpu:
            returns = self.gpu.to_gpu(returns)

        # Compute rolling correlation
        corr_dict = {}

        for col in returns.columns:
            if col != benchmark_col:
                corr = returns[col].rolling(window=window).corr(returns[benchmark_col])
                corr_dict[f"{col}_corr_{benchmark}_{window}"] = corr

        result = pd.DataFrame(corr_dict)

        logger.info(f"✓ Computed {len(result.columns)} correlation features")

        return self.gpu.to_cpu(result)

    def compute_all_indicators(
        self, prices: DataFrame, compute_macd_flag: bool = True, compute_bb_flag: bool = True
    ) -> DataFrame:
        """
        Compute all technical indicators at once

        Args:
            prices: Price data
            compute_macd_flag: Whether to compute MACD (default: True)
            compute_bb_flag: Whether to compute Bollinger Bands (default: True)

        Returns:
            DataFrame with all technical indicators
        """
        logger.info("=" * 60)
        logger.info("Computing ALL technical indicators")
        logger.info("=" * 60)

        features = []

        # 1. Returns (multiple periods)
        returns_1d = prices.pct_change()
        returns_multi = self.compute_returns(prices, periods=[1, 5, 21, 63])
        features.append(returns_multi)

        # 2. Log returns
        log_returns = self.compute_log_returns(prices)
        features.append(log_returns)

        # 3. Simple Moving Averages
        sma = self.compute_sma(prices, windows=[20, 50, 200])
        features.append(sma)

        # 4. Exponential Moving Averages
        ema = self.compute_ema(prices, spans=[12, 26, 50])
        features.append(ema)

        # 5. RSI
        rsi = self.compute_rsi(prices, window=14)
        features.append(rsi)

        # 6. MACD (optional, computationally expensive)
        if compute_macd_flag:
            macd = self.compute_macd(prices)
            features.append(macd)

        # 7. Bollinger Bands (optional)
        if compute_bb_flag:
            bb = self.compute_bollinger_bands(prices)
            features.append(bb)

        # 8. Momentum
        momentum = self.compute_momentum(prices, windows=[10, 20, 50])
        features.append(momentum)

        # 9. Volatility
        volatility = self.compute_volatility(returns_1d, windows=[20, 60, 252])
        features.append(volatility)

        # 10. Rolling correlation with SPY
        if "SPY" in prices.columns:
            corr = self.compute_rolling_correlation(returns_1d, benchmark="SPY", window=60)
            if not corr.empty:
                features.append(corr)

        # 11. Average true range
        high_cols = [col for col in prices.columns if col.endswith("_High")]
        low_cols = [col for col in prices.columns if col.endswith("_Low")]
        close_cols = [col for col in prices.columns if col.endswith("_Adj Close")]
        if len(high_cols) > 0 and len(low_cols) > 0 and len(close_cols) > 0:
            atr = self.compute_atr(
                prices[high_cols], prices[low_cols], prices[close_cols], window=14
            )
            features.append(atr)

        # Combine all features
        result = pd.concat(features, axis=1)

        logger.info("=" * 60)
        logger.info(f"✓ Total features computed: {len(result.columns)}")
        logger.info(f"✓ Date range: {result.index[0]} to {result.index[-1]}")
        logger.info(f"✓ Shape: {result.shape}")
        logger.info("=" * 60)

        return result


class FeatureEngineer:
    """
    Complete feature engineering pipeline
    Combines technical indicators with lagged features and transformations
    """

    def __init__(self, use_gpu: bool = True):
        """
        Initialize feature engineer

        Args:
            use_gpu: Whether to use GPU if available
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.tech_indicators = TechnicalIndicators(use_gpu=use_gpu)
        self.gpu = GPUAccelerator()

        logger.info(f"FeatureEngineer initialized (GPU: {self.use_gpu})")

    def create_lag_features(
        self, data: DataFrame, columns: list[str], lags: list[int] = [1, 5, 21]
    ) -> DataFrame:
        """
        Create lagged features

        Args:
            data: Input data
            columns: Columns to lag
            lags: List of lag periods

        Returns:
            DataFrame with lagged features
        """
        logger.info(f"Creating lag features for {len(columns)} columns, lags: {lags}")

        if self.use_gpu:
            data = self.gpu.to_gpu(data)

        lag_features = []

        for col in columns:
            if col in data.columns:
                for lag in lags:
                    lagged = data[col].shift(lag)
                    lagged.name = f"{col}_lag_{lag}"
                    lag_features.append(lagged)

        result = pd.concat([self.gpu.to_cpu(f) for f in lag_features], axis=1)

        logger.info(f"✓ Created {len(result.columns)} lag features")

        return result

    def create_cross_sectional_features(self, returns: DataFrame) -> DataFrame:
        """
        Create cross-sectional features (relative to universe)

        Features:
        - Rank (percentile rank within universe)
        - Z-score (standardized return)
        - Deviation from mean

        Args:
            returns: Return data

        Returns:
            DataFrame with cross-sectional features
        """
        logger.info("Creating cross-sectional features")

        if self.use_gpu:
            returns = self.gpu.to_gpu(returns)

        # Cross-sectional rank (percentile)
        rank = returns.rank(axis=1, pct=True)
        rank.columns = [f"{col}_rank" for col in returns.columns]

        # Cross-sectional z-score
        mean = returns.mean(axis=1)
        std = returns.std(axis=1)

        zscore = returns.copy()
        for col in returns.columns:
            zscore[col] = (returns[col] - mean) / std
        zscore.columns = [f"{col}_zscore" for col in returns.columns]

        # Deviation from mean
        deviation = returns.copy()
        for col in returns.columns:
            deviation[col] = returns[col] - mean
        deviation.columns = [f"{col}_dev_mean" for col in returns.columns]

        result = pd.concat(
            [self.gpu.to_cpu(rank), self.gpu.to_cpu(zscore), self.gpu.to_cpu(deviation)], axis=1
        )

        logger.info(f"✓ Created {len(result.columns)} cross-sectional features")

        return result

    def create_interaction_features(
        self, data: DataFrame, feature_pairs: list[tuple[str, str]]
    ) -> DataFrame:
        """
        Create interaction features (products of feature pairs)

        Args:
            data: Input data
            feature_pairs: List of (feature1, feature2) tuples

        Returns:
            DataFrame with interaction features
        """
        logger.info(f"Creating {len(feature_pairs)} interaction features")

        if self.use_gpu:
            data = self.gpu.to_gpu(data)

        interactions = []

        tickers = data.columns.str.split("_").str[0].unique()

        for ticker in tickers:
            for feat1, feat2 in feature_pairs:
                feat1 = f"{ticker}_{feat1}"
                feat2 = f"{ticker}_{feat2}"
                if feat1 in data.columns and feat2 in data.columns:
                    interaction = data[feat1] * data[feat2]
                    interaction.name = f"{feat1}_x_{feat2}"
                    interactions.append(interaction)

        if not interactions:
            logger.warning("No valid feature pairs found")
            return pd.DataFrame()

        result = pd.concat([self.gpu.to_cpu(i) for i in interactions], axis=1)

        logger.info(f"✓ Created {len(result.columns)} interaction features")

        return result

    def build_complete_features(
        self,
        prices: DataFrame,
        macro: DataFrame | None = None,
        sentiment: DataFrame | None = None,
        include_lags: bool = True,
        include_cross_sectional: bool = True,
        include_interaction: bool = True,
    ) -> DataFrame:
        """
        Build complete feature set for ML models

        Args:
            prices: Price data
            macro: Macro indicators (optional)
            sentiment: Sentiment indicators (optional)
            include_lags: Whether to include lag features
            include_cross_sectional: Whether to include cross-sectional features
            include_interaction: Whether to include interaction features
        Returns:
            DataFrame with complete feature set
        """
        logger.info("=" * 60)
        logger.info("BUILDING COMPLETE FEATURE SET")
        logger.info("=" * 60)

        features = []

        # 1. Technical indicators
        tech_features = self.tech_indicators.compute_all_indicators(prices)
        features.append(tech_features)

        # 2. Lag features (if requested)
        if include_lags:
            # Lag returns and key indicators
            returns_cols = [col for col in tech_features.columns if "return_daily" in col]
            if returns_cols:
                lag_features = self.create_lag_features(
                    tech_features,
                    columns=returns_cols[:10],  # Limit to first 10 assets
                    lags=[1, 5, 21],
                )
                features.append(lag_features)

        # 3. Cross-sectional features (if requested)
        if include_cross_sectional:
            returns_cols = [col for col in tech_features.columns if "return_daily" in col]
            if returns_cols:
                returns_df = tech_features[returns_cols]
                cross_features = self.create_cross_sectional_features(returns_df)
                features.append(cross_features)

        # 4. Add macro features (if provided)
        if macro is not None and not macro.empty:
            logger.info(f"Adding {len(macro.columns)} macro features")
            features.append(macro)

        # 5. Add sentiment features (if provided)
        if sentiment is not None and not sentiment.empty:
            logger.info(f"Adding {len(sentiment.columns)} sentiment features")
            features.append(sentiment)

        # 6. Add interaction features (if requested)
        if include_interaction:
            interaction_features = self.create_interaction_features(
                tech_features, feature_pairs=[("sma_20", "ema_12"), ("rsi_14", "macd_12_26_9")]
            )
            features.append(interaction_features)

        # Combine all features
        result = pd.concat(features, axis=1)

        # Remove duplicate columns (if any)
        result = result.loc[:, ~result.columns.duplicated()]

        logger.info("=" * 60)
        logger.info("✓ Complete feature set created")
        logger.info(f"✓ Total features: {len(result.columns)}")
        logger.info(f"✓ Date range: {result.index[0]} to {result.index[-1]}")
        logger.info(f"✓ Shape: {result.shape}")
        logger.info(f"✓ Missing values: {result.isnull().sum().sum()}")
        logger.info("=" * 60)

        return result

    def save_features(self, features: DataFrame, output_path: str | Path, format: str = "parquet"):
        """
        Save features to disk

        Args:
            features: Feature DataFrame
            output_path: Output file path
            format: File format ('parquet', 'csv', 'feather')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to pandas for saving
        features = self.gpu.to_cpu(features)

        if format == "parquet":
            features.to_parquet(output_path)
        elif format == "csv":
            features.to_csv(output_path)
        elif format == "feather":
            features.reset_index().to_feather(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"✓ Features saved to {output_path}")


def main():
    """Main execution function - demonstration"""

    # Load data
    logger.info("Loading price data...")
    prices = pd.read_csv("data/raw/asset_prices.csv", index_col=0, parse_dates=True)

    # Optional: load macro and sentiment
    try:
        macro = pd.read_csv("data/raw/macro_data.csv", index_col=0, parse_dates=True)
    except:
        macro = None

    try:
        sentiment = pd.read_csv("data/raw/sentiment_data.csv", index_col=0, parse_dates=True)
    except:
        sentiment = None

    # Initialize feature engineer
    engineer = FeatureEngineer(use_gpu=True)

    # Build complete feature set
    features = engineer.build_complete_features(
        prices=prices,
        macro=macro,
        sentiment=sentiment,
        include_lags=True,
        include_cross_sectional=True,
    )

    # Save features
    engineer.save_features(
        features, output_path="data/processed/features_complete.parquet", format="parquet"
    )

    # Print summary
    print("\n" + "=" * 60)
    print("FEATURE SUMMARY")
    print("=" * 60)
    print(f"Total features: {len(features.columns)}")
    print("\nFeature categories:")

    categories = {
        "Returns": len([c for c in features.columns if "return" in c]),
        "SMA": len([c for c in features.columns if "sma" in c]),
        "EMA": len([c for c in features.columns if "ema" in c]),
        "RSI": len([c for c in features.columns if "rsi" in c]),
        "MACD": len([c for c in features.columns if "macd" in c]),
        "Bollinger": len([c for c in features.columns if "bb" in c]),
        "Momentum": len([c for c in features.columns if "momentum" in c]),
        "Volatility": len([c for c in features.columns if "vol" in c]),
        "Correlation": len([c for c in features.columns if "corr" in c]),
        "Lags": len([c for c in features.columns if "lag" in c]),
        "Cross-sectional": len(
            [c for c in features.columns if any(x in c for x in ["rank", "zscore", "dev_mean"])]
        ),
        "Interaction": len([c for c in features.columns if "_x_" in c]),
    }

    for category, count in categories.items():
        if count > 0:
            print(f"  {category:20}: {count:4} features")

    print("=" * 60 + "\n")

    return features


if __name__ == "__main__":
    features = main()
