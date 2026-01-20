"""
Data Ingestion Module for Regime-Aware Portfolio Optimization
==============================================================

This module handles downloading and preprocessing of both endogenous (asset prices)
and exogenous (macro/sentiment) data for portfolio optimization.

Features:
- Multi-threaded data downloads for speed
- Automatic retry logic with exponential backoff
- Data validation and quality checks
- Caching to avoid redundant API calls
- GPU-accelerated preprocessing with cuDF
- Comprehensive logging

Author: Bruno Abreu Calfa
"""

import logging
import os
import pickle
import time
import warnings
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred

# GPU acceleration (fallback to pandas if GPU not available)
try:
    import cudf

    GPU_AVAILABLE = True
    print("✓ GPU acceleration enabled (cuDF)")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠ GPU not available, falling back to pandas")
    cudf = pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Load .env file
load_dotenv(Path.cwd() / ".env")


class DataConfig:
    """Configuration for data download"""

    # Data directories
    RAW_DATA_DIR = Path("data/raw")
    PROCESSED_DATA_DIR = Path("data/processed")
    CACHE_DIR = Path("data/cache")

    # Date range
    START_DATE = "2010-01-01"
    END_DATE = datetime.now().strftime("%Y-%m-%d")

    # Asset universe
    EQUITY_ETFS = [
        # Broad Market
        "SPY",  # S&P 500
        "QQQ",  # Nasdaq 100
        "IWM",  # Russell 2000 (Small Cap)
        "VTI",  # Total Stock Market
        "EFA",  # EAFE (International Developed)
        "EEM",  # Emerging Markets
        # Sector ETFs
        "XLK",  # Technology
        "XLF",  # Financials
        "XLE",  # Energy
        "XLV",  # Healthcare
        "XLI",  # Industrials
        "XLC",  # Communications
        "XLY",  # Consumer Discretionary
        "XLP",  # Consumer Staples
        "XLU",  # Utilities
        "XLRE",  # Real Estate
        "XLB",  # Materials
    ]

    FIXED_INCOME_ETFS = [
        "TLT",  # 20+ Year Treasury
        "IEF",  # 7-10 Year Treasury
        "SHY",  # 1-3 Year Treasury
        "LQD",  # Investment Grade Corporate
        "HYG",  # High Yield Corporate
        "TIP",  # TIPS (Inflation Protected)
        "MUB",  # Municipal Bonds
        "EMB",  # Emerging Market Bonds
    ]

    COMMODITIES_ETFS = [
        "GLD",  # Gold
        "SLV",  # Silver
        "USO",  # Oil
        "DBA",  # Agriculture
        "DBB",  # Base Metals
        "UNG",  # Natural Gas
    ]

    CURRENCY_ETFS = [
        "UUP",  # US Dollar Index
        "FXE",  # Euro
        "FXY",  # Japanese Yen
        "FXB",  # British Pound
        "FXA",  # Australian Dollar
    ]

    ALTERNATIVE_ETFS = [
        "VNQ",  # REITs
        "VIXY",  # VIX Short-Term Futures
    ]

    # Combine all assets
    ALL_ASSETS = (
        EQUITY_ETFS + FIXED_INCOME_ETFS + COMMODITIES_ETFS + CURRENCY_ETFS + ALTERNATIVE_ETFS
    )

    # FRED API economic indicators
    # You need to get a free API key from: https://fred.stlouisfed.org/docs/api/api_key.html
    FRED_API_KEY = os.getenv("FRED_API_KEY", "your_api_key_here")

    MACRO_INDICATORS = {
        # Interest Rates & Yields
        "DGS10": "10-Year Treasury Rate",
        "DGS2": "2-Year Treasury Rate",
        "DFF": "Federal Funds Rate",
        "T10Y2Y": "10Y-2Y Treasury Spread",
        "T10Y3M": "10Y-3M Treasury Spread",
        # Inflation
        "CPIAUCSL": "CPI (All Urban Consumers)",
        "PCEPI": "PCE Price Index",
        "CPILFESL": "Core CPI",
        # Employment
        "UNRATE": "Unemployment Rate",
        "PAYEMS": "Nonfarm Payrolls",
        "ICSA": "Initial Jobless Claims",
        # GDP & Growth
        "GDP": "Gross Domestic Product",
        "GDPC1": "Real GDP",
        "INDPRO": "Industrial Production Index",
        # Consumer & Business
        "UMCSENT": "U. of Michigan Consumer Sentiment",
        "RSXFS": "Retail Sales",
        "HOUST": "Housing Starts",
        # Credit & Money Supply
        "M2SL": "M2 Money Supply",
        "TOTALSL": "Total Consumer Credit",
        "DCOILWTICO": "WTI Crude Oil Price",
    }

    # Market sentiment indicators (from Yahoo Finance)
    SENTIMENT_TICKERS = {
        "^VIX": "CBOE Volatility Index",
        "^SKEW": "CBOE Skew Index",
        "^VXN": "Nasdaq Volatility",
    }

    # Cache settings
    CACHE_DAYS = 1  # Refresh data if older than N days
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds


class DataDownloader:
    """Handles downloading financial and economic data"""

    def __init__(self, config: DataConfig = None):
        self.config = config or DataConfig()
        self._setup_directories()
        self.fred_client = None

        # Initialize FRED API if key is available
        if self.config.FRED_API_KEY and self.config.FRED_API_KEY != "your_api_key_here":
            try:
                self.fred_client = Fred(api_key=self.config.FRED_API_KEY)
                logger.info("✓ FRED API initialized successfully")
            except Exception as e:
                logger.warning(f"⚠ FRED API initialization failed: {e}")
        else:
            logger.warning("⚠ FRED API key not set. Macro data download will be skipped.")
            logger.info("Get your free key at: https://fred.stlouisfed.org/docs/api/api_key.html")

    def _setup_directories(self):
        """Create necessary directories"""
        for directory in [
            self.config.RAW_DATA_DIR,
            self.config.PROCESSED_DATA_DIR,
            self.config.CACHE_DIR,
        ]:
            directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Directories created: {self.config.RAW_DATA_DIR}")

    def _get_cache_path(self, data_type: str) -> Path:
        """Get cache file path for a data type"""
        return self.config.CACHE_DIR / f"{data_type}_cache.pkl"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cached data is still valid"""
        if not cache_path.exists():
            return False

        # Check file age
        file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return file_age.days < self.config.CACHE_DAYS

    def _save_to_cache(self, data: pd.DataFrame, data_type: str):
        """Save data to cache"""
        cache_path = self._get_cache_path(data_type)
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"✓ Cached {data_type} data")

    def _load_from_cache(self, data_type: str) -> pd.DataFrame | None:
        """Load data from cache if valid"""
        cache_path = self._get_cache_path(data_type)

        if self._is_cache_valid(cache_path):
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            logger.info(f"✓ Loaded {data_type} from cache")
            return data
        return None

    @staticmethod
    def _download_ticker_worker(
        args: tuple[str, str, str, int, int],
    ) -> tuple[str, pd.Series | None]:
        """
        Worker function for downloading a single ticker in a separate process.
        Kept static to make it picklable for multiprocessing.
        """
        ticker, start_date, end_date, max_retries, retry_delay = args
        for attempt in range(max_retries):
            try:
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                )

                if data.empty:
                    logger.warning("No data for %s", ticker)
                    return ticker, None

                prices = data["Adj Close"] if "Adj Close" in data else data["Close"]
                prices.name = ticker

                return ticker, prices

            except Exception as e:  # noqa: BLE001
                if attempt < max_retries - 1:
                    logger.debug("Retry %s for %s: %s", attempt + 1, ticker, e)
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    logger.error(
                        "Failed to download %s after %s attempts",
                        ticker,
                        max_retries,
                    )
                    return ticker, None

    def download_asset_prices(
        self, tickers: list[str] = None, use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Download historical price data for assets using multiprocessing for parallelism.

        Args:
            tickers: List of ticker symbols (default: all assets from config)
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with adjusted close prices (columns = tickers)
        """
        tickers = tickers or self.config.ALL_ASSETS

        # Check cache
        if use_cache:
            cached_data = self._load_from_cache("asset_prices")
            if cached_data is not None:
                return cached_data

        logger.info("Downloading price data for %s assets (multiprocessing)...", len(tickers))

        # Prepare arguments for workers
        worker_args = [
            (
                ticker,
                self.config.START_DATE,
                self.config.END_DATE,
                self.config.MAX_RETRIES,
                self.config.RETRY_DELAY,
            )
            for ticker in tickers
        ]

        price_data: dict[str, pd.Series] = {}
        with Pool(processes=min(len(worker_args), os.cpu_count() or 1)) as pool:
            for ticker, prices in pool.imap_unordered(self._download_ticker_worker, worker_args):
                if prices is not None:
                    price_data[ticker] = prices

        if not price_data:
            raise ValueError("No price data downloaded successfully")

        df = pd.concat(price_data.values(), axis=1)
        df.index.name = "Date"

        df = df.sort_index()

        self._save_to_cache(df, "asset_prices")

        df.to_csv(self.config.RAW_DATA_DIR / "asset_prices.csv")

        logger.info("✓ Downloaded %s assets, %s days", len(df.columns), len(df))
        logger.info("  Date range: %s to %s", df.index[0], df.index[-1])
        logger.info("  Missing tickers: %s", set(tickers) - set(df.columns))

        return df

    def download_macro_data(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Download macroeconomic indicators from FRED

        Args:
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with macro indicators (Date x Indicator)
        """
        if self.fred_client is None:
            logger.warning("FRED API not initialized. Skipping macro data download.")
            return pd.DataFrame()

        # Check cache
        if use_cache:
            cached_data = self._load_from_cache("macro_data")
            if cached_data is not None:
                return cached_data

        logger.info(
            f"Downloading {len(self.config.MACRO_INDICATORS)} macro indicators from FRED..."
        )

        def download_indicator(series_id: str, name: str) -> tuple[str, pd.Series | None]:
            """Download single FRED series with retry logic"""
            for attempt in range(self.config.MAX_RETRIES):
                try:
                    data = self.fred_client.get_series(
                        series_id,
                        observation_start=self.config.START_DATE,
                        observation_end=self.config.END_DATE,
                    )

                    if data.empty:
                        logger.warning(f"No data for {series_id} ({name})")
                        return series_id, None

                    data.name = series_id
                    return series_id, data

                except Exception as e:
                    if attempt < self.config.MAX_RETRIES - 1:
                        logger.debug(f"Retry {attempt + 1} for {series_id}: {e}")
                        time.sleep(self.config.RETRY_DELAY * (attempt + 1))
                    else:
                        logger.error(f"Failed to download {series_id} ({name}): {e}")
                        return series_id, None

        # Download sequentially (FRED has relatively strict rate limits)
        macro_data = {}
        for series_id, name in self.config.MACRO_INDICATORS.items():
            series_id, data = download_indicator(series_id, name)
            if data is not None:
                macro_data[series_id] = data

        if not macro_data:
            logger.warning("No macro data downloaded")
            return pd.DataFrame()

        # Combine into DataFrame
        df = pd.DataFrame(macro_data)
        df.index.name = "Date"
        df = df.sort_index()

        # Save to cache and raw data
        self._save_to_cache(df, "macro_data")
        df.to_csv(self.config.RAW_DATA_DIR / "macro_data.csv")

        logger.info(f"✓ Downloaded {len(df.columns)} macro indicators, {len(df)} observations")
        logger.info(f"  Date range: {df.index[0]} to {df.index[-1]}")

        return df

    def download_sentiment_data(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Download market sentiment indicators (VIX, etc.)

        Args:
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with sentiment indicators
        """
        # Check cache
        if use_cache:
            cached_data = self._load_from_cache("sentiment_data")
            if cached_data is not None:
                return cached_data

        logger.info(f"Downloading {len(self.config.SENTIMENT_TICKERS)} sentiment indicators...")

        sentiment_data = {}

        for ticker, name in self.config.SENTIMENT_TICKERS.items():
            try:
                data = yf.download(
                    ticker,
                    start=self.config.START_DATE,
                    end=self.config.END_DATE,
                    progress=False,
                )

                if not data.empty:
                    # Use closing values
                    sentiment_data[ticker] = data["Close"]
                    logger.info(f"  ✓ {ticker}: {name}")
                else:
                    logger.warning(f"  ✗ {ticker}: No data")

            except Exception as e:
                logger.error(f"  ✗ {ticker}: {e}")

        if not sentiment_data:
            logger.warning("No sentiment data downloaded")
            return pd.DataFrame()

        # Combine into DataFrame
        df = pd.concat(sentiment_data.values(), axis=1)
        df.index.name = "Date"
        df = df.sort_index()

        # Save to cache and raw data
        self._save_to_cache(df, "sentiment_data")
        df.to_csv(self.config.RAW_DATA_DIR / "sentiment_data.csv")

        logger.info(f"✓ Downloaded {len(df.columns)} sentiment indicators")

        return df

    def download_all(self) -> dict[str, pd.DataFrame]:
        """
        Download all data sources

        Returns:
            Dictionary with keys: 'prices', 'macro', 'sentiment'
        """
        logger.info("=" * 60)
        logger.info("Starting complete data download pipeline")
        logger.info("=" * 60)

        data = {}

        # 1. Asset prices
        try:
            data["prices"] = self.download_asset_prices()
        except Exception as e:
            logger.error(f"Asset price download failed: {e}")
            raise

        # 2. Macro data
        try:
            data["macro"] = self.download_macro_data()
        except Exception as e:
            logger.warning(f"Macro data download failed: {e}")
            data["macro"] = pd.DataFrame()

        # 3. Sentiment data
        try:
            data["sentiment"] = self.download_sentiment_data()
        except Exception as e:
            logger.warning(f"Sentiment data download failed: {e}")
            data["sentiment"] = pd.DataFrame()

        logger.info("=" * 60)
        logger.info("Data download complete!")
        logger.info(f"  Prices: {data['prices'].shape}")
        logger.info(f"  Macro: {data['macro'].shape}")
        logger.info(f"  Sentiment: {data['sentiment'].shape}")
        logger.info("=" * 60)

        return data


class DataValidator:
    """Validates downloaded data quality"""

    @staticmethod
    def validate_prices(df: pd.DataFrame) -> dict[str, any]:
        """
        Validate price data quality

        Returns:
            Dictionary with validation results
        """
        results = {
            "n_assets": len(df.columns),
            "n_days": len(df),
            "date_range": (df.index[0], df.index[-1]),
            "missing_data": {},
            "data_issues": [],
        }

        # Check for missing data
        missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
        results["missing_data"] = missing_pct[missing_pct > 0].to_dict()

        # Check for suspicious values
        for col in df.columns:
            # Negative prices
            if (df[col] < 0).any():
                results["data_issues"].append(f"{col}: Contains negative prices")

            # Zero prices
            if (df[col] == 0).any():
                results["data_issues"].append(f"{col}: Contains zero prices")

            # Extreme returns (>50% daily)
            returns = df[col].pct_change()
            if (returns.abs() > 0.5).any():
                n_extreme = (returns.abs() > 0.5).sum()
                results["data_issues"].append(f"{col}: {n_extreme} extreme daily returns (>50%)")

        return results

    @staticmethod
    def print_validation_report(results: dict):
        """Print formatted validation report"""
        print("\n" + "=" * 60)
        print("DATA VALIDATION REPORT")
        print("=" * 60)
        print(f"Assets: {results['n_assets']}")
        print(f"Days: {results['n_days']}")
        print(f"Date Range: {results['date_range'][0]} to {results['date_range'][1]}")

        if results["missing_data"]:
            print("\nMissing Data (% of days):")
            for asset, pct in sorted(
                results["missing_data"].items(), key=lambda x: x[1], reverse=True
            )[:10]:
                print(f"  {asset}: {pct}%")
        else:
            print("\n✓ No missing data")

        if results["data_issues"]:
            print("\nData Quality Issues:")
            for issue in results["data_issues"][:10]:
                print(f"  ⚠ {issue}")
        else:
            print("\n✓ No data quality issues detected")

        print("=" * 60 + "\n")


def main():
    """Main execution function"""

    # Initialize downloader
    downloader = DataDownloader()

    # Download all data
    data = downloader.download_all()

    # Validate price data
    validator = DataValidator()
    validation_results = validator.validate_prices(data["prices"])
    validator.print_validation_report(validation_results)

    # Optional: Convert to GPU DataFrame for preprocessing
    if GPU_AVAILABLE:
        logger.info("Converting to GPU DataFrames (cuDF)...")
        try:
            gpu_prices = cudf.from_pandas(data["prices"])
            logger.info(f"✓ Prices on GPU: {gpu_prices.shape}")

            # Save GPU-ready data
            gpu_prices.to_parquet(DataConfig.PROCESSED_DATA_DIR / "prices_gpu.parquet")
            logger.info("✓ Saved GPU-ready parquet files")

        except Exception as e:
            logger.warning(f"GPU conversion failed: {e}")

    return data


if __name__ == "__main__":
    # Run data download
    data = main()
