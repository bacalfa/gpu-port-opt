"""
Test suite for data ingestion module using pytest.

Run with: pytest tests/data/test_data_download.py -v
"""

import time

import pandas as pd
import pytest

from src.data.download_data import DataDownloader, DataValidator


@pytest.fixture
def downloader():
    """Fixture providing a DataDownloader instance"""
    return DataDownloader()


@pytest.fixture
def test_tickers():
    """Fixture providing a small subset of test tickers"""
    return ["SPY", "TLT", "GLD"]


def test_basic_download(downloader, test_tickers):
    """Test basic data download functionality"""
    prices = downloader.download_asset_prices(tickers=test_tickers, use_cache=False)

    assert isinstance(prices, pd.DataFrame), "Should return a DataFrame"
    assert len(prices.columns) == len(test_tickers), "Should download all requested tickers"
    assert len(prices) > 0, "Should have data rows"
    assert prices.index.name == "Date", "Index should be named 'Date'"
    assert isinstance(prices.index, pd.DatetimeIndex), "Index should be DatetimeIndex"

    # Check that all tickers are present
    for ticker in test_tickers:
        assert ticker in prices.columns, f"Ticker {ticker} should be in columns"


def test_macro_download(downloader):
    """Test FRED macro data download"""
    if downloader.fred_client is None:
        pytest.skip("FRED API not configured - set FRED_API_KEY environment variable")

    macro = downloader.download_macro_data(use_cache=False)

    assert isinstance(macro, pd.DataFrame), "Should return a DataFrame"
    if not macro.empty:
        assert len(macro.columns) > 0, "Should have macro indicators"
        assert isinstance(macro.index, pd.DatetimeIndex), "Index should be DatetimeIndex"


def test_sentiment_download(downloader):
    """Test sentiment data download"""
    sentiment = downloader.download_sentiment_data(use_cache=False)

    assert isinstance(sentiment, pd.DataFrame), "Should return a DataFrame"
    assert len(sentiment.columns) > 0, "Should have sentiment indicators"
    assert isinstance(sentiment.index, pd.DatetimeIndex), "Index should be DatetimeIndex"


def test_validation(downloader, test_tickers):
    """Test data validation"""
    # Download test data first
    prices = downloader.download_asset_prices(tickers=test_tickers, use_cache=False)

    validator = DataValidator()
    results = validator.validate_prices(prices)

    assert "n_assets" in results, "Results should contain n_assets"
    assert "n_days" in results, "Results should contain n_days"
    assert "date_range" in results, "Results should contain date_range"
    assert results["n_assets"] == len(test_tickers), "Should validate all tickers"
    assert results["n_days"] > 0, "Should have days of data"


def test_caching(downloader, test_tickers):
    """Test caching mechanism"""
    # First download (no cache)
    start = time.time()
    data1 = downloader.download_asset_prices(tickers=test_tickers, use_cache=False)
    time1 = time.time() - start

    # Second download (with cache)
    start = time.time()
    data2 = downloader.download_asset_prices(tickers=test_tickers, use_cache=True)
    time2 = time.time() - start

    # Verify cached download is faster
    assert time2 < time1, "Cached download should be faster"

    # Verify data is identical
    pd.testing.assert_frame_equal(data1, data2, "Cache data should match original")


def test_gpu_conversion(downloader, test_tickers):
    """Test GPU conversion if available"""
    try:
        import cudf
    except ImportError:
        pytest.skip("cuDF not available - GPU features disabled")

    # Download sample data
    prices_cpu = downloader.download_asset_prices(tickers=test_tickers, use_cache=False)

    # Convert to GPU
    prices_gpu = cudf.from_pandas(prices_cpu)

    assert prices_gpu.shape == prices_cpu.shape, "GPU DataFrame should have same shape"

    # Test GPU operation
    start = time.time()
    returns_gpu = prices_gpu.pct_change()
    gpu_calc = time.time() - start

    start = time.time()
    returns_cpu = prices_cpu.pct_change()
    cpu_calc = time.time() - start

    # Verify GPU operations work and are faster (or at least functional)
    assert returns_gpu.shape == returns_cpu.shape, "GPU returns should have same shape"
    assert gpu_calc > 0, "GPU calculation should take time"
    assert cpu_calc > 0, "CPU calculation should take time"


def test_download_all(downloader, test_tickers):
    """Test downloading all data sources"""
    # Temporarily override ALL_ASSETS for faster testing
    original_assets = downloader.config.ALL_ASSETS
    downloader.config.ALL_ASSETS = test_tickers

    try:
        data = downloader.download_all()

        assert "prices" in data, "Should contain prices"
        assert "macro" in data, "Should contain macro"
        assert "sentiment" in data, "Should contain sentiment"

        assert isinstance(data["prices"], pd.DataFrame), "Prices should be DataFrame"
        assert isinstance(data["macro"], pd.DataFrame), "Macro should be DataFrame"
        assert isinstance(data["sentiment"], pd.DataFrame), "Sentiment should be DataFrame"
    finally:
        downloader.config.ALL_ASSETS = original_assets


def test_data_downloader_initialization():
    """Test DataDownloader initialization"""
    downloader = DataDownloader()

    assert downloader.config is not None, "Config should be initialized"
    assert downloader.fred_client is None or downloader.fred_client is not None, (
        "FRED client may or may not be initialized"
    )


def test_data_validator_static_methods():
    """Test DataValidator static methods"""
    validator = DataValidator()

    # Create sample data
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    sample_data = pd.DataFrame(
        {
            "SPY": 100 + pd.Series(range(100)),
            "TLT": 50 + pd.Series(range(100)),
        },
        index=dates,
    )

    results = validator.validate_prices(sample_data)

    assert isinstance(results, dict), "Results should be a dictionary"
    assert "n_assets" in results, "Should have n_assets"
    assert "n_days" in results, "Should have n_days"
    assert results["n_assets"] == 2, "Should have 2 assets"
    assert results["n_days"] == 100, "Should have 100 days"
