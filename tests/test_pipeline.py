import pytest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.data import MarketDataFetcher
from src.risk import assess_multivariate_risk

@patch('src.data.yf.download')
def test_fetch_multivariate_data(mock_download):
    # Test 1 (Data): Mock yfinance.download
    dates = pd.date_range("2024-01-01", periods=150, freq="B") # More for tail lookback
    columns = pd.MultiIndex.from_tuples([("Close", "SPY"), ("Close", "^VIX")])
    dummy_df = pd.DataFrame(
        np.column_stack((np.linspace(100.0, 150.0, 150), np.linspace(10.0, 20.0, 150))),
        index=dates,
        columns=columns
    )
    mock_download.return_value = dummy_df
    
    fetcher = MarketDataFetcher()
    primary_vol, primary_returns, macro_cov_dict, correlation_dict, primary_price = fetcher.fetch_multivariate_data("SPY", ["^VIX"], 50)
    
    assert isinstance(primary_vol, np.ndarray)
    assert isinstance(primary_returns, np.ndarray)
    assert isinstance(primary_price, pd.Series)
    assert isinstance(correlation_dict, dict)
    assert "^VIX" in correlation_dict
    assert isinstance(correlation_dict["^VIX"], float)

def test_assess_multivariate_risk():
    # Test 2 (Risk): Test Institutional metrics
    quantile_forecast = np.zeros((1, 10, 10))
    quantile_forecast[0, -1, -1] = 0.05 # Projected Max Vol
    
    historical_vol = np.ones(90) * 0.01
    historical_vol[0] = 0.011 # variance
    
    result = assess_multivariate_risk(
        quantile_forecast, historical_vol, 0.02, 2.0, 1000000.0, 0.95, 0.001, horizon=10
    )
    
    assert "sharpe_ratio" in result
    assert "sortino_ratio" in result
    assert "prob_ruin_5pct" in result
    assert 0.0 <= result["prob_ruin_5pct"] <= 1.0
