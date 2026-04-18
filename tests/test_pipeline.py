import pytest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.data import MarketDataFetcher
from src.risk import assess_multivariate_risk

@patch('src.data.yf.download')
def test_fetch_multivariate_data(mock_download):
    # Test 1 (Data): Mock yfinance.download
    dates = pd.date_range("2024-01-01", periods=50, freq="B")
    columns = pd.MultiIndex.from_tuples([("Close", "SPY"), ("Close", "^VIX")])
    dummy_df = pd.DataFrame(
        np.column_stack((np.linspace(100.0, 150.0, 50), np.linspace(10.0, 20.0, 50))),
        index=dates,
        columns=columns
    )
    mock_download.return_value = dummy_df
    
    fetcher = MarketDataFetcher()
    primary_vol, primary_returns, macro_cov_dict = fetcher.fetch_multivariate_data("SPY", ["^VIX"], 50)
    
    assert isinstance(primary_vol, np.ndarray)
    assert isinstance(primary_returns, np.ndarray)
    assert len(primary_vol) == len(primary_returns)
    assert not np.isnan(primary_returns).any()

def test_assess_multivariate_risk():
    # Test 2 (Risk): Test CVaR and Kelly logic
    quantile_forecast = np.zeros((1, 10, 10))
    quantile_forecast[0, -1, -1] = 0.05 # Projected Max Vol
    
    historical_vol = np.ones(90) * 0.01
    historical_vol[0] = 0.011 # variance
    
    # Positive Expected Return (Bullish)
    result = assess_multivariate_risk(
        quantile_forecast, historical_vol, 0.02, 2.0, 1000000.0, 0.95, 0.001
    )
    
    assert result["cvar_exposure"] > result["var_exposure"]
    assert result["kelly_fraction"] > 0
    assert result["kelly_fraction"] <= 1.0
    
    # Negative Expected Return (Bearish)
    result_bear = assess_multivariate_risk(
        quantile_forecast, historical_vol, 0.02, 2.0, 1000000.0, 0.95, -0.001
    )
    assert result_bear["kelly_fraction"] == 0.0 # No allocation for expected loss
