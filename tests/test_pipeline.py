import pytest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.data import MarketDataFetcher
from src.risk import assess_multivariate_risk

@patch('src.data.yf.download')
def test_fetch_multivariate_data(mock_download):
    # Test 1 (Data): Mock yfinance.download to return a dummy pandas DataFrame
    # with two ticker columns (SPY and ^VIX) containing 50 rows of sequential float values.
    dates = pd.date_range("2024-01-01", periods=50, freq="B")
    columns = pd.MultiIndex.from_tuples([("Close", "SPY"), ("Close", "^VIX")])
    dummy_df = pd.DataFrame(
        np.column_stack((np.linspace(100.0, 150.0, 50), np.linspace(10.0, 20.0, 50))),
        index=dates,
        columns=columns
    )
    mock_download.return_value = dummy_df
    
    fetcher = MarketDataFetcher()
    primary_vol, macro_cov_dict = fetcher.fetch_multivariate_data("SPY", ["^VIX"], 50)
    
    # Assert returns a dict for macro covariates
    assert isinstance(primary_vol, np.ndarray), "Returned primary_vol is not a numpy array"
    assert isinstance(macro_cov_dict, dict), "Returned covariates is not a dict"
    
    macro_cov = macro_cov_dict["^VIX"]
    assert isinstance(macro_cov, np.ndarray), "Returned macro_cov is not a numpy array"
    
    # Assert identical lengths
    assert len(primary_vol) == len(macro_cov), f"Length mismatch: {len(primary_vol)} vs {len(macro_cov)}"
    
    # Assert no NaNs
    assert not np.isnan(primary_vol).any(), "Found NaNs in primary_vol"
    assert not np.isnan(macro_cov).any(), "Found NaNs in macro_cov"


def test_assess_multivariate_risk():
    # Test 2 (Risk): Test the assessment logic without loading the TimesFM model
    
    # Create a dummy numpy array of shape (1, 10, 10) to mock the quantile_forecast
    quantile_forecast = np.zeros((1, 10, 10))
    
    # Set the value at [0, -1, -1] (the last horizon, highest quantile) to 0.05
    quantile_forecast[0, -1, -1] = 0.05
    
    # Create dummy historical_vol where 0.05 will be a high Z-score
    # Mean: 0.01, Std: ~0.001
    historical_vol = np.ones(90) * 0.01
    historical_vol[0] = 0.011 # minor variance
    
    # Scenario A: Both thresholds exceeded -> DANGER
    result = assess_multivariate_risk(
        quantile_forecast, 
        historical_vol, 
        risk_threshold=0.02, 
        z_threshold=2.0,
        portfolio_value=1000000.0
    )
    assert "DANGER" in result["status"], f"Expected 'DANGER' in status, got '{result['status']}'"
    assert result["z_score"] > 2.0
    assert result["var_95_exposure"] > 0
    
    # Scenario B: Absolute threshold exceeded but Z-score not (high historical vol)
    # Mean: 0.04, Std: 0.01
    high_historical_vol = np.ones(90) * 0.04
    high_historical_vol[::2] = 0.05
    high_historical_vol[1::2] = 0.03
    # Mean is 0.04, Std is 0.01. 0.05 is (0.05-0.04)/0.01 = 1.0 Z-score.
    result_high = assess_multivariate_risk(
        quantile_forecast, 
        high_historical_vol, 
        risk_threshold=0.02, 
        z_threshold=2.0,
        portfolio_value=1000000.0
    )
    assert "NORMAL" in result_high["status"], f"Expected 'NORMAL' due to Z-score {result_high['z_score']} <= 2.0"
    
    # Scenario C: Z-score exceeded but absolute threshold not
    low_forecast = np.zeros((1, 10, 10))
    low_forecast[0, -1, -1] = 0.015
    result_low_abs = assess_multivariate_risk(
        low_forecast, 
        historical_vol, 
        risk_threshold=0.02, 
        z_threshold=2.0,
        portfolio_value=1000000.0
    )
    assert "NORMAL" in result_low_abs["status"], "Expected 'NORMAL' due to low absolute volatility"
