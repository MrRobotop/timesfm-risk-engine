import numpy as np

def calculate_portfolio_var(projected_volatility: float, portfolio_value: float, confidence_level: float = 0.95) -> float:
    """
    Translates forecasted volatility into a 1-day Value-at-Risk (VaR) metric.
    Formula: VaR = Portfolio Value * Expected Volatility * Z-Score(Confidence)
    Note: For 95% confidence, Z ~= 1.645.
    """
    # Map common confidence levels to Z-scores
    z_map = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}
    z_score = z_map.get(confidence_level, 1.645)
    
    # projected_volatility is the forecasted standard deviation of log returns
    return portfolio_value * projected_volatility * z_score

def assess_multivariate_risk(quantile_forecast: np.ndarray, historical_vol: np.ndarray, risk_threshold: float, z_threshold: float = 2.0, portfolio_value: float = 1000000.0, confidence_level: float = 0.95) -> dict:
    """
    Evaluates risk using a hybrid approach:
    1. Absolute volatility threshold.
    2. Adaptive Z-score relative to the last 90 logged periods.
    3. Portfolio VaR Exposure.
    """
    # 1. Extract the projected maximal volatility (final horizon, highest quantile)
    if quantile_forecast.ndim == 3:
        projected_max_volatility = float(quantile_forecast[0, -1, -1])
    elif quantile_forecast.ndim == 2:
        projected_max_volatility = float(quantile_forecast[-1, -1])
    else:
        projected_max_volatility = float(np.ravel(quantile_forecast)[-1])
        
    # 2. Adaptive Regime Scoring: Calculate Z-score relative to last 90 periods
    recent_vol = historical_vol[-90:] if len(historical_vol) >= 90 else historical_vol
    mean_vol = np.mean(recent_vol)
    std_vol = np.std(recent_vol) + 1e-9
    
    z_score = (projected_max_volatility - mean_vol) / std_vol
    
    # 3. Portfolio Exposure Synthesis
    var_exposure = calculate_portfolio_var(projected_max_volatility, portfolio_value, confidence_level)
    
    # 4. Hybrid Evaluation & Explainability
    z_breach = z_score > z_threshold
    abs_breach = projected_max_volatility > risk_threshold
    
    if z_breach and abs_breach:
        status = "DANGER: Macro-Conditioned Volatility Spike Expected"
        explanation = f"CRITICAL: Both relative Z-score ({z_score:.2f}) and absolute volatility ({projected_max_volatility:.4f}) exceed safety bounds."
    elif z_breach:
        status = "WARNING: Regime Shift Detected"
        explanation = f"CAUTION: Relative volatility ({z_score:.2f} Z) is unusual for this asset, though absolute levels remain below threshold."
    elif abs_breach:
        status = "WARNING: High Absolute Volatility"
        explanation = f"CAUTION: Volatility ({projected_max_volatility:.4f}) is high, but consistent with the recent 90-day regime."
    else:
        status = "NORMAL: Risk within tolerances"
        explanation = "All quantitative metrics are within historical and absolute safety bounds."
        
    return {
        "status": status,
        "explanation": explanation,
        "projected_max_volatility": projected_max_volatility,
        "z_score": z_score,
        "var_exposure": var_exposure,
        "confidence": confidence_level,
        "threshold": risk_threshold,
        "z_threshold": z_threshold
    }
