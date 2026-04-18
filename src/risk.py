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

def calculate_cvar(projected_volatility: float, portfolio_value: float, confidence_level: float = 0.95) -> float:
    """
    Calculates Gaussian Expected Shortfall (CVaR).
    Factors for 90%, 95%, 99%: 1.755, 2.063, 2.665
    """
    cvar_map = {0.90: 1.755, 0.95: 2.063, 0.99: 2.665}
    factor = cvar_map.get(confidence_level, 2.063)
    return portfolio_value * projected_volatility * factor

def calculate_kelly(expected_return_daily: float, expected_volatility_daily: float, risk_free_rate: float = 0.04) -> float:
    """
    Calculates the continuous Kelly Criterion optimal fraction.
    f* = (mu - r) / sigma^2
    All rates are annualized for calculation.
    """
    mu_a = expected_return_daily * 252
    sigma_a = expected_volatility_daily * np.sqrt(252)
    
    if sigma_a < 1e-6:
        return 0.0
        
    kelly_f = (mu_a - risk_free_rate) / (sigma_a**2)
    return float(np.clip(kelly_f, 0.0, 1.0)) # Limit to no-shorting, max 1x leverage for safety

def assess_multivariate_risk(quantile_forecast: np.ndarray, historical_vol: np.ndarray, risk_threshold: float, z_threshold: float = 2.0, portfolio_value: float = 1000000.0, confidence_level: float = 0.95, expected_return_daily: float = 0.0) -> dict:
    """
    Evaluates risk using a hybrid approach:
    1. Absolute volatility threshold.
    2. Adaptive Z-score relative to the last 90 logged periods.
    3. Portfolio VaR & CVaR Exposure.
    4. Actionable Kelly Sizing.
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
    cvar_exposure = calculate_cvar(projected_max_volatility, portfolio_value, confidence_level)
    
    # 4. Actionable Alpha: Kelly Sizing
    kelly_f = calculate_kelly(expected_return_daily, projected_max_volatility)
    
    # 5. Hybrid Evaluation & Explainability
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
        "expected_return_daily": expected_return_daily,
        "z_score": z_score,
        "var_exposure": var_exposure,
        "cvar_exposure": cvar_exposure,
        "kelly_fraction": kelly_f,
        "confidence": confidence_level,
        "threshold": risk_threshold,
        "z_threshold": z_threshold
    }
