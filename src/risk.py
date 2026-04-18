import numpy as np
import pandas as pd

def calculate_historical_performance(price_series: pd.Series) -> dict:
    """
    Computes trailing returns and maximum drawdown.
    Assumes approx 252 trading days per year and 21 per month.
    """
    if len(price_series) < 2:
        return {}
        
    latest = price_series.iloc[-1]
    
    # Trailing Returns
    ret_1d = (latest / price_series.iloc[-2]) - 1 if len(price_series) >= 2 else 0.0
    ret_1m = (latest / price_series.iloc[-21]) - 1 if len(price_series) >= 21 else None
    ret_1y = (latest / price_series.iloc[-252]) - 1 if len(price_series) >= 252 else None
    ret_2y = (latest / price_series.iloc[-504]) - 1 if len(price_series) >= 504 else None
    
    # Max Drawdown
    roll_max = price_series.cummax()
    drawdown = (price_series / roll_max) - 1.0
    max_dd = float(drawdown.min())
    
    return {
        "ret_1d": float(ret_1d),
        "ret_1m": float(ret_1m) if ret_1m is not None else "N/A",
        "ret_1y": float(ret_1y) if ret_1y is not None else "N/A",
        "ret_2y": float(ret_2y) if ret_2y is not None else "N/A",
        "max_drawdown": max_dd
    }

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

def calculate_forward_ratios(expected_return_daily: float, expected_volatility_daily: float, risk_free_rate: float = 0.04) -> dict:
    """
    Computes annualized Forward Sharpe and Sortino Ratios.
    Sortino uses volatility as a proxy for downside deviation.
    """
    mu_a = expected_return_daily * 252
    sigma_a = expected_volatility_daily * np.sqrt(252)
    
    if sigma_a < 1e-6:
        return {"sharpe": 0.0, "sortino": 0.0}
        
    sharpe = (mu_a - risk_free_rate) / sigma_a
    # Proxy Sortino: assuming symmetric distribution for zero-shot, sigma is proxy for downside
    sortino = (mu_a - risk_free_rate) / (sigma_a * 1.2) # Heuristic scaling
    
    return {"sharpe": float(sharpe), "sortino": float(sortino)}

def simulate_probability_of_ruin(mu: float, sigma: float, horizon: int, threshold: float = -0.05, num_paths: int = 1000) -> float:
    """
    Monte Carlo Simulation using Geometric Brownian Motion.
    Returns the probability of hitting a specific drawdown (default -5%) within the horizon.
    """
    # S_t = S_0 * exp((mu - 0.5*sigma^2)dt + sigma*sqrt(dt)*Z)
    # We simulate step-by-step to catch breaches within the horizon (path-dependent)
    dt = 1.0 # daily
    paths = np.ones((num_paths, horizon + 1))
    
    for t in range(1, horizon + 1):
        z = np.random.standard_normal(num_paths)
        growth = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        paths[:, t] = paths[:, t-1] * growth
        
    # Minimum value reached along each path
    min_values = np.min(paths, axis=1)
    ruined = np.sum(min_values <= (1.0 + threshold))
    
    return float(ruined / num_paths)

def assess_multivariate_risk(quantile_forecast: np.ndarray, historical_vol: np.ndarray, risk_threshold: float, z_threshold: float = 2.0, portfolio_value: float = 1000000.0, confidence_level: float = 0.95, expected_return_daily: float = 0.0, horizon: int = 10) -> dict:
    """
    Evaluates risk using a hybrid approach:
    1. Absolute volatility threshold.
    2. Adaptive Z-score relative to the last 90 logged periods.
    3. Portfolio VaR & CVaR Exposure.
    4. Actionable Kelly Sizing.
    5. Institutional Ratios & Monte Carlo Ruin.
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
    
    # 4. Institutional Ratios
    ratios = calculate_forward_ratios(expected_return_daily, projected_max_volatility)
    
    # 5. Monte Carlo Probabilities (5% Drawdown)
    prob_ruin_5pct = simulate_probability_of_ruin(expected_return_daily, projected_max_volatility, horizon, threshold=-0.05)
    
    # 6. Actionable Alpha: Kelly Sizing
    kelly_f = calculate_kelly(expected_return_daily, projected_max_volatility)
    
    # 7. Hybrid Evaluation & Explainability
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
        "sharpe_ratio": ratios["sharpe"],
        "sortino_ratio": ratios["sortino"],
        "prob_ruin_5pct": prob_ruin_5pct,
        "confidence": confidence_level,
        "threshold": risk_threshold,
        "z_threshold": z_threshold
    }
