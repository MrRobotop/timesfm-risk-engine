import numpy as np
from src.data import MarketDataFetcher
from src.forecaster import RiskForecaster

df = MarketDataFetcher()
primary, cov = df.fetch_multivariate_data('SPY', ['^VIX'], 365, '1d')

f = RiskForecaster()
pf, qf = f.model.forecast_with_covariates(
    inputs=[primary * 1000],  # Scale it up massively
    dynamic_numerical_covariates={'^VIX': [np.concatenate([cov['^VIX'], np.full(10, cov['^VIX'][-1])])]}
)
print('Scaled PF:', pf)

f.model.compile(
    f.model.forecast_config,
)
