import datetime
import numpy as np
import pandas as pd
import yfinance as yf

class MarketDataFetcher:
    def fetch_multivariate_data(self, primary_ticker: str, macro_tickers: list[str], days: int, interval: str = "1d", clip_limit: float = None) -> tuple[np.ndarray, np.ndarray, dict, dict, pd.Series]:
        # Calculate date range for the last 'days' days, ensuring at least 2 years for returns math
        end_date = datetime.datetime.now()
        fetch_days = max(days + 90, 750) # Guarantee enough context and trailing return window
        start_date = end_date - datetime.timedelta(days=fetch_days)
        
        # Download historical data simultaneously
        all_tickers = [primary_ticker] + macro_tickers
        df = yf.download(all_tickers, start=start_date, end=end_date, interval=interval)
        
        # Process close prices robustly to fix disjointed intraday trading hours
        if isinstance(df.columns, pd.MultiIndex):
            close_prices = df['Close']
        else:
            close_prices = df[['Close']]
            close_prices.columns = all_tickers
            
        # Clean infinities, ffill trailing NaNs from disjointed timestamps, and drop leading NaNs
        clean_df = close_prices.replace([np.inf, -np.inf], np.nan)
        clean_df = clean_df.ffill().dropna()
        
        primary_price = clean_df[primary_ticker]
        
        # Quant Math: Calculate the daily logarithmic returns efficiently without NaN bleed
        log_returns = np.log(primary_price / primary_price.shift(1))
        
        # Calculate the 20-period Exponentially Weighted Moving standard deviation of log returns
        primary_vol = log_returns.ewm(span=20, adjust=False).std()
        
        # Consolidate back into a unified frame
        df_consolidated = clean_df.copy()
        df_consolidated['primary_volatility'] = primary_vol
        df_consolidated['primary_returns'] = log_returns
        
        # Signal Strength: Calculate 60-day Pearson correlations
        correlation_dict = {}
        for ticker in macro_tickers:
            m_returns = np.log(clean_df[ticker] / clean_df[ticker].shift(1))
            corr = log_returns.tail(60).corr(m_returns.tail(60))
            correlation_dict[ticker] = float(corr) if not np.isnan(corr) else 0.0
        
        # Drop the new leading NaNs created by the 20-period rolling window and log returns shift
        df_consolidated = df_consolidated.dropna()
        
        # Store full primary price for historical performance calculation (2y+)
        full_price_series = df_consolidated['primary_returns'].cumsum().apply(np.exp) * primary_price.iloc[0] 
        # Actually simpler to just use the clean close prices directly
        full_close_prices = clean_df[primary_ticker]
        
        # Enforce lookback window 'days' from end for forecasting
        df_forecast_view = df_consolidated.tail(days)
        
        if clip_limit is not None:
            df_forecast_view['primary_volatility'] = df_forecast_view['primary_volatility'].clip(upper=clip_limit)
        
        # Process individual macro arrays with Z-Score normalization
        macro_dict_out = {}
        for ticker in macro_tickers:
            series = df_forecast_view[ticker]
            # Standardization: (x - mean) / std (Z-score)
            z_scored = (series - series.mean()) / (series.std() + 1e-9)
            macro_dict_out[ticker] = z_scored.to_numpy()
            
        primary_vol_array = df_forecast_view['primary_volatility'].to_numpy()
        primary_returns_array = df_forecast_view['primary_returns'].to_numpy()
        
        return primary_vol_array, primary_returns_array, macro_dict_out, correlation_dict, full_close_prices
