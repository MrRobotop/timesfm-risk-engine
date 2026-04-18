import datetime
import numpy as np
import pandas as pd
import yfinance as yf

class MarketDataFetcher:
    def fetch_multivariate_data(self, primary_ticker: str, macro_tickers: list[str], days: int, interval: str = "1d", clip_limit: float = None) -> tuple[np.ndarray, dict]:
        # Calculate date range for the last 'days' days
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days)
        
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
        
        # Drop the new leading NaNs created by the 20-period rolling window
        df_consolidated = df_consolidated.dropna()
        
        if clip_limit is not None:
            df_consolidated['primary_volatility'] = df_consolidated['primary_volatility'].clip(upper=clip_limit)
        
        # Process individual macro arrays with Z-Score normalization
        macro_dict_out = {}
        for ticker in macro_tickers:
            series = df_consolidated[ticker]
            # Standardization: (x - mean) / std (Z-score)
            z_scored = (series - series.mean()) / (series.std() + 1e-9)
            macro_dict_out[ticker] = z_scored.to_numpy()
            
        primary_vol_array = df_consolidated['primary_volatility'].to_numpy()
        
        return primary_vol_array, macro_dict_out
