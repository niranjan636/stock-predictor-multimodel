from datetime import date
import pandas as pd
import yfinance as yf

def get_data(ticker: str, start: date, end: date) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return df

    # If MultiIndex columns exist (happens with multiple tickers or some data sources)
    if isinstance(df.columns, pd.MultiIndex):
        # If it's a single ticker, just drop the second level
        if len(df.columns.levels[1]) == 1:
            df.columns = df.columns.droplevel(1)
        else:
            # Multi-ticker: flatten to strings like 'Close_AAPL'
            df.columns = ['_'.join([str(c) for c in col if c]) for col in df.columns]

    # Standardize column names to title case
    df.columns = [str(c).title() for c in df.columns]

    return df
