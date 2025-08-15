import numpy as np
import pandas as pd

def _as_close_series(obj: pd.Series | pd.DataFrame) -> pd.Series:
    """
    Return a 1-D float Series aligned to the original index,
    no matter if the input is a Series or a (n,1) DataFrame.
    """
    if isinstance(obj, pd.DataFrame):
        s = obj.squeeze("columns")          # (n,1) -> (n,)
    else:
        s = obj
    # Coerce to numeric float and ensure it's a pandas Series
    s = pd.to_numeric(s, errors="coerce").astype(float)
    if not isinstance(s, pd.Series):
        s = pd.Series(s, index=getattr(obj, "index", None), dtype=float)
    return s

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    s = _as_close_series(series)
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    roll_up = gain.rolling(period).mean()
    roll_down = loss.rolling(period).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature table from OHLCV dataframe with a robust 1-D Close series.
    Features: returns, SMA/EMA (5/10/20), RSI(14), lags(1/2/3/5)
    Target: next-day Close.
    """
    data = df.copy()
    close = _as_close_series(data["Close"])  # robust 1-D series

    data["Return"] = close.pct_change()

    for w in [5, 10, 20]:
        data[f"SMA{w}"] = close.rolling(w).mean()

    for w in [5, 10, 20]:
        data[f"EMA{w}"] = close.ewm(span=w, adjust=False).mean()

    data["RSI14"] = rsi(close, 14)

    for l in [1, 2, 3, 5]:
        data[f"Lag{l}"] = close.shift(l)

    # Next-day close as target
    data["Target"] = close.shift(-1)

    return data.dropna()

def train_test_split_time(data: pd.DataFrame, test_size: float = 0.2):
    """Time-based split (no shuffling)."""
    n = len(data)
    split = int(n * (1 - test_size))
    return data.iloc[:split], data.iloc[split:]
