"""Price data layer: download, clean and cache daily adjusted closing prices.

The Trading 212 API has no historical price feed, so research data comes from
Yahoo Finance (via yfinance). Prices are total-return adjusted (dividends
reinvested), cleaned for bad ticks, and cached locally in data/prices.csv.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_FILE = Path("data") / "prices.csv"


def _download_one(ticker: str, start: str):
    """Adjusted close for one ticker, or None if the download fails."""
    try:
        df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    except Exception as exc:  # network or symbol problems
        print(f"  ! {ticker}: download failed ({exc})")
        return None
    if df is None or df.empty:
        print(f"  ! {ticker}: no data returned")
        return None
    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # newer yfinance returns 2-level columns
        close = close.iloc[:, 0]
    return close.rename(ticker)


def clean_prices(prices: pd.DataFrame, spike: float = 0.20) -> pd.DataFrame:
    """Remove bad ticks and fill short gaps.

    A bad tick is a one-day move larger than `spike` that is immediately
    reversed by a move of similar size in the opposite direction. London-listed
    ETFs on Yahoo occasionally show these (for example pence/pound mix-ups).
    """
    cleaned = prices.copy()
    for col in cleaned.columns:
        s = cleaned[col]
        r_in = s.pct_change()
        r_out = s.pct_change().shift(-1)
        bad = (r_in.abs() > spike) & (r_out.abs() > spike) & (r_in * r_out < 0)
        if bad.any():
            print(f"  cleaned {int(bad.sum())} bad tick(s) in {col}")
            cleaned.loc[bad, col] = float("nan")
    return cleaned.ffill(limit=5)


def load_prices(tickers, start: str = "2010-01-01", refresh: bool = False) -> pd.DataFrame:
    """Daily adjusted closes for `tickers`, from cache if available.

    Rows before every ticker has data are dropped, so all series share a
    common start date. Pass refresh=True to force a fresh download.
    """
    tickers = list(tickers)
    if CACHE_FILE.exists() and not refresh:
        cached = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
        if set(tickers) <= set(cached.columns):
            return cached[tickers]

    print("Downloading prices from Yahoo Finance...")
    series = [s for s in (_download_one(t, start) for t in tickers) if s is not None]
    if not series:
        raise RuntimeError("No price data could be downloaded.")

    prices = pd.concat(series, axis=1).sort_index()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = clean_prices(prices).dropna(how="any")

    CACHE_FILE.parent.mkdir(exist_ok=True)
    prices.to_csv(CACHE_FILE)
    print(f"Saved {len(prices)} days x {prices.shape[1]} assets to {CACHE_FILE}")
    return prices
