"""Index calculation engine.

Turns a rule (a function that maps price history to target weights) into a
daily index level, the way a rules-based index is calculated:

- Weights are decided on each rebalance date using only data up to that date.
- They are implemented `lag` trading days later, so there is no look-ahead.
- Between rebalances, holdings drift with market moves.
- Transaction costs are deducted in proportion to turnover.
- Anything not invested sits in cash.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each calendar month."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period("M")).max().values)


def run_index(
    prices: pd.DataFrame,
    weight_fn,
    cost_bps: float = 10.0,
    lag: int = 1,
    cash_rate: float = 0.0,
    base: float = 100.0,
) -> dict:
    """Calculate an index from prices and a weighting rule.

    weight_fn(history) receives prices up to and including the rebalance date
    and returns a Series of target weights (summing to at most 1), or None if
    there is not yet enough history.

    Returns a dict with:
      levels   - daily index level, starting at `base` on the first trade date
      weights  - target weights at each rebalance (rows = trade dates)
      turnover - one-way-sum turnover at each rebalance
    """
    dates = prices.index
    returns = prices.pct_change().fillna(0.0).to_numpy()
    signal_dates = set(month_end_dates(dates))

    holdings = np.zeros(prices.shape[1])
    cash = base
    pending = {}
    levels, weight_rows, turnover_rows = {}, {}, {}
    started = False

    for i, date in enumerate(dates):
        if i > 0:
            holdings = holdings * (1.0 + returns[i])
            cash *= 1.0 + cash_rate / TRADING_DAYS
        nav = holdings.sum() + cash

        if date in pending:  # implement the weights decided `lag` days ago
            target = pending.pop(date)
            turnover = np.abs(target - holdings / nav).sum()
            nav -= nav * turnover * cost_bps / 1e4
            holdings = target * nav
            cash = nav - holdings.sum()
            weight_rows[date] = target
            turnover_rows[date] = turnover
            started = True

        if date in signal_dates and i + lag < len(dates):
            w = weight_fn(prices.iloc[: i + 1])
            if w is not None:
                w = w.reindex(prices.columns).fillna(0.0).to_numpy(dtype=float)
                if (w < -1e-12).any() or w.sum() > 1.0 + 1e-9:
                    raise ValueError("weights must be long-only and sum to <= 1")
                pending[dates[i + lag]] = w

        if started:
            levels[date] = nav

    levels = pd.Series(levels, name="level")
    levels = levels / levels.iloc[0] * base
    return {
        "levels": levels,
        "weights": pd.DataFrame.from_dict(weight_rows, orient="index", columns=prices.columns),
        "turnover": pd.Series(turnover_rows, name="turnover"),
    }


def performance_stats(levels: pd.Series) -> dict:
    """Standard performance and risk statistics for an index level series."""
    rets = levels.pct_change().dropna()
    years = (levels.index[-1] - levels.index[0]).days / 365.25
    cagr = (levels.iloc[-1] / levels.iloc[0]) ** (1 / years) - 1
    vol = rets.std() * np.sqrt(TRADING_DAYS)
    drawdown = levels / levels.cummax() - 1
    max_dd = drawdown.min()
    return {
        "Annual return": cagr,
        "Annual volatility": vol,
        "Sharpe ratio (rf=0)": cagr / vol if vol > 0 else float("nan"),
        "Max drawdown": max_dd,
        "Calmar ratio": cagr / abs(max_dd) if max_dd < 0 else float("nan"),
    }
