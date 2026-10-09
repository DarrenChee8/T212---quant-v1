"""Volatility-Targeted Multi-Asset Trend Index: backtest.

Rules (decided at each month end, implemented the next trading day):
  1. Risk budget: each asset gets a weight proportional to 1 / its volatility.
  2. Trend filter: an asset keeps its weight only if its 12-month return is
     positive. Otherwise that weight is held in cash.
  3. Volatility target: the remaining weights are scaled so that expected
     portfolio volatility equals the target, with total exposure capped at
     100% (long-only, no leverage).

Benchmark: the same assets, equally weighted, rebalanced monthly.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write the chart to a file; no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qis_data import load_prices
from qis_index import TRADING_DAYS, performance_stats, run_index

# ----------------------------------------------------------------- settings
UNIVERSE = {
    "VWRL.L": "Global equities",
    "IGLT.L": "UK gilts",
    "INXG.L": "UK index-linked gilts",
    "SGLN.L": "Gold",
    "COMM.L": "Broad commodities",
}
TREND_LOOKBACK = 252   # trading days (about 12 months)
VOL_WINDOW = 60        # trading days used to estimate risk
TARGET_VOL = 0.08      # 8% a year
COST_BPS = 10          # cost per unit of turnover, in basis points
OUTPUT_DIR = Path("output")


# -------------------------------------------------------------------- rules
def trend_weights(history: pd.DataFrame):
    """Target weights for the trend index, or None if history is too short."""
    if len(history) <= TREND_LOOKBACK:
        return None

    momentum = history.iloc[-1] / history.iloc[-1 - TREND_LOOKBACK] - 1
    recent = history.pct_change().iloc[-VOL_WINDOW:]
    vol = recent.std() * np.sqrt(TRADING_DAYS)

    risk_budget = (1 / vol) / (1 / vol).sum()
    weights = risk_budget * (momentum > 0)
    if weights.sum() == 0:
        return weights  # nothing is trending: 100% cash

    cov = recent.cov() * TRADING_DAYS
    port_vol = float(np.sqrt(weights @ cov @ weights))
    weights = weights * (TARGET_VOL / port_vol)
    if weights.sum() > 1:
        weights = weights / weights.sum()
    return weights


def equal_weights(history: pd.DataFrame):
    """Benchmark: equal weight in every asset."""
    if len(history) <= TREND_LOOKBACK:
        return None
    n = history.shape[1]
    return pd.Series(1 / n, index=history.columns)


# --------------------------------------------------------------------- main
def main():
    prices = load_prices(UNIVERSE.keys())
    missing = [t for t in UNIVERSE if t not in prices.columns]
    if missing:
        print("Warning, no data for:", missing)
    print(f"Data: {prices.index[0].date()} to {prices.index[-1].date()}, "
          f"{prices.shape[1]} assets\n")

    trend = run_index(prices, trend_weights, cost_bps=COST_BPS)
    bench = run_index(prices, equal_weights, cost_bps=COST_BPS)

    stats = pd.DataFrame({
        "Trend index": performance_stats(trend["levels"]),
        "Equal weight": performance_stats(bench["levels"]),
    })
    exposure = trend["weights"].sum(axis=1)
    years = len(trend["levels"]) / TRADING_DAYS
    stats.loc["Average exposure"] = [exposure.mean(), bench["weights"].sum(axis=1).mean()]
    stats.loc["Annual turnover"] = [trend["turnover"].sum() / years,
                                    bench["turnover"].sum() / years]

    pct_rows = [r for r in stats.index if "ratio" not in r.lower()]
    shown = stats.copy().astype(object)
    for row in stats.index:
        fmt = "{:.1%}" if row in pct_rows else "{:.2f}"
        shown.loc[row] = [fmt.format(v) for v in stats.loc[row]]
    print(shown.to_string())

    latest = trend["weights"].iloc[-1]
    print(f"\nLatest target weights ({trend['weights'].index[-1].date()}):")
    for ticker, w in latest.items():
        print(f"  {ticker:8s} {UNIVERSE[ticker]:24s} {w:6.1%}")
    print(f"  {'Cash':33s} {1 - latest.sum():6.1%}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    pd.DataFrame({"trend_index": trend["levels"],
                  "equal_weight": bench["levels"]}).to_csv(OUTPUT_DIR / "index_levels.csv")
    trend["weights"].to_csv(OUTPUT_DIR / "trend_weights.csv")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(trend["levels"], label="Trend index")
    ax1.plot(bench["levels"], label="Equal weight benchmark")
    ax1.set_title("Volatility-Targeted Multi-Asset Trend Index (base 100)")
    ax1.set_ylabel("Index level")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.stackplot(trend["weights"].index, trend["weights"].T.to_numpy(),
                  labels=[UNIVERSE[t] for t in trend["weights"].columns], step="post")
    ax2.set_ylabel("Target weight")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper left", fontsize=8, ncol=3)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "trend_index.png", dpi=150)
    print(f"\nSaved chart and CSV files to the {OUTPUT_DIR} folder")


if __name__ == "__main__":
    main()
