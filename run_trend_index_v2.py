"""Volatility-Targeted Multi-Asset Trend Index, version 2.

Changes from version 1, each fixing a weakness seen in the first backtest:
  1. Blended trend signal. Instead of one 12-month test (fully in or out),
     each asset is scored by the share of 3, 6 and 12-month returns that are
     positive (0, 1/3, 2/3 or 1). Positions scale in and out gradually, and
     the index reacts faster to sharp reversals.
  2. Per-asset cap. No asset may exceed MAX_WEIGHT. Anything above the cap is
     held in cash, which stops one asset dominating the index.

The script also runs a robustness grid: the same rules under different
settings, to check the result does not depend on one lucky choice.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qis_data import load_prices
from qis_index import TRADING_DAYS, performance_stats, run_index
from run_trend_index import COST_BPS, OUTPUT_DIR, UNIVERSE, equal_weights
from run_trend_index import trend_weights as trend_weights_v1

# ----------------------------------------------------------------- settings
LOOKBACKS = (63, 126, 252)  # about 3, 6 and 12 months
VOL_WINDOW = 60
TARGET_VOL = 0.08
MAX_WEIGHT = 0.35
MIN_HISTORY = 252           # every variant starts on the same date


# -------------------------------------------------------------------- rules
def make_trend_rule(lookbacks=LOOKBACKS, target_vol=TARGET_VOL,
                    max_weight=MAX_WEIGHT, vol_window=VOL_WINDOW):
    """Build a weighting rule with the given settings."""

    def rule(history: pd.DataFrame):
        if len(history) <= MIN_HISTORY:
            return None

        last = history.iloc[-1]
        score = sum((last / history.iloc[-1 - lb] - 1 > 0).astype(float)
                    for lb in lookbacks) / len(lookbacks)

        recent = history.pct_change().iloc[-vol_window:]
        vol = recent.std() * np.sqrt(TRADING_DAYS)
        weights = (1 / vol) / (1 / vol).sum() * score
        if weights.sum() == 0:
            return weights  # nothing trending: 100% cash

        cov = recent.cov() * TRADING_DAYS
        port_vol = float(np.sqrt(weights @ cov @ weights))
        weights = (weights * target_vol / port_vol).clip(upper=max_weight)
        if weights.sum() > 1:
            weights = weights / weights.sum()
        return weights

    return rule


def summarise(result: dict) -> dict:
    """Performance statistics plus exposure, turnover and concentration."""
    stats = performance_stats(result["levels"])
    years = len(result["levels"]) / TRADING_DAYS
    stats["Average exposure"] = result["weights"].sum(axis=1).mean()
    stats["Annual turnover"] = result["turnover"].sum() / years
    stats["Largest single weight"] = result["weights"].max().max()
    return stats


def show(table: pd.DataFrame) -> pd.DataFrame:
    """Format a statistics table: ratios to 2 decimals, the rest as percent."""
    out = table.copy().astype(object)
    for col in table.columns:
        fmt = "{:.2f}" if "ratio" in col.lower() else "{:.1%}"
        out[col] = [fmt.format(v) for v in table[col]]
    return out


# --------------------------------------------------------------------- main
def main():
    prices = load_prices(UNIVERSE.keys())
    print(f"Data: {prices.index[0].date()} to {prices.index[-1].date()}\n")

    runs = {
        "V2 (blend + cap)": run_index(prices, make_trend_rule(), cost_bps=COST_BPS),
        "V1 (12m only)": run_index(prices, trend_weights_v1, cost_bps=COST_BPS),
        "Equal weight": run_index(prices, equal_weights, cost_bps=COST_BPS),
    }
    comparison = pd.DataFrame({k: summarise(v) for k, v in runs.items()}).T
    print("=== Version comparison ===")
    print(show(comparison).T.to_string(), "\n")

    # Robustness grid: change one setting at a time
    variants = {"Base case": {}}
    for name, lbs in [("12m only", (252,)), ("6m only", (126,)),
                      ("3m only", (63,)), ("6m + 12m", (126, 252))]:
        variants[f"Lookback: {name}"] = {"lookbacks": lbs}
    for tv in (0.06, 0.10):
        variants[f"Target vol: {tv:.0%}"] = {"target_vol": tv}
    for cap in (0.25, 0.50):
        variants[f"Cap: {cap:.0%}"] = {"max_weight": cap}
    for vw in (30, 120):
        variants[f"Vol window: {vw}d"] = {"vol_window": vw}

    grid = pd.DataFrame({
        name: summarise(run_index(prices, make_trend_rule(**kw), cost_bps=COST_BPS))
        for name, kw in variants.items()
    }).T
    keep = ["Annual return", "Annual volatility", "Sharpe ratio (rf=0)",
            "Max drawdown", "Annual turnover"]
    print("=== Robustness grid (one setting changed at a time) ===")
    print(show(grid[keep]).to_string(), "\n")
    sharpe = grid["Sharpe ratio (rf=0)"]
    print(f"Sharpe range across variants: {sharpe.min():.2f} to {sharpe.max():.2f} "
          f"(equal weight: {comparison.loc['Equal weight', 'Sharpe ratio (rf=0)']:.2f})")

    v2 = runs["V2 (blend + cap)"]
    latest = v2["weights"].iloc[-1]
    print(f"\nLatest V2 target weights ({v2['weights'].index[-1].date()}):")
    for ticker, w in latest.items():
        print(f"  {ticker:8s} {UNIVERSE[ticker]:24s} {w:6.1%}")
    print(f"  {'Cash':33s} {1 - latest.sum():6.1%}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    comparison.to_csv(OUTPUT_DIR / "v2_comparison.csv")
    grid.to_csv(OUTPUT_DIR / "v2_robustness.csv")
    v2["weights"].to_csv(OUTPUT_DIR / "v2_weights.csv")
    pd.DataFrame({k: v["levels"] for k, v in runs.items()}).to_csv(
        OUTPUT_DIR / "v2_index_levels.csv")

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(10, 11), sharex=True,
        gridspec_kw={"height_ratios": [2, 1, 1]})
    for name, res in runs.items():
        ax1.plot(res["levels"], label=name)
    ax1.set_title("Volatility-Targeted Multi-Asset Trend Index v2 (base 100)")
    ax1.set_ylabel("Index level")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    for name, res in runs.items():
        dd = res["levels"] / res["levels"].cummax() - 1
        ax2.plot(dd, label=name)
    ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax2.grid(alpha=0.3)

    w = v2["weights"]
    ax3.stackplot(w.index, w.T.to_numpy(), step="post",
                  labels=[UNIVERSE[t] for t in w.columns])
    ax3.set_ylabel("V2 target weight")
    ax3.set_ylim(0, 1)
    ax3.grid(alpha=0.3)
    ax3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=5,
               fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "trend_index_v2.png", dpi=150)
    print(f"\nSaved chart and CSV files to the {OUTPUT_DIR} folder")


if __name__ == "__main__":
    main()
