"""Rebalance the Trading 212 PRACTICE account to the trend index weights.

Usage:
    python rebalance.py            # dry run: shows the trades, sends nothing
    python rebalance.py --execute  # sends the orders to the practice account

Safety rules built in:
- Practice (demo) account only. The script refuses to run against live.
- Dry run unless --execute is given.
- Stops if any order is already pending, so a second run cannot double up.
- Sells go before buys, buys are limited to the cash available, and a small
  cash buffer is kept back.
- Trades smaller than MIN_TRADE_PCT of the account are skipped (a no-trade
  band, which also cuts turnover).
- Orders are never retried automatically.
"""

import sys
import time
from datetime import datetime

import pandas as pd
import yfinance as yf

from qis_data import load_prices
from run_trend_index import OUTPUT_DIR, UNIVERSE
from run_trend_index_v2 import make_trend_rule
from t212_client import T212Client

# Yahoo Finance ticker -> Trading 212 ticker (GBP/GBX London listings)
T212_TICKERS = {
    "VWRL.L": "VWRLl_EQ",
    "IGLT.L": "IGLTl_EQ",
    "INXG.L": "INXGl_EQ",
    "SGLN.L": "SGLNl_EQ",
    "COMM.L": "COMMl_EQ",
}
CASH_BUFFER = 0.02     # keep 2% of the account in cash
MIN_TRADE_PCT = 0.01   # skip trades smaller than 1% of the account
QTY_DECIMALS = 2
PAUSE_SECONDS = 2      # gap between orders, to respect rate limits


def prices_in_pounds(prices: pd.DataFrame) -> pd.Series:
    """Latest price per share in pounds. London ETFs are often quoted in pence."""
    out = {}
    for ticker in prices.columns:
        currency = yf.Ticker(ticker).fast_info["currency"]
        last = float(prices[ticker].iloc[-1])
        if currency in ("GBp", "GBX"):
            out[ticker] = last / 100
        elif currency == "GBP":
            out[ticker] = last
        else:
            raise RuntimeError(f"{ticker} is quoted in {currency}, expected GBP or pence")
    return pd.Series(out)


def build_plan(weights, price_gbp, held_qty, account_value, free_cash) -> pd.DataFrame:
    """Work out the trades needed to reach the target weights."""
    investable = account_value * (1 - CASH_BUFFER)
    plan = pd.DataFrame({
        "t212_ticker": pd.Series(T212_TICKERS),
        "price_gbp": price_gbp,
        "held_qty": held_qty,
        "target_weight": weights,
    }).fillna({"held_qty": 0.0})
    plan["held_value"] = plan["held_qty"] * plan["price_gbp"]
    plan["target_value"] = plan["target_weight"] * investable
    plan["trade_value"] = plan["target_value"] - plan["held_value"]

    too_small = plan["trade_value"].abs() < MIN_TRADE_PCT * account_value
    plan.loc[too_small, "trade_value"] = 0.0

    # Never plan to spend more than free cash plus the proceeds of the sells
    buys = plan["trade_value"].clip(lower=0).sum()
    sells = -plan["trade_value"].clip(upper=0).sum()
    budget = free_cash + sells - CASH_BUFFER * account_value
    if buys > budget > 0:
        plan.loc[plan["trade_value"] > 0, "trade_value"] *= budget / buys
    elif buys > 0 and budget <= 0:
        plan.loc[plan["trade_value"] > 0, "trade_value"] = 0.0

    plan["trade_qty"] = (plan["trade_value"] / plan["price_gbp"]).round(QTY_DECIMALS)
    # Never sell more than is held
    plan["trade_qty"] = plan[["trade_qty"]].apply(
        lambda col: col.where(col >= -plan["held_qty"], -plan["held_qty"]))["trade_qty"]
    return plan


def main():
    execute = "--execute" in sys.argv
    client = T212Client()  # demo by default
    if client.live:
        sys.exit("This script only runs against the practice account.")

    pending = client.get_orders()
    if pending:
        sys.exit(f"{len(pending)} order(s) already pending. Wait for them to fill "
                 "or cancel them before rebalancing again.")

    prices = load_prices(UNIVERSE.keys(), refresh=True)
    weights = make_trend_rule()(prices)
    price_gbp = prices_in_pounds(prices)

    cash = client.get_cash()
    account_value, free_cash = float(cash["total"]), float(cash["free"])
    by_t212 = {p["ticker"]: float(p["quantity"]) for p in client.get_portfolio()}
    held_qty = pd.Series({y: by_t212.get(t, 0.0) for y, t in T212_TICKERS.items()})

    plan = build_plan(weights, price_gbp, held_qty, account_value, free_cash)

    print(f"\nPrices as of {prices.index[-1].date()} | account value "
          f"£{account_value:,.2f} | free cash £{free_cash:,.2f}\n")
    shown = plan.copy()
    shown.insert(0, "asset", [UNIVERSE[t] for t in shown.index])
    shown["target_weight"] = shown["target_weight"].map("{:.1%}".format)
    for col in ("price_gbp", "held_value", "target_value", "trade_value"):
        shown[col] = shown[col].map("£{:,.2f}".format)
    print(shown.to_string(index=False))
    print("\nCheck the price_gbp column looks like pounds per share before executing.")

    trades = plan[plan["trade_qty"] != 0].sort_values("trade_qty")  # sells first
    if trades.empty:
        print("\nNo trades needed.")
        return
    if not execute:
        print("\nDRY RUN: nothing was sent. Run with --execute to place these "
              "orders on the practice account.")
        return

    log = []
    for yahoo, row in trades.iterrows():
        try:
            order = client.place_market_order(row["t212_ticker"], float(row["trade_qty"]),
                                              dry_run=False)
            status, order_id = order.get("status"), order.get("id")
            print(f"Sent {row['t212_ticker']} qty {row['trade_qty']:+.2f} -> {status}")
        except Exception as exc:  # stop at the first failure; never retry
            print(f"FAILED {row['t212_ticker']}: {exc}\nStopping. Check the account "
                  "before running again.")
            log.append({"time": datetime.now(), "ticker": row["t212_ticker"],
                        "qty": row["trade_qty"], "status": f"FAILED: {exc}", "id": None})
            break
        log.append({"time": datetime.now(), "ticker": row["t212_ticker"],
                    "qty": row["trade_qty"], "status": status, "id": order_id})
        time.sleep(PAUSE_SECONDS)

    OUTPUT_DIR.mkdir(exist_ok=True)
    log_file = OUTPUT_DIR / "rebalance_log.csv"
    pd.DataFrame(log).to_csv(log_file, mode="a", index=False,
                             header=not log_file.exists())
    print(f"\nLogged {len(log)} order(s) to {log_file}")


if __name__ == "__main__":
    main()
