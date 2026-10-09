"""Minimal Trading 212 Public API (v0) client.

Design choices:
- Defaults to the DEMO (practice) environment. Live must be requested explicitly.
- Order placement defaults to dry_run=True, so nothing is sent unless you say so.
- GET requests are retried on rate limits (HTTP 429). POST requests are NEVER
  retried automatically, because order placement is not idempotent in the beta
  API and a retry could double-fill.
- Credentials are read from a local .env file that is excluded from Git.
"""

import os
import time

import requests
from dotenv import load_dotenv

DEMO_URL = "https://demo.trading212.com/api/v0"
LIVE_URL = "https://live.trading212.com/api/v0"


class T212Error(Exception):
    """Raised when the API returns an error response."""


class T212Client:
    def __init__(self, live: bool = False, timeout: int = 15):
        load_dotenv()
        key = os.environ["T212_API_KEY"]
        secret = os.getenv("T212_API_SECRET")

        self.live = live
        self.base = LIVE_URL if live else DEMO_URL
        self.timeout = timeout
        self.session = requests.Session()
        if secret:  # newer keys: key + secret via basic auth
            self.session.auth = (key, secret)
        else:  # older keys: key only
            self.session.headers["Authorization"] = key

    # ------------------------------------------------------------------ core
    def _request(self, method: str, path: str, max_retries: int = 2, **kwargs):
        url = f"{self.base}{path}"
        retries = max_retries if method == "GET" else 0
        for attempt in range(retries + 1):
            r = self.session.request(method, url, timeout=self.timeout, **kwargs)
            if r.status_code == 429 and attempt < retries:
                time.sleep(self._wait_seconds(r))
                continue
            if not r.ok:
                raise T212Error(f"{method} {path} -> {r.status_code}: {r.text}")
            return r.json() if r.text else None

    @staticmethod
    def _wait_seconds(response) -> float:
        """Seconds until the rate limit resets, capped at 60."""
        reset = response.headers.get("x-ratelimit-reset")
        if reset:
            try:
                return min(max(float(reset) - time.time(), 1.0), 60.0)
            except ValueError:
                pass
        return 5.0

    # --------------------------------------------------------------- account
    def get_cash(self) -> dict:
        return self._request("GET", "/equity/account/cash")

    def get_account_info(self) -> dict:
        return self._request("GET", "/equity/account/info")

    # ------------------------------------------------------------- portfolio
    def get_portfolio(self) -> list:
        """All open positions."""
        return self._request("GET", "/equity/portfolio")

    def get_position(self, ticker: str) -> dict:
        return self._request("GET", f"/equity/portfolio/{ticker}")

    # -------------------------------------------------------------- metadata
    def get_instruments(self) -> list:
        """Full tradable instrument list. Heavily rate limited: cache the result."""
        return self._request("GET", "/equity/metadata/instruments")

    def get_exchanges(self) -> list:
        return self._request("GET", "/equity/metadata/exchanges")

    # ---------------------------------------------------------------- orders
    def get_orders(self) -> list:
        """Pending (unfilled) orders."""
        return self._request("GET", "/equity/orders")

    def get_order(self, order_id: int) -> dict:
        return self._request("GET", f"/equity/orders/{order_id}")

    def cancel_order(self, order_id: int):
        return self._request("DELETE", f"/equity/orders/{order_id}")

    def place_market_order(self, ticker: str, quantity: float, dry_run: bool = True):
        """Place a market order. Positive quantity buys, negative sells.

        With dry_run=True (the default) the order is only printed, not sent.
        """
        if quantity == 0:
            raise ValueError("quantity must be non-zero")
        payload = {"ticker": ticker, "quantity": round(quantity, 6)}
        env = "LIVE" if self.live else "DEMO"
        if dry_run:
            print(f"[DRY RUN | {env}] market order: {payload}")
            return None
        return self._request("POST", "/equity/orders/market", json=payload)

    # --------------------------------------------------------------- history
    def get_order_history(self, limit: int = 50) -> dict:
        return self._request("GET", "/equity/history/orders", params={"limit": limit})

    def get_dividends(self, limit: int = 50) -> dict:
        return self._request("GET", "/equity/history/dividends", params={"limit": limit})
