#!/usr/bin/env python3
"""
Quiet Money Engine — price/volume data layer.

Pulls daily OHLCV history from FMP (reuses FMP_API_KEY). This is the feed the
cross-sectional signals are computed from. Defensive across FMP's stable vs v3
endpoints, throttled to be polite, returns bars oldest -> newest.
"""
import os
import time
import logging

import requests

log = logging.getLogger("data_layer")

FMP_API_KEY = os.getenv("FMP_API_KEY", "")
_MIN_INTERVAL = float(os.getenv("FMP_MIN_INTERVAL", "0.2"))
_last = 0.0
_HTTP_TIMEOUT = 20


def _polite_get(url: str, params: dict) -> requests.Response:
    global _last
    wait = _MIN_INTERVAL - (time.monotonic() - _last)
    if wait > 0:
        time.sleep(wait)
    _last = time.monotonic()
    return requests.get(url, params=params, timeout=_HTTP_TIMEOUT)


def get_price_history(ticker: str, days: int = 400) -> list[dict]:
    """Return [{date, close, volume}, ...] oldest -> newest, or [] on failure."""
    if not FMP_API_KEY:
        log.warning("FMP_API_KEY not set")
        return []

    attempts = [
        ("https://financialmodelingprep.com/stable/historical-price-eod/full",
         {"symbol": ticker, "apikey": FMP_API_KEY}),
        ("https://financialmodelingprep.com/api/v3/historical-price-full/" + ticker,
         {"apikey": FMP_API_KEY, "timeseries": days}),
    ]
    for url, params in attempts:
        try:
            r = _polite_get(url, params)
            if r.status_code in (401, 403):
                log.warning("FMP rejected key (HTTP %s) — check FMP_API_KEY", r.status_code)
                return []
            if r.status_code != 200:
                continue
            data = r.json()
            rows = data.get("historical") if isinstance(data, dict) else data
            if not rows:
                continue
            bars = [
                {"date": row.get("date"),
                 "close": float(row.get("adjClose") or row.get("close") or 0),
                 "volume": float(row.get("volume") or 0)}
                for row in rows if row.get("date")
            ]
            bars = [b for b in bars if b["close"] > 0]
            bars.sort(key=lambda b: b["date"])  # oldest -> newest
            return bars[-days:]
        except (requests.RequestException, ValueError, TypeError, KeyError):
            continue
    return []
