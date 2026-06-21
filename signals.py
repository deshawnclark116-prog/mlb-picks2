#!/usr/bin/env python3
"""
Quiet Money Engine — signal library.

Each signal is a pure function: data dict -> float (higher = more bullish) or
None when there isn't enough data. The scoring engine z-scores each signal
across the universe, so raw units don't need to match between signals.

To add signals 2-6: write a function here, then register it in SIGNALS.
"""
TRADING_DAYS_YEAR = 252
TRADING_DAYS_MONTH = 21


def momentum_12_1(data: dict) -> float | None:
    """12-month price return skipping the most recent month (classic 12-1
    momentum — the most replicated price signal). Skipping the last month
    avoids the short-term reversal that contaminates raw 12-month return."""
    bars = data.get("bars") or []
    if len(bars) < TRADING_DAYS_YEAR + 1:
        return None
    closes = [b["close"] for b in bars]
    start = closes[-(TRADING_DAYS_YEAR + 1)]   # ~12 months ago
    end = closes[-(TRADING_DAYS_MONTH + 1)]    # ~1 month ago
    if start <= 0:
        return None
    return end / start - 1.0


# >>> Signals 2-6 plug in here, same signature (data -> float|None):
#   high_volume_premium(data)    -> recent volume vs its own baseline
#   pead_surprise(data)          -> earnings surprise (needs earnings data)
#   estimate_revision(data)      -> analyst revision momentum (needs estimates)
#   insider_buy_strength(data)   -> from your Form 4 parser output
#   short_interest_pressure(data)-> FINRA short interest (contrarian/risk)

SIGNALS = {
    "momentum_12_1": momentum_12_1,
}
