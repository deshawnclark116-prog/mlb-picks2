#!/usr/bin/env python3
"""
Quiet Money Engine — cross-sectional scoring engine.

Computes each signal across the whole universe, z-scores each signal (so weak
signals combine on equal footing), averages them equal-weight (the robust
default — optimized weights rarely beat equal weight out of sample), and ranks.
Returns a ranked watchlist with each signal's z-score attached for transparency.
"""
import math
import logging

log = logging.getLogger("scoring")


def _zscores(raw: dict) -> dict:
    """raw: {ticker: value-or-None} -> {ticker: zscore} over the non-None values."""
    vals = [(t, v) for t, v in raw.items() if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if len(vals) < 2:
        return {t: 0.0 for t, _ in vals}
    xs = [v for _, v in vals]
    mean = sum(xs) / len(xs)
    std = math.sqrt(sum((x - mean) ** 2 for x in xs) / len(xs))
    if std == 0:
        return {t: 0.0 for t, _ in vals}
    return {t: (v - mean) / std for t, v in vals}


def score_universe(universe_data: dict, signals: dict, weights: dict | None = None) -> list[dict]:
    """universe_data: {ticker: data_dict}; signals: {name: fn}.
    Returns ranked list: [{ticker, composite, signals: {name: z}}, ...]."""
    # 1) z-score each signal across the universe
    z_by_signal = {}
    for name, fn in signals.items():
        raw = {}
        for ticker, data in universe_data.items():
            try:
                raw[ticker] = fn(data)
            except Exception:
                raw[ticker] = None
        z_by_signal[name] = _zscores(raw)

    # 2) composite = weighted mean of whatever z-scores a ticker actually has
    rows = []
    for ticker in universe_data:
        zs = {name: z_by_signal[name][ticker] for name in signals if ticker in z_by_signal[name]}
        if not zs:
            continue  # no signal could be computed for this name
        if weights:
            num = sum(weights.get(n, 1.0) * z for n, z in zs.items())
            den = sum(weights.get(n, 1.0) for n in zs)
            composite = num / den if den else 0.0
        else:
            composite = sum(zs.values()) / len(zs)  # equal weight
        rows.append({"ticker": ticker, "composite": composite, "signals": zs})

    rows.sort(key=lambda r: r["composite"], reverse=True)
    return rows
