#!/usr/bin/env python3
"""
Quiet Money Engine — daily cross-sectional scorer (entrypoint).

Fetches price history for the universe, computes every signal, z-scores and
ranks them into a watchlist. Designed to run as a daily Render Cron Job after
market close. For now it prints the ranking; persistence to Postgres is the next
stage. Start with a SMALL universe to sanity-check, then expand.
"""
import os
import logging

from data_layer import get_price_history
from signals import SIGNALS
from scoring import score_universe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("scorer")

# Starter universe — keep it small first to confirm the machinery works, then
# grow it. (Later we auto-build this from the tradability-gated set + the names
# your insider worker is already flagging.)
UNIVERSE = [t.strip().upper() for t in os.getenv(
    "UNIVERSE", "AAPL,MSFT,NVDA,AMD,INTC,F,GM,PLUG,RIOT,SOFI"
).split(",") if t.strip()]


def build_universe_data(tickers: list[str]) -> dict:
    data = {}
    for t in tickers:
        bars = get_price_history(t, days=400)
        if bars:
            data[t] = {"bars": bars}
        else:
            log.warning("No price history for %s; skipping", t)
    return data


def main() -> None:
    log.info("Scoring %d tickers on signals: %s", len(UNIVERSE), ", ".join(SIGNALS))
    data = build_universe_data(UNIVERSE)
    if not data:
        log.error("No data fetched — check FMP_API_KEY")
        return
    ranked = score_universe(data, SIGNALS)
    log.info("=== Ranked watchlist (%d names) ===", len(ranked))
    for i, row in enumerate(ranked, 1):
        sig_str = " ".join(f"{n}={z:+.2f}" for n, z in row["signals"].items())
        log.info("%2d. %-6s  composite %+.2f | %s", i, row["ticker"], row["composite"], sig_str)


if __name__ == "__main__":
    main()
