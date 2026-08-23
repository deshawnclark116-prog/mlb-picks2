"""
build.py - Bridges the ML API to GitHub Pages.
Pulls finished predictions + games from the live API and writes them as
static JSON into docs/, which the GitHub Action commits and GitHub Pages
serves. The app reads docs/predictions.json unchanged.

Run by .github/workflows/daily.yml (twice daily) or manually.
"""
import json, time, datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo
import urllib.request

# Migrated BACK to Render 2026-08-23 -- Cloud Run (used 2026-08-06 through
# 2026-08-23) turned out to have a recurring network-latency stall against
# MLB's Stats API: /run/now hung for 20+ minutes multiple times (2026-08-11,
# 2026-08-16, 2026-08-23), each time silently serving a stale morning cache
# all day since build.py's poll budget gives up long before the stall
# clears. Same head-to-head test on 2026-08-23 (Render finished a full
# 70-pick board including batter_hits/batter_home_runs in under 2 minutes
# while the Cloud Run run from the same moment was still stuck past 20
# minutes) confirmed Render doesn't have this problem. Render's billing
# lapse that originally motivated leaving (2026-08-06) has been resolved.
API_BASE = "https://prop-edge-api.onrender.com"
DOCS = Path("docs")
DOCS.mkdir(exist_ok=True)

ET = ZoneInfo("America/New_York")

# /run/now used to run synchronously and this script just called it then read
# /predictions. On a full slate that synchronous call can run past Render's
# platform request timeout and get killed outright (observed: hard 500 after
# ~59s), so /run/now is now fire-and-forget (returns immediately, finishes in
# a background thread) and must be polled via /run/now/status instead.
RUN_NOW_POLL_BUDGET_S = 100
RUN_NOW_POLL_INTERVAL_S = 4


def fetch(path, timeout=120):
    """GET JSON from the API. Long timeout because Render free tier
    can cold-start slowly, and /predictions may generate on the fly."""
    url = f"{API_BASE}{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "build-bridge/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def trigger_and_wait_for_run_now():
    """Kick off /run/now (background job on the server) and poll
    /run/now/status until it finishes or our budget runs out. Always safe to
    call even if a run is already in progress (server dedupes)."""
    fetch("/run/now")
    waited = 0
    while waited < RUN_NOW_POLL_BUDGET_S:
        time.sleep(RUN_NOW_POLL_INTERVAL_S)
        waited += RUN_NOW_POLL_INTERVAL_S
        try:
            status = fetch("/run/now/status")
        except Exception as e:
            print(f"  status poll failed: {e}")
            continue
        if not status.get("running"):
            print(f"  run/now finished after ~{waited}s: "
                  f"{status.get('last_result')} total={status.get('last_total')}")
            return
    print(f"  run/now still running after {RUN_NOW_POLL_BUDGET_S}s budget; "
          f"reading /predictions with whatever is currently cached")


def main():
    # The server decides what "today" means (and which games are pregame)
    # using Eastern time (today_et() in api.py). The runner's own clock is
    # UTC, which is a different calendar date for ~4h every night (8pm-
    # midnight ET) -- using it here mislabeled the last 1-2 runs of each ET
    # day under TOMORROW's filename instead of finishing out today's, and
    # made tomorrow's file start with a stale duplicate of today's board.
    today = dt.datetime.now(ET).date().isoformat()
    print(f"Building static JSON for {today} from {API_BASE}")

    # 1. Predictions — trigger a fresh run (background job), wait for it to
    # finish (bounded), then read whatever's in /predictions.
    preds = []
    try:
        trigger_and_wait_for_run_now()
        preds = fetch("/predictions")
        if not isinstance(preds, list):
            preds = []
        print(f"  Got {len(preds)} predictions")
    except Exception as e:
        print(f"  predictions fetch failed: {e}")
        # fall back to whatever the API already has
        try:
            preds = fetch("/predictions")
        except Exception as e2:
            print(f"  fallback also failed: {e2}")

    # 2. Games
    games = []
    try:
        games = fetch("/games")
        if not isinstance(games, list):
            games = []
        print(f"  Got {len(games)} games")
    except Exception as e:
        print(f"  games fetch failed: {e}")

    # 3. Record (for the app's Record tab)
    record = {}
    try:
        record = fetch("/record")
        print(f"  Got record")
    except Exception as e:
        print(f"  record fetch failed: {e}")

    # 4. Health
    health = {
        "status": "ok" if preds or games else "empty",
        "predictions_today": len(preds),
        "games_today": len(games),
        "last_updated": dt.datetime.now(dt.timezone.utc).isoformat(),
        "date": today,
    }

    # write the plain files the app reads (undated) + a dated archive copy
    (DOCS / "predictions.json").write_text(json.dumps(preds))
    (DOCS / f"predictions_{today}.json").write_text(json.dumps(preds))
    (DOCS / "games.json").write_text(json.dumps(games))
    (DOCS / "record.json").write_text(json.dumps(record))
    (DOCS / "health.json").write_text(json.dumps(health))

    print(f"  Wrote predictions.json ({len(preds)}), games.json ({len(games)}), "
          f"record.json, health.json")


if __name__ == "__main__":
    main()
