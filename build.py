#!/usr/bin/env python3
"""
build.py - Fetches real ML predictions from Render API
and writes them to GitHub Pages JSON files.
"""
import os, json, datetime as dt, requests

RENDER_URL = os.environ.get("RENDER_URL", "").rstrip("/")
APP_API_KEY = os.environ.get("APP_API_KEY", "")
HEADERS = {"X-API-Key": APP_API_KEY}


def fetch(path):
    try:
        r = requests.get(f"{RENDER_URL}{path}", headers=HEADERS, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  Failed {path}: {e}")
        return None


def main():
    os.makedirs("docs", exist_ok=True)

    if not RENDER_URL:
        print("No RENDER_URL set — skipping.")
        return

    print("Triggering daily predictions on Render...")
    result = fetch("/run/daily")
    print(f"  {result}")

    print("Fetching predictions...")
    preds = fetch("/predictions") or []

    print("Fetching games...")
    games = fetch("/games") or []

    print("Fetching record...")
    record = fetch("/record") or {
        "summary": {"total": 0, "hits": 0, "misses": 0, "hit_rate": 0},
        "by_prop": {}, "by_confidence": {}, "results": [],
    }

    today = dt.date.today().isoformat()
    health = {
        "status": "ok" if preds else "no_predictions",
        "predictions_today": len(preds),
        "games_today": len(games),
        "last_updated": dt.datetime.now(dt.timezone.utc).isoformat(),
        "date": today,
    }

    json.dump(preds,   open("docs/predictions.json",  "w"), indent=2)
    json.dump(preds,   open(f"docs/predictions_{today}.json", "w"), indent=2)
    json.dump(games,   open("docs/games.json",         "w"), indent=2)
    json.dump(health,  open("docs/health.json",        "w"), indent=2)
    json.dump(record,  open("docs/record.json",        "w"), indent=2)

    print(f"Done. {len(preds)} predictions, {len(games)} games.")


if __name__ == "__main__":
    main()
