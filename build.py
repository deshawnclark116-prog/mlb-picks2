"""
build.py - Bridges the Render ML API to GitHub Pages.
Pulls finished predictions + games from the live API and writes them as
static JSON into docs/, which the GitHub Action commits and GitHub Pages
serves. The app reads docs/predictions.json unchanged.

Run by .github/workflows/daily.yml (twice daily) or manually.
"""
import json, datetime as dt
from pathlib import Path
import urllib.request

API_BASE = "https://prop-edge-api.onrender.com"
DOCS = Path("docs")
DOCS.mkdir(exist_ok=True)


def fetch(path, timeout=120):
    """GET JSON from the API. Long timeout because Render free tier
    can cold-start slowly, and /predictions may generate on the fly."""
    url = f"{API_BASE}{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "build-bridge/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main():
    today = dt.date.today().isoformat()
    print(f"Building static JSON for {today} from {API_BASE}")

    # 1. Predictions — trigger a fresh run, then read them
    preds = []
    try:
        # /run/now generates + returns counts; then /predictions has the data
        fetch("/run/now")
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
