"""
FANDUEL_LINES_A

FanDuel's public pregame player lines, shown next to our own projection
on the board (never used as a model input). Main lines only -- not the
alt ladders -- plus anytime-TD odds.

  fetch("nfl") -> {(normalized player name): {"rushing_yards": 72.5,
                   "receiving_yards": 48.5, "anytime_td_odds": -115, ...}}
"""
import re
import unicodedata
from datetime import datetime, timedelta, timezone

import requests

BASE = "https://sbapi.nj.sportsbook.fanduel.com/api"
AK = "FhMFpcPWXMeyZxOx"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
           "Accept": "application/json"}
PAGE = {"nfl": "nfl", "cfb": "ncaaf"}
LINE_RE = re.compile(r"^PLAYER_X_(RUSHING|RECEIVING|PASSING)_YARDS(_LOW|_MEDIUM|_HIGH)?$")
TABS = ("rushing-props", "receiving-props", "passing-props", "td-scorer-props")


def norm(n):
    n = unicodedata.normalize("NFKD", n or "")
    n = "".join(c for c in n if not unicodedata.combining(c)).lower()
    n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\.?\b", "", n)
    n = re.sub(r"[^a-z ]", "", n)
    return re.sub(r"\s+", " ", n).strip()


def _get(path, **params):
    r = requests.get(f"{BASE}/{path}", params={"_ak": AK, **params}, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()


def fetch(sport="nfl", days_ahead=7):
    out = {}
    now = datetime.now(timezone.utc)
    page = _get("content-managed-page", page="CUSTOM", customPageId=PAGE[sport])
    events = []
    for eid, e in page.get("attachments", {}).get("events", {}).items():
        if " @ " not in e.get("name", ""):
            continue
        try:
            t = datetime.fromisoformat(e["openDate"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            continue
        if now - timedelta(hours=4) <= t <= now + timedelta(days=days_ahead):
            events.append(eid)
    for eid in events:
        for tab in TABS:
            try:
                mk = _get("event-page", eventId=eid, tab=tab).get("attachments", {}).get("markets", {})
            except Exception as e:
                print(f"    fanduel: {eid}/{tab} failed ({e})")
                continue
            for m in mk.values():
                mt = m.get("marketType", "")
                lm = LINE_RE.match(mt)
                if lm:
                    player = m.get("marketName", "").split(" - ")[0]
                    over = next((r for r in m.get("runners", []) if r.get("runnerName", "").endswith("Over")), None)
                    if over is None or over.get("handicap") is None:
                        continue
                    odds = (over.get("winRunnerOdds", {}).get("americanDisplayOdds") or {}).get("americanOdds")
                    rec = out.setdefault(norm(player), {})
                    rec[f"{lm.group(1).lower()}_yards"] = float(over["handicap"])
                    rec[f"{lm.group(1).lower()}_yards_over_odds"] = odds
                elif mt == "ANY_TIME_TOUCHDOWN_SCORER":
                    for r in m.get("runners", []):
                        odds = (r.get("winRunnerOdds", {}).get("americanDisplayOdds") or {}).get("americanOdds")
                        if odds is not None:
                            out.setdefault(norm(r.get("runnerName")), {})["anytime_td_odds"] = odds
    print(f"    fanduel: {len(events)} {sport} events, {len(out)} players with lines")
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(fetch(), indent=1)[:2000])
