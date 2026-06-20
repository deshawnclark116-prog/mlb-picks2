"""
lineupk.py - Opponent lineup strikeout expectation from INDIVIDUAL batters.
Sums each of the 9 hitters' own K-rate (vs the pitcher's handedness) into a
real strikeout expectation for the start. Individual batters, not team average.
"""
import time
import requests

MLB = "https://statsapi.mlb.com/api/v1"
MLB11 = "https://statsapi.mlb.com/api/v1.1"
LEAGUE_AVG_K = 0.22

S = requests.Session()
S.headers["User-Agent"] = "prop-edge-lineupk/1.0"


def _get(url, **params):
    for attempt in range(3):
        try:
            r = S.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(1.0 * (attempt + 1))
    return {}


def batter_k_rate_vs_hand(batter_id, throws, season):
    """A single batter's K-rate vs LHP (throws='L') or RHP ('R').
    Falls back to overall, then league avg. Returns (k_rate, had_sample)."""
    code = "vl" if throws == "L" else "vr"
    d = _get(f"{MLB}/people/{batter_id}/stats",
             stats="statSplits", sitCodes=code, group="hitting", season=season)
    try:
        st = d["stats"][0]["splits"][0]["stat"]
        pa = int(st.get("plateAppearances", 0) or 0)
        so = int(st.get("strikeOuts", 0) or 0)
        if pa >= 15:
            return so / pa, True
    except Exception:
        pass
    # fallback: overall season K-rate
    d2 = _get(f"{MLB}/people/{batter_id}/stats", stats="season",
              group="hitting", season=season)
    try:
        st = d2["stats"][0]["splits"][0]["stat"]
        pa = int(st.get("plateAppearances", 0) or 0)
        so = int(st.get("strikeOuts", 0) or 0)
        if pa >= 15:
            return so / pa, True
    except Exception:
        pass
    return LEAGUE_AVG_K, False


def lineup_k_expectation(opp_batter_ids, pitcher_throws, season, expected_bf):
    """Project Ks from the opposing lineup's individual K-rates.
    Returns (expected_ks_from_lineup, avg_lineup_k_rate, n_with_sample)."""
    if not opp_batter_ids:
        return None, None, 0
    rates = []
    n_sample = 0
    for bid in opp_batter_ids[:9]:
        kr, had = batter_k_rate_vs_hand(bid, pitcher_throws, season)
        rates.append(kr)
        if had:
            n_sample += 1
        time.sleep(0.06)
    if not rates:
        return None, None, 0
    avg_k = sum(rates) / len(rates)
    # the pitcher faces ~expected_bf batters; each PA's K chance ≈ the lineup's
    # average individual K-rate (cycling through the order). Expected Ks:
    expected_ks = avg_k * expected_bf
    return expected_ks, avg_k, n_sample


def get_pitcher_throws(pitcher_id):
    d = _get(f"{MLB}/people/{pitcher_id}")
    try:
        return d["people"][0]["pitchHand"]["code"]
    except Exception:
        return "R"


if __name__ == "__main__":
    # test on a posted lineup
    import datetime as dt
    from zoneinfo import ZoneInfo
    d = dt.datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    s = _get(f"{MLB}/schedule", sportId=1, date=d)
    games = [g["gamePk"] for day in s.get("dates", []) for g in day.get("games", [])]
    for pk in games:
        f = _get(f"{MLB11}/game/{pk}/feed/live")
        try:
            order = f["liveData"]["boxscore"]["teams"]["away"]["battingOrder"]
            if len(order) == 9:
                ek, avg, n = lineup_k_expectation(order, "R", 2026, 24)
                print(f"Game {pk}: lineup avg K-rate={avg:.3f}, "
                      f"expected Ks (24 BF) = {ek:.1f}, {n}/9 with sample")
                break
        except Exception:
            continue
