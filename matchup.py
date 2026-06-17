"""
matchup.py - Stage 1: real opponent-lineup strikeout rates.
Pulls the opponent's confirmed lineup and each hitter's K-rate split by the
pitcher's handedness. Feeds the Monte Carlo K simulator (Stages 3-4).
"""
import time, datetime as dt
import requests

MLB = "https://statsapi.mlb.com/api/v1"
MLB11 = "https://statsapi.mlb.com/api/v1.1"
SEASON = dt.date.today().year
LEAGUE_AVG_K = 0.22

S = requests.Session()
S.headers["User-Agent"] = "prop-edge-matchup/1.0"


def _get(url, **params):
    for attempt in range(3):
        try:
            r = S.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"  retry {url}: {e}")
            time.sleep(1.2 * (attempt + 1))
    return {}


def get_pitcher_throws(pid):
    """Return 'L' or 'R' for the pitcher's throwing hand."""
    d = _get(f"{MLB}/people/{pid}")
    try:
        return d["people"][0]["pitchHand"]["code"]
    except Exception:
        return "R"


def get_batter_k_rate_vs(pid, hand):
    """Batter's K-rate vs LHP (hand='L') or RHP (hand='R').
    Falls back to overall K-rate if the split is thin/missing."""
    code = "vl" if hand == "L" else "vr"
    d = _get(f"{MLB}/people/{pid}/stats",
             stats="statSplits", sitCodes=code, group="hitting", season=SEASON)
    try:
        for s in d.get("stats", []):
            for sp in s.get("splits", []):
                st = sp.get("stat", {})
                pa = int(st.get("plateAppearances", 0) or 0)
                so = int(st.get("strikeOuts", 0) or 0)
                if pa >= 20:                       # enough sample to trust the split
                    return so / pa
    except Exception:
        pass
    # fallback: overall season K-rate
    d2 = _get(f"{MLB}/people/{pid}/stats", stats="season", group="hitting", season=SEASON)
    try:
        st = d2["stats"][0]["splits"][0]["stat"]
        pa = int(st.get("plateAppearances", 0) or 0)
        so = int(st.get("strikeOuts", 0) or 0)
        if pa >= 20:
            return so / pa
    except Exception:
        pass
    return LEAGUE_AVG_K                            # last resort: league average


def get_confirmed_lineup(game_pk):
    data = _get(f"{MLB11}/game/{game_pk}/feed/live")
    out = {"home": [], "away": []}
    try:
        teams = data["liveData"]["boxscore"]["teams"]
        for side in ("home", "away"):
            order = teams.get(side, {}).get("battingOrder", [])
            out[side] = [int(pid) for pid in order]
    except Exception as e:
        print(f"  lineup fetch failed for {game_pk}: {e}")
    return out


def opponent_lineup_k_rates(game_pk, pitcher_pid, pitcher_side):
    """For a pitcher, return the list of K-rates for the 9 opposing hitters,
    split by the pitcher's handedness. pitcher_side is 'home' or 'away'.
    Returns (k_rates_list, pitcher_throws)."""
    throws = get_pitcher_throws(pitcher_pid)
    lineup = get_confirmed_lineup(game_pk)
    opp_side = "away" if pitcher_side == "home" else "home"
    opp_batters = lineup.get(opp_side, [])
    k_rates = []
    for bpid in opp_batters[:9]:
        kr = get_batter_k_rate_vs(bpid, throws)
        k_rates.append(round(kr, 3))
        time.sleep(0.1)
    return k_rates, throws


if __name__ == "__main__":
    # TEST on a real game — pass game_pk, pitcher_id, and side as args
    import sys
    if len(sys.argv) >= 4:
        gpk, ppid, side = sys.argv[1], int(sys.argv[2]), sys.argv[3]
        rates, throws = opponent_lineup_k_rates(gpk, ppid, side)
        print(f"Pitcher throws: {throws}")
        print(f"Opponent lineup K-rates: {rates}")
        if rates:
            print(f"Average lineup K-rate: {round(sum(rates)/len(rates), 3)}")
            print(f"(League avg is {LEAGUE_AVG_K})")
    else:
        print("Usage: python matchup.py <game_pk> <pitcher_id> <home|away>")
