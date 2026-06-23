"""
lineupk.py - Opponent lineup strikeout expectation, HEAD-TO-HEAD first.
Per batter: blend their actual K-rate vs THIS pitcher (head-to-head) with their
general K-rate vs the pitcher's handedness. Head-to-head leads (40%), general
steadies (60%). No head-to-head history -> fall back to the general rate.
Sum the 9 batters -> the lineup's strikeout expectation.

Tested across 14+ games: this weighting gave avg bias +0.28, avg miss 1.56 Ks.
Weights live at the top so they're easy to tweak as the record grows.
"""
import time
import requests

MLB = "https://statsapi.mlb.com/api/v1"
MLB11 = "https://statsapi.mlb.com/api/v1.1"
LEAGUE_AVG_K = 0.22

# --- tunable weights (we'll likely tweak these as results come in) ---
H2H_WEIGHT = 0.40      # weight on head-to-head (batter vs THIS pitcher)
GEN_WEIGHT = 0.60      # weight on general K-rate vs handedness
MIN_H2H_PA = 2         # minimum head-to-head PA to use it at all (work with what we have)

S = requests.Session()
S.headers["User-Agent"] = "prop-edge-lineupk/2.0"


def _get(url, **params):
    for attempt in range(3):
        try:
            r = S.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(1.0 * (attempt + 1))
    return {}


def general_k_rate_vs_hand(batter_id, throws, season):
    """Batter's general K-rate vs LHP/RHP. Returns (k_rate, had_sample)."""
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


def head_to_head_k_rate(batter_id, pitcher_id):
    """Batter's actual K-rate vs THIS pitcher. Returns (k_rate, pa) or (None, 0)."""
    d = _get(f"{MLB}/people/{batter_id}/stats",
             stats="vsPlayer", opposingPlayerId=pitcher_id, group="hitting")
    pa_tot = 0
    k_tot = 0
    try:
        for s in d["stats"][0]["splits"]:
            if s.get("stat"):
                st = s["stat"]
                pa_tot += int(st.get("plateAppearances", 0) or 0)
                k_tot += int(st.get("strikeOuts", 0) or 0)
    except Exception:
        pass
    if pa_tot >= MIN_H2H_PA:
        return k_tot / pa_tot, pa_tot
    return None, 0


def blended_batter_k_rate(batter_id, pitcher_id, throws, season):
    """Head-to-head leads, general steadies. No head-to-head -> general only.
    Returns (k_rate, used_h2h)."""
    gen_kr, _ = general_k_rate_vs_hand(batter_id, throws, season)
    h2h_kr, h2h_pa = head_to_head_k_rate(batter_id, pitcher_id)
    if h2h_kr is not None:
        rate = H2H_WEIGHT * h2h_kr + GEN_WEIGHT * gen_kr
        return rate, True
    return gen_kr, False


def lineup_k_expectation(opp_batter_ids, pitcher_throws, season, expected_bf,
                         pitcher_id=None):
    """Sum the 9 batters' blended (head-to-head + general) K-rates into the
    lineup's expected strikeouts. Returns (expected_ks, avg_k_rate, n_with_data).
    If pitcher_id is None, falls back to general-only (no head-to-head)."""
    if not opp_batter_ids:
        return None, None, 0
    rates = []
    n_data = 0
    for bid in opp_batter_ids[:9]:
        if pitcher_id is not None:
            kr, used = blended_batter_k_rate(bid, pitcher_id, pitcher_throws, season)
        else:
            kr, used = general_k_rate_vs_hand(bid, pitcher_throws, season)
        rates.append(kr)
        if used:
            n_data += 1
        time.sleep(0.06)
    if not rates:
        return None, None, 0
    avg_k = sum(rates) / len(rates)
    expected_ks = avg_k * expected_bf
    return expected_ks, avg_k, n_data


def get_pitcher_throws(pitcher_id):
    d = _get(f"{MLB}/people/{pitcher_id}")
    try:
        return d["people"][0]["pitchHand"]["code"]
    except Exception:
        return "R"


if __name__ == "__main__":
    # test on the Woodruff/Reds game that started all this
    wood = 605540
    f = _get(f"{MLB11}/game/824502/feed/live")
    try:
        order = f["liveData"]["boxscore"]["teams"]["home"]["battingOrder"][:9]
        throws = get_pitcher_throws(wood)
        ek, avg, n = lineup_k_expectation(order, throws, 2026, 24, pitcher_id=wood)
        print(f"Woodruff/Reds: lineup avg K-rate={avg:.3f}, "
              f"expected Ks={ek:.1f}, {n}/9 with data (actual: he threw 10)")
        # compare general-only
        ek2, avg2, _ = lineup_k_expectation(order, throws, 2026, 24, pitcher_id=None)
        print(f"  general-only would be: {ek2:.1f} Ks")
    except Exception as e:
        print(f"test error: {e}")
