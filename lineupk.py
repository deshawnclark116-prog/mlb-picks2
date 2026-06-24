"""
lineupk.py - Head-to-head engine helpers.
STRIKEOUTS: each batter's K-rate vs THIS pitcher (40%) + general vs handedness
(60%), summed across lineup.
HITS: sum-score method (validated 4-window sweep, ~1950 batters) - gen_avg +
h2h_avg RANKS hit likelihood better than blending. No head-to-head -> falls back
to general so good hitters without matchup history aren't buried.
"""
import time
import requests

MLB = "https://statsapi.mlb.com/api/v1"
MLB11 = "https://statsapi.mlb.com/api/v1.1"
LEAGUE_AVG_K = 0.22

H2H_WEIGHT = 0.40
GEN_WEIGHT = 0.60
MIN_H2H_PA = 2

S = requests.Session()
S.headers["User-Agent"] = "prop-edge-lineupk/3.0"


def _get(url, **params):
    for attempt in range(3):
        try:
            r = S.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(1.0 * (attempt + 1))
    return {}


# ---------- STRIKEOUT side ----------

def general_k_rate_vs_hand(batter_id, throws, season):
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
    gen_kr, _ = general_k_rate_vs_hand(batter_id, throws, season)
    h2h_kr, h2h_pa = head_to_head_k_rate(batter_id, pitcher_id)
    if h2h_kr is not None:
        return H2H_WEIGHT * h2h_kr + GEN_WEIGHT * gen_kr, True
    return gen_kr, False


def lineup_k_expectation(opp_batter_ids, pitcher_throws, season, expected_bf,
                         pitcher_id=None):
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
    return avg_k * expected_bf, avg_k, n_data


def get_pitcher_throws(pitcher_id):
    d = _get(f"{MLB}/people/{pitcher_id}")
    try:
        return d["people"][0]["pitchHand"]["code"]
    except Exception:
        return "R"


# ---------- HIT side ----------

def general_batting_avg(batter_id, season):
    d = _get(f"{MLB}/people/{batter_id}/stats", stats="season",
             group="hitting", season=season)
    try:
        st = d["stats"][0]["splits"][0]["stat"]
        ab = int(st.get("atBats", 0) or 0)
        h = int(st.get("hits", 0) or 0)
        if ab >= 20:
            return h / ab, True
    except Exception:
        pass
    return 0.0, False


def head_to_head_avg(batter_id, pitcher_id):
    d = _get(f"{MLB}/people/{batter_id}/stats",
             stats="vsPlayer", opposingPlayerId=pitcher_id, group="hitting")
    ab_tot = 0
    h_tot = 0
    try:
        for s in d["stats"][0]["splits"]:
            if s.get("stat"):
                st = s["stat"]
                ab_tot += int(st.get("atBats", 0) or 0)
                h_tot += int(st.get("hits", 0) or 0)
    except Exception:
        pass
    if ab_tot >= 4:
        return h_tot / ab_tot, ab_tot
    return None, 0


def batter_hit_sum_score(batter_id, pitcher_id, season):
    """SUM-SCORE (validated 4-window sweep): gen_avg + h2h_avg ranks hit
    likelihood. Tiers from backtest hit rates. No head-to-head -> score is just
    gen_avg, tier 'no_h2h' (handled separately so good hitters aren't buried)."""
    gen_avg, _ = general_batting_avg(batter_id, season)
    h2h_avg, h2h_ab = head_to_head_avg(batter_id, pitcher_id)
    if h2h_avg is not None:
        score = gen_avg + h2h_avg
        if score >= 0.70:
            tier = "sum_premium"   # ~91%
        elif score >= 0.60:
            tier = "sum_strong"    # ~82%
        elif score >= 0.50:
            tier = "sum_good"      # ~72%
        elif score >= 0.40:
            tier = "sum_lean"      # ~65%
        else:
            tier = "sum_avoid"     # ~27%
        return {"score": score, "tier": tier, "gen_avg": gen_avg,
                "h2h_avg": h2h_avg, "h2h_ab": h2h_ab, "has_h2h": True}
    return {"score": gen_avg, "tier": "no_h2h", "gen_avg": gen_avg,
            "h2h_avg": None, "h2h_ab": 0, "has_h2h": False}


if __name__ == "__main__":
    wood = 605540
    f = _get(f"{MLB11}/game/824502/feed/live")
    try:
        order = f["liveData"]["boxscore"]["teams"]["home"]["battingOrder"][:9]
        throws = get_pitcher_throws(wood)
        ek, avg, n = lineup_k_expectation(order, throws, 2026, 24, pitcher_id=wood)
        print(f"K test: lineup avg K-rate={avg:.3f}, expected Ks={ek:.1f}, {n}/9")
    except Exception as e:
        print(f"test error: {e}")
