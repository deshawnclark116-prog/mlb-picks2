"""
lineupk.py - Head-to-head engine helpers.
STRIKEOUTS: each batter's K-rate vs THIS pitcher (40%) + general vs handedness
(60%), summed across lineup.
HITS: sum-score method (gen_avg + h2h_avg) validated 4-window sweep ~1950 batters.
HR: 3-signal score (season SLG + h2h_SLG + recent ISO).
Validated: score >= 1.30 → 41.9% HR rate vs 4.8% below, 4/4 windows.
h2h_SLG capped at 1.000 to prevent small-sample inflation.
Profitable at any HR prop +238 or better (FanDuel typically +300-+500).
"""
import time
import datetime as dt
import requests

MLB = "https://statsapi.mlb.com/api/v1"
MLB11 = "https://statsapi.mlb.com/api/v1.1"
LEAGUE_AVG_K = 0.22

H2H_WEIGHT = 0.40
GEN_WEIGHT = 0.60
MIN_H2H_PA = 2

S = requests.Session()
S.headers["User-Agent"] = "prop-edge-lineupk/4.1"


def _prior_date_str(as_of_date):
    if not as_of_date:
        return None

    try:
        game_date = dt.date.fromisoformat(str(as_of_date)[:10])
    except Exception:
        return None

    return (game_date - dt.timedelta(days=1)).isoformat()


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


def general_k_rate_vs_hand(batter_id, throws, season, as_of_date=None):
    """
    General batter K-rate with optional strict D-1 date bounding.

    D0 usable-data sources:
      - handedness split with >= 15 prior PA
      - current-season overall K rate with >= 15 prior PA

    When as_of_date is supplied, requests are bounded through the prior
    calendar date. No unbounded same-day fallback is used.
    """
    code = "vl" if throws == "L" else "vr"
    prior_date = _prior_date_str(as_of_date)

    hand_params = {
        "stats": "statSplits",
        "sitCodes": code,
        "group": "hitting",
        "season": season,
    }
    if prior_date:
        hand_params["endDate"] = prior_date

    d = _get(f"{MLB}/people/{batter_id}/stats", **hand_params)
    try:
        st = d["stats"][0]["splits"][0]["stat"]
        pa = int(st.get("plateAppearances", 0) or 0)
        so = int(st.get("strikeOuts", 0) or 0)
        if pa >= 15:
            return so / pa, True
    except Exception:
        pass

    season_params = {
        "stats": "season",
        "group": "hitting",
        "season": season,
    }
    if prior_date:
        season_params["endDate"] = prior_date

    d2 = _get(f"{MLB}/people/{batter_id}/stats", **season_params)
    try:
        st = d2["stats"][0]["splits"][0]["stat"]
        pa = int(st.get("plateAppearances", 0) or 0)
        so = int(st.get("strikeOuts", 0) or 0)
        if pa >= 15:
            return so / pa, True
    except Exception:
        pass

    return LEAGUE_AVG_K, False




def head_to_head_k_rate(batter_id, pitcher_id, as_of_date=None):
    params = {
        "stats": "vsPlayer",
        "opposingPlayerId": pitcher_id,
        "group": "hitting",
    }

    prior_date = _prior_date_str(as_of_date)
    if prior_date:
        params["endDate"] = prior_date

    d = _get(f"{MLB}/people/{batter_id}/stats", **params)

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




def blended_batter_k_rate(
    batter_id,
    pitcher_id,
    throws,
    season,
    as_of_date=None,
):
    """
    D0_FIXED_LINEUP_ACTIVATION parity.

    Availability is True when ANY validated K-data source is usable:
      - handedness split >= 15 prior PA
      - season K rate >= 15 prior PA
      - H2H K history >= 2 prior PA

    The validated 40% H2H / 60% general rate blend is unchanged.
    """
    gen_kr, gen_used = general_k_rate_vs_hand(
        batter_id,
        throws,
        season,
        as_of_date=as_of_date,
    )

    h2h_kr, h2h_pa = head_to_head_k_rate(
        batter_id,
        pitcher_id,
        as_of_date=as_of_date,
    )

    if h2h_kr is not None:
        return H2H_WEIGHT * h2h_kr + GEN_WEIGHT * gen_kr, True

    return gen_kr, gen_used




def lineup_k_expectation(
    opp_batter_ids,
    pitcher_throws,
    season,
    expected_bf,
    pitcher_id=None,
    as_of_date=None,
):
    if not opp_batter_ids:
        return None, None, 0

    rates = []
    n_data = 0

    for bid in opp_batter_ids[:9]:
        if pitcher_id is not None:
            kr, used = blended_batter_k_rate(
                bid,
                pitcher_id,
                pitcher_throws,
                season,
                as_of_date=as_of_date,
            )
        else:
            kr, used = general_k_rate_vs_hand(
                bid,
                pitcher_throws,
                season,
                as_of_date=as_of_date,
            )

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
    """SUM-SCORE validated 4-window sweep ~1950 batters.
    Tiers from backtest hit rates. No h2h falls back to gen_avg."""
    gen_avg, _ = general_batting_avg(batter_id, season)
    h2h_avg, h2h_ab = head_to_head_avg(batter_id, pitcher_id)
    if h2h_avg is not None:
        score = gen_avg + h2h_avg
        if score >= 0.70:
            tier = "sum_premium"
        elif score >= 0.60:
            tier = "sum_strong"
        elif score >= 0.50:
            tier = "sum_good"
        elif score >= 0.40:
            tier = "sum_lean"
        else:
            tier = "sum_avoid"
        return {"score": score, "tier": tier, "gen_avg": gen_avg,
                "h2h_avg": h2h_avg, "h2h_ab": h2h_ab, "has_h2h": True}
    return {"score": gen_avg, "tier": "no_h2h", "gen_avg": gen_avg,
            "h2h_avg": None, "h2h_ab": 0, "has_h2h": False}


# ---------- HR side ----------

def batter_hr_score(batter_id, pitcher_id, season):
    """3-signal HR score: season SLG + h2h_SLG vs pitcher + recent 7-game ISO.
    h2h_SLG CAPPED at 1.000 to prevent small-sample inflation
    (e.g. 2-for-4 with a HR = 2.000 raw SLG → capped to 1.000).
    Validated across 27 days / 178 entries:
      score >= 1.30 → 41.9% HR rate vs 4.8% below (4/4 windows)
      score >= 1.40 → 44.4% HR rate
    Profitable at any FanDuel HR prop +238 or better."""

    # season SLG
    season_slg = 0.0
    d = _get(f"{MLB}/people/{batter_id}/stats", stats="season",
             group="hitting", season=season)
    try:
        st = d["stats"][0]["splits"][0]["stat"]
        pa = int(st.get("plateAppearances", 0) or 0)
        if pa >= 20:
            season_slg = float(st.get("slg", 0) or 0)
    except Exception:
        pass

    # h2h SLG vs this pitcher — CAPPED at 1.000
    h2h_slg = 0.0
    h2h_ab = 0
    d2 = _get(f"{MLB}/people/{batter_id}/stats",
              stats="vsPlayer", opposingPlayerId=pitcher_id, group="hitting")
    ab_tot = 0
    tb_tot = 0
    try:
        for s in d2["stats"][0]["splits"]:
            if s.get("stat"):
                st = s["stat"]
                ab_tot += int(st.get("atBats", 0) or 0)
                tb_tot += int(st.get("totalBases", 0) or 0)
    except Exception:
        pass
    if ab_tot >= 4:
        h2h_slg = min(tb_tot / ab_tot, 1.000)  # cap prevents small-sample inflation
        h2h_ab = ab_tot

    # recent 7-game ISO
    recent_iso = 0.0
    g = _get(f"{MLB}/people/{batter_id}/stats", stats="gameLog",
             group="hitting", season=season)
    try:
        splits = g["stats"][0]["splits"]
        if len(splits) >= 3:
            recent = splits[-7:]
            h = tb = ab = 0
            for sp in recent:
                st = sp["stat"]
                h  += int(st.get("hits", 0) or 0)
                tb += int(st.get("totalBases", 0) or 0)
                ab += int(st.get("atBats", 0) or 0)
            if ab > 0:
                recent_iso = (tb - h) / ab
    except Exception:
        pass

    score = season_slg + h2h_slg + recent_iso

    if score >= 1.50:
        tier = "hr_elite"
    elif score >= 1.30:
        tier = "hr_strong"
    elif score >= 1.00:
        tier = "hr_lean"
    else:
        tier = "hr_none"

    return {
        "score": round(score, 3),
        "tier": tier,
        "season_slg": season_slg,
        "h2h_slg": h2h_slg,
        "h2h_ab": h2h_ab,
        "recent_iso": recent_iso,
        "fires": score >= 1.30,
    }


if __name__ == "__main__":
    import json, urllib.request
    def get(u): return json.loads(urllib.request.urlopen(
        urllib.request.Request(u, headers={"User-Agent": "x"}), timeout=30).read())
    f = get(f"{MLB11}/game/824502/feed/live")
    try:
        order = f["liveData"]["boxscore"]["teams"]["home"]["battingOrder"][:9]
        throws = get_pitcher_throws(605540)
        ek, avg, n = lineup_k_expectation(order, throws, 2026, 24, pitcher_id=605540)
        print(f"K test: lineup avg K-rate={avg:.3f}, expected Ks={ek:.1f}, {n}/9")
        sig = batter_hr_score(order[0], 605540, 2026)
        print(f"HR score test: {sig}")
    except Exception as e:
        print(f"test error: {e}")
