"""
bvp.py - Batter-vs-Pitcher history signal (full profile).
Pulls the complete BvP line: hits, XBH (2B/3B/HR), RBI, total bases, steals,
walks, strikeouts — plus computed power metrics (TB/AB, ISO). Each prop type
can draw the BvP stat that matters for it:
  hits -> avg/h | total_bases -> tb_per_ab/iso | rbi -> rbi | strikeouts -> k_rate
Two uses: per-batter flags, and lineup-wide aggregate for the K projection.
Experimental — tested against the record before being trusted.
"""
import time
import datetime as dt
import requests

MLB = "https://statsapi.mlb.com/api/v1"
S = requests.Session()
S.headers["User-Agent"] = "prop-edge-bvp/1.1"


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



def batter_vs_pitcher(
    batter_id,
    pitcher_id,
    as_of_date=None,
):
    """
    Full BvP line with optional strict D-1 upper date bound.

    Existing hitter callers that omit as_of_date stay on the legacy path.
    The pitcher-K lineup nudge passes as_of_date explicitly.
    """
    params = {
        "stats": "vsPlayer",
        "group": "hitting",
        "opposingPlayerId": pitcher_id,
        "sportId": 1,
    }

    prior_date = _prior_date_str(as_of_date)
    if prior_date:
        params["endDate"] = prior_date

    d = _get(f"{MLB}/people/{batter_id}/stats", **params)

    try:
        for s in d.get("stats", []):
            for sp in s.get("splits", []):
                st = sp.get("stat", {})
                ab = int(st.get("atBats", 0) or 0)
                h = int(st.get("hits", 0) or 0)
                dbl = int(st.get("doubles", 0) or 0)
                trp = int(st.get("triples", 0) or 0)
                hr = int(st.get("homeRuns", 0) or 0)
                rbi = int(st.get("rbi", 0) or 0)
                tb = int(st.get("totalBases", 0) or 0)
                sb = int(st.get("stolenBases", 0) or 0)
                so = int(st.get("strikeOuts", 0) or 0)
                bb = int(st.get("baseOnBalls", 0) or 0)
                pa = int(st.get("plateAppearances", 0) or 0)

                if pa > 0:
                    return {
                        "ab": ab,
                        "h": h,
                        "doubles": dbl,
                        "triples": trp,
                        "hr": hr,
                        "rbi": rbi,
                        "tb": tb,
                        "sb": sb,
                        "so": so,
                        "bb": bb,
                        "pa": pa,
                        "avg": (h / ab) if ab else 0.0,
                        "k_rate": (so / pa) if pa else 0.0,
                        "tb_per_ab": (tb / ab) if ab else 0.0,
                        "iso": ((tb - h) / ab) if ab else 0.0,
                    }
    except Exception:
        pass

    return None



def classify_batter(bvp):
    """Classify a batter's history vs this pitcher.
    Checks 'hits him' FIRST so a strong average isn't overridden by K-rate."""
    if not bvp or bvp["pa"] == 0:
        return "none"
    avg = bvp["avg"]; krate = bvp["k_rate"]; pa = bvp["pa"]
    if pa < 3:
        return "none"
    # hits him: strong average OR real power is the dominant signal
    if avg >= 0.300 or bvp["iso"] >= 0.250:
        return "hits"
    # struggles: weak average OR very high strikeout rate
    if avg <= 0.180 or krate >= 0.40:
        return "struggles"
    return "neutral"


def power_flag(bvp):
    """Classify the batter's POWER vs this pitcher (for total bases / HR picks).
    Returns 'power' (squares him up), 'weak', or 'neutral'/'none'."""
    if not bvp or bvp["pa"] < 3:
        return "none"
    # tb_per_ab >= 0.6 is roughly a .600 SLG vs him; iso >= .25 is real pop
    if bvp["tb_per_ab"] >= 0.60 or bvp["iso"] >= 0.250:
        return "power"
    if bvp["tb_per_ab"] <= 0.25:
        return "weak"
    return "neutral"



def lineup_vs_pitcher(
    batter_ids,
    pitcher_id,
    as_of_date=None,
):
    """Aggregate lineup BvP for the pitcher-K nudge."""
    tot_ab = tot_h = tot_so = tot_pa = tot_tb = 0
    per_batter = {}

    for bid in batter_ids:
        batter_bvp = batter_vs_pitcher(
            bid,
            pitcher_id,
            as_of_date=as_of_date,
        )

        per_batter[bid] = batter_bvp

        if batter_bvp:
            tot_ab += batter_bvp["ab"]
            tot_h += batter_bvp["h"]
            tot_so += batter_bvp["so"]
            tot_pa += batter_bvp["pa"]
            tot_tb += batter_bvp["tb"]

        time.sleep(0.08)

    if tot_pa == 0:
        return {
            "lineup_avg": None,
            "lineup_k_rate": None,
            "lineup_slg": None,
            "k_nudge": 1.0,
            "per_batter": per_batter,
            "sample_pa": 0,
        }

    lineup_avg = tot_h / tot_ab if tot_ab else 0
    lineup_k_rate = tot_so / tot_pa
    lineup_slg = tot_tb / tot_ab if tot_ab else 0

    # Validated nudge semantics unchanged.
    raw = lineup_k_rate / 0.22
    k_nudge = max(0.85, min(1.15, raw))

    return {
        "lineup_avg": round(lineup_avg, 3),
        "lineup_k_rate": round(lineup_k_rate, 3),
        "lineup_slg": round(lineup_slg, 3),
        "k_nudge": round(k_nudge, 3),
        "per_batter": per_batter,
        "sample_pa": tot_pa,
    }



if __name__ == "__main__":
    import urllib.request, json
    def g(u): return json.loads(urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'x'}),timeout=20).read())
    feed = g("https://statsapi.mlb.com/api/v1.1/game/823451/feed/live")
    order = feed["liveData"]["boxscore"]["teams"]["away"]["battingOrder"]
    print("Marlins lineup vs Luzardo (666200) — full BvP profile:\n")
    for bid in order:
        pd = g(f"https://statsapi.mlb.com/api/v1/people/{bid}")
        name = pd["people"][0]["fullName"]
        bvp = batter_vs_pitcher(bid, 666200)
        cls = classify_batter(bvp); pwr = power_flag(bvp)
        if bvp:
            print(f"  {name[:16]:16} {bvp['ab']:3}AB {bvp['h']}H {bvp['doubles']}2B "
                  f"{bvp['hr']}HR {bvp['rbi']}RBI {bvp['tb']}TB {bvp['so']}K "
                  f"avg={bvp['avg']:.3f} iso={bvp['iso']:.3f} -> {cls.upper()}/{pwr.upper()}")
        else:
            print(f"  {name[:16]:16} no history -> NONE")
    print()
    agg = lineup_vs_pitcher(order, 666200)
    print(f"Lineup combined: avg={agg['lineup_avg']} slg={agg['lineup_slg']} "
          f"k_rate={agg['lineup_k_rate']} (sample {agg['sample_pa']} PA)")
    print(f"K-nudge for Luzardo: {agg['k_nudge']}x")
