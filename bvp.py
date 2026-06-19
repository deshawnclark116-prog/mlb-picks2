"""
bvp.py - Batter-vs-Pitcher history signal.
For a batter+pitcher, returns their head-to-head line and a classification.
Two uses:
  1. Per-batter: nudge that batter's HITS pick (hits him -> boost, struggles -> fade)
  2. Lineup-wide: aggregate the 9 hitters' BvP vs the pitcher -> nudge his K projection
Tested against the record before trusting — this is an experiment with a switch.
"""
import time
import requests

MLB = "https://statsapi.mlb.com/api/v1"
S = requests.Session()
S.headers["User-Agent"] = "prop-edge-bvp/1.0"


def _get(url, **params):
    for attempt in range(3):
        try:
            r = S.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(1.0 * (attempt + 1))
    return {}


def batter_vs_pitcher(batter_id, pitcher_id):
    """Career BvP line. Returns dict or None if no history."""
    d = _get(f"{MLB}/people/{batter_id}/stats",
             stats="vsPlayer", group="hitting",
             opposingPlayerId=pitcher_id, sportId=1)
    try:
        for s in d.get("stats", []):
            for sp in s.get("splits", []):
                st = sp.get("stat", {})
                ab = int(st.get("atBats", 0) or 0)
                h = int(st.get("hits", 0) or 0)
                hr = int(st.get("homeRuns", 0) or 0)
                so = int(st.get("strikeOuts", 0) or 0)
                pa = int(st.get("plateAppearances", 0) or 0)
                if pa > 0:
                    return {"ab": ab, "h": h, "hr": hr, "so": so, "pa": pa,
                            "avg": (h / ab) if ab else 0.0,
                            "k_rate": (so / pa) if pa else 0.0}
    except Exception:
        pass
    return None


def classify_batter(bvp):
    """Classify a batter's history vs this pitcher.
    Returns 'hits' (does well), 'struggles' (does poorly), or 'neutral'/'none'."""
    if not bvp or bvp["pa"] == 0:
        return "none"
    avg = bvp["avg"]; krate = bvp["k_rate"]; pa = bvp["pa"]
    # only classify when there's at least a little history
    if pa < 3:
        return "none"
    if avg >= 0.300 and krate < 0.30:
        return "hits"
    if avg <= 0.180 or krate >= 0.35:
        return "struggles"
    return "neutral"


def lineup_vs_pitcher(batter_ids, pitcher_id):
    """Aggregate the lineup's BvP vs the pitcher.
    Returns combined line + a K-nudge factor for the pitcher's projection."""
    tot_ab = tot_h = tot_so = tot_pa = 0
    per_batter = {}
    for bid in batter_ids:
        bvp = batter_vs_pitcher(bid, pitcher_id)
        per_batter[bid] = bvp
        if bvp:
            tot_ab += bvp["ab"]; tot_h += bvp["h"]
            tot_so += bvp["so"]; tot_pa += bvp["pa"]
        time.sleep(0.08)
    if tot_pa == 0:
        return {"lineup_avg": None, "lineup_k_rate": None,
                "k_nudge": 1.0, "per_batter": per_batter, "sample_pa": 0}
    lineup_avg = tot_h / tot_ab if tot_ab else 0
    lineup_k_rate = tot_so / tot_pa
    # K-nudge: if the lineup historically whiffs a lot vs him, nudge his Ks up;
    # if they historically hit him, nudge down. Centered at league ~0.22.
    # Capped to a modest +/-15% so it informs rather than dominates.
    raw = lineup_k_rate / 0.22
    k_nudge = max(0.85, min(1.15, raw))
    return {"lineup_avg": round(lineup_avg, 3),
            "lineup_k_rate": round(lineup_k_rate, 3),
            "k_nudge": round(k_nudge, 3),
            "per_batter": per_batter,
            "sample_pa": tot_pa}


if __name__ == "__main__":
    # TEST: Marlins lineup vs Luzardo (game 823451), and per-batter classification
    import urllib.request, json
    def g(u): return json.loads(urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'x'}),timeout=20).read())
    feed = g("https://statsapi.mlb.com/api/v1.1/game/823451/feed/live")
    order = feed["liveData"]["boxscore"]["teams"]["away"]["battingOrder"]
    print("Marlins lineup vs Luzardo (666200):\n")
    for bid in order:
        pd = g(f"https://statsapi.mlb.com/api/v1/people/{bid}")
        name = pd["people"][0]["fullName"]
        bvp = batter_vs_pitcher(bid, 666200)
        cls = classify_batter(bvp)
        if bvp:
            print(f"  {name[:18]:18} {bvp['ab']:3}AB {bvp['h']}H {bvp['so']}K "
                  f"avg={bvp['avg']:.3f} -> {cls.upper()}")
        else:
            print(f"  {name[:18]:18} no history -> NONE")
    print()
    agg = lineup_vs_pitcher(order, 666200)
    print(f"Lineup combined: avg={agg['lineup_avg']} k_rate={agg['lineup_k_rate']} "
          f"(sample {agg['sample_pa']} PA)")
    print(f"K-nudge for Luzardo's projection: {agg['k_nudge']}x")
