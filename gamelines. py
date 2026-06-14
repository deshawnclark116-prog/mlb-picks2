"""
gamelines.py - Moneyline model for MLB game lines.
Uses Pythagorean expectation (win rate from runs scored vs allowed) + log5
(head-to-head combination) + home-field edge, then de-vigs the market
moneyline and computes edge. Transparent, proven, no training needed.

Pulls team runs scored/allowed from the MLB Stats API (same source the rest
of the system uses).
"""
import time, datetime as dt
import requests

MLB = "https://statsapi.mlb.com/api/v1"
SEASON = dt.date.today().year
PYTH_EXP = 1.83          # baseball Pythagorean exponent (well-established)
HOME_EDGE = 0.035        # ~3.5% home-field bump

S = requests.Session()
S.headers["User-Agent"] = "prop-edge-gl/1.0"


def _get(url, **params):
    for attempt in range(3):
        try:
            r = S.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"  retry {url}: {e}")
            time.sleep(1.5 * (attempt + 1))
    return {}


def team_run_table():
    """Return {team_name: {'rs':runs_scored,'ra':runs_allowed,'g':games}}
    for the current season, from MLB Stats API team stats."""
    table = {}
    # hitting (runs scored) and pitching (runs allowed) come from team stats
    hit = _get(f"{MLB}/teams/stats", season=SEASON, group="hitting",
               stats="season", sportIds=1)
    pit = _get(f"{MLB}/teams/stats", season=SEASON, group="pitching",
               stats="season", sportIds=1)

    def splits(d):
        try:
            return d["stats"][0]["splits"]
        except Exception:
            return []

    for sp in splits(hit):
        t = sp.get("team", {}).get("name")
        st = sp.get("stat", {})
        if not t:
            continue
        table.setdefault(t, {})
        table[t]["rs"] = float(st.get("runs", 0) or 0)
        table[t]["g"] = float(st.get("gamesPlayed", 0) or 0)

    for sp in splits(pit):
        t = sp.get("team", {}).get("name")
        st = sp.get("stat", {})
        if not t:
            continue
        table.setdefault(t, {})
        table[t]["ra"] = float(st.get("runs", 0) or 0)
        if "g" not in table[t]:
            table[t]["g"] = float(st.get("gamesPlayed", 0) or 0)

    return table


def pythag_win_pct(rs, ra):
    """Expected win rate from runs scored & allowed."""
    if rs <= 0 and ra <= 0:
        return 0.5
    num = rs ** PYTH_EXP
    den = rs ** PYTH_EXP + ra ** PYTH_EXP
    return num / den if den else 0.5


def log5(p_a, p_b):
    """Probability team A beats team B given each team's win rate (log5)."""
    den = p_a + p_b - 2 * p_a * p_b
    if den == 0:
        return 0.5
    return (p_a - p_a * p_b) / den


def moneyline_prob(home_team, away_team, table):
    """Return (home_win_prob, away_win_prob) with home-field edge applied."""
    h = table.get(home_team)
    a = table.get(away_team)
    if not h or not a or "rs" not in h or "ra" not in h or "rs" not in a or "ra" not in a:
        return None
    ph = pythag_win_pct(h["rs"], h["ra"])
    pa = pythag_win_pct(a["rs"], a["ra"])
    home_base = log5(ph, pa)
    # apply home-field edge, clamp to sane range
    home_win = min(0.95, max(0.05, home_base + HOME_EDGE))
    return home_win, 1 - home_win


# ── odds math (shared shape with api.py) ──────────────────────────────────────

def american_to_prob(odds):
    if odds < 0:
        return -odds / (-odds + 100)
    return 100 / (odds + 100)


def no_vig_two_way(a_odds, b_odds):
    pa = american_to_prob(a_odds); pb = american_to_prob(b_odds)
    tot = pa + pb
    if tot == 0:
        return 0.5, 0.5
    return pa / tot, pb / tot


def value_edge(model_p, fair_p):
    if fair_p <= 0:
        return 0.0
    return (model_p - fair_p) / fair_p


def kelly_fraction(model_p, american_odds, cap=0.25):
    b = (american_odds / 100) if american_odds > 0 else (100 / -american_odds)
    q = 1 - model_p
    f = (b * model_p - q) / b if b else 0
    return max(0.0, min(f, cap))


if __name__ == "__main__":
    # quick self-test: print a few teams' Pythagorean win rates
    t = team_run_table()
    print(f"Loaded {len(t)} teams")
    for name in list(t)[:5]:
        d = t[name]
        if "rs" in d and "ra" in d:
            print(f"  {name}: RS={d['rs']:.0f} RA={d['ra']:.0f} "
                  f"G={d.get('g',0):.0f} pythag={pythag_win_pct(d['rs'],d['ra']):.3f}")
