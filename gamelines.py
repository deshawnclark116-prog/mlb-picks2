"""
gamelines.py - MLB game-line models: moneyline, totals, team totals, run line.
Pythagorean expectation + log5 + run estimation. Transparent, no training.
Pulls team runs scored/allowed from the MLB Stats API.
"""
import time, datetime as dt
import requests

MLB = "https://statsapi.mlb.com/api/v1"
SEASON = dt.date.today().year
PYTH_EXP = 1.83
HOME_EDGE = 0.035
LEAGUE_RPG = 4.5          # approx league average runs per team per game (fallback)
WIN_BY_2PLUS = 0.57       # ~57% of MLB wins are by 2+ runs (run-line factor)

S = requests.Session()
S.headers["User-Agent"] = "prop-edge-gl/2.0"


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
    """Return {team_name: {'rs','ra','g','rs_pg','ra_pg'}} for the season."""
    table = {}
    hit = _get(f"{MLB}/teams/stats", season=SEASON, group="hitting",
               stats="season", sportIds=1)
    pit = _get(f"{MLB}/teams/stats", season=SEASON, group="pitching",
               stats="season", sportIds=1)

    def splits(d):
        try: return d["stats"][0]["splits"]
        except Exception: return []

    for sp in splits(hit):
        t = sp.get("team", {}).get("name"); st = sp.get("stat", {})
        if not t: continue
        table.setdefault(t, {})
        table[t]["rs"] = float(st.get("runs", 0) or 0)
        table[t]["g"] = float(st.get("gamesPlayed", 0) or 0)

    for sp in splits(pit):
        t = sp.get("team", {}).get("name"); st = sp.get("stat", {})
        if not t: continue
        table.setdefault(t, {})
        table[t]["ra"] = float(st.get("runs", 0) or 0)
        if "g" not in table[t]:
            table[t]["g"] = float(st.get("gamesPlayed", 0) or 0)

    # per-game rates
    for t, d in table.items():
        g = d.get("g", 0) or 0
        d["rs_pg"] = (d.get("rs", 0) / g) if g else LEAGUE_RPG
        d["ra_pg"] = (d.get("ra", 0) / g) if g else LEAGUE_RPG
    return table


def pythag_win_pct(rs, ra):
    if rs <= 0 and ra <= 0: return 0.5
    num = rs ** PYTH_EXP
    den = rs ** PYTH_EXP + ra ** PYTH_EXP
    return num / den if den else 0.5


def log5(p_a, p_b):
    den = p_a + p_b - 2 * p_a * p_b
    if den == 0: return 0.5
    return (p_a - p_a * p_b) / den


def moneyline_prob(home_team, away_team, table):
    h = table.get(home_team); a = table.get(away_team)
    if not h or not a or "rs" not in h or "ra" not in h or "rs" not in a or "ra" not in a:
        return None
    ph = pythag_win_pct(h["rs"], h["ra"]); pa = pythag_win_pct(a["rs"], a["ra"])
    home_base = log5(ph, pa)
    home_win = min(0.95, max(0.05, home_base + HOME_EDGE))
    return home_win, 1 - home_win


def expected_runs(home_team, away_team, table):
    """Estimate each team's expected runs by blending their offense with the
    opponent's defense, scaled to league average. Returns (home_exp, away_exp)."""
    h = table.get(home_team); a = table.get(away_team)
    if not h or not a: return None
    # team offense * opponent defense / league avg (log-style blend, simplified)
    home_exp = (h["rs_pg"] + a["ra_pg"]) / 2
    away_exp = (a["rs_pg"] + h["ra_pg"]) / 2
    return home_exp, away_exp


def total_runs(home_team, away_team, table):
    er = expected_runs(home_team, away_team, table)
    if not er: return None
    return er[0] + er[1]


def run_line_prob(home_team, away_team, table):
    """P(home covers -1.5) and P(away covers +1.5 i.e. away wins or loses by 1).
    Approximation using moneyline win prob and the win-by-2+ factor."""
    probs = moneyline_prob(home_team, away_team, table)
    if not probs: return None
    home_win, away_win = probs
    # home -1.5 hits only if home wins AND by 2+
    home_cover = home_win * WIN_BY_2PLUS
    # away +1.5 hits if away wins OR loses by exactly 1
    away_cover = away_win + home_win * (1 - WIN_BY_2PLUS)
    return home_cover, away_cover


# ── odds math ─────────────────────────────────────────────────────────────────

def american_to_prob(odds):
    if odds < 0: return -odds / (-odds + 100)
    return 100 / (odds + 100)


def no_vig_two_way(a_odds, b_odds):
    pa = american_to_prob(a_odds); pb = american_to_prob(b_odds)
    tot = pa + pb
    if tot == 0: return 0.5, 0.5
    return pa / tot, pb / tot


def value_edge(model_p, fair_p):
    if fair_p <= 0: return 0.0
    return (model_p - fair_p) / fair_p


def kelly_fraction(model_p, american_odds, cap=0.25):
    b = (american_odds / 100) if american_odds > 0 else (100 / -american_odds)
    q = 1 - model_p
    f = (b * model_p - q) / b if b else 0
    return max(0.0, min(f, cap))


# ── totals probability (normal approx around expected total) ──────────────────
import math

def prob_total_over(expected_total, line, sd=3.0):
    """P(actual total > line) using a normal approx. SD ~3 runs for MLB totals."""
    if sd <= 0: return 0.5
    z = (line - expected_total) / sd
    # P(X > line) = 1 - CDF(z)
    cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return 1 - cdf


def prob_team_over(expected_team_runs, line, sd=2.4):
    if sd <= 0: return 0.5
    z = (line - expected_team_runs) / sd
    cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return 1 - cdf


if __name__ == "__main__":
    t = team_run_table()
    print(f"Loaded {len(t)} teams")
    names = list(t)[:2]
    if len(names) == 2:
        h, a = names
        print(f"Test {a} @ {h}")
        print("  ML:", moneyline_prob(h, a, t))
        print("  Total:", total_runs(h, a, t))
        print("  Run line:", run_line_prob(h, a, t))
