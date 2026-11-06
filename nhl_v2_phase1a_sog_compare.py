"""NHL V2 Phase1A-SOG comparators: Phase0 simple and competent-human baseline FORMULAS evaluated on the V2 official-source data with the same
cutoff rule as the B2 rows. Poisson count distributions (Phase0 locked dispersion index 1.107 <= 1.15). Research only; no betting-market inputs."""
import bisect
from collections import defaultdict

import numpy as np

import nhl_v2_phase1a_sog_data as D

FIT_SEASONS = (2018, 2019, 2020, 2021, 2022)
MIN_PRIOR = 5
MIN_TOI3 = 480.0
SHRINK_GAMES = 20.0
OPP_DAMP = 0.5


def pos_group(code):
    return "D" if code == "D" else "F"


def comparator_features(games, rows, tab, horizon_min=90):
    """Arrays aligned with `tab` rows: same-season prior-appearance statistics from games usable at the row's cutoff (start <= target start - (horizon+210) min)."""
    back = (horizon_min + D.GAME_MAX_MINUTES) * 60
    gstart = {gid: D.epoch(g["game_start_utc"]) for gid, g in games.items()}
    apps = defaultdict(list)
    for r in rows:
        apps[r["player_id"]].append((gstart[r["game_id"]], r["game_id"], r["sog"], r["toi_sec"] or 0, r["position"]))
    for p in apps:
        apps[p].sort(key=lambda a: (a[0], a[1]))
    starts = {p: [a[0] for a in v] for p, v in apps.items()}
    team_sog = defaultdict(int)
    for r in rows:
        team_sog[(r["game_id"], r["team_id"])] += r["sog"]
    tg = []                                                          # (start, season, team, sog_for, sog_against)
    for gid, g in games.items():
        h, a = g["home_team_id"], g["away_team_id"]
        if (gid, h) in team_sog or (gid, a) in team_sog:
            s = gstart[gid]; season = int(str(gid)[:4])
            tg.append((s, season, h, team_sog[(gid, h)], team_sog[(gid, a)])); tg.append((s, season, a, team_sog[(gid, a)], team_sog[(gid, h)]))
    tg.sort()
    by_team = defaultdict(list)
    for t in tg:
        by_team[(t[1], t[2])].append(t)
    team_arr = {k: ([t[0] for t in v], np.cumsum([t[4] for t in v])) for k, v in by_team.items()}      # sog allowed cumulative
    season_start = defaultdict(list)
    for t in tg:
        season_start[t[1]].append(t)
    season_arr = {s: ([t[0] for t in v], np.cumsum([t[3] for t in v])) for s, v in season_start.items()}
    n = len(tab["sog"])
    out = {k: np.full(n, np.nan) for k in ("n_prior", "m3", "m5", "m10", "ewma", "S", "T_hr", "pred_toi", "opp_allowed", "league")}
    out["pos"] = np.empty(n, dtype=object)
    for i in range(n):
        pid = int(tab["player_id"][i]); s0 = int(tab["start"][i]); limit = s0 - back; season = int(tab["season"][i])
        k = bisect.bisect_right(starts.get(pid, []), limit)
        al = [a for a in apps.get(pid, [])[:k] if int(str(a[1])[:4]) == season]
        n_p = len(al)
        out["n_prior"][i] = n_p
        out["pos"][i] = pos_group(al[-1][4]) if al else "F"
        if n_p:
            sogs = [a[2] for a in al]
            out["m3"][i] = np.mean(sogs[-3:]); out["m5"][i] = np.mean(sogs[-5:]); out["m10"][i] = np.mean(sogs[-10:])
            e = sogs[0]
            for v in sogs[1:]:
                e = 0.2 * v + 0.8 * e
            out["ewma"][i] = e
            out["S"][i] = float(sum(sogs)); out["T_hr"][i] = sum(a[3] for a in al) / 3600.0
            out["pred_toi"][i] = float(np.mean([a[3] for a in al[-3:]]))
        opp = int(tab["opp_id"][i])
        arr = team_arr.get((season, opp))
        if arr:
            j = bisect.bisect_right(arr[0], limit)
            if j > 0:
                out["opp_allowed"][i] = arr[1][j - 1] / j
        sa = season_arr.get(season)
        if sa:
            j = bisect.bisect_right(sa[0], limit)
            if j >= 60:
                out["league"][i] = sa[1][j - 1] / j
    return out


def eligible(cf):
    return (cf["n_prior"] >= MIN_PRIOR) & (np.nan_to_num(cf["pred_toi"], nan=-1.0) >= MIN_TOI3)


def fit_constants(cf, tab, rows):
    """Phase0 constants: position-group SOG per hour (sum SOG / sum TOI hours) and mean TOI hours per game, from FIT-season eligible rows that PLAYED
    (target values of FIT seasons only; never of an evaluation season)."""
    actual = {(r["game_id"], r["player_id"]): (r["sog"], (r["toi_sec"] or 0) / 3600.0) for r in rows if r["season_start_year"] in FIT_SEASONS}
    m = eligible(cf) & (tab["played"] == 1) & np.isin(tab["season"], FIT_SEASONS)
    c = {}
    for g in ("F", "D"):
        idx = np.where(m & (cf["pos"] == g))[0]
        sog = np.array([actual[(int(tab["game_id"][i]), int(tab["player_id"][i]))][0] for i in idx], float)
        hrs = np.array([actual[(int(tab["game_id"][i]), int(tab["player_id"][i]))][1] for i in idx], float)
        c[g] = {"rate_hr": float(sog.sum() / hrs.sum()), "toi_hr": float(hrs.mean()), "n": int(len(idx))}
    return c


def human_mu(cf, c):
    n = len(cf["n_prior"]); mu = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(cf["n_prior"][i]) or cf["n_prior"][i] < MIN_PRIOR or np.isnan(cf["pred_toi"][i]):
            continue
        g = cf["pos"][i]; r0, h0 = c[g]["rate_hr"], SHRINK_GAMES * c[g]["toi_hr"]
        rate = (cf["S"][i] + r0 * h0) / (cf["T_hr"][i] + h0)
        f = 1.0
        if not np.isnan(cf["opp_allowed"][i]) and not np.isnan(cf["league"][i]) and cf["league"][i] > 0:
            f = 1.0 + OPP_DAMP * (cf["opp_allowed"][i] / cf["league"][i] - 1.0)
        mu[i] = (cf["pred_toi"][i] / 3600.0) * rate * f
    return mu


def simple_means(cf, c):
    toi_hr = np.nan_to_num(cf["pred_toi"], nan=0.0) / 3600.0
    with np.errstate(invalid="ignore", divide="ignore"):
        rate_x_toi = cf["S"] / cf["T_hr"] * toi_hr
    return {"prior10_mean": cf["m10"], "prior5_mean": cf["m5"], "prior3_mean": cf["m3"], "ewma_0.2": cf["ewma"], "season_rate_x_prior3_toi": rate_x_toi}
