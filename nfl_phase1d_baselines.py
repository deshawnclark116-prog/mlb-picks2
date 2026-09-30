"""
NFL_PHASE1D_BASELINES  (Phase 1D)  -- the pre-registered baselines of protocol Amendment G, produced and logged next to every forecast

HB1  historical baseline for rushing yards, receiving yards, receptions, passing yards and anytime TD.
     For a candidate with p = the FROZEN status-only P(active) (the Phase 1A status lookup table of the weekly artifact, T-90m game-day INA => 0):
       * qualifying games = the player's own games (as-of, from the snapshot's stats_player_week files) with >= 1 opportunity of the relevant kind
         (carries / targets / attempts); values = the last 8 of them;
       * if he has < 3 qualifying games the empirical pool is the position group's most recent 400 qualifying player-games (as-of, league-wide);
       * forecast distribution = (1 - p) x point mass at 0  +  p x uniform over the empirical values;
       * anytime TD: P = p x (TDs_in_last_8_qualifying + 2 x pos_rate) / (n + 2), pos_rate frozen from 2023-2024 (QB .1426, RB .2573, WR .2117, TE .1522).
     No model, no sportsbook input, no target-game information.

The records use the SAME schema as forecast records (quantile grids, p_zero, lattice cdf, event probability) so the score logger grades them unchanged.
"""
import csv
import io
from collections import defaultdict

import numpy as np

import nfl_phase1_store as ST

BASELINE_VERSION = "HB1"
POS_TD_RATE = {"QB": 0.1426, "RB": 0.2573, "WR": 0.2117, "TE": 0.1522}
WINDOW, MIN_OWN, POOL = 8, 3, 400
GRID19 = [round(x, 2) for x in np.arange(0.05, 0.951, 0.05)]
GRID99 = [round(x, 2) for x in np.arange(0.01, 0.995, 0.01)]
FAMILY = {"rush_yds": ("carries", "rushing_yards"), "rec_yds": ("targets", "receiving_yards"), "rec": ("targets", "receptions"), "pass_yds": ("attempts", "passing_yards")}
COUNT_STEP = {"rec": 1.0, "atd": 1.0}


def game_logs(stats_blobs):
    """{gsis: [(season, week, row-dict), ...] chronological} plus {pos_group: [(season, week, row, gsis)]} from the snapshot's stats files (completed games only)."""
    per, pos_rows = defaultdict(list), defaultdict(list)
    for b in stats_blobs:
        for r in csv.DictReader(io.StringIO(b.decode("utf-8"))):
            if r.get("season_type", "REG") != "REG":
                continue
            d = {k: (float(r[k]) if r.get(k) not in (None, "", "NA") else 0.0) for k in ("carries", "targets", "attempts", "rushing_yards", "receiving_yards", "receptions",
                                                                                          "passing_yards", "rushing_tds", "receiving_tds")}
            s, w = int(r["season"]), int(r["week"])
            pos = {"FB": "RB"}.get(r["position"], r["position"])
            per[r["player_id"]].append((s, w, d, pos))
            pos_rows[pos].append((s, w, d, r["player_id"]))
    for v in per.values():
        v.sort(key=lambda x: (x[0], x[1]))
    for v in pos_rows.values():
        v.sort(key=lambda x: (x[0], x[1]))
    return per, pos_rows


def qualifying(rows, opp_key, val_key=None):
    out = []
    for s, w, d, *_ in rows:
        if d[opp_key] >= 1:
            out.append(d[val_key] if val_key else d)
    return out


def mixture(values, p):
    """Discrete distribution: mass (1-p) at 0 plus p spread uniformly over `values` (list). Returns (support, prob) sorted."""
    v = np.array(sorted(values), float) if len(values) else np.array([0.0])
    sup = np.concatenate([[0.0], v]); pr = np.concatenate([[1.0 - p], np.full(len(v), p / len(v))])
    o = np.argsort(sup, kind="stable")
    return sup[o], pr[o]


def inv_cdf(sup, pr, taus):
    cdf = np.cumsum(pr)
    return [float(sup[min(int(np.searchsorted(cdf, t - 1e-12)), len(sup) - 1)]) for t in taus]


def dist_record(sup, pr, event=False):
    cdf = np.cumsum(pr)
    mean = float((sup * pr).sum())
    sd = float(np.sqrt(max(((sup - mean) ** 2 * pr).sum(), 0.0)))
    q19 = inv_cdf(sup, pr, GRID19); q99 = inv_cdf(sup, pr, GRID99)
    p_zero = float(pr[sup <= 0].sum())
    return {"mean": mean, "median": q19[GRID19.index(0.5)], "sd": sd, "quantile_grid": q19, "quantile_grid_99": q99, "p_zero": p_zero,
            "quantiles": {f"p{int(round(p * 100)):02d}": inv_cdf(sup, pr, [p])[0] for p in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)}}


def lattice(sup, pr, step=1.0, kmax=40):
    K = int(min(np.ceil(float(sup.max()) / step), kmax))
    return {"step": step, "cdf": [float(pr[sup <= k * step + 1e-9].sum()) for k in range(K + 1)]}


def build_records(pack_games, status_lookup, stats_blobs, season, week, game_id, teams, horizon, kickoff_iso, cutoff_iso, gsis_pos, prov):
    """pack_games: {team: Phase 1A game inputs (types with ids/pact_lookup)}. status_lookup(team, type, j) -> frozen status-only P(active) (already 0 for T90 INA)."""
    per, pos_rows = game_logs(stats_blobs)
    recs = []
    pools = {}

    def pool(pos, opp_key, val_key):
        k = (pos, opp_key, val_key)
        if k not in pools:
            q = [d[val_key] for s, w, d, _ in pos_rows.get(pos, []) if d[opp_key] >= 1]
            pools[k] = q[-POOL:]
        return pools[k]
    for tm in teams:
        g = pack_games.get(tm)
        if g is None:
            continue
        need = {"rush_yds": "carry", "rec_yds": "target", "rec": "target", "pass_yds": "qb_att"}
        seen_atd = {}
        for outcome, tname in need.items():
            t = g["types"].get(tname)
            if not t:
                continue
            opp_key, val_key = FAMILY[outcome]
            for j, gid in enumerate(t["ids"]):
                p = float(status_lookup(tm, tname, j))
                pos = {"FB": "RB"}.get(gsis_pos.get(gid) or t["pos"][j], gsis_pos.get(gid) or t["pos"][j])
                own = qualifying(per.get(gid, []), opp_key, val_key)[-WINDOW:]
                vals = own if len(own) >= MIN_OWN else pool(pos, opp_key, val_key)
                sup, pr = mixture(vals, p)
                rec = {"id": ST.make_id(BASELINE_VERSION, game_id, gid, outcome, horizon, cutoff_iso), "model_version": BASELINE_VERSION, "horizon": horizon, "cutoff": cutoff_iso,
                       "kickoff": kickoff_iso, "season": season, "week": week, "game_id": game_id, "player_id": gid, "team": tm, "outcome": outcome, "p_active": p,
                       "event_probability_ge1": None, "cdf_lattice": lattice(sup, pr, 1.0) if outcome in COUNT_STEP else None, "baseline_input_snapshots": prov,
                       "n_own_qualifying_games": len(own), **dist_record(sup, pr)}
                recs.append(rec)
                if outcome in ("rush_yds", "rec_yds") and gid not in seen_atd:
                    seen_atd[gid] = (tm, p, pos, tname)
            if outcome == "rec_yds":
                pass
        for gid, (tm_, p, pos, _) in seen_atd.items():
            q_own = [1.0 if (d["rushing_tds"] + d["receiving_tds"]) >= 1 else 0.0 for s, w, d, _ps in per.get(gid, []) if d["carries"] + d["targets"] + d["attempts"] >= 1][-WINDOW:]
            rate = (sum(q_own) + 2.0 * POS_TD_RATE.get(pos, 0.15)) / (len(q_own) + 2.0)
            pe = float(p * rate)
            sup, pr = np.array([0.0, 1.0]), np.array([1.0 - pe, pe])
            rec = {"id": ST.make_id(BASELINE_VERSION, game_id, gid, "atd", horizon, cutoff_iso), "model_version": BASELINE_VERSION, "horizon": horizon, "cutoff": cutoff_iso,
                   "kickoff": kickoff_iso, "season": season, "week": week, "game_id": game_id, "player_id": gid, "team": tm_, "outcome": "atd", "p_active": p,
                   "event_probability_ge1": pe, "cdf_lattice": lattice(sup, pr, 1.0), "baseline_input_snapshots": prov, "n_own_qualifying_games": len(q_own), **dist_record(sup, pr)}
            recs.append(rec)
    return recs
