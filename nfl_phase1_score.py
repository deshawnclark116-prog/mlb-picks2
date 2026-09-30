"""
NFL_PHASE1_SCORE  (Phase 1C, shadow research)  -- the score logger

Matches OFFICIAL outcomes to immutable forecasts and appends score records. Never modifies a forecast.

  * Outcomes come from an official stats file (nflverse stats_player_week_<season>.csv); the sha256 of the exact bytes graded against is recorded.
  * A forecast whose game is not final (no result in games.csv) or whose week is absent from the stats file stays UNGRADED: nothing is written.
  * A player with no stats row in a FINAL game is graded 0 (pregame-universe convention), flagged no_stats_row = true.
  * score id = make_id(forecast id, source sha256). Re-running with the same official file is a verified no-op. If the official file is REVISED
    (different bytes) a NEW score record is appended with an incremented revision number; the earlier score is kept, never overwritten.
  * Scores per forecast: outcome, error of the median / mean, quantile (pinball) score over the stored grid (a CRPS proxy = 2 x mean pinball),
    coverage indicators for the 50 / 80 / 90 % intervals, and for event outcomes the realized indicator with log loss / Brier.
"""
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

import nfl_phase1_store as ST

STAT_COL = {"rush_yds": ("rushing_yards",), "rec_yds": ("receiving_yards",), "rec": ("receptions",), "pass_yds": ("passing_yards",), "pass_td": ("passing_tds",),
            "int": ("passing_interceptions",), "rush_td": ("rushing_tds",), "rec_td": ("receiving_tds",), "atd": ("rushing_tds", "receiving_tds"),
            "tackles": ("def_tackles_solo", "def_tackle_assists"), "sacks": ("def_sacks",), "def_int": ("def_interceptions",)}
GRID = [round(x, 2) for x in np.arange(0.05, 0.951, 0.05)]


def fnum(v):
    try:
        x = float(v)
        return 0.0 if np.isnan(x) else x
    except (TypeError, ValueError):
        return 0.0


def load_official(stats_bytes, games_bytes):
    """-> (outcomes {(season, week, gsis): {col: value}}, final_games {(season, week, team)}, sha256 of the stats bytes)"""
    out = {}
    for r in csv.DictReader(io.StringIO(stats_bytes.decode("utf-8"))):
        if r.get("season_type", "REG") != "REG":
            continue
        out[(int(r["season"]), int(r["week"]), r["player_id"])] = r
    final = set()
    for r in csv.DictReader(io.StringIO(games_bytes.decode("utf-8"))):
        if r["game_type"] == "REG" and r["result"] not in ("", "NA"):
            final.add((int(r["season"]), int(r["week"]), r["home_team"])); final.add((int(r["season"]), int(r["week"]), r["away_team"]))
    return out, final, hashlib.sha256(stats_bytes).hexdigest()


def load_participation(snap_bytes, players_bytes):
    """{(season, week, gsis)} of players with an offensive or defensive snap in the official snap-count file (pfr id mapped to gsis through players.csv)."""
    pfr2g = {}
    for r in csv.DictReader(io.StringIO(players_bytes.decode("utf-8"))):
        if r.get("pfr_id") and r.get("gsis_id"):
            pfr2g[r["pfr_id"]] = r["gsis_id"]
    out = set()
    for r in csv.DictReader(io.StringIO(snap_bytes.decode("utf-8"))):
        if r.get("game_type", "REG") != "REG":
            continue
        if fnum(r.get("offense_snaps")) > 0 or fnum(r.get("defense_snaps")) > 0:
            g = pfr2g.get(r["pfr_player_id"])
            if g:
                out.add((int(r["season"]), int(r["week"]), g))
    return out


def actual_value(outcome, row):
    if row is None:
        return 0.0
    return sum(fnum(row.get(c)) for c in STAT_COL[outcome])


def pinball(qgrid, y):
    return float(np.mean([max(q * (y - v), (q - 1) * (y - v)) for q, v in zip(GRID, qgrid)]))


GRID99 = [round(x, 2) for x in np.arange(0.01, 0.995, 0.01)]


def pit_bounds(f, y):
    """[lo, hi] of the randomized-PIT interval of the realized value: P(X < y) and P(X <= y) under the stored forecast distribution.
    Discrete outcomes use the stored exact lattice cdf; yardage outcomes use the atom at 0 (p_zero) plus the 99-point quantile grid. Returns None if the
    record predates these fields. The randomization itself happens at evaluation time with a seed derived from the forecast id."""
    lat = f.get("cdf_lattice")
    if lat:
        cdf, step = lat["cdf"], lat["step"]
        k = int(round(y / step))
        if k < 0:
            return 0.0, 0.0
        hi = cdf[k] if k < len(cdf) else 1.0
        lo = cdf[k - 1] if 1 <= k <= len(cdf) else (0.0 if k == 0 else cdf[-1])
        return float(lo), float(hi)
    if f.get("quantile_grid_99") is not None and f.get("p_zero") is not None:
        if y <= 0:
            return 0.0, float(f["p_zero"])
        F = float(np.interp(y, f["quantile_grid_99"], GRID99, left=0.005, right=0.995))
        F = max(F, float(f["p_zero"]))
        return F, F
    return None


def score_one(f, y):
    q = f["quantile_grid"]
    med_i = GRID.index(0.5)
    out = {"outcome_value": y, "error_median": f["median"] - y, "abs_error_median": abs(f["median"] - y), "error_mean": f["mean"] - y,
           "pinball_mean": pinball(q, y), "crps_proxy": 2.0 * pinball(q, y)}
    lo = lambda a: np.interp(a, GRID, q)
    for lv in (0.5, 0.8, 0.9):
        a = (1 - lv) / 2
        out[f"in_{int(lv * 100)}"] = bool(lo(a) <= y <= lo(1 - a))
    pb = pit_bounds(f, y)
    if pb is not None:
        out["pit_lo"], out["pit_hi"] = pb
    p = f.get("event_probability_ge1")
    if p is not None:
        ev = 1.0 if y >= 1 else 0.0
        pc = min(max(p, 1e-6), 1 - 1e-6)
        out.update(event=ev, event_p=p, event_logloss=float(-(ev * np.log(pc) + (1 - ev) * np.log(1 - pc))), event_brier=float((p - ev) ** 2))
    return out


def grade(forecast_store, score_store, stats_bytes, games_bytes, source_name="stats_player_week", snap_bytes=None, players_bytes=None):
    """Append scores for every gradable forecast. Returns a status dict. With snap_bytes + players_bytes each score also records `played` (an offensive / defensive
    snap or an opportunity in the official files) for the availability gate."""
    official, final, src_sha = load_official(stats_bytes, games_bytes)
    part = load_participation(snap_bytes, players_bytes) if (snap_bytes and players_bytes) else None
    if part is not None:
        import hashlib as _h
        src_sha = _h.sha256(stats_bytes + snap_bytes).hexdigest()
    idx = score_store.index()
    existing = {}
    for r in score_store.all_records():
        existing.setdefault(r["forecast_id"], []).append(r)
    seasons_in_file = {k[0] for k in official}
    graded = ungraded = dup = 0
    revisions = 0
    batch = {}
    for f in [r for b in forecast_store.batch_files() for r in forecast_store.read_batch(b)[1]]:
        s, w = f["season"], f["week"]
        if (s, w, f["team"]) not in final or s not in seasons_in_file or not any(k[0] == s and k[1] == w for k in official):
            ungraded += 1
            continue
        row = official.get((s, w, f["player_id"]))
        y = actual_value(f["outcome"], row)
        sid = ST.make_id(f["id"], src_sha)
        revs = existing.get(f["id"], [])
        rec = {"id": sid, "forecast_id": f["id"], "model_version": f["model_version"], "horizon": f["horizon"], "season": s, "week": w, "game_id": f["game_id"],
               "player_id": f["player_id"], "outcome": f["outcome"], "team": f["team"], "source": source_name, "source_sha256": src_sha, "no_stats_row": row is None,
               "revision": len({x["source_sha256"] for x in revs} - {src_sha}), **score_one(f, y)}
        if part is not None:
            rec["played"] = bool((s, w, f["player_id"]) in part or (row is not None and sum(fnum(row.get(c)) for c in ("carries", "targets", "attempts")) > 0))
        if sid in idx:
            dup += 1
            continue
        if revs:
            revisions += 1
        batch.setdefault(f"{s}_wk{w:02d}_{f['horizon']}_{f['game_id']}_{src_sha[:12]}", []).append(rec)
        graded += 1
    for name, recs in batch.items():
        score_store.append_batch(name, {"source_sha256": src_sha, "source": source_name}, recs)
    return {"graded_new": graded, "verified_duplicates": dup, "ungraded": ungraded, "revisions_appended": revisions, "source_sha256": src_sha}


def summarize(score_store, by=("outcome", "horizon")):
    recs = score_store.all_records()
    latest = {}
    for r in recs:                                             # latest revision per forecast for the summary; all revisions stay in the log
        cur = latest.get(r["forecast_id"])
        if cur is None or r["revision"] >= cur["revision"]:
            latest[r["forecast_id"]] = r
    groups = {}
    for r in latest.values():
        groups.setdefault(tuple(r[k] for k in by), []).append(r)
    table = {}
    for k, rs in sorted(groups.items()):
        table["|".join(map(str, k))] = {"n": len(rs), "mae_median": float(np.mean([x["abs_error_median"] for x in rs])), "bias_mean": float(np.mean([x["error_mean"] for x in rs])),
                                        "crps_proxy": float(np.mean([x["crps_proxy"] for x in rs])), "cov50": float(np.mean([x["in_50"] for x in rs])),
                                        "cov80": float(np.mean([x["in_80"] for x in rs])), "cov90": float(np.mean([x["in_90"] for x in rs]))}
        ev = [x for x in rs if "event_p" in x]
        if ev:
            table["|".join(map(str, k))].update(event_logloss=float(np.mean([x["event_logloss"] for x in ev])), event_brier=float(np.mean([x["event_brier"] for x in ev])))
    return table
