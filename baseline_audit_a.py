#!/usr/bin/env python3
"""
BASELINE_AUDIT_A

Does each served model actually beat a naive "he/they averaged X, so X
again" guess on real outcomes it was never trained or tuned on?

Every champion gate in this repo checked the model against a coin flip
(AUC >= 0.58, logloss gain vs a constant). That mostly measures "starters
vs backups" / "good teams vs bad teams" -- which a plain season average
also captures. This audit asks the harder question the gates never did.

Method, identical for every binary market:
  - model score = the served model's raw prediction
  - naive score = ONE number a person could compute by hand (a season-
    to-date or prior-season average of the same stat / team strength)
  - both get their own 1-D Platt calibration fit on a CALIBRATION season,
    then both are scored on a later EVAL season neither saw
  - paired bootstrap on per-row log loss difference; a market only
    counts as BEATS_NAIVE if the model's log loss is lower by a
    meaningful margin AND that gap is statistically real

Run
---
python -u baseline_audit_a.py            # all sports it has data for
"""
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
OUT = REPO / "docs" / "baseline_audit.json"
B_BOOT = 2000
SEED = 20260927
MIN_LOGLOSS_GAIN = 0.003   # per-row, in nats -- below this is noise-sized
MAX_P = 0.05


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def platt_fit(x, y, iters=50):
    """1-D logistic regression y ~ sigmoid(a*x + b), Newton's method."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    mu, sd = x.mean(), x.std() or 1.0
    z = (x - mu) / sd
    a, b = 0.0, 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a * z + b)))
        w = p * (1 - p) + 1e-9
        g = np.array([np.sum((p - y) * z), np.sum(p - y)])
        H = np.array([[np.sum(w * z * z), np.sum(w * z)], [np.sum(w * z), np.sum(w)]])
        step = np.linalg.solve(H + 1e-6 * np.eye(2), g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-8:
            break
    return lambda xx: 1 / (1 + np.exp(-(a * (np.asarray(xx, float) - mu) / sd + b)))


def auc(s, y):
    s = np.asarray(s); y = np.asarray(y)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # average ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    npos = y.sum(); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def row_logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def compare(name, model_cal, naive_cal, y_cal, model_ev, naive_ev, y_ev, note=""):
    y_cal = np.asarray(y_cal, float); y_ev = np.asarray(y_ev, float)
    m_cal, m_ev = logit(np.asarray(model_cal)), logit(np.asarray(model_ev))
    n_cal, n_ev = np.nan_to_num(np.asarray(naive_cal, float)), np.nan_to_num(np.asarray(naive_ev, float))
    pm = platt_fit(m_cal, y_cal)(m_ev)
    pn = platt_fit(n_cal, y_cal)(n_ev)
    base = np.full(len(y_ev), y_cal.mean())
    llm, lln, llb = row_logloss(pm, y_ev), row_logloss(pn, y_ev), row_logloss(base, y_ev)
    diff = lln - llm  # >0 means model better
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(diff), size=(B_BOOT, len(diff)))
    boot = diff[idx].mean(axis=1)
    p_not_better = float((boot <= 0).mean())
    gain = float(diff.mean())
    verdict = ("BEATS_NAIVE" if gain >= MIN_LOGLOSS_GAIN and p_not_better < MAX_P
               else "WORSE_THAN_NAIVE" if gain < 0 and p_not_better > 1 - MAX_P
               else "NO_REAL_EDGE")
    res = {
        "market": name, "n_calibration": int(len(y_cal)), "n_eval": int(len(y_ev)),
        "base_rate_eval": round(float(y_ev.mean()), 4),
        "model": {"auc": round(auc(pm, y_ev), 4), "logloss": round(float(llm.mean()), 4),
                  "brier": round(float(((pm - y_ev) ** 2).mean()), 4),
                  "pick_accuracy": round(float(((pm >= 0.5) == (y_ev == 1)).mean()), 4)},
        "naive": {"auc": round(auc(pn, y_ev), 4), "logloss": round(float(lln.mean()), 4),
                  "brier": round(float(((pn - y_ev) ** 2).mean()), 4),
                  "pick_accuracy": round(float(((pn >= 0.5) == (y_ev == 1)).mean()), 4)},
        "constant_logloss": round(float(llb.mean()), 4),
        "logloss_gain_vs_naive": round(gain, 5),
        "p_model_not_better": round(p_not_better, 4),
        "verdict": verdict,
        "note": note,
    }
    m, n = res["model"], res["naive"]
    print(f"  {name:38s} n={len(y_ev):6d}  AUC model {m['auc']:.3f} vs naive {n['auc']:.3f}  "
          f"logloss {m['logloss']:.4f} vs {n['logloss']:.4f}  gain {gain:+.4f} "
          f"(p={p_not_better:.3f})  -> {verdict}")
    return res


def xgb_predict(xgb, model_path, cols, rows, use_best_iteration=True):
    bst = xgb.Booster(); bst.load_model(str(model_path))
    X = np.array([[r.get(c) if r.get(c) is not None else np.nan for c in cols] for r in rows],
                 dtype=np.float32)
    d = xgb.DMatrix(X, feature_names=cols)
    if use_best_iteration:
        try:
            return bst.predict(d, iteration_range=(0, bst.best_iteration + 1))
        except Exception:
            pass
    return bst.predict(d)


def model_cols(model_dir, stem):
    return json.loads((model_dir / f"{stem}_columns.json").read_text())


# ------------------------------------------------------------------ NHL
def audit_nhl(xgb, cal_season=2024, eval_season=2025):
    print(f"\nNHL  (calibrate on {cal_season}, evaluate on {eval_season} -- a full real season "
          f"no NHL model was trained or gated on)")
    sys.path.insert(0, str(REPO))
    import nhl_points_clean_baseline_a as pts_b
    import nhl_shots_on_goal_clean_baseline_a as sog_b
    import nhl_moneyline_clean_baseline_a as ml_b
    import nhl_prior_season_points_gate_a as pts_ps
    import nhl_prior_season_shots_on_goal_gate_a as sog_ps
    import nhl_prior_season_moneyline_gate_a as ml_ps

    db = REPO / "nhl_models" / "nhl_model.sqlite"
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    md = REPO / "nhl_models"
    results = []

    def in_season(name, rows, model_dir, stem, target, naive_col):
        cal = [r for r in rows if r["season"] == cal_season]
        ev = [r for r in rows if r["season"] == eval_season]
        cols = model_cols(model_dir, stem)
        results.append(compare(
            name,
            xgb_predict(xgb, model_dir / f"{stem}.json", cols, cal),
            [r[naive_col] for r in cal], [r[target] for r in cal],
            xgb_predict(xgb, model_dir / f"{stem}.json", cols, ev),
            [r[naive_col] for r in ev], [r[target] for r in ev],
            note=f"naive = {naive_col} alone"))

    in_season("nhl points (anytime point)", pts_b.build_rows(conn),
              md / "nhl_points_walkforward_stability_a_work", "nhl_points", "over_line", "season_avg_points")
    in_season("nhl shots_on_goal (O/U 2.5)", sog_b.build_rows(conn),
              md / "nhl_shots_on_goal_walkforward_stability_a_work", "nhl_shots_on_goal", "over_line",
              "season_avg_shots")
    ml_rows = ml_b.build_rows(conn)
    for r in ml_rows:
        r["naive_strength"] = r["projected_margin"] + (0.15 if r["is_home"] else -0.15)
    in_season("nhl moneyline", ml_rows, md / "nhl_moneyline_walkforward_stability_a_work",
              "nhl_moneyline", "team_won", "naive_strength")

    def prior_rows(mod, season):
        return mod.build_rows(mod.load_weeks123_rows(conn, season),
                              mod.load_prior_season_player_stats(conn, season - 1),
                              mod.load_prior_season_opp_allowed(conn, season - 1),
                              mod.load_prior_season_team_stats(conn, season - 1))

    for mod, name, stem in ((pts_ps, "nhl points_early_season", "nhl_prior_season_points"),
                            (sog_ps, "nhl shots_on_goal_early_season", "nhl_prior_season_shots_on_goal")):
        cal, ev = prior_rows(mod, cal_season), prior_rows(mod, eval_season)
        mdir = md / f"{stem.replace('nhl_prior_season_', 'nhl_prior_season_')}_gate_a_work"
        cols = model_cols(mdir, stem)
        results.append(compare(
            name,
            xgb_predict(xgb, mdir / f"{stem}.json", cols, cal, use_best_iteration=False),
            [r["prior_avg"] for r in cal], [r["target"] for r in cal],
            xgb_predict(xgb, mdir / f"{stem}.json", cols, ev, use_best_iteration=False),
            [r["prior_avg"] for r in ev], [r["target"] for r in ev],
            note="naive = last season's per-game average alone"))

    def ml_prior_rows(season):
        return ml_ps.build_rows(ml_ps.load_weeks123(conn, season),
                                ml_ps.load_full_season_team_stats(conn, season - 1))
    cal, ev = ml_prior_rows(cal_season), ml_prior_rows(eval_season)
    for r in cal + ev:
        r["naive_strength"] = r["prior_projected_margin"] + (0.15 if r["is_home"] else -0.15)
    mdir = md / "nhl_prior_season_moneyline_gate_a_work"
    cols = model_cols(mdir, "nhl_prior_season_moneyline")
    results.append(compare(
        "nhl moneyline_early_season",
        xgb_predict(xgb, mdir / "nhl_prior_season_moneyline.json", cols, cal, use_best_iteration=False),
        [r["naive_strength"] for r in cal], [r["team_won"] for r in cal],
        xgb_predict(xgb, mdir / "nhl_prior_season_moneyline.json", cols, ev, use_best_iteration=False),
        [r["naive_strength"] for r in ev], [r["team_won"] for r in ev],
        note="naive = last season's goal differential gap + home ice"))
    conn.close()
    return results


# ------------------------------------------------------------------ CFB
def audit_cfb(xgb, cal_season=2024, eval_season=2025):
    print(f"\nCFB  (calibrate on {cal_season}, evaluate on {eval_season} -- the in-season models' "
          f"untouched holdout season)")
    sys.path.insert(0, str(REPO))
    import cfb_serving_builder_a as sb
    import cfb_prior_season_early_gate_a as ps_early
    import cfb_prior_season_anytime_touchdowns_gate_a as ps_td
    import cfb_prior_season_moneyline_gate_a as ps_ml

    db = REPO / "cfb_models" / "cfb_model.sqlite"
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    results = []

    naive_feat = {"rushing_yards": "season_avg_rush_yards", "passing_touchdowns": "season_avg_pass_td"}
    for mkt, cfg in sb.MARKETS.items():
        line = cfg["line"]
        cal = sb.SeasonEngine(con, mkt, cal_season).replay()
        ev = sb.SeasonEngine(con, mkt, eval_season).replay()
        cols = cfg["features"]
        path = cfg["model_dir"] / f"{cfg['stem']}.json"
        results.append(compare(
            f"cfb {mkt} (O/U {line})",
            xgb_predict(xgb, path, cols, [r[5] for r in cal]),
            [r[5][naive_feat[mkt]] for r in cal], [1 if r[6] > line else 0 for r in cal],
            xgb_predict(xgb, path, cols, [r[5] for r in ev]),
            [r[5][naive_feat[mkt]] for r in ev], [1 if r[6] > line else 0 for r in ev],
            note=f"naive = {naive_feat[mkt]} alone"))

    cal = sb.AnytimeTouchdownEngine(con, cal_season).replay()
    ev = sb.AnytimeTouchdownEngine(con, eval_season).replay()
    path = sb.ANYTIME_TD_MODEL_DIR / "cfb_anytime_touchdowns.json"
    results.append(compare(
        "cfb anytime_touchdowns",
        xgb_predict(xgb, path, sb.ANYTIME_TD_FEATURES, [r[5] for r in cal]),
        [r[5]["season_avg_total_td"] for r in cal], [1 if r[6] >= 1 else 0 for r in cal],
        xgb_predict(xgb, path, sb.ANYTIME_TD_FEATURES, [r[5] for r in ev]),
        [r[5]["season_avg_total_td"] for r in ev], [1 if r[6] >= 1 else 0 for r in ev],
        note="naive = season_avg_total_td alone"))

    cal = sb.MoneylineEngine(con, cal_season).replay()
    ev = sb.MoneylineEngine(con, eval_season).replay()
    path = sb.MONEYLINE_MODEL_DIR / "cfb_moneyline.json"
    nv = lambda f: f["projected_margin"] + (0 if f["is_neutral_site"] else (2.5 if f["is_home"] else -2.5))
    results.append(compare(
        "cfb moneyline",
        xgb_predict(xgb, path, sb.MONEYLINE_FEATURES, [r[3] for r in cal]),
        [nv(r[3]) for r in cal], [r[4] for r in cal],
        xgb_predict(xgb, path, sb.MONEYLINE_FEATURES, [r[3] for r in ev]),
        [nv(r[3]) for r in ev], [r[4] for r in ev],
        note="naive = season-to-date point differential gap + home field"))

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    for mkt in ("rushing_yards", "passing_touchdowns"):
        cfg, line = ps_early.MARKETS[mkt], ps_early.LINES[mkt]
        def rows(season):
            rr = ps_early.build_rows(ps_early.load_weeks123(conn, season, cfg["position"]),
                                     ps_early.load_full_season_by_player(conn, season - 1, cfg["position"]),
                                     cfg, line)
            return [r for r in rr if r["matched"]]
        cal, ev = rows(cal_season), rows(eval_season)
        path = REPO / "cfb_models" / f"cfb_prior_season_{mkt}.json"
        cols = model_cols(REPO / "cfb_models", f"cfb_prior_season_{mkt}")
        results.append(compare(
            f"cfb {mkt}_early_season",
            xgb_predict(xgb, path, cols, cal, use_best_iteration=False),
            [r["prior_season_avg_stat"] for r in cal], [r["over_line"] for r in cal],
            xgb_predict(xgb, path, cols, ev, use_best_iteration=False),
            [r["prior_season_avg_stat"] for r in ev], [r["over_line"] for r in ev],
            note="naive = last season's per-game average alone"))

    def td_rows(season):
        rr = ps_td.build_rows(ps_td.load_weeks123(conn, season), ps_td.load_full_season_by_player(conn, season - 1))
        return [r for r in rr if r["matched"]]
    cal, ev = td_rows(cal_season), td_rows(eval_season)
    path = REPO / "cfb_models" / "cfb_prior_season_anytime_touchdowns.json"
    cols = model_cols(REPO / "cfb_models", "cfb_prior_season_anytime_touchdowns")
    results.append(compare(
        "cfb anytime_touchdowns_early_season",
        xgb_predict(xgb, path, cols, cal, use_best_iteration=False),
        [r["prior_season_avg_stat"] for r in cal], [r["over_line"] for r in cal],
        xgb_predict(xgb, path, cols, ev, use_best_iteration=False),
        [r["prior_season_avg_stat"] for r in ev], [r["over_line"] for r in ev],
        note="naive = last season's TDs per game alone"))

    def ml_rows(season):
        return ps_ml.build_rows(ps_ml.load_weeks123(conn, season), ps_ml.load_full_season_team_stats(conn, season - 1))
    cal, ev = ml_rows(cal_season), ml_rows(eval_season)
    nv2 = lambda f: f["prior_projected_margin"] + (0 if f["is_neutral_site"] else (2.5 if f["is_home"] else -2.5))
    mdir = REPO / "cfb_models" / "cfb_prior_season_moneyline_gate_a_work"
    cols = model_cols(mdir, "cfb_prior_season_moneyline")
    results.append(compare(
        "cfb moneyline_early_season",
        xgb_predict(xgb, mdir / "cfb_prior_season_moneyline.json", cols, cal, use_best_iteration=False),
        [nv2(r) for r in cal], [r["team_won"] for r in cal],
        xgb_predict(xgb, mdir / "cfb_prior_season_moneyline.json", cols, ev, use_best_iteration=False),
        [nv2(r) for r in ev], [r["team_won"] for r in ev],
        note="naive = last season's point differential gap + home field"))
    con.close(); conn.close()
    return results


# ------------------------------------------------------------------ NFL
def audit_nfl(xgb, db, csv_dir, cal_season=2024, eval_season=2025):
    """db: an nfl_model.sqlite-schema db holding real 2023+ player_games
    (build one from nflverse CSVs with nfl_player_games_foundation_a.py
    --player-stats-csv/--schedules-csv). csv_dir: stats_player_week_{yr}.csv
    files, for the CSV-based prior-season early gate."""
    print(f"\nNFL  (calibrate on {cal_season}, evaluate on {eval_season}; early-season markets "
          f"also evaluated on the live 2026 weeks 1-3)")
    sys.path.insert(0, str(REPO))
    import nfl_serving_builder_a as sb
    import nfl_prior_season_early_gate_a as ps_early
    import nfl_prior_season_anytime_touchdowns_gate_a as ps_td
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    results = []

    naive_feat = {"rushing_yards": "season_avg_rush_yards", "receiving_yards": "season_avg_rec_yards",
                  "sacks": "season_avg_sacks"}
    for mkt, cfg in sb.MARKETS.items():
        line = cfg["line"]
        cal = sb.SeasonEngine(con, mkt, cal_season).replay()
        ev = sb.SeasonEngine(con, mkt, eval_season).replay()
        path = cfg["model_dir"] / f"{cfg['stem']}.json"
        results.append(compare(
            f"nfl {mkt} (O/U {line})",
            xgb_predict(xgb, path, cfg["features"], [r[5] for r in cal]),
            [r[5][naive_feat[mkt]] for r in cal], [1 if r[6] > line else 0 for r in cal],
            xgb_predict(xgb, path, cfg["features"], [r[5] for r in ev]),
            [r[5][naive_feat[mkt]] for r in ev], [1 if r[6] > line else 0 for r in ev],
            note=f"naive = {naive_feat[mkt]} alone"))

    cal = sb.AnytimeTouchdownEngine(con, cal_season).replay()
    ev = sb.AnytimeTouchdownEngine(con, eval_season).replay()
    path = sb.ANYTIME_TD_MODEL_DIR / "nfl_anytime_touchdowns.json"
    results.append(compare(
        "nfl anytime_touchdowns",
        xgb_predict(xgb, path, sb.ANYTIME_TD_FEATURES, [r[5] for r in cal]),
        [r[5]["season_avg_total_td"] for r in cal], [1 if r[6] >= 1 else 0 for r in cal],
        xgb_predict(xgb, path, sb.ANYTIME_TD_FEATURES, [r[5] for r in ev]),
        [r[5]["season_avg_total_td"] for r in ev], [1 if r[6] >= 1 else 0 for r in ev],
        note="naive = season_avg_total_td alone"))

    ps_early.CSV_DIR = Path(csv_dir)
    raw = {y: ps_early.load_season_csv(y) for y in (cal_season - 1, cal_season, eval_season, 2026)}
    for mkt, cfg in ps_early.MARKETS.items():
        def rows(season):
            rr = ps_early.build_rows(ps_early.load_weeks123(raw[season], cfg["position"]),
                                     ps_early.load_full_season_by_player(raw[season - 1], cfg["position"]), cfg)
            return [r for r in rr if r["matched"]]
        cal = rows(cal_season)
        path = REPO / "nfl_models" / f"nfl_prior_season_{mkt}.json"
        cols = model_cols(REPO / "nfl_models", f"nfl_prior_season_{mkt}")
        for ev_season in (eval_season, 2026):
            ev = rows(ev_season)
            results.append(compare(
                f"nfl {mkt}_early_season [{ev_season}]",
                xgb_predict(xgb, path, cols, cal, use_best_iteration=False),
                [r["prior_season_avg_yards"] for r in cal], [r["over_line"] for r in cal],
                xgb_predict(xgb, path, cols, ev, use_best_iteration=False),
                [r["prior_season_avg_yards"] for r in ev], [r["over_line"] for r in ev],
                note="naive = last season's per-game average alone"))

    def td_rows(season):
        rr = ps_td.build_rows(ps_td.load_weeks123(con, season), ps_td.load_full_season_by_player(con, season - 1))
        return [r for r in rr if r.get("matched", True)]
    cal = td_rows(cal_season)
    mdir = REPO / "nfl_models" / "nfl_prior_season_anytime_touchdowns_gate_a_work"
    cols = model_cols(mdir, "nfl_prior_season_anytime_touchdowns")
    for ev_season in (eval_season, 2026):
        ev = td_rows(ev_season)
        results.append(compare(
            f"nfl anytime_touchdowns_early_season [{ev_season}]",
            xgb_predict(xgb, mdir / "nfl_prior_season_anytime_touchdowns.json", cols, cal, use_best_iteration=False),
            [r["prior_season_avg_stat"] for r in cal], [r["over_line"] for r in cal],
            xgb_predict(xgb, mdir / "nfl_prior_season_anytime_touchdowns.json", cols, ev, use_best_iteration=False),
            [r["prior_season_avg_stat"] for r in ev], [r["over_line"] for r in ev],
            note="naive = last season's TDs per game alone"))
    con.close()
    return results


# --------------------------------------------------------------- TENNIS
def audit_tennis():
    """Tennis's own champion gates already scored every market against a
    naive baseline on large real holdouts -- read those verdicts directly
    rather than re-deriving them."""
    print("\nTENNIS  (from each market's own gate report, which already compared against naive)")
    out = []
    for mkt in ("total_games", "moneyline", "set_betting"):
        for tour, suffix in (("atp", ""), ("wta", "_wta")):
            path = REPO / f"tennis_{mkt}_gate_report{suffix}.json"
            if not path.exists():
                continue
            d = json.loads(path.read_text())
            if "naive_holdout_mae" in d:
                gain = d["mae_improvement_pct_vs_naive_holdout"]
                p = d["p_model_error_worse_than_naive_holdout"]
                detail = {"model_mae": round(d["model_holdout_mae"], 3), "naive_mae": round(d["naive_holdout_mae"], 3),
                          "mae_improvement_pct": round(gain * 100, 2)}
                verdict = "BEATS_NAIVE" if gain >= 0.02 and p < MAX_P else "NO_REAL_EDGE"
            elif "naive_holdout" in d:
                ma, na = d["holdout"]["accuracy"], d["naive_holdout"]["accuracy"]
                p = 1 - d["p_naive_worse_than_model_holdout"]
                detail = {"model_accuracy": round(ma, 4), "naive_accuracy": round(na, 4)}
                verdict = "BEATS_NAIVE" if ma - na >= 0.01 and p < MAX_P else "NO_REAL_EDGE"
            else:
                p = d.get("p_model_not_better_than_naive_holdout", 1.0)
                detail = {"naive_accuracy": d.get("holdout_naive_accuracy")}
                verdict = "BEATS_NAIVE" if d.get("passed") and p < MAX_P else "NO_REAL_EDGE"
            res = {"market": f"tennis {mkt} ({tour})", "verdict": verdict, "p_model_not_better": round(p, 4),
                   "gate_passed": d.get("passed"), **detail,
                   "note": f"from {path.name} (large real holdout, n={d.get('n_holdout_observations', '?')})"}
            print(f"  {res['market']:38s} {detail}  p={p:.3f}  -> {verdict}")
            out.append(res)
    return out


def audit_nfl_projections(log_path, csv_dir):
    """The real-odds yardage simulator's continuous projections (what 201 of
    202 NFL picks on a typical board actually are) vs last season's per-game
    average, on every logged projection with a real result."""
    import csv, re, statistics, unicodedata
    def norm(n):
        n = unicodedata.normalize("NFKD", n or ""); n = "".join(c for c in n if not unicodedata.combining(c)).lower()
        n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\.?\b", "", n); n = re.sub(r"[^a-z ]", "", n)
        return re.sub(r"\s+", " ", n).strip()
    hist = {}
    for yr in (2025, 2026):
        p = Path(csv_dir) / f"stats_player_week_{yr}.csv"
        for r in csv.DictReader(open(p)):
            if r.get("season_type", "REG") != "REG":
                continue
            hist.setdefault(norm(r["player_display_name"]), {})[(yr, int(r["week"]))] = r
    sys_err, naive_err = [], []
    for line in open(log_path):
        if not line.strip():
            continue
        l = json.loads(line)
        if l.get("market") not in ("rushing_yards", "receiving_yards") or l.get("season") != 2026:
            continue
        proj = l.get("projected_mean") or l.get("projected_median")
        h = hist.get(norm(l.get("player")), {})
        act = h.get((2026, l.get("week")))
        prior = [float(v[l["market"]] or 0) for (y, _), v in h.items() if y == 2025]
        if proj is None or act is None or not prior:
            continue
        a = float(act[l["market"]] or 0)
        sys_err.append(abs(a - proj)); naive_err.append(abs(a - statistics.mean(prior)))
    if not sys_err:
        return []
    ms, mn = statistics.mean(sys_err), statistics.mean(naive_err)
    verdict = "BEATS_NAIVE" if (mn - ms) / mn >= 0.05 else "NO_REAL_EDGE"
    print(f"\nNFL yardage projections (simulator)   n={len(sys_err)}  avg miss {ms:.1f} yds vs last-season avg "
          f"{mn:.1f} yds  -> {verdict}")
    return [{"market": "nfl rushing/receiving projections (simulator)", "n_eval": len(sys_err),
             "model_mae_yds": round(ms, 2), "naive_mae_yds": round(mn, 2), "verdict": verdict,
             "note": "naive = last season's per-game average; every logged 2026 projection with a real result"}]


def main():
    import argparse
    import xgboost as xgb
    ap = argparse.ArgumentParser()
    ap.add_argument("--nfl-db", help="nfl_model.sqlite-schema db with real 2023+ player_games")
    ap.add_argument("--nfl-csv-dir", help="dir of nflverse stats_player_week_{year}.csv files")
    args = ap.parse_args()
    print("BASELINE_AUDIT_A\n================")
    report = {"generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
              "method": __doc__.strip().split("\n\n")[2], "markets": []}
    report["markets"] += audit_nhl(xgb)
    report["markets"] += audit_cfb(xgb)
    if args.nfl_db and args.nfl_csv_dir:
        report["markets"] += audit_nfl(xgb, args.nfl_db, args.nfl_csv_dir)
        report["markets"] += audit_nfl_projections(REPO / "docs" / "nfl_picks_log.jsonl", args.nfl_csv_dir)
    else:
        print("\nNFL skipped (needs --nfl-db and --nfl-csv-dir)")
    report["markets"] += audit_tennis()
    from collections import Counter
    report["summary"] = dict(Counter(m["verdict"] for m in report["markets"]))
    print(f"\nsummary: {report['summary']}")
    OUT.write_text(json.dumps(report, indent=2))
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    raise SystemExit(main())
