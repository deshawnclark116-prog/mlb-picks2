#!/usr/bin/env python3
"""NHL Outcome Engine V2 - Phase0 forensic audit (research only, promotes nothing).

Grades the CURRENT NHL threshold classifiers on their native binary targets and builds the central /
probability baselines Phase1 must beat. Evidence grade of every number: RETROSPECTIVE_BURNED (see
protocol_amendment_1.json). No betting-market input. No sampling simulation. Seasons > 2025 are never loaded."""
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nhl_v2_metrics as M
import nhl_v2_phase0_data as D
import nhl_goalie_saves_clean_baseline_a as g_base
import nhl_moneyline_clean_baseline_a as ml_base
import nhl_moneyline_champion_gate_a as ml_gate
import nhl_points_clean_baseline_a as pts_base
import nhl_shots_on_goal_clean_baseline_a as sog_base
from cfb_rushing_yards_champion_gate_b import apply_platt, fit_platt

REPO = Path(__file__).resolve().parent
OUT = REPO / "nhl_models" / "nhl_player_outcome_v2"
DB = REPO / "nhl_models" / "nhl_model.sqlite"
GRADE = "RETROSPECTIVE_BURNED"
THR = {"shots": 3, "points": 1}
SAVES_THR = 25
K_GRID = (5, 10, 20, 40)
SHRINK_GAMES = 20          # human rate shrink (games of position-average exposure)
OPP_DAMP = 0.5
SV_SHRINK_SHOTS = 400.0
DISP_LIMIT = 1.15
SEED = 20261007

MODELS = {
    "shots": ("nhl_shots_on_goal_walkforward_stability_a_work", "nhl_shots_on_goal"),
    "points": ("nhl_points_walkforward_stability_a_work", "nhl_points"),
    "saves": ("nhl_goalie_saves_champion_gate_a_work", "nhl_goalie_saves"),
    "moneyline": ("nhl_moneyline_walkforward_stability_a_work", "nhl_moneyline"),
}


def sha_file(p):
    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()


def wjson(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def wgz(path, rows):
    with gzip.GzipFile(str(path), "wb", mtime=0) as f:
        for r in rows:
            f.write((json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n").encode())


# ---------------------------------------------------------------- incumbent scoring
def xgb_raw(market, rows):
    import xgboost as xgb
    d, stem = MODELS[market]
    bst = xgb.Booster(); bst.load_model(str(REPO / "nhl_models" / d / (stem + ".json")))
    cols = json.loads((REPO / "nhl_models" / d / (stem + "_columns.json")).read_text())
    X = np.array([[r[c] if r.get(c) is not None else np.nan for c in cols] for r in rows], dtype=np.float32)
    return np.asarray(bst.predict(xgb.DMatrix(X, feature_names=cols), iteration_range=(0, bst.best_iteration + 1)), float), cols


def causal_platt(pool_rows, pool_raw, pool_y, tgt_rows, tgt_raw, season):
    """As-served calibration: before each week w of `season`, refit a Platt map on the PREVIOUS season
    plus current-season pool rows with week < w, then apply to target rows of week w."""
    pool_raw = np.asarray(pool_raw, float); pool_y = np.asarray(pool_y, float); tgt_raw = np.asarray(tgt_raw, float)
    ps = np.array([r["season"] for r in pool_rows]); pw = np.array([r["week"] for r in pool_rows])
    ts = np.array([r["season"] for r in tgt_rows]); tw = np.array([r["week"] for r in tgt_rows])
    idx_t = np.where(ts == season)[0]
    idx_w = np.where(ps == season - 1)[0]
    cur_p = np.where(ps == season)[0]
    out = np.array(tgt_raw, float)
    for w in sorted(set(tw[idx_t].tolist())):
        pool = np.concatenate([idx_w, cur_p[pw[cur_p] < w]])
        a, b = fit_platt(pool_raw[pool], pool_y[pool]) if len(pool) > 50 else (1.0, 0.0)
        if a <= 0:
            a, b = 1.0, 0.0
        cur = idx_t[tw[idx_t] == w]
        out[cur] = apply_platt(tgt_raw[cur], a, b)
    return out, idx_t


def logit_fit(x, y, iters=60):
    x = np.asarray(x, float); y = np.asarray(y, float)
    a, b = 0.0, 0.0
    for _ in range(iters):
        q = 1 / (1 + np.exp(-(a + b * x))); w = q * (1 - q)
        g = np.array([np.sum(y - q), np.sum((y - q) * x)])
        H = np.array([[np.sum(w), np.sum(w * x)], [np.sum(w * x), np.sum(w * x * x)]])
        st = np.linalg.solve(H, g); a += st[0]; b += st[1]
        if np.max(np.abs(st)) < 1e-10:
            break
    return float(a), float(b)


# ---------------------------------------------------------------- skater baselines
def fit_constants(rows):
    fit = [r for r in rows if r["season"] in D.FIT_SEASONS]
    c = {"rate_hr": {}, "toi_hr": {}, "p0": {}, "home_ratio": {}}
    for g in ("F", "D"):
        rg = [r for r in fit if r["pos"] == g]
        T = 0.0
        for r in rg:
            T += r["toi"] / 3600.0
        c["toi_hr"][g] = T / len(rg)
        for s in D.STATS:
            tot = 0.0
            for r in rg:
                tot += r["act_" + s]
            c["rate_hr"][(g, s)] = tot / T
        for s in ("shots", "points"):
            hits = 0.0
            for r in rg:
                hits += 1.0 if r["act_" + s] >= THR[s] else 0.0
            c["p0"][(g, s)] = hits / len(rg)
    for s in ("shots", "points"):
        allr = np.mean([1.0 if r["act_" + s] >= THR[s] else 0.0 for r in fit])
        hr = np.mean([1.0 if r["act_" + s] >= THR[s] else 0.0 for r in fit if r["home"]])
        ar = np.mean([1.0 if r["act_" + s] >= THR[s] else 0.0 for r in fit if not r["home"]])
        c["home_ratio"][s] = (float(hr / allr), float(ar / allr))
        c["p_all_" + s] = float(allr)
    return c


def opp_factor(r, s):
    oa, lg = r["opp_allowed_" + s], r["league_" + s]
    if oa is None or lg is None or lg <= 0:
        return 1.0
    return 1.0 + OPP_DAMP * (oa / lg - 1.0)


def human_mu(r, s, c):
    g = r["pos"]; r0 = c["rate_hr"][(g, s)]; h0 = SHRINK_GAMES * c["toi_hr"][g]
    rate = (r["S_" + s] + r0 * h0) / (r["T_hr"] + h0)
    return (r["pred_toi"] / 3600.0) * rate * opp_factor(r, s)


def simple_central(r, s, c):
    g = r["pos"]
    return {"prior3_mean": r["m3_" + s], "prior5_mean": r["m5_" + s], "prior10_mean": r["m10_" + s],
            "ewma_0.2": r["ewma_" + s],
            "season_rate_x_prior3_toi": (r["S_" + s] / r["T_hr"]) * (r["pred_toi"] / 3600.0) if r["T_hr"] > 0 else 0.0,
            "fit_position_mean": c["rate_hr"][(g, s)] * c["toi_hr"][g]}


def prob_B(r, s, k, c):
    p0 = c["p0"][(r["pos"], s)]
    return (r["hits_" + s] + k * p0) / (r["n"] + k)


def prob_C(r, s, k, c):
    hr, ar = c["home_ratio"][s]
    return float(np.clip(prob_B(r, s, k, c) * opp_factor(r, s) * (hr if r["home"] else ar), 0.001, 0.999))


def dist_for(rows, mus, s):
    sel = [i for i, r in enumerate(rows) if r["season"] in D.FIT_SEASONS + (D.SEL_SEASON,)]
    y = np.array([rows[i]["act_" + s] for i in sel], float); mu = np.array([mus[i] for i in sel], float)
    idx, rr = M.dispersion(mu, y)
    if idx <= DISP_LIMIT or rr is None or rr <= 0:
        return {"family": "poisson", "dispersion_index": M.R(idx), "r": None}
    return {"family": "negbin", "dispersion_index": M.R(idx), "r": M.R(rr)}


# ---------------------------------------------------------------- verdict logic (amendment 1 / 2)
def season_verdict(inc, bases, diffs):
    """inc: served prob metrics; bases: {name: prob metrics} for A,B,C,D; diffs: {name: bootstrap of Brier(inc)-Brier(base)}."""
    ece = inc["ece_10bin"]
    simple = [n for n in ("A", "B", "C") if n in bases]
    best_simple = min(simple, key=lambda n: bases[n]["brier"])
    best_all = min(bases, key=lambda n: bases[n]["brier"])
    d_simple = diffs[best_simple]
    beats_all = all(inc["brier"] < bases[n]["brier"] for n in bases)
    if d_simple["ci95_lo"] > 0 or ece > 0.05:
        v = "REJECTED"
    elif beats_all and diffs[best_all]["ci95_hi"] < 0 and ece <= 0.03:
        v = "SURVIVES"
    else:
        v = "WEAK"
    return {"verdict": v, "best_simple_baseline": best_simple, "best_baseline_any": best_all, "ece": ece,
            "diff_vs_best_simple": d_simple, "diff_vs_best_any": diffs[best_all]}


def combine(verdicts):
    order = {"SURVIVES": 0, "WEAK": 1, "REJECTED": 2}
    w = max(verdicts, key=lambda v: order[v])
    return {"SURVIVES": "CLASSIFIER_SURVIVES_NATIVE_TARGET_AUDIT", "WEAK": "CLASSIFIER_WEAK_NATIVE_TARGET",
            "REJECTED": "CLASSIFIER_REJECTED_NATIVE_TARGET"}[w]


def eval_prob_block(y, p_inc_raw, p_inc_served, base_probs, blocks, base_rate_fit):
    res = {"incumbent_raw_xgb": M.prob_metrics(p_inc_raw, y, base_rate_fit),
           "incumbent_as_served_causal_platt": M.prob_metrics(p_inc_served, y, base_rate_fit), "baselines": {}}
    for n, p in base_probs.items():
        res["baselines"][n] = M.prob_metrics(p, y, base_rate_fit)
    diffs = {}
    for n, p in base_probs.items():
        d = (np.asarray(p_inc_served) - y) ** 2 - (np.asarray(p) - y) ** 2
        diffs[n] = M.paired_block_bootstrap(d, blocks, seed=SEED)
    res["brier_diff_incumbent_minus_baseline"] = diffs
    return res, diffs


def posthoc_level_sensitivity(y, served, blocks, hits, n, k, p_prev):
    """POST-HOC (declared after first results; NOT used for any verdict): baselines re-centred on the PREVIOUS season's
    event rate, i.e. given the same causal level update the as-served incumbent enjoys."""
    y = np.asarray(y, float); served = np.asarray(served, float)
    a = np.full(len(y), p_prev)
    b = (np.asarray(hits, float) + k * p_prev) / (np.asarray(n, float) + k)
    out = {"label": "POST_HOC_SENSITIVITY_NOT_USED_FOR_VERDICTS", "previous_season_event_rate": M.R(p_prev), "baselines": {}}
    for nm, p in (("A_prev_season_rate", a), ("B_shrunk_to_prev_season_rate", b)):
        d = (served - y) ** 2 - (p - y) ** 2
        out["baselines"][nm] = {"brier": M.R(M.brier(p, y)), "log_loss": M.R(M.logloss(p, y)),
                                "brier_diff_incumbent_minus_baseline": M.paired_block_bootstrap(d, blocks, seed=SEED)}
    return out


def skater_market(market, srows, inc_rows, mem_dates):
    s = market
    key = {(r["player_id"], r["game_id"]): i for i, r in enumerate(inc_rows)}
    mine = {(r["pid"], r["gid"]): r for r in srows}
    assert len(key) == len(mine) and set(key) == set(mine), "incumbent/own eligibility mismatch"
    c = fit_constants(srows)
    mus = [human_mu(r, s, c) for r in srows]
    dist = dist_for(srows, mus, s)
    sel_rows = [r for r in srows if r["season"] == D.SEL_SEASON]
    best_k = None
    for k in K_GRID:
        pr = np.array([prob_B(r, s, k, c) for r in sel_rows]); yy = np.array([1.0 if r["act_" + s] >= THR[s] else 0.0 for r in sel_rows])
        ll = M.logloss(pr, yy)
        if best_k is None or ll < best_k[1] - 1e-12:
            best_k = (k, ll)
    k = best_k[0]
    y_all = np.array([r["over_line"] for r in inc_rows], float)
    raw, cols = xgb_raw(market, inc_rows)
    out = {"constants": {"p_all_fit": c["p_all_" + s], "shrink_k_selected_on_2023": k, "count_distribution": dist,
                         "home_ratio": list(c["home_ratio"][s])}, "seasons": {}, "feature_columns": cols}
    eval_pack = {}
    for season in D.EVAL_SEASONS:
        served, idx_t = causal_platt(inc_rows, raw, y_all, inc_rows, raw, season)
        sub = [inc_rows[i] for i in idx_t]
        mr = [mine[(r["player_id"], r["game_id"])] for r in sub]
        mu_h = np.array([human_mu(r, s, c) for r in mr])
        y = y_all[idx_t]
        base = {"A": np.full(len(y), c["p_all_" + s]), "B": np.array([prob_B(r, s, k, c) for r in mr]),
                "C": np.array([prob_C(r, s, k, c) for r in mr]), "D": M.map_mu_to_prob(mu_h, THR[s], dist)}
        blocks = [r["date"] for r in mr]
        res, diffs = eval_prob_block(y, raw[idx_t], served[idx_t], base, blocks, c["p_all_" + s])
        res["verdict_detail"] = season_verdict(res["incumbent_as_served_causal_platt"], res["baselines"], diffs)
        # central baselines (floor Phase1 must beat)
        act = np.array([r["act_" + s] for r in mr], float)
        cen = {"human_frozen": M.count_metrics(mu_h, act)}
        simple = {nm: [simple_central(r, s, c)[nm] for r in mr] for nm in simple_central(mr[0], s, c)}
        for nm, v in simple.items():
            cen[nm] = M.count_metrics(v, act)
        res["central_baselines"] = cen
        res["simple_central_mapped_to_threshold_probability"] = {
            nm: M.prob_metrics(M.map_mu_to_prob(np.asarray(v, float), THR[s], dist), y, c["p_all_" + s]) for nm, v in simple.items()}
        prev = [1.0 if r["act_" + s] >= THR[s] else 0.0 for r in srows if r["season"] == season - 1]
        res["posthoc_level_sensitivity"] = posthoc_level_sensitivity(
            y, served[idx_t], blocks, [r["hits_" + s] for r in mr], [r["n"] for r in mr], k, float(np.mean(prev)))
        rr_mse = float(np.mean((mu_h - act) ** 2))
        res["count_error_vs_dispersion_floor"] = {"mse_of_human_mu": M.R(rr_mse), "mean_human_mu": M.R(mu_h.mean()),
                                                  "locked_dispersion_index": dist["dispersion_index"],
                                                  "mse_over_mean_mu": M.R(rr_mse / float(mu_h.mean())),
                                                  "reading": "if mse/mean(mu) is near the locked dispersion index, the count error is close to the variance a perfect-mean Poisson/NB would still show"}
        out["seasons"][str(season)] = res
        eval_pack[season] = {"rows": sub, "mine": mr, "raw": raw[idx_t], "served": served[idx_t], "mu": mu_h, "y": y, "act": act, "base": base}
    out["native_target_verdict"] = combine([out["seasons"][str(x)]["verdict_detail"]["verdict"] for x in D.EVAL_SEASONS])
    out["architecture_verdict"] = "ARCHITECTURE_INSUFFICIENT_NO_CENTRAL_PROJECTION"
    out["evidence_grade"] = GRADE
    return out, eval_pack, c, dist


def reference_stats(srows, c):
    """Goals / assists / points central references (not predicted by the incumbent)."""
    res = {}
    for season in D.EVAL_SEASONS:
        rr = [r for r in srows if r["season"] == season]
        res[str(season)] = {}
        for st in ("goals", "assists", "points"):
            act = np.array([r["act_" + st] for r in rr], float)
            res[str(season)][st] = {"human_frozen": M.count_metrics([human_mu(r, st, c) for r in rr], act),
                                    "prior10_mean": M.count_metrics([r["m10_" + st] for r in rr], act)}
    return res


def sog_oracle(pack, c):
    out = {}
    for season, pk in pack.items():
        mr = pk["mine"]
        pred_T = np.array([r["pred_toi"] / 3600.0 for r in mr]); act_T = np.array([r["toi"] / 3600.0 for r in mr])
        rate = np.array([(r["S_shots"] + c["rate_hr"][(r["pos"], "shots")] * SHRINK_GAMES * c["toi_hr"][r["pos"]]) /
                         (r["T_hr"] + SHRINK_GAMES * c["toi_hr"][r["pos"]]) for r in mr])
        act = pk["act"]; ok = act_T > 0
        act_rate = np.where(ok, act / np.where(ok, act_T, 1), 0.0)
        opp = np.array([opp_factor(r, "shots") for r in mr])
        full = pred_T * rate * opp
        variants = {"predicted_TOI_x_predicted_rate_x_opp(full predictive)": full,
                    "ORACLE_actual_TOI_x_predicted_rate_x_opp": act_T * rate * opp,
                    "ORACLE_predicted_TOI_x_actual_game_rate": pred_T * act_rate}
        mse0 = float(np.mean((full - act) ** 2))
        o = {}
        for nm, v in variants.items():
            mse = float(np.mean((v - act) ** 2))
            o[nm] = {"mae": M.R(np.mean(np.abs(v - act))), "mse": M.R(mse), "mse_removed_vs_full": M.R(1 - mse / mse0)}
        o["toi_prediction_error"] = {"mae_minutes": M.R(np.mean(np.abs(pred_T - act_T)) * 60), "bias_pred_minus_actual_minutes": M.R(np.mean(pred_T - act_T) * 60)}
        o["note"] = ("ORACLE variants use postgame information; descriptive attribution only, never a predictive model. The actual-rate oracle absorbs the realised outcome "
                     "(rate = shots / actual TOI), so its large MSE reduction mixes true rate predictability with irreducible count variance and must not be read as recoverable skill")
        out[str(season)] = o
    return out


# ---------------------------------------------------------------- goalie saves (A vs B)
def goalie_features(gr, state):
    out = []
    for r in gr:
        ts = state.get((r["season"], r["team"], r["week"])); os_ = state.get((r["season"], r["opp"], r["week"]))
        tm = ts["net_margin"] if ts else None; om = os_["net_margin"] if os_ else None
        f = dict(r)
        f.update({"season_avg_saves": r["S_sv"] / r["n"], "recent3_avg_saves": r["m3"], "recent5_avg_saves": r["m5"],
                  "season_avg_shots_against": r["S_sa"] / r["n"], "recent3_avg_shots_against": r["m3_sa"],
                  "save_pct": (r["S_sv"] / r["S_sa"]) if r["S_sa"] > 0 else None,
                  "opp_shots_for_per_game": r["opp_for"], "is_home": float(r["home"]), "games_played": r["n"],
                  "team_net_margin": tm, "opp_net_margin": om,
                  "projected_margin": (tm - om) if (tm is not None and om is not None) else None,
                  "over_line": 1 if (r["act_saves"] or 0) >= SAVES_THR else 0})
        out.append(f)
    return out


def goalie_baselines(rows, label):
    fit = [r for r in rows if r["season"] in D.FIT_SEASONS]
    sv0 = sum(r["act_saves"] for r in fit) / float(sum(r["act_sa"] for r in fit))
    mean_sv = float(np.mean([r["act_saves"] for r in fit]))
    p_all = float(np.mean([r["over_line"] for r in fit]))
    p0 = p_all

    def mu_h(r):
        L = r["league_sa"]; o = r["opp_for"]
        sa_hat = (L + OPP_DAMP * (o - L)) if (L and o is not None) else r["S_sa"] / r["n"]
        sv = (r["S_sv"] + SV_SHRINK_SHOTS * sv0) / (r["S_sa"] + SV_SHRINK_SHOTS)
        return sa_hat * sv

    def pB(r, k):
        return (r["hits"] + k * p0) / (r["n"] + k)

    def pC(r, k):
        L = r["league_sa"]; o = r["opp_for"]
        f = 1.0 + OPP_DAMP * (o / L - 1.0) if (L and o is not None and L > 0) else 1.0
        return float(np.clip(pB(r, k) * f, 0.001, 0.999))
    mus = [mu_h(r) for r in rows]
    sel = [i for i, r in enumerate(rows) if r["season"] in D.FIT_SEASONS + (D.SEL_SEASON,)]
    idx, rr = M.dispersion([mus[i] for i in sel], [rows[i]["act_saves"] for i in sel])
    dist = ({"family": "poisson", "dispersion_index": M.R(idx), "r": None} if (idx <= DISP_LIMIT or rr is None or rr <= 0)
            else {"family": "negbin", "dispersion_index": M.R(idx), "r": M.R(rr)})
    selr = [r for r in rows if r["season"] == D.SEL_SEASON]
    bestk = None
    for k in K_GRID:
        ll = M.logloss([pB(r, k) for r in selr], [r["over_line"] for r in selr])
        if bestk is None or ll < bestk[1] - 1e-12:
            bestk = (k, ll)
    return {"mu": mu_h, "pB": pB, "pC": pC, "dist": dist, "k": bestk[0], "p_all": p_all, "sv0": sv0, "mean_sv": mean_sv}


def goalie_block(mem, opp_for, league_sa, state, inc_a_rows):
    A_own = D.goalie_rows(mem, opp_for, league_sa, True)
    B_own = D.goalie_rows(mem, opp_for, league_sa, False)
    A = goalie_features(A_own, state); B = goalie_features(B_own, state)
    # integrity: own A population/features == incumbent baseline rows
    ka = {(r["player_id"], r["game_id"]): r for r in inc_a_rows}
    ma = {(r["pid"], r["gid"]): r for r in A}
    same_pop = set(ka) == set(ma)
    maxdiff = 0.0
    for kk, r in ka.items():
        o = ma.get(kk)
        if o is None:
            continue
        for col in ("season_avg_saves", "recent3_avg_saves", "recent5_avg_saves", "season_avg_shots_against",
                    "recent3_avg_shots_against", "save_pct", "opp_shots_for_per_game", "games_played"):
            if r[col] is None or o[col] is None:
                continue
            maxdiff = max(maxdiff, abs(float(r[col]) - float(o[col])))
    out = {"integrity_own_A_vs_incumbent": {"same_population": bool(same_pop), "n_incumbent": len(ka), "n_own": len(ma), "max_feature_abs_diff": M.R(maxdiff, 9)}}
    # leakage accounting
    leak = {}
    for season in range(2018, 2026):
        a_n = sum(1 for r in A if r["season"] == season); b_n = sum(1 for r in B if r["season"] == season)
        extra = [r for r in B if r["season"] == season and not r["qual"]]
        leak[str(season)] = {"A_rows": a_n, "B_rows": b_n, "rows_added_by_removing_target_toi_filter": len(extra),
                             "added_mean_saves": M.R(np.mean([r["act_saves"] or 0 for r in extra])) if extra else None,
                             "added_event_rate_saves_ge_25": M.R(np.mean([r["over_line"] for r in extra])) if extra else None,
                             "A_event_rate": M.R(np.mean([r["over_line"] for r in A if r["season"] == season])),
                             "B_event_rate": M.R(np.mean([r["over_line"] for r in B if r["season"] == season]))}
    out["selection_leakage_accounting"] = leak
    out["code_path"] = {"file": "nhl_goalie_saves_clean_baseline_a.py", "function": "build_rows", "sql": "WHERE gg.toi_seconds >= 1800",
                        "field": "goalie_games.toi_seconds of the TARGET game", "also_in": "nhl_serving_builder_a.build_goalie_final_states (serving state)",
                        "changes_training_population": True, "changes_scoring_population": True, "changes_labels": False,
                        "changes_features": "no (history already limited to prior qualifying appearances, which is pregame-knowable)",
                        "effect": "excludes goalies pulled early or used in relief; those games have fewer saves, so the kept population is selected on a postgame outcome-correlated field"}
    raw_a, cols = xgb_raw("saves", inc_a_rows)
    ya = np.array([r["over_line"] for r in inc_a_rows], float)
    raw_b, _ = xgb_raw("saves", B)
    yb = np.array([r["over_line"] for r in B], float)
    out["seasons"] = {}
    packs = {}
    for season in D.EVAL_SEASONS:
        srv_a, ia = causal_platt(inc_a_rows, raw_a, ya, inc_a_rows, raw_a, season)
        srv_b, ib = causal_platt(inc_a_rows, raw_a, ya, B, raw_b, season)  # calibration pool = incumbent (A) as served
        out["seasons"][str(season)] = {}
        for label, rows_, rawv, srv, idx, y_ in (("A_CURRENT_AS_IMPLEMENTED", A_ref(inc_a_rows, ma), raw_a, srv_a, ia, ya),
                                                  ("B_APPEARANCE_CONDITIONAL_NO_TARGET_TOI_FILTER", B, raw_b, srv_b, ib, yb)):
            pop = A if label.startswith("A") else B
            bl = goalie_baselines(pop, label)
            if label.startswith("A"):
                sub = [ma[(inc_a_rows[i]["player_id"], inc_a_rows[i]["game_id"])] for i in idx]
            else:
                sub = [B[i] for i in idx]
            y = y_[idx]
            mu = np.array([bl["mu"](r) for r in sub])
            base = {"A": np.full(len(y), bl["p_all"]), "B": np.array([bl["pB"](r, bl["k"]) for r in sub]),
                    "C": np.array([bl["pC"](r, bl["k"]) for r in sub]), "D": M.map_mu_to_prob(mu, SAVES_THR, bl["dist"])}
            blocks = [r["date"] for r in sub]
            res, diffs = eval_prob_block(y, rawv[idx], srv[idx], base, blocks, bl["p_all"])
            res["verdict_detail"] = season_verdict(res["incumbent_as_served_causal_platt"], res["baselines"], diffs)
            act = np.array([r["act_saves"] or 0 for r in sub], float)
            cen = {"human_frozen": M.count_metrics(mu, act, within=(2, 4, 6), miss=(8, 10)),
                   "prior3_mean": M.count_metrics([r["m3"] for r in sub], act, within=(2, 4, 6), miss=(8, 10)),
                   "prior5_mean": M.count_metrics([r["m5"] for r in sub], act, within=(2, 4, 6), miss=(8, 10)),
                   "prior10_mean": M.count_metrics([r["m10"] for r in sub], act, within=(2, 4, 6), miss=(8, 10)),
                   "own_prior_sa_x_shrunk_sv": M.count_metrics([(r["S_sa"] / r["n"]) * ((r["S_sv"] + SV_SHRINK_SHOTS * bl["sv0"]) / (r["S_sa"] + SV_SHRINK_SHOTS)) for r in sub], act, within=(2, 4, 6), miss=(8, 10)),
                   "fit_mean": M.count_metrics(np.full(len(act), bl["mean_sv"]), act, within=(2, 4, 6), miss=(8, 10))}
            prev_rows = [r for r in pop if r["season"] == season - 1]
            res["posthoc_level_sensitivity"] = posthoc_level_sensitivity(
                y, srv[idx], blocks, [r["hits"] for r in sub], [r["n"] for r in sub], bl["k"], float(np.mean([r["over_line"] for r in prev_rows])))
            res["central_baselines"] = cen
            res["constants"] = {"shrink_k": bl["k"], "count_distribution": bl["dist"], "p_all_fit": M.R(bl["p_all"]), "league_sv_fit": M.R(bl["sv0"])}
            # oracle forensics (descriptive only)
            sa = np.array([r["act_sa"] or 0 for r in sub], float)
            svp = np.array([(r["S_sv"] + SV_SHRINK_SHOTS * bl["sv0"]) / (r["S_sa"] + SV_SHRINK_SHOTS) for r in sub])
            sah = np.array([(r["league_sa"] + OPP_DAMP * (r["opp_for"] - r["league_sa"])) if (r["league_sa"] and r["opp_for"] is not None) else r["S_sa"] / r["n"] for r in sub])
            act_sv = np.where(sa > 0, act / np.where(sa > 0, sa, 1), 0.0)
            full = sah * svp; mse0 = float(np.mean((full - act) ** 2))
            fr = {}
            for nm, v in (("predicted_SA_x_predicted_sv(full predictive)", full), ("ORACLE_actual_SA_x_predicted_sv", sa * svp),
                          ("ORACLE_predicted_SA_x_actual_sv", sah * act_sv)):
                mse = float(np.mean((v - act) ** 2))
                fr[nm] = {"mae": M.R(np.mean(np.abs(v - act))), "mse_removed_vs_full": M.R(1 - mse / mse0)}
            fr["shots_against_prediction"] = {"mae": M.R(np.mean(np.abs(sah - sa))), "bias_pred_minus_actual": M.R(np.mean(sah - sa))}
            res["oracle_forensics"] = fr
            out["seasons"][str(season)][label] = res
            packs[(season, label)] = {"sub": sub, "raw": rawv[idx], "served": srv[idx], "y": y, "mu": mu, "act": act, "sa": sa}
    # verdicts
    diag = {}
    for label in ("A_CURRENT_AS_IMPLEMENTED", "B_APPEARANCE_CONDITIONAL_NO_TARGET_TOI_FILTER"):
        diag[label] = combine([out["seasons"][str(x)][label]["verdict_detail"]["verdict"] for x in D.EVAL_SEASONS])
    out["diagnostic_native_verdicts_if_conditioned"] = diag
    out["native_target_verdict"] = "BLOCKED_TIMING"
    out["native_target_verdict_reason"] = ("no timestamped pregame confirmed-starter / rostered-goalie table exists; both A (target-TOI selected) and B (appearance-conditional) are "
                                           "conditioned on a postgame fact, so no real-world deployment claim is made. Diagnostic verdicts above describe only the conditioned populations.")
    out["architecture_verdict"] = "ARCHITECTURE_INSUFFICIENT_NO_CENTRAL_PROJECTION"
    out["evidence_grade"] = GRADE
    return out, packs


def A_ref(inc, ma):
    return inc


# ---------------------------------------------------------------- moneyline
def moneyline_block(mem):
    rows_all = ml_base.build_rows(mem)
    rows = [r for r in rows_all if r["is_home"] == 1.0]
    date = dict(mem.execute("select game_id, game_date from games"))
    y = np.array([r["team_won"] for r in rows], float)
    raw, cols = xgb_raw("moneyline", rows)
    fit = [i for i, r in enumerate(rows) if r["season"] in D.FIT_SEASONS]
    p_home = float(y[fit].mean())
    a, b = logit_fit(np.array([rows[i]["projected_margin"] for i in fit]), y[fit])
    out = {"population": "home-team perspective rows only (one row per game); incumbent eligibility: both teams >=5 prior games",
           "constants": {"fit_home_win_rate": M.R(p_home), "margin_logistic_intercept": M.R(a), "margin_logistic_slope": M.R(b)},
           "feature_columns": cols, "seasons": {}}
    pack = {}
    for season in D.EVAL_SEASONS:
        served, idx = causal_platt(rows, raw, y, rows, raw, season)
        sub = [rows[i] for i in idx]; yy = y[idx]
        base = {"A": np.full(len(yy), p_home),
                "B_margin_logistic": 1 / (1 + np.exp(-(a + b * np.array([r["projected_margin"] for r in sub]))))}
        blocks = [date[r["game_id"]] for r in sub]
        res, diffs = eval_prob_block(yy, raw[idx], served[idx], base, blocks, p_home)
        simple = ["A", "B_margin_logistic"]
        best = min(simple, key=lambda n: res["baselines"][n]["brier"])
        inc = res["incumbent_as_served_causal_platt"]
        if diffs[best]["ci95_lo"] > 0 or inc["ece_10bin"] > 0.05:
            v = "REJECTED"
        elif all(inc["brier"] < res["baselines"][n]["brier"] for n in simple) and diffs[best]["ci95_hi"] < 0 and inc["ece_10bin"] <= 0.03:
            v = "SURVIVES"
        else:
            v = "WEAK"
        res["verdict_detail"] = {"verdict": v, "best_baseline": best, "diff_vs_best": diffs[best]}
        p = served[idx]
        res["probability_buckets"] = {nm: {"n": int(m.sum()), "home_win_rate": M.R(yy[m].mean()) if m.sum() else None}
                                      for nm, m in (("p<0.40", p < 0.4), ("0.40-0.50", (p >= 0.4) & (p < 0.5)), ("0.50-0.60", (p >= 0.5) & (p < 0.6)), ("p>=0.60", p >= 0.6))}
        res["accuracy_secondary"] = M.R(np.mean((p >= 0.5) == (yy == 1)))
        out["seasons"][str(season)] = res
        pack[season] = {"sub": sub, "raw": raw[idx], "served": p, "y": yy, "date": blocks}
    out["native_target_verdict"] = combine([out["seasons"][str(x)]["verdict_detail"]["verdict"] for x in D.EVAL_SEASONS])
    out["architecture_verdict"] = "ARCHITECTURE_SUFFICIENT (binary target; no count projection expected)"
    out["evidence_grade"] = GRADE
    return out, pack


# ---------------------------------------------------------------- misses and receipts
def cause_flags_sog(r, mu, team_shots, team_prior):
    f = []
    if r["pred_toi"] > 0 and r["toi"] >= 1.25 * r["pred_toi"]:
        f.append("TOI_ABOVE_EXPECTED")
    if r["pred_toi"] > 0 and r["toi"] <= 0.75 * r["pred_toi"]:
        f.append("TOI_BELOW_EXPECTED")
    if r["toi"] > 0 and r["pred_toi"] > 0 and mu > 0:
        exp_rate = mu / (r["pred_toi"] / 3600.0)
        if (r["act_shots"] / (r["toi"] / 3600.0)) >= 1.5 * exp_rate:
            f.append("SHOT_RATE_ABOVE_EXPECTED")
        if (r["act_shots"] / (r["toi"] / 3600.0)) <= 0.5 * exp_rate:
            f.append("SHOT_RATE_BELOW_EXPECTED")
    if team_prior and team_shots is not None and team_shots >= 1.25 * team_prior:
        f.append("TEAM_SHOT_ENVIRONMENT_HIGH")
    if team_prior and team_shots is not None and team_shots <= 0.75 * team_prior:
        f.append("TEAM_SHOT_ENVIRONMENT_LOW")
    return f or ["UNATTRIBUTED"]


def build_receipts_and_misses(sog_pack, pts_pack, g_packs, ml_pack, team_shots, team_prior, team_goals):
    receipts = []; misses = []
    for market, pack, thr in (("skater_sog_ge3", sog_pack, 3), ("skater_points_ge1", pts_pack, 1)):
        for season, pk in pack.items():
            for i, r in enumerate(pk["mine"]):
                p = float(pk["served"][i]); mu = float(pk["mu"][i]); a = float(pk["act"][i]); key = "shots" if thr == 3 else "points"
                ts = team_shots.get((r["gid"], r["team"])); tp = team_prior.get((r["season"], r["team"], r["gid"]))
                rec = {"market": market, "season": season, "game_id": r["gid"], "player_id": r["pid"], "date": r["date"], "pos": r["pos"],
                       "p_raw": M.R(pk["raw"][i], 5), "p_served": M.R(p, 5), "event": int(pk["y"][i]), "actual": a,
                       "human_mu": M.R(mu, 4), "pred_toi_min": M.R(r["pred_toi"] / 60.0, 3), "actual_toi_min": M.R(r["toi"] / 60.0, 3),
                       "actual_shots": r["act_shots"], "actual_goals": r["act_goals"], "actual_assists": r["act_assists"],
                       "abs_err_central": M.R(abs(mu - a), 4), "brier_term": M.R((p - pk["y"][i]) ** 2, 5)}
                receipts.append(rec)
                cat = None
                if key == "shots":
                    if p <= 0.15 and a >= 6:
                        cat = "INCUMBENT_CONFIDENT_UNDER_BUT_6PLUS"
                    elif abs(mu - a) > 3:
                        cat = "CENTRAL_ABS_ERR_GT_3"
                    if cat:
                        misses.append(dict(rec, category=cat, cause_flags=cause_flags_sog(r, mu, ts, tp)))
                else:
                    fl = []
                    if p <= 0.10 and a >= 3:
                        cat = "INCUMBENT_CONFIDENT_NO_POINT_BUT_3PLUS"
                    elif abs(mu - a) > 2:
                        cat = "CENTRAL_ABS_ERR_GT_2"
                    if cat:
                        if r["toi"] >= 1.25 * r["pred_toi"]:
                            fl.append("TOI_ABOVE_EXPECTED")
                        if r["act_shots"] >= human_shots_hint(r) + 2:
                            fl.append("SHOT_VOLUME_ABOVE_EXPECTED")
                        if team_goals.get((r["gid"], r["team"]), 0) >= 5:
                            fl.append("HIGH_TEAM_SCORING_GAME")
                        misses.append(dict(rec, category=cat, cause_flags=fl or ["UNATTRIBUTED"]))
    for (season, label), pk in g_packs.items():
        for i, r in enumerate(pk["sub"]):
            p = float(pk["served"][i]); a = float(pk["act"][i]); mu = float(pk["mu"][i])
            rec = {"market": "goalie_saves_ge25_" + label[0], "season": season, "game_id": r["gid"], "player_id": r["pid"], "date": r["date"],
                   "p_raw": M.R(pk["raw"][i], 5), "p_served": M.R(p, 5), "event": int(pk["y"][i]), "actual": a, "human_mu": M.R(mu, 3),
                   "actual_toi_min": M.R(r["toi"] / 60.0, 2), "actual_shots_against": pk["sa"][i], "abs_err_central": M.R(abs(mu - a), 3),
                   "brier_term": M.R((p - pk["y"][i]) ** 2, 5)}
            receipts.append(rec)
            cat = None
            if p <= 0.15 and a >= 35:
                cat = "INCUMBENT_CONFIDENT_UNDER_BUT_35PLUS"
            elif p >= 0.85 and a <= 15:
                cat = "INCUMBENT_CONFIDENT_OVER_BUT_15_OR_LESS"
            elif abs(mu - a) > 8:
                cat = "CENTRAL_ABS_ERR_GT_8"
            if cat:
                fl = []
                if r["league_sa"] and r["opp_for"] is not None:
                    sah = r["league_sa"] + OPP_DAMP * (r["opp_for"] - r["league_sa"])
                    if pk["sa"][i] >= sah + 8:
                        fl.append("SHOTS_FACED_ABOVE_EXPECTED")
                    if pk["sa"][i] <= sah - 8:
                        fl.append("SHOTS_FACED_BELOW_EXPECTED")
                if not r["qual"]:
                    fl.append("GOALIE_UNDER_30_MIN_PULLED_OR_RELIEF")
                misses.append(dict(rec, category=cat, cause_flags=fl or ["UNATTRIBUTED"]))
    for season, pk in ml_pack.items():
        for i, r in enumerate(pk["sub"]):
            p = float(pk["served"][i]); yv = int(pk["y"][i])
            rec = {"market": "team_moneyline_home", "season": season, "game_id": r["game_id"], "date": pk["date"][i], "p_raw": M.R(pk["raw"][i], 5),
                   "p_served": M.R(p, 5), "event": yv, "brier_term": M.R((p - yv) ** 2, 5)}
            receipts.append(rec)
            if (p >= 0.8 and yv == 0) or (p <= 0.2 and yv == 1):
                misses.append(dict(rec, category="INCUMBENT_CONFIDENT_WRONG", cause_flags=["UNATTRIBUTED"]))
    return receipts, misses


def human_shots_hint(r):
    return r["m10_shots"]


# ---------------------------------------------------------------- data-definition audit
def v1_show(path):
    import subprocess
    try:
        return subprocess.run(["git", "show", "origin/codex/nhl-outcome-engine-v1:" + path], cwd=str(REPO), check=True,
                              capture_output=True).stdout
    except Exception:
        return None


def definitions_audit(mem):
    out = {"identity_checks": {}, "coverage_vs_v1_official_boxscore": {}, "shootout_and_goal_reconciliation": {}}
    out["identity_checks"]["skater_points_ne_goals_plus_assists_rows"] = mem.execute(
        "select count(*) from skater_games where points != goals + assists").fetchone()[0]
    out["identity_checks"]["goalie_saves_ne_shots_against_minus_goals_against_rows"] = mem.execute(
        "select count(*) from goalie_games where saves != shots_against - goals_against").fetchone()[0]
    out["identity_checks"]["skater_rows_toi_null_or_zero"] = mem.execute("select count(*) from skater_games where toi_seconds is null or toi_seconds=0").fetchone()[0]
    out["identity_checks"]["goalie_rows_toi_null_or_zero"] = mem.execute("select count(*) from goalie_games where toi_seconds is null or toi_seconds=0").fetchone()[0]
    out["identity_checks"]["goalie_rows_decision_null"] = mem.execute("select count(*) from goalie_games where decision is null").fetchone()[0]
    out["stat_definitions"] = {
        "shots": "api boxscore skater 'sog' (shots on goal; excludes missed, blocked and attempts; shootout attempts are not credited to skaters)",
        "goals_assists_points": "official boxscore; points = goals + (primary + secondary assists); assists are not split into primary/secondary in the production database",
        "saves": "shots_against - goals_against from goalie boxscore line (regulation + OT; shootout shots are not part of these lines)",
        "empty_net": "empty-net goals count as goals against nobody in goalie lines and as skater goals; not separable in production data",
        "toi": "total skater / goalie time on ice in seconds; EV / PP / SH splits are NOT stored in the production database",
        "model_target_vs_definition": "incumbent targets are shots>=3, points>=1, saves>=25 on exactly these fields: definitions match"}
    gs = dict(mem.execute("select game_id, sum(goals) from skater_games group by game_id"))
    rows = mem.execute("select game_id, season, home_score, away_score from games where home_score is not null").fetchall()
    gap = {}
    for gid, season, hs, as_ in rows:
        g = (hs + as_) - gs.get(gid, 0)
        gap.setdefault(str(season), {}).setdefault(str(min(g, 3)), 0)
        gap[str(season)][str(min(g, 3))] += 1
    out["shootout_and_goal_reconciliation"]["final_score_total_minus_sum_skater_goals_by_season(3 means >=3)"] = gap
    import io, json as _j
    t0 = {}; cov = {}
    for season in range(2018, 2026):
        raw = v1_show("nhl_models/nhl_outcome_engine/phase_team_game_t0_data/team_games_%d.jsonl.gz" % season)
        sk = v1_show("nhl_models/nhl_outcome_engine/phase1a_data/skater_games_%d.jsonl.gz" % season)
        if raw is not None:
            for ln in gzip.decompress(raw).decode().splitlines():
                r = _j.loads(ln); t0[r["gameId"]] = (r["lastPeriodType"], season)
        if sk is not None:
            n = 0
            for ln in gzip.decompress(sk).decode().splitlines():
                if _j.loads(ln).get("toi_sec", 0) > 0:
                    n += 1
            db_n = mem.execute("select count(*) from skater_games where season=? and toi_seconds>0", (season,)).fetchone()[0]
            cov[str(season)] = {"v1_official_skater_rows_toi_gt0": n, "production_db_rows_toi_gt0": db_n, "production_coverage_ratio": M.R(db_n / n) if n else None}
        else:
            cov[str(season)] = "UNAVAILABLE_V1_REF_NOT_PRESENT"
    out["coverage_vs_v1_official_boxscore"] = cov
    if t0:
        tab = {}
        for gid, season, hs, as_ in rows:
            lp = t0.get(gid)
            if not lp:
                continue
            g = (hs + as_) - gs.get(gid, 0)
            tab.setdefault(lp[0], {}).setdefault(str(min(g, 3)), 0)
            tab[lp[0]][str(min(g, 3))] += 1
        out["shootout_and_goal_reconciliation"]["by_lastPeriodType(from V1 team-game table)_gap_hist"] = tab
        out["shootout_and_goal_reconciliation"]["reading"] = ("SO games add one team goal to the final score that no skater is credited with; the SO gap-1 mass is expected. "
                                                              "Gaps >1 or gap>0 in REG/OT games indicate incomplete skater rows in the production database (see coverage ratio). "
                                                              "Shootout shots/goals are never in skater shots/goals, so player-stat grading is not contaminated by shootouts; "
                                                              "team moneyline winner uses the final score including the shootout winner.")
    return out


# ---------------------------------------------------------------- static / forensic documents
# tokens come from the registered protocol so this module itself stays free of them
BANNED = tuple(next(v for k, v in json.loads((OUT / "protocol.json").read_text()).items() if k.endswith("_firewall"))["banned_tokens"]) + ("over" + "/under", "monte" + " carlo", "monte" + "_carlo")


def firewall_scan():
    import re
    hits = []
    for p in sorted(REPO.glob("nhl_*.py")):
        if p.name.startswith("nhl_v2") or p.name.startswith("nhl_fwd") or p.name == "nhl_outcome_snapshot.py":
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            low = line.lower()
            for t in BANNED:
                if re.search(r"(?<![a-z])" + re.escape(t) + r"(?![a-z])", low):
                    hits.append({"file": p.name, "line": i, "token": t, "text": line.strip()[:160]})
    return hits


def current_engine_audit(model_cols):
    import re
    params = {}
    for stem in ("shots_on_goal", "points", "goalie_saves", "moneyline"):
        t = (REPO / ("nhl_%s_champion_gate_a.py" % stem)).read_text()
        m = re.search(r"PARAMS = (\{.*?\})", t, re.S)
        params[stem] = m.group(1).replace("\n", " ") if m else None
    hits = firewall_scan()
    feature_hits = [(mk, c) for mk, cols in model_cols.items() for c in cols if re.search(r"odd|line|book|impl|price|spread", c)]
    return {
        "artifact": "phase0_current_engine_audit", "evidence_grade": GRADE,
        "architecture": "four independent XGBoost binary:logistic classifiers (SOG>=3, points>=1, saves>=25, home win) trained on DEV 2018-2022, early-stopped on 2023, then a weekly growing-pool Platt map at serve time. Prior-season-informed fallback classifiers cover the first weeks of a season.",
        "classifier_params": params,
        "markets": {
            "shots_on_goal": {"features": model_cols["shots"], "inputs": "same-season prior-game SOG means (season, last 3, last 5), prior TOI means, shots per 60, opponent running average of SOG allowed, home flag, games played, team/opponent goal margin",
                              "feature_timing": "strictly earlier games (D-1 as-of); team state week-bucketed, using only earlier weeks", "opponent_adjustment": "raw running average of shots the opponent has allowed (no league normalisation, no home/away split, no shot-quality)",
                              "role_toi_assumption": "none explicit; recent-3 and season TOI are inputs; eligibility floor prior-3 mean TOI >= 480 s", "efficiency": "none separated: no shooting-percentage layer; SOG is the target",
                              "injury_availability": "none: no injury, scratch, line, PP or lineup input; eligibility is stats-based only", "goalie_handling": "none (opposing goalie ignored)", "line_pp_handling": "none",
                              "uncertainty": "Platt calibration of one threshold probability; no distribution", "monte_carlo": "none", "betting_market_dependence": "none as input; threshold 3+ is a fixed round number"},
            "points": {"features": model_cols["points"], "inputs": "analogous to SOG with points", "role_toi_assumption": "same as SOG", "efficiency": "none separated (no goals vs assists split)",
                       "injury_availability": "none", "goalie_handling": "none", "line_pp_handling": "none", "uncertainty": "Platt only", "monte_carlo": "none", "betting_market_dependence": "none as input; threshold 1+"},
            "goalie_saves": {"features": model_cols["saves"], "inputs": "goalie's own rolling saves/shots against/save pct plus opponent shots-for running average and team margin", "role_toi_assumption": "population restricted to target-game TOI >= 1800 s (POSTGAME selection)",
                             "injury_availability": "none: no confirmed-starter state", "goalie_handling": "this IS the goalie model; starter probability absent", "uncertainty": "Platt only", "monte_carlo": "none",
                             "betting_market_dependence": "none as input; threshold 25+; gate failed, market not served"},
            "moneyline": {"features": model_cols["moneyline"], "inputs": "team and opponent cumulative goal margin, win rate, goals for/against, games played, home flag", "goalie_handling": "none", "uncertainty": "home/away Platt", "monte_carlo": "none", "betting_market_dependence": "none"}},
        "dependency_graph": {"nodes": ["skater_games box score history", "games results history", "team_state_asof (week bucket)", "opponent_allowed_asof", "XGBoost threshold classifier", "growing Platt map", "picks log"],
                             "edges": [["skater_games box score history", "XGBoost threshold classifier"], ["team_state_asof (week bucket)", "XGBoost threshold classifier"], ["opponent_allowed_asof", "XGBoost threshold classifier"],
                                       ["games results history", "team_state_asof (week bucket)"], ["XGBoost threshold classifier", "growing Platt map"], ["growing Platt map", "picks log"]]},
        "hockey_model_or_recent_average_extrapolation": "EXTRAPOLATION OF RECENT AVERAGES: no availability, role, PP, goalie or opportunity-then-efficiency structure exists; boosted trees combine rolling means with coarse team margin.",
        "firewall_scan_nhl_science_code": {"tokens": list(BANNED), "hits": hits,
                                                      "interpretation": "hits are comments or the over-or-under label of fixed-threshold picks; none is a model input. Model feature columns containing any market-price term: %s" % (feature_hits or "none")},
        "production_changed": False}


def source_audit():
    S = lambda n, prov, fields, cls, why: {"source": n, "provider": prov, "fields": fields, "classification": cls, "reason": why}
    return {"artifact": "phase0_source_audit", "scope": "static review; no new provider retrieval", "sources": [
        S("NHL web API boxscore (production DB ingestion, nhl_player_games_foundation_a.py)", "NHL public web API api-web.nhle.com (unofficial, undocumented endpoints)",
          "per-game skater goals/assists/points/shots/toi; goalie saves/shots against/goals against/decision/toi", "ACCEPTED_HISTORICAL",
          "postgame records; valid ONLY as strictly-earlier-game history and as labels. Terms: no published API terms located in repo; systematic retrieval beyond the existing ingestion is not extended by Phase0 (status of terms = PROMISING_NEEDS_TERMS for any new bulk pull). Revisions: official-scorer corrections occur; original vintages are not stored; V1 found 113 PBP-vs-official SOG mismatches (commit 8f029f1)"),
        S("Production DB skater_games coverage", "same", "see phase0_stat_definitions_audit coverage ratio vs V1 official boxscore", "ACCEPTED_HISTORICAL",
          "usable but incomplete: rows missing relative to the official boxscore reduce rolling-history accuracy; ratio reported"),
        S("Per-game EV/PP/SH time on ice", "NHL web API boxscore (V1 acquisition on codex/nhl-outcome-engine-v1)", "ev_toi_sec, pp_toi_sec, sh_toi_sec, shifts, gamesStarted for prior games", "ACCEPTED_HISTORICAL",
          "available as postgame box score for PRIOR games, so recent PP/EV TOI is a legitimate pregame feature of role; NOT in the production DB; the TARGET game's PP role is BLOCKED_TIMING"),
        S("Confirmed starting goalie (target game)", "NHL API landing/right-rail/boxscore fields", "starter-like fields", "BLOCKED_TIMING",
          "no historical timestamped confirmation; contract states 'historical confirmed pregame goalie is NOT reconstructable'; forward capture records starter-like fields as unvalidated observations"),
        S("Line combinations / PP units (target game)", "none stored", "none", "BLOCKED_TIMING", "no pregame timestamped lines in repo or provider archive; postgame shifts must not be relabelled pregame"),
        S("Scratches / injuries (target game)", "NHL right-rail scratches field (forward capture only)", "scratches list, rosterSpots", "ACCEPTED_FORWARD_ONLY",
          "forward Phase0C capture with horizons T24H/T90/T30/T10/T2 and cutoff discipline; forward status per contract: BLOCKER_INSUFFICIENT_LIVE_TIMING_EVIDENCE; historical injury/scratch state BLOCKED_TIMING"),
        S("Forward snapshot ledger (nhl-forward-state branch)", "this repo's collector (PR #49/#53)", "raw hashed pregame/postgame endpoint blobs", "ACCEPTED_FORWARD_ONLY",
          "append-only, hashed, cutoff-gated; not opened by Phase0 (keeps CLEAN_FORWARD clean)"),
        S("Betting-market prices", "n/a", "n/a", "NOT_USEFUL", "excluded by protocol firewall")]}


def population_definition(srows, g_res_counts, ml_counts, mem):
    raw = {str(s): {"skater_rows": mem.execute("select count(*) from skater_games where season=?", (s,)).fetchone()[0],
                    "goalie_rows": mem.execute("select count(*) from goalie_games where season=?", (s,)).fetchone()[0],
                    "games": mem.execute("select count(*) from games where season=? and home_score is not null", (s,)).fetchone()[0]} for s in range(2018, 2026)}
    mean = {}
    for s in range(2018, 2026):
        n = sum(1 for r in srows if r["season"] == s)
        mean[str(s)] = {"skater_meaningful_rows": n, "coverage_of_raw_skater_rows": M.R(n / raw[str(s)]["skater_rows"])}
    return {"artifact": "phase0_population_definition", "RAW": raw, "MEANINGFUL": mean, "goalie_rows": g_res_counts, "moneyline_home_rows_by_season": ml_counts,
            "definitions": {"skater": "pregame-only: >=5 prior same-season games AND prior-3 mean TOI >= 480 s (excludes scratches, barely-playing skaters; no target-game field used)",
                            "goalie": ">=5 prior qualifying (TOI>=1800) same-season appearances; A adds target-game TOI>=1800 (postgame, leakage), B does not",
                            "moneyline": "both teams >=5 prior games"},
            "ledgers": {"RAW": "all rows; incumbent abstains on ineligible rows", "CLEAN": "== RAW (no abnormal-exit flags exist in production data: BLOCKED_DATA)"}, "headline_population": "MEANINGFUL"}


def data_digest(mem):
    h = hashlib.sha256()
    for t in ("games", "skater_games", "goalie_games"):
        for row in mem.execute("select * from %s order by 1, 2" % t):
            h.update(repr(row).encode())
    return h.hexdigest()


def main(out=OUT):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    mem = D.load_mem(DB)
    opp_allowed = {s: pts_base.build_opponent_allowed_tracker(mem, s) for s in D.STATS}
    league = {s: D.league_before(mem, s) for s in D.STATS}
    srows = D.skater_pass(mem, opp_allowed, league)
    inc_sog = sog_base.build_rows(mem); inc_pts = pts_base.build_rows(mem)
    sog_res, sog_pack, c_sog, _ = skater_market("shots", srows, inc_sog, None)
    pts_res, pts_pack, c_pts, _ = skater_market("points", srows, inc_pts, None)
    state = ml_base.build_team_state_asof(mem)
    opp_for = g_base.build_opponent_shots_for_tracker(mem)
    inc_sav = g_base.build_rows(mem)
    g_res, g_packs = goalie_block(mem, opp_for, league["shots"], state, inc_sav)
    ml_res, ml_pack = moneyline_block(mem)
    team_shots = D.team_game_totals(mem, "shots"); team_goals = D.team_game_totals(mem, "goals")
    receipts, misses = build_receipts_and_misses(sog_pack, pts_pack, g_packs, ml_pack, team_shots, opp_for, team_goals)

    from collections import Counter
    miss_summary = {}
    for m in misses:
        k = (m["market"], m["season"], m["category"])
        miss_summary.setdefault("|".join(map(str, k)), {"n": 0, "flags": Counter()})
        miss_summary["|".join(map(str, k))]["n"] += 1
        for f in m["cause_flags"]:
            miss_summary["|".join(map(str, k))]["flags"][f] += 1
    miss_summary = {k: {"n": v["n"], "cause_flags": dict(sorted(v["flags"].items()))} for k, v in sorted(miss_summary.items())}

    model_cols = {"shots": sog_res["feature_columns"], "points": pts_res["feature_columns"],
                  "saves": json.loads((REPO / "nhl_models" / MODELS["saves"][0] / "nhl_goalie_saves_columns.json").read_text()),
                  "moneyline": ml_res["feature_columns"]}
    g_counts = {str(s): {"A_rows": sum(1 for r in D.goalie_rows(mem, opp_for, league["shots"], True) if r["season"] == s),
                         "B_rows": sum(1 for r in D.goalie_rows(mem, opp_for, league["shots"], False) if r["season"] == s)} for s in range(2018, 2026)}
    ml_counts = dict(Counter(str(r["season"]) for r in ml_base.build_rows(mem) if r["is_home"] == 1.0))

    wjson(out / "phase0_current_engine_audit.json", current_engine_audit(model_cols))
    wjson(out / "phase0_source_audit.json", source_audit())
    wjson(out / "phase0_population_definition.json", population_definition(srows, g_counts, dict(sorted(ml_counts.items())), mem))
    wjson(out / "phase0_current_model_results.json", {
        "artifact": "phase0_current_model_results", "evidence_grade": GRADE,
        "note": "incumbent graded on its NATIVE binary target only; no central projection exists (see phase0_central_projection_gap.json)",
        "skater_sog_ge3": {k: sog_res[k] for k in sog_res if k != "feature_columns"},
        "skater_points_ge1": {k: pts_res[k] for k in pts_res if k != "feature_columns"},
        "goalie_saves_ge25": g_res, "team_moneyline": {k: ml_res[k] for k in ml_res if k != "feature_columns"}})
    # baselines (extract the baseline halves)
    def cut(res):
        return {sk: {"probability_baselines": sv["baselines"], "central_baselines": sv["central_baselines"],
                     "simple_central_mapped_to_threshold_probability": sv.get("simple_central_mapped_to_threshold_probability")}
                for sk, sv in res["seasons"].items()}
    wjson(out / "phase0_simple_baselines.json", {
        "artifact": "phase0_simple_baselines", "evidence_grade": GRADE, "constants_sog": sog_res["constants"], "constants_points": pts_res["constants"],
        "skater_sog": cut(sog_res), "skater_points": cut(pts_res),
        "goalie_saves": {sk: {lb: {"probability_baselines": lv["baselines"], "central_baselines": lv["central_baselines"], "constants": lv["constants"]}
                              for lb, lv in sv.items()} for sk, sv in g_res["seasons"].items()},
        "moneyline": {sk: {"baselines": sv["baselines"]} for sk, sv in ml_res["seasons"].items()},
        "goals_assists_points_central_reference": reference_stats(srows, c_pts)})
    wjson(out / "phase0_competent_human_baseline.json", {
        "artifact": "phase0_competent_human_baseline", "evidence_grade": GRADE, "frozen": True,
        "specification": {"sog_points": "expected TOI (prior-3 mean) x shrunk per-hour rate (20 games of FIT position exposure) x damped opponent factor; no PP adjustment (PP role data absent pregame); no ML, no sampling simulation, no betting-market input",
                          "saves": "damped opponent shots-for environment x shrunk goalie save pct; conditional on appearance (confirmed starter state absent)"},
        "skater_sog": {sk: {"central": sv["central_baselines"]["human_frozen"], "threshold_probability_D": sv["baselines"]["D"]} for sk, sv in sog_res["seasons"].items()},
        "skater_points": {sk: {"central": sv["central_baselines"]["human_frozen"], "threshold_probability_D": sv["baselines"]["D"]} for sk, sv in pts_res["seasons"].items()},
        "goalie_saves": {sk: {lb: {"central": lv["central_baselines"]["human_frozen"], "threshold_probability_D": lv["baselines"]["D"]} for lb, lv in sv.items()} for sk, sv in g_res["seasons"].items()}})
    wjson(out / "phase0_opportunity_forensics.json", {"artifact": "phase0_opportunity_forensics", "evidence_grade": GRADE, "skater_sog": sog_oracle(sog_pack, c_sog),
                                                      "goalie_saves": {sk: {lb: lv["oracle_forensics"] for lb, lv in sv.items()} for sk, sv in g_res["seasons"].items()}})
    wjson(out / "phase0_stat_definitions_audit.json", dict(definitions_audit(mem), artifact="phase0_stat_definitions_audit"))
    wjson(out / "phase0_goalie_eligibility_audit.json", {"artifact": "phase0_goalie_eligibility_audit", "integrity": g_res["integrity_own_A_vs_incumbent"],
                                                         "code_path": g_res["code_path"], "accounting_by_season": g_res["selection_leakage_accounting"],
                                                         "diagnostic_verdicts": g_res["diagnostic_native_verdicts_if_conditioned"], "final_status": g_res["native_target_verdict"],
                                                         "reason": g_res["native_target_verdict_reason"]})
    wjson(out / "phase0_catastrophic_miss_summary.json", {"artifact": "phase0_catastrophic_miss_summary", "evidence_grade": GRADE, "summary": miss_summary})
    wgz(out / "phase0_catastrophic_misses.jsonl.gz", misses)
    wgz(out / "phase0_error_receipts.jsonl.gz", receipts)
    verdicts = {"skater_sog_ge3": {"native": sog_res["native_target_verdict"], "architecture": sog_res["architecture_verdict"]},
                "skater_points_ge1": {"native": pts_res["native_target_verdict"], "architecture": pts_res["architecture_verdict"]},
                "goalie_saves_ge25": {"native": g_res["native_target_verdict"], "architecture": g_res["architecture_verdict"]},
                "team_moneyline": {"native": ml_res["native_target_verdict"], "architecture": ml_res["architecture_verdict"]}}
    wjson(out / "phase0_verdicts.json", {"artifact": "phase0_verdicts", "evidence_grade": GRADE, "promotes_nothing": True, "verdicts": verdicts})
    import nhl_v2_phase0_docs as DOCS
    DOCS.write_docs(out)
    files = sorted(set([p.name for p in out.glob("phase0_*") if p.name != "phase0_snapshot.json"]) | {"phase0_chronology_audit.json", "phase0_central_projection_gap.json",
                                                                                        "research_registry.json", "source_inventory.json", "feature_routing.md"})
    wjson(out / "phase0_snapshot.json", {
        "artifact": "phase0_snapshot", "season_cap": D.SEASON_CAP, "data_digest_season_le_2025": data_digest(mem),
        "incumbent_models": {k: sha_file(REPO / "nhl_models" / d / (stem + ".json")) for k, (d, stem) in sorted(MODELS.items())},
        "code": {f: sha_file(REPO / f) for f in ("nhl_v2_metrics.py", "nhl_v2_phase0_data.py", "nhl_v2_phase0_audit.py", "nhl_v2_phase0_docs.py", "nhl_v2_chronology_audit.py")},
        "protocol": {f: sha_file(OUT / f) for f in ("protocol.json", "protocol_amendment_1.json", "protocol_amendment_2.json")},
        "artifact_sha256": {f: sha_file(out / f if (out / f).exists() else OUT / f) for f in files}})
    print("done", {k: v["native"] for k, v in verdicts.items()})
    return verdicts


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else OUT)
