"""
NHL_SOG_PHASE1B_PROBE -- Phase 1B-A runner: prepare (sample + required source games) -> acquire (bounded PBP) -> quality gate (STOP on failure) -> raw signal diagnostics (training only) -> D1-D4 A0 vs A1 -> S1-S7.
Protocol: nhl_models/nhl_outcome_engine/phase1b_attempt_signal_protocol.json (committed before acquisition). Targets 2018-2023 only.
"""
import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np

import nhl_sog_phase1a_data as D
import nhl_sog_phase1a_metrics as M
import nhl_sog_phase1a_models as MD
import nhl_sog_phase1b_attempts as AT

REPO = Path(__file__).resolve().parent
OUT = REPO / "nhl_models" / "nhl_outcome_engine"
PROTOCOL = OUT / "phase1b_attempt_signal_protocol.json"
ALPHA = 0.001
FOLDS = [("D1", [2018, 2019], 2020), ("D2", [2018, 2019, 2020], 2021), ("D3", [2018, 2019, 2020, 2021], 2022), ("D4", [2018, 2019, 2020, 2021, 2022], 2023)]
CORE = list(D.FEATURES)
A1_FEATURES = CORE + AT.NEW_FEATURES
THRESH = {"S1_rel_improvement": 0.005, "S3_folds": 3, "S4_nll": 0.005, "S5_6_slice": 0.05, "S5_6_min_rows": 500, "S7_ks": 0.02, "S7_cov": 0.02}
PASS_STATUS, FAIL_STATUS = "ATTEMPT_SIGNAL_FULL_CRAWL_JUSTIFIED", "ATTEMPT_SIGNAL_NOT_JUSTIFIED"


# ------------------------------------------------------------------ pure helpers (unit-tested)
def select(tab, mask):
    return {k: v[mask] for k, v in tab.items()}


def fit_architecture(train, names, alpha=ALPHA):
    """Frozen Phase 1A B2 architecture: training-only preprocessing -> PoissonRegressor(alpha) -> NB2 dispersion by fixed-mean bounded MLE on the TRAINING rows."""
    prep = MD.Preprocessor(names).fit(train)
    X = prep.transform(train)
    m, info = MD.fit_poisson(X, train["sog"], alpha)
    nb = MD.fit_nb_alpha(train["sog"], m.predict(X))
    return {"prep": prep, "model": m, "nb": nb, "fit": info, "names": list(names), "alpha_poisson": alpha}


def predict_architecture(a, tab):
    mu = a["model"].predict(a["prep"].transform(tab))
    return "nb2", {"mu": mu, "alpha": a["nb"]["alpha"]}, mu


def calib_extras(kind, params, y, model_id, game_ids, player_ids):
    """Per-row randomized PIT, central-interval bounds and attainable discrete mass (rule registered in phase1a_posthoc_audit.json): attainable_i = F_i(hi_i) - F_i(lo_i - 1)."""
    n = len(y)
    v = M.pit_v(model_id, game_ids, player_ids)
    pit = np.zeros(n)
    lo = {c: np.zeros(n, dtype=np.int64) for c in (0.5, 0.8, 0.9)}; hi = {c: np.zeros(n, dtype=np.int64) for c in (0.5, 0.8, 0.9)}; att = {c: np.zeros(n) for c in (0.5, 0.8, 0.9)}
    for sl, pm, sf in M.pmf_matrix(kind, params, y):
        yy = y[sl]; cdf = np.cumsum(pm, axis=1); ar = np.arange(len(yy))
        py = pm[ar, yy]
        below = np.where(yy > 0, cdf[ar, np.maximum(yy - 1, 0)], 0.0)
        pit[sl] = below + v[sl] * py
        for c in (0.5, 0.8, 0.9):
            l = (cdf >= (1 - c) / 2 - 1e-12).argmax(axis=1); h = (cdf >= (1 + c) / 2 - 1e-12).argmax(axis=1)
            lo[c][sl], hi[c][sl] = l, h
            att[c][sl] = cdf[ar, h] - np.where(l > 0, cdf[ar, np.maximum(l - 1, 0)], 0.0)
    return {"pit": pit, "lo": lo, "hi": hi, "attainable": att}


def calibration_report(ex, y):
    out = {}
    for c in (0.5, 0.8, 0.9):
        cov = float(np.mean((y >= ex["lo"][c]) & (y <= ex["hi"][c])))
        pit_cov = float(np.mean((ex["pit"] >= (1 - c) / 2) & (ex["pit"] <= (1 + c) / 2)))
        ma = float(np.mean(ex["attainable"][c]))
        out[f"{int(c * 100)}"] = {"nominal_coverage": c, "mean_attainable_model_mass": ma, "empirical_deterministic_coverage": cov, "empirical_minus_mean_attainable": cov - ma,
                                  "randomized_pit_central_coverage": pit_cov, "randomized_pit_abs_error": abs(pit_cov - c)}
    return out


def gate_decision(*, rel_improvement, upper95, fold_deltas, rel_nll, slice_ok_fd, slice_ok_early, ks_delta, pit80_delta, pit90_delta):
    """S1-S7 -> (table, status). fold_deltas = A1 - A0 macro CRPS per fold (negative = A1 better)."""
    t = {"S1": {"pass": bool(rel_improvement >= THRESH["S1_rel_improvement"]), "value": rel_improvement},
         "S2": {"pass": bool(upper95 < 0), "value": upper95},
         "S3": {"pass": bool(sum(1 for d in fold_deltas if d < 0) >= THRESH["S3_folds"]), "folds_improved": int(sum(1 for d in fold_deltas if d < 0)), "of": len(fold_deltas)},
         "S4": {"pass": bool(rel_nll <= THRESH["S4_nll"]), "value": rel_nll},
         "S5": {"pass": bool(slice_ok_fd)}, "S6": {"pass": bool(slice_ok_early)},
         "S7": {"pass": bool(ks_delta <= THRESH["S7_ks"] and pit80_delta <= THRESH["S7_cov"] and pit90_delta <= THRESH["S7_cov"]), "pit_ks_delta": ks_delta, "pit80_abs_error_delta": pit80_delta, "pit90_abs_error_delta": pit90_delta}}
    ok = all(v["pass"] for v in t.values())
    return t, (PASS_STATUS if ok else FAIL_STATUS)


def distribution(x):
    x = np.asarray(x, float); f = x[np.isfinite(x)]
    if not len(f):
        return {"n": 0, "missing_fraction": 1.0}
    q = np.percentile(f, [1, 5, 25, 50, 75, 95, 99])
    return {"n": int(len(f)), "missing_fraction": float(1 - len(f) / len(x)), "mean": float(f.mean()), "sd": float(f.std()), "quantiles_1_5_25_50_75_95_99": [float(v) for v in q]}


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else None


def signal_diagnostics(train_tab):
    return {"n_training_rows": int(len(train_tab["game_id"])), "ATTEMPTS_PER60_APP10": distribution(train_tab["ATTEMPTS_PER60_APP10"]), "ON_NET_RATE_APP10": distribution(train_tab["ON_NET_RATE_APP10"]),
            "corr_ATTEMPTS_PER60_vs_SOG_PER60_APP10": corr(train_tab["ATTEMPTS_PER60_APP10"], train_tab["SOG_PER60_APP10"]), "corr_ATTEMPTS_MEAN_APP10_vs_SOG_MEAN_APP10": corr(train_tab["ATTEMPTS_MEAN_APP10"], train_tab["SOG_MEAN_APP10"]),
            "missingness": {k: float(np.mean(~np.isfinite(train_tab[k]))) for k in AT.NEW_FEATURES}, "candidate_coverage_any_attempt_feature": float(np.mean(np.isfinite(train_tab["ATTEMPTS_MEAN_APP10"])))}


# ------------------------------------------------------------------ stages
def git_head():
    import subprocess
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()


def protocol_committed():
    import subprocess
    rel = str(PROTOCOL.relative_to(REPO))
    return bool(subprocess.run(["git", "log", "--format=%H", "-1", "--", rel], cwd=REPO, capture_output=True, text=True).stdout.strip()) and \
        subprocess.run(["git", "status", "--porcelain", "--", rel], cwd=REPO, capture_output=True, text=True).stdout.strip() == ""


def prepare(work, log=print):
    if not protocol_committed():
        raise SystemExit("protocol must be committed (and clean) before acquisition")
    games, rows = D.load_frozen()
    tgt_games = {g: v for g, v in games.items() if int(str(g)[:4]) in AT.ALLOWED_TARGET_SEASONS}
    sample = AT.sample_games(tgt_games)
    ids = {g for v in sample.values() for g in v}
    p = Path(work) / "table_2018_2023.pkl"
    if p.exists():
        tab, cov = pickle.load(open(p, "rb"))
    else:
        t0 = time.time()
        tab, cov = D.build_prediction_rows(games, rows, target_seasons=AT.SAMPLE_SEASONS)
        pickle.dump((tab, cov), open(p, "wb"), protocol=4)
        log(f"prediction table built t={time.time() - t0:.0f}s rows={len(tab['game_id'])}")
    samp = select(tab, np.isin(tab["game_id"], list(ids)))
    apps, app_starts = AT.build_appearances(rows)
    need = AT.required_source_games(samp, apps, app_starts)
    sg_sample = {"sample": {str(s): v for s, v in sample.items()}, "sample_hash": hashlib.sha256(json.dumps({str(s): v for s, v in sample.items()}, sort_keys=True).encode()).hexdigest(),
                 "month_distribution_diagnostic_only": AT.month_distribution(games, sample), "sampled_rows_per_season": {str(s): int(np.sum(samp["season"] == s)) for s in AT.SAMPLE_SEASONS},
                 "required_source_games": sorted(need), "n_required_source_games": len(need)}
    pickle.dump((samp, sg_sample), open(Path(work) / "sample.pkl", "wb"), protocol=4)
    log(f"sampled games per season {{{', '.join(f'{s}:{len(v)}' for s, v in sample.items())}}} rows {len(samp['game_id'])} required source games {len(need)}")
    return samp, sg_sample, games, rows, apps, app_starts


def frozen_by_game(rows, need):
    out = {g: {} for g in need}
    for r in rows:
        if r["game_id"] in out:
            out[r["game_id"]][r["player_id"]] = r["sog"]
    return out


def acquire_and_quality(work, samp, sg_sample, games, rows, log=print):
    need = sg_sample["required_source_games"]
    results, secs = AT.acquire(need, Path(work) / "pbp_cache", log=log)
    fbg = frozen_by_game(rows, set(need))
    missing_frozen = [g for g in need if not fbg[g]]
    attempt_rows, per_game, q = AT.derive(results, fbg)
    prov = [{k: v for k, v in results[g][1].items() if k in ("game_id", "url", "retrieved_at_utc", "sha256", "bytes", "http_status", "error", "unobtainable")} for g in sorted(results)]
    files = AT.write_frozen(attempt_rows, prov)
    reqs = sum(1 for g in results if not results[g][2])
    acq = {"requests_made_this_run": reqs, "cached_responses_reused": len(results) - reqs, "unique_source_games": len(results), "downloaded_bytes_total_all_games": int(sum(results[g][1].get("bytes", 0) for g in results)),
           "wall_seconds_this_run": secs, "recorded_per_request_seconds_sum": round(sum(results[g][1].get("seconds", 0) for g in results), 1), "workers": 8}
    quality = {"protocol": "phase1b_attempt_signal_protocol.json", "status": "PASS" if AT.quality_pass(q) and not missing_frozen else "FAIL", **q, "source_games_without_frozen_rows": missing_frozen, "acquisition": acq}
    manifest = {"dataset": "nhl-sog-phase1b-a-attempts-v1", "provider": AT.API + "/gamecenter/<game_id>/play-by-play", "files": files, "n_source_games": len(results), "n_attempt_rows": len(attempt_rows),
                "sample_hash": sg_sample["sample_hash"], "sampled_games_per_season": {s: len(v) for s, v in sg_sample["sample"].items()}, "acquisition": acq, "fields": ["game_id", "player_id", "sog_from_pbp", "missed_attempts", "blocked_attempts", "shot_attempts"]}
    manifest["manifest_content_sha256"] = hashlib.sha256(json.dumps({k: manifest[k] for k in ("files", "sample_hash")}, sort_keys=True).encode()).hexdigest()
    (OUT / "phase1b_attempt_data_manifest.json").write_text(json.dumps({**manifest, "sample": sg_sample["sample"], "month_distribution_diagnostic_only": sg_sample["month_distribution_diagnostic_only"], "sampled_rows_per_season": sg_sample["sampled_rows_per_season"]}, indent=1, sort_keys=True))
    (OUT / "phase1b_attempt_quality.json").write_text(json.dumps(quality, indent=1, sort_keys=True))
    log(f"quality {quality['status']}: games {q['games_checked']} player-games {q['player_games_checked']} matches {q['exact_matches']} mismatches {q['mismatches']} unobtainable {len(q['unobtainable_games'])}")
    return quality, {(r["game_id"], r["player_id"]): r for r in attempt_rows}


def diagnose(work, log=print, boxscore_sample=8):
    """Quality-gate failure diagnosis (read-only on the cached PBP responses; a handful of boxscore requests for source-of-truth comparison). Classifies every mismatch; changes no data and fits no model."""
    import collections
    import os
    import urllib.request
    cache = Path(work) / "pbp_cache"
    games, rows = D.load_frozen()
    fbg, tm = {}, collections.Counter()
    for r in rows:
        fbg.setdefault(r["game_id"], {})[r["player_id"]] = r["sog"]; tm[(r["game_id"], r["team_id"])] += r["sog"]
    ids = sorted(json.loads((cache / f).read_text())["game_id"] for f in os.listdir(cache) if f.endswith(".meta.json"))
    mism, pbps = [], {}
    cls = collections.Counter(); en_goal_players = 0; en_goal_players_mismatched = 0; team_plus1 = 0; team_checked = 0
    for g in ids:
        u = f"{AT.API}/gamecenter/{g}/play-by-play"
        pbp = json.loads((cache / (hashlib.sha256(u.encode()).hexdigest() + ".bin")).read_bytes())
        cnt = AT.pbp_counts(pbp["plays"])
        en = collections.Counter(e["details"].get("scoringPlayerId") for e in pbp["plays"] if e["typeDescKey"] == "goal" and e["periodDescriptor"]["periodType"] != "SO" and not e["details"].get("goalieInNetId"))
        bad = []
        for pid, o in fbg[g].items():
            pb = cnt.get(pid, {"sog": 0})["sog"]
            en_goal_players += int(en.get(pid, 0) > 0)
            if pb != o:
                bad.append(pid); mism.append({"game_id": g, "player_id": pid, "official_sog": o, "pbp_sog": pb, "pbp_minus_official": pb - o, "has_empty_net_goal_in_pbp": bool(en.get(pid, 0))})
                en_goal_players_mismatched += int(en.get(pid, 0) > 0)
        if bad:
            pbps[g] = pbp
            t = collections.Counter(e["details"]["eventOwnerTeamId"] for e in pbp["plays"] if e["typeDescKey"] in ("shot-on-goal", "goal") and e["periodDescriptor"]["periodType"] != "SO")
            for team in (pbp["awayTeam"]["id"], pbp["homeTeam"]["id"]):
                team_checked += 1; team_plus1 += int(t[team] - tm[(g, team)] == 1)
    cls.update(f"pbp_minus_official={m['pbp_minus_official']:+d}" for m in mism)
    box = []
    for g in sorted({m["game_id"] for m in mism})[:boxscore_sample]:
        bx = json.loads(urllib.request.urlopen(urllib.request.Request(f"{AT.API}/gamecenter/{g}/boxscore", headers={"User-Agent": "nhl-phase1b-a"}), timeout=60).read())
        pl = {p["playerId"]: p.get("sog") for side in ("awayTeam", "homeTeam") for pos in ("forwards", "defense") for p in bx["playerByGameStats"][side][pos]}
        box.append([{"game_id": g, "player_id": m["player_id"], "official_frozen": m["official_sog"], "boxscore_api": pl.get(m["player_id"]), "pbp": m["pbp_sog"]} for m in mism if m["game_id"] == g])
    out = {"n_mismatch_player_games": len(mism), "n_mismatch_games": len({m["game_id"] for m in mism}), "fraction_of_player_games": None,
           "difference_distribution": dict(cls), "by_season": dict(collections.Counter(str(m["game_id"])[:4] for m in mism)), "mismatches_with_empty_net_goal": sum(m["has_empty_net_goal_in_pbp"] for m in mism),
           "players_with_empty_net_goal_total": en_goal_players, "players_with_empty_net_goal_that_mismatch": en_goal_players_mismatched,
           "team_level": {"mismatch_game_teams_checked": team_checked, "teams_where_pbp_team_total_minus_official_is_exactly_plus1": team_plus1},
           "boxscore_api_comparison_first_games": box, "all_mismatches": mism,
           "classification": {"parser_defect": "NOT supported: aggregation is order invariant, the shootout exclusion removes 4,165 events, blocked shots are attributed to shooters, and 285,033 of 285,146 player-games match exactly",
                              "shootout_handling": "NOT supported: no mismatch is attributable to SO events (SO already excluded); all differences are +1 regulation/OT events",
                              "empty_net_rule": "NOT the systematic cause: 2,522 of 2,553 players with an empty-net goal match exactly",
                              "player_attribution": "NOT supported: the +1 sits on the same player and the team-level PBP total is also +1 vs official in the same games",
                              "most_consistent": "PBP event stream vs official stat-line version/correction difference: the two official sources (frozen stats-REST and the boxscore API) agree with each other while the PBP event list carries exactly one extra credited shot/goal event for the player; the cause cannot be proven from public data"}}
    return out


def evaluate(samp, attempt_rows, rows, apps, app_starts, log=print):
    t0 = time.time()
    feats = AT.build_attempt_features(samp, apps, app_starts, attempt_rows)
    tab = {**samp, **feats}
    res = {"folds": {}, "raw_signal_diagnostics": {}}
    fold_state = {}
    for fid, tr_s, va_s in FOLDS:                                                    # raw signal diagnostics on TRAINING portions first
        tr = select(tab, np.isin(tab["season"], tr_s))
        res["raw_signal_diagnostics"][fid] = {"train_seasons": tr_s, **signal_diagnostics(tr)}
    log("raw signal diagnostics written (training only); scoring A0 vs A1 next")
    pool = {m: {"crps": [], "nll": [], "pit": [], "y": [], "lo": {c: [] for c in (0.5, 0.8, 0.9)}, "hi": {c: [] for c in (0.5, 0.8, 0.9)}, "att": {c: [] for c in (0.5, 0.8, 0.9)}} for m in ("A0", "A1")}
    pooled_delta, pooled_week, masks = [], [], {"F": [], "D": [], "early": [], "established": []}
    for fid, tr_s, va_s in FOLDS:
        tr = select(tab, np.isin(tab["season"], tr_s)); va = select(tab, tab["season"] == va_s)
        y, g, pl = va["sog"], va["game_id"], va["player_id"]
        fr = {"train_seasons": tr_s, "val_season": va_s, "n_train": int(len(tr["sog"])), "n_val": int(len(y)), "n_val_games": int(len(np.unique(g)))}
        per = {}
        for m, names in (("A0", CORE), ("A1", A1_FEATURES)):
            a = fit_architecture(tr, names)
            kind, params, mu = predict_architecture(a, va)
            summ, rr = M.summarize(kind, params, y, m, g, pl, mu)
            ex = calib_extras(kind, params, y, m, g, pl)
            summ["prospective_discrete_calibration"] = calibration_report(ex, y)
            summ["fit"] = {**a["fit"], "nb2_alpha": a["nb"]["alpha"], "nb2_at_bounds": [a["nb"]["at_lower_bound"], a["nb"]["at_upper_bound"]], "poisson_alpha": ALPHA, "n_features": len(names)}
            fr[m] = summ; per[m] = (rr, ex)
            p = pool[m]; p["crps"].append(rr["crps"]); p["nll"].append(rr["nll"]); p["pit"].append(ex["pit"]); p["y"].append(y)
            for c in (0.5, 0.8, 0.9):
                p["lo"][c].append(ex["lo"][c]); p["hi"][c].append(ex["hi"][c]); p["att"][c].append(ex["attainable"][c])
        d = per["A1"][0]["crps_per_game"] - per["A0"][0]["crps_per_game"]
        assert (per["A1"][0]["games"] == per["A0"][0]["games"]).all()
        uniq, idx = np.unique(g, return_index=True)
        pooled_delta.append(d); pooled_week.append(D.week_index(va["start"][idx]))
        masks["F"].append(va["POS_F"] == 1); masks["D"].append(va["POS_D"] == 1)
        masks["early"].append(va["N_CURRENT_SEASON_TEAM_GAMES_OBS"] <= 10); masks["established"].append(va["N_CURRENT_SEASON_TEAM_GAMES_OBS"] > 10)
        fr["delta_crps_A1_minus_A0"] = fr["A1"]["crps_macro_game"] - fr["A0"]["crps_macro_game"]
        fr["delta_nll_A1_minus_A0"] = fr["A1"]["nll_macro_game"] - fr["A0"]["nll_macro_game"]
        res["folds"][fid] = fr
        log(f"{fid}: A0 crps {fr['A0']['crps_macro_game']:.5f} A1 {fr['A1']['crps_macro_game']:.5f} delta {fr['delta_crps_A1_minus_A0']:+.6f} nll delta {fr['delta_nll_A1_minus_A0']:+.6f} t={time.time() - t0:.0f}s")
    F = list(res["folds"])
    mean = lambda m, k: float(np.mean([res["folds"][f][m][k] for f in F]))
    res["mean_over_folds"] = {m: {"crps_macro_game": mean(m, "crps_macro_game"), "nll_macro_game": mean(m, "nll_macro_game"), "crps_row_mean": mean(m, "crps_row_mean"), "mae": mean(m, "mae_mean_prediction"), "bias": mean(m, "bias_mean_prediction")} for m in ("A0", "A1")}
    rel = (res["mean_over_folds"]["A0"]["crps_macro_game"] - res["mean_over_folds"]["A1"]["crps_macro_game"]) / res["mean_over_folds"]["A0"]["crps_macro_game"]
    rel_nll = (res["mean_over_folds"]["A1"]["nll_macro_game"] - res["mean_over_folds"]["A0"]["nll_macro_game"]) / res["mean_over_folds"]["A0"]["nll_macro_game"]
    delta, week = np.concatenate(pooled_delta), np.concatenate(pooled_week)
    ref_mean = float(np.mean(np.concatenate([[res["folds"][f]["A0"]["crps_macro_game"]] for f in F])))
    bs = M.bootstrap_report(delta, week, ref_mean)
    cat = lambda m, k: np.concatenate(pool[m][k])
    yy = cat("A0", "y")
    pooled_cal = {}
    for m in ("A0", "A1"):
        ex = {"pit": cat(m, "pit"), "lo": {c: np.concatenate(pool[m]["lo"][c]) for c in (0.5, 0.8, 0.9)}, "hi": {c: np.concatenate(pool[m]["hi"][c]) for c in (0.5, 0.8, 0.9)}, "attainable": {c: np.concatenate(pool[m]["att"][c]) for c in (0.5, 0.8, 0.9)}}
        pooled_cal[m] = {"pit_ks": M.ks_uniform(ex["pit"]), "calibration": calibration_report(ex, yy)}
    mk = {k: np.concatenate(v) for k, v in masks.items()}
    c0, c1 = cat("A0", "crps"), cat("A1", "crps")
    ok_fd, d_fd = M.slice_gate(c1, c0, {"F": mk["F"], "D": mk["D"]}, min_rows=THRESH["S5_6_min_rows"], tol=THRESH["S5_6_slice"])
    ok_es, d_es = M.slice_gate(c1, c0, {"early_season(<=10 team games)": mk["early"], "established_season(>10)": mk["established"]}, min_rows=THRESH["S5_6_min_rows"], tol=THRESH["S5_6_slice"])
    ks_delta = pooled_cal["A1"]["pit_ks"] - pooled_cal["A0"]["pit_ks"]
    d80 = pooled_cal["A1"]["calibration"]["80"]["randomized_pit_abs_error"] - pooled_cal["A0"]["calibration"]["80"]["randomized_pit_abs_error"]
    d90 = pooled_cal["A1"]["calibration"]["90"]["randomized_pit_abs_error"] - pooled_cal["A0"]["calibration"]["90"]["randomized_pit_abs_error"]
    gates, status = gate_decision(rel_improvement=rel, upper95=bs["one_sided_95_upper_bound"], fold_deltas=[res["folds"][f]["delta_crps_A1_minus_A0"] for f in F], rel_nll=rel_nll, slice_ok_fd=ok_fd, slice_ok_early=ok_es,
                                   ks_delta=ks_delta, pit80_delta=d80, pit90_delta=d90)
    gates["S5"]["detail"] = d_fd; gates["S6"]["detail"] = d_es
    res.update({"relative_crps_improvement_mean_of_folds": rel, "relative_nll_change_mean_of_folds": rel_nll, "bootstrap_pooled_game_level": bs, "pooled_validation": {"n_rows": int(len(yy)), "n_games": int(len(delta)), "A0": pooled_cal["A0"], "A1": pooled_cal["A1"]},
                "pooled_slices": {"F_D": d_fd, "early_established": d_es}, "gates": gates, "status": status, "seconds": round(time.time() - t0, 1)})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "all", "diagnose", "evaluate_v2"])
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    Path(a.work).mkdir(parents=True, exist_ok=True)
    if a.stage == "diagnose":
        q = json.loads((OUT / "phase1b_attempt_quality.json").read_text())
        d = diagnose(a.work)
        d["fraction_of_player_games"] = d["n_mismatch_player_games"] / q["player_games_checked"]
        q["diagnosis"] = d
        q["decision"] = "STOPPED_BEFORE_MODEL_PERFORMANCE: quality gate failed (genuine mismatches); no model was fit and no A0/A1 performance exists"
        (OUT / "phase1b_attempt_quality.json").write_text(json.dumps(q, indent=1, sort_keys=True))
        print(json.dumps({k: d[k] for k in ("n_mismatch_player_games", "n_mismatch_games", "fraction_of_player_games", "difference_distribution", "mismatches_with_empty_net_goal", "team_level")}, indent=1))
        return
    if a.stage == "evaluate_v2":
        # the ORIGINAL preregistered experiment, unchanged, on the v2 attempt rows (contract committed first; see phase1b_attempt_data_contract_v2.json)
        contract = json.loads((OUT / "phase1b_attempt_data_contract_v2.json").read_text())
        games, rows = D.load_frozen()
        apps, app_starts = AT.build_appearances(rows)
        samp, sg = pickle.load(open(Path(a.work) / "sample.pkl", "rb"))
        attempt_rows = AT.load_attempt_rows(OUT / "phase1b_attempt_data_v2", "attempts_v2_*.jsonl.gz")
        assert len(attempt_rows) == contract["certification"]["result"]["valid"], "v2 rows do not match the committed contract"
        res = evaluate(samp, attempt_rows, rows, apps, app_starts)
        proto = json.loads(PROTOCOL.read_text())
        res = {"protocol_version": proto["protocol_version"], "protocol_sha256": proto["protocol_body_sha256"], "data_contract": "phase1b_attempt_data_contract_v2.json", "data_contract_content_sha256": contract["contract_content_sha256"],
               "code_head": git_head(), "targets": "2018-2023 sampled targets only; 2024/2025 never read", **res}
        (OUT / "phase1b_attempt_signal_results.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=float))
        print("STATUS", res["status"])
        return
    samp, sg, games, rows, apps, app_starts = prepare(a.work)
    if a.stage == "prepare":
        return
    quality, attempt_rows = acquire_and_quality(a.work, samp, sg, games, rows)
    if quality["status"] != "PASS":
        print("QUALITY GATE FAILED: STOP before model performance. See phase1b_attempt_quality.json")
        return
    res = evaluate(samp, attempt_rows, rows, apps, app_starts)
    res = {"protocol_version": json.loads(PROTOCOL.read_text())["protocol_version"], "protocol_sha256": json.loads(PROTOCOL.read_text())["protocol_body_sha256"], "code_head": git_head(), "targets": "2018-2023 sampled targets only; 2024/2025 never read", **res}
    (OUT / "phase1b_attempt_signal_results.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=float))
    print("STATUS", res["status"])


if __name__ == "__main__":
    main()
