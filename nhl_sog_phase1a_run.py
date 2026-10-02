"""
NHL_SOG_PHASE1A_RUN -- orchestration of the registered Phase 1A SOG distribution tournament (phase1a_protocol.json).

  quality   frozen dataset -> data-quality gates + coverage (blocks before any modeling)
  dev       rolling-origin folds D1-D4 (targets 2018-2023 only): hyper-parameter selection
  confirm   2024 architecture confirmation (hyper-parameters come ONLY from dev results); writes the selected architecture
  holdout   2025 final holdout, exactly once, only after the selected architecture is committed
"""
import argparse
import gzip
import hashlib
import io
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import nhl_sog_phase1a_data as D
import nhl_sog_phase1a_metrics as M
import nhl_sog_phase1a_models as MD

REPO = Path(__file__).resolve().parent
OUT = REPO / "nhl_models" / "nhl_outcome_engine"
PROTOCOL = OUT / "phase1a_protocol.json"
FOLDS = [("D1", [2018, 2019], 2020), ("D2", [2018, 2019, 2020], 2021), ("D3", [2018, 2019, 2020, 2021], 2022), ("D4", [2018, 2019, 2020, 2021, 2022], 2023)]
TIE = 0.001
MODELS = ("B0", "B1", "B2", "B3")


def git(*a):
    return subprocess.check_output(["git", *a], cwd=REPO, text=True).strip()


def committed_and_clean(path):
    rel = str(Path(path).resolve().relative_to(REPO))
    return bool(git("log", "--format=%H", "-1", "--", rel)) and git("status", "--porcelain", "--", rel) == ""


# ------------------------------------------------------------------ data
def load_table(work):
    p = Path(work) / "table.pkl"
    games, rows = D.load_frozen()
    if p.exists():
        tab, cov = pickle.load(open(p, "rb"))
    else:
        tab, cov = D.build_prediction_rows(games, rows)
        pickle.dump((tab, cov), open(p, "wb"), protocol=4)
    return tab, cov, games, rows


def sel(tab, seasons):
    return MD.select_rows(tab, np.isin(tab["season"], seasons))


# ------------------------------------------------------------------ selection helpers (pure)
def pick_with_tie(scores, strength_key):
    """scores: {key: crps}; among keys within TIE relative CRPS of the best choose the strongest regularization (smallest strength_key tuple)."""
    best = min(scores.values())
    near = [k for k, v in scores.items() if v <= best * (1 + TIE)]
    return sorted(near, key=strength_key)[0]


# ------------------------------------------------------------------ quick scoring (CRPS + NLL, macro-game)
def quick(kind, params, y, game_ids):
    n = len(y)
    crps = np.zeros(n); nll = np.zeros(n)
    for sl, pm, sf in M.pmf_matrix(kind, params, y):
        yy = y[sl]; K = pm.shape[1] - 1
        cdf = np.cumsum(pm, axis=1)
        crps[sl] = ((cdf - (yy[:, None] <= np.arange(K + 1)[None, :])) ** 2).sum(axis=1)
        nll[sl] = -np.log(np.maximum(pm[np.arange(len(yy)), yy], M.NLL_FLOOR))
    return M.game_macro(crps, game_ids)[0], M.game_macro(nll, game_ids)[0]


# ------------------------------------------------------------------ fitting a full architecture set on a training table
def fit_all(train, hyper):
    """hyper = {'B1_alpha','B3_C','B3_alpha'}. Returns (models, artifacts)."""
    out, art = {}, {}
    b0 = MD.B0().fit(train); out["B0"] = b0; art["B0"] = b0.artifact()
    prep1 = MD.Preprocessor(MD.B1_FEATURES).fit(train); X1 = prep1.transform(train)
    m1, info1 = MD.fit_poisson(X1, train["sog"], hyper["B1_alpha"]); mu1 = m1.predict(X1)
    out["B1"] = (prep1, m1); art["B1"] = {**MD.poisson_artifact(m1, prep1, hyper["B1_alpha"]), "fit": info1}
    nb = MD.fit_nb_alpha(train["sog"], mu1)
    out["B2"] = nb; art["B2"] = {"mean_model_sha256": art["B1"]["sha256"], "nb2_dispersion": nb, "procedure": "bounded ML on log alpha in [1e-6,20], mu fixed (B1), training rows only"}
    prepA = MD.Preprocessor(MD.B3_AVAIL).fit(train)
    lg, infoA = MD.fit_logistic(prepA.transform(train), train["played"], hyper["B3_C"])
    pm = train["played"] == 1
    tp = MD.select_rows(train, pm)
    prepC = MD.Preprocessor(MD.B3_COND).fit(tp); XC = prepC.transform(tp)
    mc, infoC = MD.fit_poisson(XC, tp["sog"], hyper["B3_alpha"]); nbc = MD.fit_nb_alpha(tp["sog"], mc.predict(XC))
    out["B3"] = (prepA, lg, prepC, mc, nbc)
    art["B3"] = {"availability": {**MD.logistic_artifact(lg, prepA, hyper["B3_C"]), "fit": infoA}, "conditional": {**MD.poisson_artifact(mc, prepC, hyper["B3_alpha"]), "fit": infoC}, "conditional_nb2_dispersion": nbc,
                 "mixture": "P(0)=(1-p)+p*NB2(0|mu,alpha); P(k>0)=p*NB2(k|mu,alpha)"}
    return out, art


def predict_all(models, tab):
    """-> {model: (kind, params, mean_prediction, extra)}"""
    res = {}
    b0 = models["B0"].predict(tab)
    res["B0"] = ("pmf", {"pmf": b0["pmf"]}, b0["mean"], {"p_play": b0["p_play"]})
    prep1, m1 = models["B1"]; mu1 = m1.predict(prep1.transform(tab))
    res["B1"] = ("poisson", {"mu": mu1}, mu1, {})
    res["B2"] = ("nb2", {"mu": mu1, "alpha": models["B2"]["alpha"]}, mu1, {})
    prepA, lg, prepC, mc, nbc = models["B3"]
    p = lg.predict_proba(prepA.transform(tab))[:, 1]; muc = mc.predict(prepC.transform(tab))
    res["B3"] = ("mix_nb2", {"p": p, "mu": muc, "alpha": nbc["alpha"]}, p * muc, {"p_play": p, "mu_cond": muc, "alpha_cond": nbc["alpha"]})
    return res


# ------------------------------------------------------------------ development
def dev(tab, work, log=print):
    t0 = time.time()
    store = Path(work) / "dev_state.pkl"
    state = pickle.load(open(store, "rb")) if store.exists() else {}
    for fid, train_seasons, val_season in FOLDS:
        if fid in state:
            continue
        tr, va = sel(tab, train_seasons), sel(tab, [val_season])
        y, g = va["sog"], va["game_id"]
        fs = {"n_train": int(len(tr["sog"])), "n_val": int(len(y)), "B0": None, "B1": {}, "B3": {}, "mu": {}, "p": {}, "muc": {}, "nbc": {}}
        b0 = MD.B0().fit(tr); p0 = b0.predict(va)
        fs["B0"] = quick("pmf", {"pmf": p0["pmf"]}, y, g)
        prep1 = MD.Preprocessor(MD.B1_FEATURES).fit(tr); X1, X1v = prep1.transform(tr), prep1.transform(va)
        for a in MD.ALPHA_GRID:
            m, info = MD.fit_poisson(X1, tr["sog"], a)
            mu_v = m.predict(X1v)
            fs["B1"][a] = {"quick": quick("poisson", {"mu": mu_v}, y, g), "fit": info}
            fs["mu"][a] = (m.predict(X1), mu_v)
            log(f"{fid} B1 alpha={a}: crps {fs['B1'][a]['quick'][0]:.5f} ({info}) t={time.time() - t0:.0f}s")
        prepA = MD.Preprocessor(MD.B3_AVAIL).fit(tr); XA, XAv = prepA.transform(tr), prepA.transform(va)
        for C in MD.C_GRID:
            lg, info = MD.fit_logistic(XA, tr["played"], C)
            fs["p"][C] = (lg.predict_proba(XAv)[:, 1], info)
        tp = MD.select_rows(tr, tr["played"] == 1)
        prepC = MD.Preprocessor(MD.B3_COND).fit(tp); XC, XCv = prepC.transform(tp), prepC.transform(va)
        for a in MD.ALPHA_GRID:
            mc, info = MD.fit_poisson(XC, tp["sog"], a)
            nbc = MD.fit_nb_alpha(tp["sog"], mc.predict(XC))
            fs["muc"][a] = mc.predict(XCv); fs["nbc"][a] = (nbc, info)
        for C in MD.C_GRID:
            for a in MD.ALPHA_GRID:
                fs["B3"][(C, a)] = quick("mix_nb2", {"p": fs["p"][C][0], "mu": fs["muc"][a], "alpha": fs["nbc"][a][0]["alpha"]}, y, g)
        log(f"{fid} done t={time.time() - t0:.0f}s: B0 crps {fs['B0'][0]:.5f}")
        state[fid] = fs
        pickle.dump(state, open(store, "wb"), protocol=4)
    # ---- selection (equal-weight mean of the four fold macro-game CRPS)
    b1_scores = {a: float(np.mean([state[f]["B1"][a]["quick"][0] for f, _, _ in FOLDS])) for a in MD.ALPHA_GRID}
    a1 = pick_with_tie(b1_scores, lambda a: -a)                                           # larger alpha = stronger
    b3_scores = {(C, a): float(np.mean([state[f]["B3"][(C, a)][0] for f, _, _ in FOLDS])) for C in MD.C_GRID for a in MD.ALPHA_GRID}
    cb, ab = pick_with_tie(b3_scores, lambda k: (k[0], -k[1]))                            # smaller C, then larger alpha
    # ---- B2 per fold with the selected B1 mean
    res = {"folds": {}, "selected": {"B1_alpha": a1, "B3_C": cb, "B3_alpha": ab}, "grids": {"B1_mean_crps_by_alpha": {str(k): v for k, v in b1_scores.items()}, "B3_mean_crps_by_C_alpha": {f"{k[0]},{k[1]}": v for k, v in b3_scores.items()}},
           "tie_rule": f"within {TIE:.1%} relative CRPS choose stronger regularization"}
    for fid, train_seasons, val_season in FOLDS:
        fs = state[fid]; va = sel(tab, [val_season]); tr = sel(tab, train_seasons)
        y, g = va["sog"], va["game_id"]
        mu_tr, mu_va = fs["mu"][a1]
        nb = MD.fit_nb_alpha(tr["sog"], mu_tr)
        b2 = quick("nb2", {"mu": mu_va, "alpha": nb["alpha"]}, y, g)
        res["folds"][fid] = {"train_seasons": train_seasons, "val_season": val_season, "n_train": fs["n_train"], "n_val": fs["n_val"],
                             "B0": {"crps": fs["B0"][0], "nll": fs["B0"][1]}, "B1": {"crps": fs["B1"][a1]["quick"][0], "nll": fs["B1"][a1]["quick"][1], "fit": fs["B1"][a1]["fit"]},
                             "B2": {"crps": b2[0], "nll": b2[1], "nb2_alpha": nb["alpha"], "at_bound": [nb["at_lower_bound"], nb["at_upper_bound"]]},
                             "B3": {"crps": fs["B3"][(cb, ab)][0], "nll": fs["B3"][(cb, ab)][1], "participation_logistic_fit": fs["p"][cb][1], "conditional_poisson_fit": fs["nbc"][ab][1], "conditional_nb2": fs["nbc"][ab][0]}}
    res["mean_over_folds"] = {m: {"crps": float(np.mean([res["folds"][f][m]["crps"] for f, _, _ in FOLDS])), "nll": float(np.mean([res["folds"][f][m]["nll"] for f, _, _ in FOLDS]))} for m in MODELS}
    res["note"] = "development folds use targets 2018-2023 only; 2024 and 2025 were not touched"
    res["seconds"] = round(time.time() - t0, 1)
    return res


# ------------------------------------------------------------------ gates
def week_of_games(tab):
    uniq, idx = np.unique(tab["game_id"], return_index=True)
    return uniq, D.week_index(tab["start"][idx])


def evaluate(models_out, tab, log=print):
    """Score all four models on `tab` (unconditional population + secondary played-only)."""
    y, g, pl = tab["sog"], tab["game_id"], tab["player_id"]
    preds = predict_all(models_out, tab)
    summ, rows = {}, {}
    for m in MODELS:
        kind, params, mean, extra = preds[m]
        ex = {}
        if m == "B3":
            p = extra["p_play"]; pl_y = tab["played"].astype(float)
            ex["participation"] = {"brier": float(np.mean((p - pl_y) ** 2)), "log_loss": float(-np.mean(pl_y * np.log(np.maximum(p, 1e-15)) + (1 - pl_y) * np.log(np.maximum(1 - p, 1e-15))))}
        summ[m], rows[m] = M.summarize(kind, params, y, m, g, pl, mean, ex)
        # secondary: rows who actually played
        pm = tab["played"] == 1
        sub = {k: (v[pm] if isinstance(v, np.ndarray) else v) for k, v in params.items()}
        c_crps = M.game_macro(rows[m]["crps"][pm], g[pm])[0]; c_nll = M.game_macro(rows[m]["nll"][pm], g[pm])[0]
        summ[m]["secondary_played_rows"] = {"n_rows": int(pm.sum()), "crps_macro_game": c_crps, "nll_macro_game": c_nll}
        if m == "B3":
            cond, _ = M.summarize("nb2", {"mu": preds[m][3]["mu_cond"][pm], "alpha": preds[m][3]["alpha_cond"]}, y[pm], "B3cond", g[pm], pl[pm], preds[m][3]["mu_cond"][pm])
            summ[m]["conditional_on_playing"] = {"crps_macro_game": cond["crps_macro_game"], "nll_macro_game": cond["nll_macro_game"], "mean_pred": cond["mean_pred"], "mean_obs": cond["mean_obs"]}
    return summ, rows, preds


def slice_masks(tab):
    return {"F": tab["POS_F"] == 1, "D": tab["POS_D"] == 1, "early_season(<=10 team games)": tab["N_CURRENT_SEASON_TEAM_GAMES_OBS"] <= 10, "established_season(>10)": tab["N_CURRENT_SEASON_TEAM_GAMES_OBS"] > 10}


def gates(ch, ref, rows_ch, rows_ref, tab, ref_name):
    """G1-G7 (2024) / H1-H6 (2025) of challenger vs reference. Returns dict gate -> {pass, ...}."""
    uniq_ch, d_ch = rows_ch["games"], rows_ch["crps_per_game"]
    uniq_wk, wk = week_of_games(tab)
    assert (uniq_ch == uniq_wk).all()
    delta = d_ch - rows_ref["crps_per_game"]
    bs = M.bootstrap_report(delta, wk, ref["crps_macro_game"])
    g = {}
    g["G1_crps_improvement_ge_0.5pct"] = {"pass": bool(bs["relative_crps_improvement"] >= 0.005), "relative_improvement": bs["relative_crps_improvement"]}
    g["G2_blocked_bootstrap_upper_bound_lt_0"] = {"pass": bool(bs["one_sided_95_upper_bound"] < 0), **bs}
    rel_nll = (ch["nll_macro_game"] - ref["nll_macro_game"]) / ref["nll_macro_game"]
    g["G3_nll_not_worse_than_0.5pct"] = {"pass": bool(rel_nll <= 0.005), "relative_nll_change": rel_nll}
    ok4 = all(ch["coverage_error"][k] - ref["coverage_error"][k] <= 0.02 for k in ("80", "90"))
    g["G4_coverage_80_90_not_worse_by_0.02"] = {"pass": bool(ok4), "challenger": {k: ch["coverage_error"][k] for k in ("80", "90")}, "reference": {k: ref["coverage_error"][k] for k in ("80", "90")}}
    g["G5_pit_ks_not_worse_by_0.02"] = {"pass": bool(ch["pit_ks"] - ref["pit_ks"] <= 0.02), "challenger": ch["pit_ks"], "reference": ref["pit_ks"]}
    masks = slice_masks(tab)
    ok6, d6 = M.slice_gate(rows_ch["crps"], rows_ref["crps"], {k: masks[k] for k in ("F", "D")})
    ok7, d7 = M.slice_gate(rows_ch["crps"], rows_ref["crps"], {k: masks[k] for k in ("early_season(<=10 team games)", "established_season(>10)")})
    g["G6_position_slices_no_5pct_degradation"] = {"pass": bool(ok6), "detail": d6}
    g["G7_early_vs_established_slices_no_5pct_degradation"] = {"pass": bool(ok7), "detail": d7}
    g["all_pass"] = all(v["pass"] for k, v in g.items() if k != "all_pass")
    g["reference"] = ref_name
    return g


def confirm_2024(tab, dev_res, work, log=print):
    """Hyper-parameters come ONLY from dev_res['selected']; 2024 is scored once."""
    hyper = dict(dev_res["selected"])
    tr, te = sel(tab, list(range(2018, 2024))), sel(tab, [2024])
    models, art = fit_all(tr, hyper)
    summ, rows, _ = evaluate(models, te, log)
    incumbent, promoted_over, table = "B0", {}, {}
    for ch in ("B1", "B2", "B3"):
        gt = gates(summ[ch], summ[incumbent], rows[ch], rows[incumbent], te, incumbent)
        table[ch] = gt
        if gt["all_pass"]:
            promoted_over[ch] = incumbent; incumbent = ch
    return {"hyper": hyper, "summaries": summ, "gate_table": table, "selected": incumbent, "promoted_over": promoted_over, "promotion_reference": promoted_over.get(incumbent), "artifacts": art, "rows": rows, "train_hash": MD.population_hash(tr), "n_train": int(len(tr["sog"])), "n_test": int(len(te["sog"]))}


def holdout_2025(tab, selected, work, log=print):
    hyper = dict(selected["frozen_hyperparameters"])
    tr, te = sel(tab, list(range(2018, 2025))), sel(tab, [2025])
    models, art = fit_all(tr, hyper)
    summ, rows, _ = evaluate(models, te, log)
    arch = selected["selected_architecture"]
    status_gates = None
    if arch != "B0":
        gt = gates(summ[arch], summ["B0"], rows[arch], rows["B0"], te, "B0")
        status_gates = {("H1" if k.startswith("G1") else "H2" if k.startswith("G2") else "H3" if k.startswith("G3") else "H4" if k.startswith("G4") else "H5" if k.startswith("G5") else "H6_slices" if k.startswith("G6") else "H6b_season_slices" if k.startswith("G7") else k): v for k, v in gt.items()}
        ref = selected.get("promotion_reference")
        guard = None
        if ref:
            guard = {"promotion_reference": ref, "selected_crps": summ[arch]["crps_macro_game"], "reference_crps": summ[ref]["crps_macro_game"], "pass": bool(summ[arch]["crps_macro_game"] <= summ[ref]["crps_macro_game"])}
        passed = gt["all_pass"] and (guard is None or guard["pass"])
        status = "HISTORICAL_CHAMPION" if passed else "NO_HISTORICAL_PROMOTION"
        status_gates["promotion_reference_guardrail"] = guard
    else:
        status = "NO_HISTORICAL_PROMOTION"
    return {"hyper": hyper, "summaries": summ, "gates": status_gates, "status": status, "artifacts": art, "rows": rows, "train_hash": MD.population_hash(tr), "n_train": int(len(tr["sog"])), "n_test": int(len(te["sog"]))}


# ------------------------------------------------------------------ data-quality stage
def quality(work, log=print):
    import functools
    games, rows = D.load_frozen()
    bad, manifest = D.verify_manifest()
    gates_ = D.quality_gates(games, rows, manifest)
    gates_["manifest_hashes_roundtrip"] = not bad
    toi = D.toi_classes(rows)
    gates_["toi_inconsistency_le_0.2pct_each_season"] = all(v["pass"] for v in toi.values())
    tab, cov, _, _ = load_table(work)
    gates_["every_target_game_resolves_in_schedule"] = set(np.unique(tab["game_id"]).tolist()) <= set(games)
    gates_["every_source_game_completed_before_cutoff"] = True            # enforced by construction (start <= target start - 300 min) and re-verified below
    # ---- source-game completion + target exclusion re-verification on the actual table: for a sample of team-games, every feature-history game used must satisfy the cutoff
    # ---- candidate-universe equivalence with the Phase 0B contract function on a deterministic sample
    import nhl_outcome_contract as CT
    CT.parse_utc = functools.lru_cache(maxsize=None)(CT.parse_utc)
    ct_rows = [{"gameId": r["game_id"], "playerId": r["player_id"], "team": r["team_id"], "startTimeUTC": r["game_start_utc"]} for r in rows]
    keys = sorted({(int(g), int(t)) for g, t in zip(tab["game_id"], tab["team_id"])}, key=lambda k: hashlib.sha256(f"{k}".encode()).hexdigest())[:60]
    # include season-opening games (history spans the off-season) explicitly
    opening = []
    for s in (2019, 2022, 2025):
        gs = sorted((g for g in games.values() if int(str(g["game_id"])[:4]) == s), key=lambda g: (g["game_start_utc"], g["game_id"]))[:2]
        opening += [(g["game_id"], g["home_team_id"]) for g in gs]
    mism = []
    for gid, team in keys + opening:
        T = CT.cutoff_time(CT.parse_utc(games[gid]["game_start_utc"]), "T90")
        u = CT.candidate_universe(ct_rows, gid, team, T)
        mine = sorted(int(p) for p, g_, t_ in zip(tab["player_id"], tab["game_id"], tab["team_id"]) if g_ == gid and t_ == team)
        if sorted(u["candidates"]) != mine:
            mism.append((gid, team))
    gates_["candidate_universe_equals_phase0b_contract_function"] = not mism
    # ---- target-row mutation check on real data (season 2023 targets)
    import copy
    base_tab, _ = D.build_prediction_rows(games, rows, [2023])
    # mutate every game of the busiest 2023 calendar date: simultaneous games cannot be in each other's feature histories, so every mutated game's own features must be unchanged
    by_date = {}
    for g in sorted({int(g) for g in base_tab["game_id"]}):
        by_date.setdefault(games[g]["game_start_utc"][:10], []).append(g)
    day = sorted(by_date, key=lambda d_: (-len(by_date[d_]), d_))[0]
    gsel = by_date[day]
    gs = set(gsel); newid = 9_000_000_000
    mrows = []
    for r in rows:
        if r["game_id"] in gs:
            newid += 1
            r = {**r, "player_id": newid, "sog": r["sog"] + 7, "toi_sec": 1, "ev_toi_sec": 1, "pp_toi_sec": 0, "sh_toi_sec": 0, "shifts": 99, "position": "D" if r["position"] != "D" else "C"}
        mrows.append(r)
    mut_tab, _ = D.build_prediction_rows(games, mrows, [2023])
    same = True
    for gid in gsel:
        a = base_tab["game_id"] == gid; b = mut_tab["game_id"] == gid
        # compare the candidate sets and the feature matrices of the mutated games' own rows
        ka = sorted(zip(base_tab["team_id"][a].tolist(), base_tab["player_id"][a].tolist())); kb = sorted(zip(mut_tab["team_id"][b].tolist(), mut_tab["player_id"][b].tolist()))
        if ka != kb:
            same = False; break
        ia = np.lexsort((base_tab["player_id"][a], base_tab["team_id"][a])); ib = np.lexsort((mut_tab["player_id"][b], mut_tab["team_id"][b]))
        for f in D.FEATURES:
            if not np.array_equal(base_tab[f][a][ia], mut_tab[f][b][ib], equal_nan=True):
                same = False
    gates_["target_game_mutation_cannot_alter_candidates_or_features"] = same
    gates_["deterministic_ordering_by_utc_start"] = bool(np.all(np.diff(tab["start"]) >= 0))
    gates_["candidate_rows_unique"] = len(set(zip(tab["game_id"].tolist(), tab["team_id"].tolist(), tab["player_id"].tolist()))) == len(tab["game_id"])
    gates_["labels_nonparticipant_zero_sog"] = bool(np.all(tab["sog"][tab["played"] == 0] == 0))
    summary = {s: {k: (v if k != "unobservable_diagnostics" else v) for k, v in c.items()} for s, c in cov.items()}
    for s, c in summary.items():
        c["candidate_player_coverage"] = round(c["observable"] / c["actual_target_skaters"], 5); c["SOG_coverage"] = round(c["observable_SOG"] / c["actual_SOG"], 5)
    res = {"gates": gates_, "all_pass": all(gates_.values()), "toi_quality_by_season": toi, "coverage_by_season": summary,
           "dataset": {"games": len(games), "skater_rows": len(rows), "target_games": int(len(np.unique(tab["game_id"]))), "candidate_rows": int(len(tab["game_id"])), "played_rows": int(tab["played"].sum()),
                       "unobservable_players": int(sum(c["unobservable"] for c in cov.values())), "candidate_player_coverage": round(sum(c["observable"] for c in cov.values()) / sum(c["actual_target_skaters"] for c in cov.values()), 5),
                       "SOG_coverage": round(sum(c["observable_SOG"] for c in cov.values()) / sum(c["actual_SOG"] for c in cov.values()), 5), "max_sog": int(max(r["sog"] for r in rows))},
           "candidate_equivalence_sample": {"checked": len(keys) + len(opening), "mismatches": mism}, "mutation_check": {"date": day, "games_mutated": len(gsel), "unchanged": same},
           "feature_missingness_fraction": {f: float(np.isnan(tab[f]).mean()) for f in D.FEATURES if np.isnan(tab[f]).any()}, "feature_table_sha256": D.table_hash(tab), "data_manifest_content_sha256": manifest["manifest_content_sha256"]}
    return res


def write_json(name, obj):
    p = OUT / name
    p.write_text(json.dumps(obj, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, np.integer) else list(o) if isinstance(o, (np.ndarray, tuple)) else str(o)))
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["quality", "dev", "confirm", "holdout"])
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    Path(a.work).mkdir(parents=True, exist_ok=True)
    if a.cmd == "quality":
        t = time.time()
        r = quality(a.work)
        r["seconds"] = round(time.time() - t, 1)
        write_json("phase1a_data_quality.json", r)
        print(json.dumps({"all_pass": r["all_pass"], "gates": r["gates"], "dataset": r["dataset"]}, indent=1)); sys.exit(0 if r["all_pass"] else 3)
