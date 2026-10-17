#!/usr/bin/env python3
"""Historical reproduction of the fixed B2 (BURNED_REPRODUCTION_ONLY) and comparator evaluation on identical rows.
Writes phase1a_sog_burned_reproduction.json and phase1a_sog_comparators.json. Nothing here validates or promotes anything."""
import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nhl_v2_phase1a_sog_compare as C
import nhl_v2_phase1a_sog_data as D
import nhl_v2_phase1a_sog_model as M
from nhl_v2_phase1a_sog_quality import meaningful_mask

OUT = D.OUT
GRADE = "BURNED_REPRODUCTION_ONLY"
V1_REF = json.loads((OUT / "phase1a_v1_migration_audit.json").read_text())["burned_reference_numbers"]["b2_2025_holdout"]


def fit_base_rates(tab, pop_mask):
    """FIT 2018-2022 event rate P(SOG>=k) by position class within the population definition (F, D, U->global)."""
    fit = np.isin(tab["season"], C.FIT_SEASONS) & pop_mask
    rates = {}
    for name, m in (("F", tab["POS_F"] == 1), ("D", tab["POS_D"] == 1), ("U", tab["POS_UNKNOWN"] == 1)):
        sel = fit & m if (fit & m).sum() >= 200 else fit
        rates[name] = [float((tab["sog"][sel] >= k).mean()) for k in range(1, 6)]
    return rates


def row_base_rates(tab, rates):
    out = np.zeros((len(tab["sog"]), 5))
    for name, m in (("F", tab["POS_F"] == 1), ("D", tab["POS_D"] == 1), ("U", tab["POS_UNKNOWN"] == 1)):
        out[m] = rates[name]
    return out


def sub(tab, mask):
    return M.select_rows(tab, mask)


def evaluate_pop(tab_te, mu, alpha, pop_mask, rates, label):
    t = sub(tab_te, pop_mask)
    s, rows = M.summarize("nb2", {"mu": mu[pop_mask], "alpha": alpha}, t["sog"], "B2", t["game_id"], t["player_id"], mu[pop_mask], row_base_rates(t, rates))
    s["population"] = label
    return s, rows


def main():
    games, rows = D.load_frozen()
    cache = Path("/tmp/nhl_v2_tab90.pkl")
    if cache.exists():
        tab, cov = pickle.load(open(cache, "rb"))
    else:
        tab, cov = D.build_rows(games, rows, horizon_min=90)
        pickle.dump((tab, cov), open(cache, "wb"), protocol=4)
    mm = meaningful_mask(tab)
    res = {"artifact": "phase1a_sog_burned_reproduction", "evidence_grade": GRADE, "horizon": "T90", "table_sha256": D.table_hash(tab), "runs": {}}
    arts = {}
    for label, fit_seasons, score in (("fit2018_2024_score2025", list(range(2018, 2025)), 2025), ("fit2018_2023_score2024", list(range(2018, 2024)), 2024)):
        train = sub(tab, np.isin(tab["season"], fit_seasons)); te_mask = tab["season"] == score
        art = M.fit_b2(train, "T90"); arts[label] = art
        te = sub(tab, te_mask); mu = M.predict_mu(art, te); alpha = art["nb2"]["alpha"]
        mm_te = mm[te_mask]; pl_te = te["played"] == 1
        full_rates = fit_base_rates(tab, np.ones(len(tab["sog"]), bool)); mean_rates = fit_base_rates(tab, mm)
        run = {"train_seasons": fit_seasons, "score_season": score, "n_train_rows": art["n_train_rows"], "nb2_alpha": art["nb2"], "poisson_converged": art["converged"], "model_sha256": art["sha256"], "populations": {}}
        for pname, pm, rates in (("FULL_PREGAME_CANDIDATE_UNIVERSE", np.ones(len(mu), bool), full_rates), ("MEANINGFUL_EXPECTED_PARTICIPANT_UNIVERSE", mm_te, mean_rates), ("PLAYED_ONLY_SECONDARY", pl_te, full_rates)):
            s, _ = evaluate_pop(te, mu, alpha, pm, rates, pname)
            run["populations"][pname] = s
        res["runs"][label] = run
    r25 = res["runs"]["fit2018_2024_score2025"]["populations"]["FULL_PREGAME_CANDIDATE_UNIVERSE"]
    ref = V1_REF
    d = {"n_rows_equal": r25["n_rows"] == ref["n_rows"], "dCRPS": r25["crps_macro_game"] - ref["crps_macro_game"], "dNLL": r25["nll_macro_game"] - ref["nll_macro_game"], "dMAE": r25["central"]["mae_mean"] - ref["mae_mean_prediction"],
         "dBias": r25["central"]["bias"] - ref["bias_mean_prediction"], "dBrier_P3": r25["thresholds"]["P(SOG>=3)"]["brier"] - ref["p_ge_3_brier"]}
    ok = d["n_rows_equal"] and abs(d["dCRPS"]) <= 1e-4 and abs(d["dNLL"]) <= 1e-4 and abs(d["dMAE"]) <= 1e-4
    res["reproduction_vs_v1"] = {"v1_reference": ref, "v2": {"n_rows": r25["n_rows"], "crps_macro_game": r25["crps_macro_game"], "nll_macro_game": r25["nll_macro_game"], "mae_mean": r25["central"]["mae_mean"], "bias": r25["central"]["bias"], "p_ge_3_brier": r25["thresholds"]["P(SOG>=3)"]["brier"]},
                                 "differences": d, "criterion": "same n_rows and |dCRPS|,|dNLL|,|dMAE| <= 1e-4", "status": "B2_REPRODUCED" if ok else "B2_NOT_REPRODUCED"}
    (OUT / "phase1a_sog_burned_reproduction.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print("repro", res["reproduction_vs_v1"]["status"], d)

    # ---------------- comparators on identical rows (2025 fit2018_2024 model; 2024 fit2018_2023 model)
    ccache = Path("/tmp/nhl_v2_cf90.pkl")
    if ccache.exists():
        cf = pickle.load(open(ccache, "rb"))
    else:
        cf = C.comparator_features(games, rows, tab, 90)
        pickle.dump(cf, open(ccache, "wb"), protocol=4)
    const = C.fit_constants(cf, tab, rows)
    mu_h = C.human_mu(cf, const); simple = C.simple_means(cf, const)
    elig = C.eligible(cf)
    receipts = {}
    for l in gzip.decompress((OUT / "phase0_error_receipts.jsonl.gz").read_bytes()).decode().splitlines():
        r = json.loads(l)
        if r["market"] == "skater_sog_ge3":
            receipts[(r["game_id"], r["player_id"])] = r["p_served"]
    comp = {"artifact": "phase1a_sog_comparators", "evidence_grade": GRADE, "comparator_constants_fit_2018_2022": const, "poisson_mean_floor": 0.05, "seasons": {}}
    for label, score in (("fit2018_2024_score2025", 2025), ("fit2018_2023_score2024", 2024)):
        art = arts[label]; te_mask = tab["season"] == score
        te = sub(tab, te_mask); mu = M.predict_mu(art, te); alpha = art["nb2"]["alpha"]
        sc = {}
        for pname, pop in (("FULL_PREGAME_CANDIDATE_UNIVERSE", np.ones(len(mu), bool)), ("MEANINGFUL_EXPECTED_PARTICIPANT_UNIVERSE", mm[te_mask])):
            idr = pop & elig[te_mask]
            t = sub(te, idr)
            rates = fit_base_rates(tab, np.ones(len(tab["sog"]), bool) if pname.startswith("FULL") else mm)
            br = row_base_rates(t, rates)
            y = t["sog"]; g = t["game_id"]; pl = t["player_id"]
            block = {"n_rows": int(idr.sum()), "models": {}}
            cands = {"V2_B2": (mu[idr], "nb2", alpha), "human_frozen": (mu_h[te_mask][idr], "poisson", None)}
            for nm, arr in simple.items():
                cands["simple_" + nm] = (arr[te_mask][idr], "poisson", None)
            per_game = {}
            for nm, (m_, kind, al) in cands.items():
                mf = np.maximum(m_, 0.05) if kind == "poisson" else m_
                s, rws = M.summarize(kind, {"mu": mf, **({"alpha": al} if al else {})}, y, nm, g, pl, m_, br)
                block["models"][nm] = {"central": s["central"], "crps_macro_game": s["crps_macro_game"], "nll_macro_game": s["nll_macro_game"], "interval_coverage": s["interval_coverage"],
                                       "pit_ks": s["pit_ks"], "thresholds": {k: {kk: vv for kk, vv in v.items() if kk != "reliability"} for k, v in s["thresholds"].items()}}
                per_game[nm] = rws
            wk = D.week_index(t["start"][np.unique(g, return_index=True)[1]])
            uniq = np.unique(g)
            block["paired_crps_vs_V2_B2"] = {}
            for nm in cands:
                if nm == "V2_B2":
                    continue
                delta = per_game["V2_B2"]["crps_pg"] - per_game[nm]["crps_pg"]
                block["paired_crps_vs_V2_B2"][nm] = dict(M.blocked_bootstrap(delta, wk), note="negative mean = V2_B2 lower CRPS than the comparator", relative_crps_change=float(delta.mean() / block["models"][nm]["crps_macro_game"]))
            # production classifier on identical (played) rows
            prod = np.array([receipts.get((int(a), int(b)), np.nan) for a, b in zip(g, pl)])
            ok_ = ~np.isnan(prod) & (t["played"] == 1)
            if ok_.sum():
                e3 = (y[ok_] >= 3).astype(float)
                p_b2 = per_game["V2_B2"]["p_ge"][ok_, 2]
                p_h = per_game["human_frozen"]["p_ge"][ok_, 2]
                block["production_classifier_vs_V2_native_P(SOG>=3)"] = {"n_rows_identical_played_with_both": int(ok_.sum()), "brier_production_classifier": float(np.mean((prod[ok_] - e3) ** 2)), "brier_V2_B2_unconditional": float(np.mean((p_b2 - e3) ** 2)),
                                                                          "brier_human_poisson": float(np.mean((p_h - e3) ** 2)), "mean_p_production": float(prod[ok_].mean()), "mean_p_V2_B2": float(p_b2.mean()), "observed_rate": float(e3.mean()),
                                                                          "caveat": "production ledger exists only for played, production-database rows; B2 probability includes participation uncertainty so it is biased low on played rows; descriptive only"}
            sc[pname] = block
        comp["seasons"][str(score)] = sc
    (OUT / "phase1a_sog_comparators.json").write_text(json.dumps(comp, indent=1, sort_keys=True) + "\n")
    print("comparators done")


if __name__ == "__main__":
    main()
