"""
Assemble the Phase 1A hardening report (items C and D) from the raw study outputs.

  python nfl_phase1_hardening_report.py --raw DIR --out nfl_models/nfl_player_outcome_phase1b

DIR holds: decomp.pkl, cal_*.json (interval calibration study), policy_*.json (outside-bucket policy on the depth-chart universe),
policyORIG_*.json (same on the accepted universe), depthuni_report.json, phase1a_report.json.
Decision rule (fixed before reading results): a change is adopted only if (1) week-block bootstrap p < 0.10 for CRPS, (2) the CRPS
improvement is >= 0.5% of the incumbent CRPS (materiality), (3) for calibration changes the randomized-PIT 80% / 50% coverage is
within 0.03 of nominal. Everything is DEVELOPMENT data (2025 + 2026 wk1-3, burned), never a holdout.
"""
import argparse
import glob
import json
import pickle
import shutil
from pathlib import Path

MATERIAL = 0.005


def load_glob(raw, pat):
    out = {}
    for f in sorted(glob.glob(str(Path(raw) / pat))):
        out.update(json.load(open(f)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rawdir = out / "hardening_raw"
    rawdir.mkdir(exist_ok=True)
    for f in glob.glob(str(raw / "cal_*.json")) + glob.glob(str(raw / "policy*_*.json")):
        shutil.copy(f, rawdir / Path(f).name)
    decomp = {k: v["decomposition"] for k, v in pickle.load(open(raw / "decomp.pkl", "rb")).items()}
    cal = load_glob(raw, "cal_*.json")
    pol_depth = load_glob(raw, "policy_*.json")
    pol_orig = load_glob(raw, "policyORIG_*.json")
    rep = {"label": "DEVELOPMENT (2025 + 2026 wk1-3; burned, not holdout)", "decision_rule": __doc__.split("Decision rule")[1].split("Everything")[0].strip(),
           "A_changes": ["historical injury wording -> 'historical final-weekly-report proxy evaluated under the documented nflverse timing assumption' (data.py A2, availability docstring, Phase 1A README, protocol)",
                         "forward immutable snapshot system nfl_phase1_snapshots.py (download -> sha256 -> retrieval timestamp -> read-only file -> forecast only from snapshot -> append-only manifest; T-24h and T-90m independent downloads)",
                         "protocol availability model corrected from logistic to gradient boosting (amendment E) + consistency test; forward snapshot procedure (amendment F)",
                         "opportunity module: outside-bucket policies (fixed / proportional / rolling) and depth-chart candidate universe as OPTIONS; defaults reproduce accepted Phase 1A"],
           "C_named_player_mass_decomposition": decomp}
    # ---- item C decisions
    sel = {"interval_calibration": {}, "outside_bucket_policy": {}, "candidate_universe": {}}
    table = {}
    for t, r in cal.items():
        base = r["candidates"]["R0_as_is"]
        row = {"as_is": {k: base[k] for k in ("crps", "mae_median", "pit_cov80", "pit_cov50", "cov80_empirical_interval", "cov80_implied_by_forecast", "mean_width80")},
               "candidates": {}}
        best = None
        for n, m in r["candidates"].items():
            if n == "R0_as_is":
                continue
            vs = m["vs_R0"]
            rel = vs["crps_improvement"] / base["crps"]
            ok = vs["p_crps"] < 0.10 and rel >= MATERIAL and abs(m["pit_cov80"] - 0.8) <= 0.03 and abs(m["pit_cov50"] - 0.5) <= 0.03 and vs["mae_improvement"] >= -1e-9
            row["candidates"][n] = {"crps": m["crps"], "crps_improvement_vs_as_is": vs["crps_improvement"], "relative": round(rel, 4), "p": vs["p_crps"], "mae_improvement": vs["mae_improvement"],
                                    "pit_cov80": m["pit_cov80"], "pit_cov50": m["pit_cov50"], "adopt": bool(ok)}
            if ok and (best is None or vs["crps_improvement"] > best[1]):
                best = (n, vs["crps_improvement"])
        row["decision"] = best[0] if best else "R0_as_is (no change)"
        table[t] = row
        sel["interval_calibration"][t] = {"decision": row["decision"]}
        if best and best[0] == "R2_pit_recalibration":
            sel["interval_calibration"][t]["pit_map"] = r["pit_map"]
    sel["def_snap"] = sel["interval_calibration"].get("def_snap", {})
    if sel["def_snap"].get("pit_map"):
        sel["def_snap"]["downstream_check"] = ("DEVELOPMENT end-to-end (nfl_phase1b_evaluate): feeding the recalibrated def_snap samples into the defensive-event assembly made tackle CRPS WORSE "
                                               "(0.9737 -> 0.9767, p_not_better 0.99) and left sacks / interceptions unchanged, so the Phase 1B assembly keeps the raw Phase 1A def_snap samples. "
                                               "The recalibration is adopted for the def_snap count distribution itself only (CRPS 7.412 -> 7.240, coverage within band); the conflict is an unresolved limitation.")
    rep["D_interval_calibration"] = {
        "finding_carry": "The nominal-80% carry interval covers 91% empirically only because carries are discrete: the forecast's OWN implied 10-90 coverage is 90.5% (atoms at the quantiles). "
                         "The randomized PIT (the discreteness-correct calibration test) covers 80.6% of the 0.1-0.9 band and 50.4% of the 0.25-0.75 band: the carry distribution is calibrated as-is.",
        "table": table}
    # ---- outside bucket policy
    def policy_decisions(P):
        d = {}
        for t, r in P.items():
            base = r["V0_fixed_train_mean(Phase1A)"]
            best = None
            for n, m in r.items():
                vs = m.get("vs_fixed_train_mean")
                if not vs:
                    continue
                rel = vs["crps_improvement"] / base["crps"]
                if vs["p_crps"] < 0.10 and rel >= MATERIAL and (best is None or vs["crps_improvement"] > best[1]):
                    best = (n, vs["crps_improvement"], rel, vs["p_crps"])
            d[t] = {"decision": best[0] if best else "V0 fixed train mean (no change)", "detail": None if not best else {"crps_improvement": best[1], "relative": round(best[2], 4), "p": best[3]},
                    "variants": {n: {"crps": m["crps"], "named_share_error": m["named_share_error"], "vs_V0": m.get("vs_fixed_train_mean")} for n, m in r.items()}}
        return d
    pd_orig, pd_depth = policy_decisions(pol_orig), policy_decisions(pol_depth)
    rep["D_outside_bucket_policy"] = {"accepted_universe": pd_orig, "depth_chart_universe": pd_depth,
                                      "types_not_studied_on_accepted_universe": ["rz_carry", "rz_target"],
                                      "note": "the accepted-universe study covers carry, target, qb_att; rz_carry and rz_target were studied on the depth-chart universe only and no policy passed the rule there"}
    for t, x in pd_orig.items():
        sel["outside_bucket_policy"][t] = x["decision"]
    for t in ("rz_carry", "rz_target"):
        sel["outside_bucket_policy"][t] = "V0 fixed train mean (no change; no policy passed on the depth-chart universe)"
    # ---- universe
    A = json.load(open(raw / "phase1a_report.json"))["opportunity"]["types"]
    Dp = json.load(open(raw / "depthuni_report.json"))["opportunity"]["types"]
    uni = {}
    for t in ("carry", "target", "qb_att", "rz_carry", "rz_target"):
        ra, rd = A[t]["reconciliation"], Dp[t]["reconciliation"]
        uni[t] = {"rows_scored_accepted": A[t]["selected_metrics"]["n"], "rows_scored_depth": Dp[t]["selected_metrics"]["n"],
                  "actual_named_share_accepted": ra["actual_named_share_of_team_total"], "actual_named_share_depth": rd["actual_named_share_of_team_total"],
                  "forecast_named_share_accepted": ra["forecast_named_share_of_team_total"], "forecast_named_share_depth": rd["forecast_named_share_of_team_total"],
                  "crps_accepted": A[t]["selected_metrics"]["crps"], "crps_depth": Dp[t]["selected_metrics"]["crps"]}
    rep["D_candidate_universe"] = {"per_type": uni,
                                   "finding": "Adding depth-chart skill players raises the ACTUAL named-player carry share from 0.940 to 0.967 but the FORECAST share only from 0.909 to 0.913: the added players receive almost no propensity, so the deficit is not primarily a universe omission. "
                                              "CRPS on the extended universe is not comparable with the accepted universe (different scored rows: added candidates are mostly zeros)."}
    sel["candidate_universe"] = {"decision": "keep the accepted Phase 1A universe",
                                 "reason": "no common-player accuracy evidence that the extension helps; scored populations differ; the extension does not close the forecast-share gap. Reported as an unresolved limitation."}
    rep["D_decisions"] = sel
    (out / "phase1a_hardening_report.json").write_text(json.dumps(rep, indent=1, default=float))
    (out / "phase1a_hardening_selected.json").write_text(json.dumps(sel, indent=1, default=float))
    print(json.dumps(sel["interval_calibration"], default=float)[:600]); print(sel["outside_bucket_policy"])


if __name__ == "__main__":
    main()
