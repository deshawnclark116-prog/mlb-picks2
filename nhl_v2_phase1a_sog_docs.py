#!/usr/bin/env python3
"""Generates phase1a_sog_findings.md and phase1a_sog_snapshot.json deterministically from the committed Phase1A-SOG artifacts."""
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
OUT = REPO / "nhl_models" / "nhl_player_outcome_v2"


def J(n):
    return json.loads((OUT / n).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def f(x, nd=4):
    return "n/a" if x is None else ("%." + str(nd) + "f") % x


def main():
    mig, src, man, q = J("phase1a_v1_migration_audit.json"), J("phase1a_sog_source_audit.json"), J("phase1a_sog_data_manifest.json"), J("phase1a_sog_data_quality.json")
    rep, comp, lock = J("phase1a_sog_burned_reproduction.json"), J("phase1a_sog_comparators.json"), J("phase1a_sog_engine_lock_v1_1.json")
    L = []
    A = L.append
    A("# NHL V2 Phase1A-SOG - count-distribution certification and forward lock\n")
    A("Evidence grade of every historical number: **BURNED_REPRODUCTION_ONLY**. 2024 and 2025 are burned; V1 is a burned prior-research corpus. The first clean confirmation is forward 2026 data created after the engine lock. Nothing here validates or promotes anything.\n")
    A("## V1 migration\n")
    cls = {}
    for c in mig["components"]:
        cls.setdefault(c["classification"], []).append(c["component"][:90])
    for k in sorted(cls):
        A("- **%s**: %d component(s)" % (k, len(cls[k])))
    A("\nPhase1B shot-attempt extension: REJECTED_DO_NOT_REOPEN (mean relative CRPS gain %s vs 0.5%% materiality). B3 is not ported; participation/availability stays a separate uncertified layer.\n" % f(mig["burned_reference_numbers"]["phase1b_attempt"]["relative_crps_improvement_mean_of_folds"], 5))
    A("## Data source\n")
    ident = sum(s["rows_identical"] for s in src["samples"]); tot = sum(s["frozen_rows_same_games"] for s in src["samples"])
    A("Official NHL source (api-web schedule + stats-REST skater summary/time-on-ice), %d windows, retrieved 2026-10-02, vendored with sha256 equal to the V1 manifest (%s). Source re-check: %d / %d sampled rows byte-identical across 6 windows; both hosts reachable. No published terms found: research-only, low-rate use. Gates all pass: %s. The pinned V1 builder and the V2 builder produce a bit-identical table; V1's recorded table hash is not reproducible and is not used.\n" % (man["v1_acquisition"]["n_windows"], man["all_files_equal_v1_manifest"], ident, tot, q["all_pass"]))
    A("| season | candidate player coverage | SOG coverage | full-universe played fraction | meaningful rows | meaningful played fraction |\n|---|---|---|---|---|---|")
    for s in range(2018, 2026):
        c = q["coverage_by_season_T90"][str(s)]; p = q["candidate_population_audit_T90"][str(s)]
        A("| %d | %s | %s | %s | %d | %s |" % (s, f(c["candidate_player_coverage"], 4), f(c["sog_coverage"], 4), f(p["full_played_fraction"], 3), p["meaningful_rows"], f(p["meaningful_played_fraction"], 3)))
    A("\nThe full candidate universe is ~77% participants; ~23% of rows are non-participant zeros that make any unconditional score look easier. The MEANINGFUL expected-participant universe (prior information only) is ~93% participants and is the headline population.\n")
    r = rep["reproduction_vs_v1"]
    A("## Burned reproduction of the fixed B2 (fit 2018-2024, score 2025, T90)\n")
    A("Status: **%s** (n equal: %s; dCRPS %.2e, dNLL %.2e, dMAE %.2e). The V2 pipeline reproduces V1 to ~1e-7." % (r["status"], r["differences"]["n_rows_equal"], r["differences"]["dCRPS"], r["differences"]["dNLL"], r["differences"]["dMAE"]))
    A("\n| population | n | MAE(mean) | bias | median AE | MAE(median fcst) | CRPS | NLL | 50/80/90 coverage | P(3+) Brier | Brier skill | ECE | slope |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for lab in ("fit2018_2024_score2025", "fit2018_2023_score2024"):
        for pn, s in rep["runs"][lab]["populations"].items():
            c = s["central"]; t3 = s["thresholds"]["P(SOG>=3)"]
            A("| %s %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (lab[-4:], pn.split("_")[0], s["n_rows"], f(c["mae_mean"]), f(c["bias"]), f(c["median_ae"], 3), f(c["mae_median_forecast"]), f(s["crps_macro_game"], 5), f(s["nll_macro_game"], 5),
                                                                     "/".join(f(v, 3) for v in s["interval_coverage"].values()), f(t3["brier"], 5), f(t3["brier_skill"], 4), f(t3["ece_10_equal_count_bins"], 4), f(t3["calibration_slope"], 3)))
    A("\nCentral 50/80/90% intervals over-cover because the V1 interval definition on discrete counts is conservative; they are reported as defined, not tuned.\n")
    A("## Comparators on identical rows (burned; not validation)\n")
    A("| season | population | model | n | MAE | bias | CRPS | NLL | P(3+) Brier |\n|---|---|---|---|---|---|---|---|---|")
    for s, sc in comp["seasons"].items():
        for pn, b in sc.items():
            for m in ("V2_B2", "human_frozen", "simple_prior10_mean", "simple_season_rate_x_prior3_toi"):
                v = b["models"][m]
                A("| %s | %s | %s | %d | %s | %s | %s | %s | %s |" % (s, pn.split("_")[0], m, b["n_rows"], f(v["central"]["mae_mean"]), f(v["central"]["bias"]), f(v["crps_macro_game"], 5), f(v["nll_macro_game"], 5), f(v["thresholds"]["P(SOG>=3)"]["brier"], 5)))
    A("\nOn the MEANINGFUL expected-participant rows B2 improves game-macro CRPS over the Phase0 human baseline by about 2.3-2.6% and over prior-10 by about 4.7-4.9% (blocked-bootstrap intervals exclude zero). On the full universe the gap (~12%) is mostly participation modelling and should not be read as skill at shot volume.")
    pc = comp["seasons"]["2025"]["MEANINGFUL_EXPECTED_PARTICIPANT_UNIVERSE"]["production_classifier_vs_V2_native_P(SOG>=3)"]
    A("\nProduction SOG>=3 classifier (played rows only; biased toward the classifier because B2 mixes participation): Brier %s vs V2 B2 %s vs human %s on %d identical rows. The classifier is slightly better on this restricted task; B2 has no advantage there and none is claimed.\n" % (f(pc["brier_production_classifier"], 5), f(pc["brier_V2_B2_unconditional"], 5), f(pc["brier_human_poisson"], 5), pc["n_rows_identical_played_with_both"]))
    A("## Engine lock and forward\n")
    A("- Locked engine `%s` (fixed B2, per horizon T24H/T90/T30, targets 2018-2025; supersedes `%s`, which produced %d real forecasts - reason %s; model files byte-identical); lock eligible_from_cutoff_utc %s; availability layer recorded but NOT used (OBSERVED_NOT_CERTIFIED); no refit during the forward window." % (lock["engine_version"], lock["supersedes"], lock["v1_0_real_forecasts"], lock["reason"], lock["eligible_from_cutoff_utc"]))
    A("- Forward ledger (v1.1): decision key = game, horizon, scheduled start (revisions create new keys; MISSED_REVISED_CUTOFF); per-run content-addressed source manifests with retrieval provenance; hard source-completeness gate (SOURCE_INCOMPLETE / SOURCE_FETCH_FAILED are never turned into predictions); append-only hash chain; grader built before the first forecast (RAW and CLEAN ledgers, fixed-ratio large-miss forensics); MISSED_CUTOFF and INVALID_LATE are recorded, never backfilled.")
    A("- Scheduling limitation: scheduled workflows fire only from the default branch and this branch must not be merged, so forecasts exist only for windows in which a run actually happened (manual/dispatch or a session). Every other window is recorded MISSED_CUTOFF.")
    A("- No betting-market input anywhere; no simulation.\n")
    (OUT / "phase1a_sog_findings.md").write_text("\n".join(L) + "\n")
    names = sorted(p.name for p in OUT.glob("phase1a_*") if p.is_file() and p.name != "phase1a_sog_snapshot.json")
    data = sorted(p.name for p in (OUT / "phase1a_data").glob("*"))
    snap = {"artifact": "phase1a_sog_snapshot", "artifact_sha256": {n: sha(OUT / n) for n in names}, "vendored_data_sha256": {n: sha(OUT / "phase1a_data" / n) for n in data},
            "models_sha256": {p.name: sha(p) for p in sorted((OUT / "phase1a_sog_models").glob("*.json"))},
            "code_sha256": {p.name: sha(p) for p in sorted(REPO.glob("nhl_v2_phase1a_*.py"))}, "lock_sha256": sha(OUT / "phase1a_sog_engine_lock_v1_1.json")}
    (OUT / "phase1a_sog_snapshot.json").write_text(json.dumps(snap, indent=1, sort_keys=True) + "\n")
    print("docs ok")


if __name__ == "__main__":
    main()
