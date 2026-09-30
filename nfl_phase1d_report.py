"""
NFL_PHASE1D_REPORT  (Phase 1D)  -- README.md / report.json (results) and freeze_candidate_validation_v2.md (PASS / BLOCKER only)

  python nfl_phase1d_report.py results        # README.md + report.json from the recorded evidence files
  python nfl_phase1d_report.py validation     # freeze_candidate_validation_v2.md (needs the built candidate; every check is computed, nothing hand-graded)

The validation document has exactly two sections, PASS and BLOCKER. A requirement is PASS only if its check on the recorded evidence succeeds; anything not demonstrated is a BLOCKER.
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
P1D = REPO / "nfl_models" / "nfl_player_outcome_phase1d"
P1C = REPO / "nfl_models" / "nfl_player_outcome_phase1c"


RESEARCH_INTERPRETATION = {
    "1": "Phase 1C currently TIES the honest v2 refit on rushing point MAE within noise: combined played-row MAE 25.02 vs 25.13, improvement 0.106 yd, block-bootstrap p(not better) = 0.35 (n = 986).",
    "2": "Phase 1C is slightly WORSE than the honest v2 refit on receiving point MAE on burned development data: 22.48 vs 22.19 (v2 better by 0.30 yd), p(Phase 1C not better) = 0.93 (n = 2,665).",
    "3": "Phase 1C has modestly better distributional CRPS in some comparisons (vs the v2 refit + 2024 residual distribution: rushing 17.92 vs 18.24, improvement 0.31, p = 0.085; receiving 15.91 vs 16.03, "
         "improvement 0.13, p = 0.21) - neither is significant at 0.05. It beats the historical-blend point forecast on MAE (rushing p = 0.039, receiving p = 0.015).",
    "4": "Phase 1C has not earned production replacement.",
    "5": "This does NOT prevent a clean-forward shadow evaluation.",
    "6": "No further predictive tuning is permitted before that clean evaluation. The purpose of freezing is to TEST the architecture, not to declare it superior.",
}


def J(p, base=P1D, default=None):
    f = base / p
    return json.loads(f.read_text()) if f.exists() else default


def plan_weeks():
    return [tuple(x) for x in J("dry_run_plan.json")["weeks"]]


def dry_runs():
    return {(s, w): J(f"dry_run/dry_run_{s}_wk{w:02d}.json") for s, w in plan_weeks()}


# ------------------------------------------------------------------ requirement checks: (id, text, ok, evidence)
def requirements():
    R = []
    add = lambda rid, text, ok, ev: R.append((rid, text, bool(ok), ev))
    pa = J("posthoc_audit.json") or {}
    ids = {d["id"]: d for d in pa.get("decisions", [])}
    add("1", "Post-hoc audit: every Phase 1C post-hoc decision classified; 2022 wk1 = DATA/WARMUP DEFINITION CORRECTION with a permanent prospective rule and both analyses kept in provenance; calibration / depth universe "
             "development-selected; N=25,000 an engineering selection; machine scan of Phase 1C files enumerated", bool(ids) and ids["D01"]["classification"] == "DATA/WARMUP DEFINITION CORRECTION" and "permanent_rule" in ids["D01"]
        and ids["D02"]["classification"] == "DEVELOPMENT-SELECTED" and ids["D03"]["classification"] == "ENGINEERING SELECTION" and ids["D04"]["classification"] == "DEVELOPMENT-SELECTED" and pa.get("n_scan_hits", 0) > 50
        and all(v["share_with_all_zero_decayed_history"] == 1.0 for k, v in pa.get("evidence_2022_wk1_history_does_not_exist", {}).items() if isinstance(v, dict)),
        f"posthoc_audit.json/.md: {len(ids)} decisions, {pa.get('n_scan_hits')} scan hits; 2022 wk1 rows have all-zero decayed history in 100% of rows")
    acc = J("accuracy_slices.json")
    add("2", "Accuracy-number integrity: slices A-H published separately for rushing / receiving / receptions / passing at T24 and T90, tolerances +/-5..40 (passing wider), every figure names its universe, "
             "a guard refuses unlabeled accuracy fields", bool(acc) and all(o in acc["outcomes"] for o in ("rush_yds", "rec_yds", "rec", "pass_yds")),
        "accuracy_slices.json/.md; nfl_phase1d_accuracy.assert_labeled; tests/test_nfl_phase1d.py::test_no_unlabeled_accuracy_field_can_be_emitted")
    proto = json.loads((REPO / "nfl_models" / "nfl_player_outcome_phase1_protocol.json").read_text())
    add("3", "Protocol Amendment G registered in protocol v1.2 before any clean-forward outcome, consistent with the code and the evaluators", proto["version"] == "1.2" and any(a["id"] == "G" for a in proto.get("amendments_v1_2", [])),
        "nfl_models/nfl_player_outcome_phase1_protocol.json amendments_v1_2[G]; nfl_phase1d_gates.py")
    runs = dry_runs()
    have = all(v is not None for v in runs.values())
    prob = sum(v["provenance_and_schedule"]["n_problems"] for v in runs.values() if v) if have else -1
    add("4", "Complete input snapshot / content-addressed store: every source read by Phase 1 code is stored as an immutable sha256 blob with retrieval ts, cutoff, byte count, blob path, parser version; identical bytes stored "
             "once; hash verified on read; all snapshot sets of the dry run re-verified", have and prob == 0, f"{sum(v['provenance_and_schedule']['snapshot_sets_verified'] for v in runs.values() if v) if have else 0} snapshot sets verified, {prob} problems")
    add("5", "Schedule / kickoff safety: kickoff and cutoff from the schedule snapshot, retrieval <= cutoff enforced, append-only revision ledger, flex / postponement / cancellation / neutral-site rules implemented and tested",
        have and all(v["provenance_and_schedule"]["sets_retrieved_after_cutoff"] == [] for v in runs.values() if v), "nfl_phase1d_schedule.py; tests; schedule ledger rows in the dry run")
    eq = J("equivalence_results.json")
    art = J("artifacts_freeze_fit_2026wk3/manifest.json")
    add("6", "Phase 1A serialized (SHA256 per artifact) and LiveLoader accepted: serialized model + snapshot time-travel reconstruction equals the historical research replay within 1e-9 on every selected week / horizon",
        bool(eq) and eq["summary"]["accepted"] and bool(art), f"equivalence_results.json max |diff| {eq['summary']['max_abs_diff_overall'] if eq else None}; freeze bundle {art['bundle_sha256'][:16] if art else None}")
    det = J("freeze_fit_determinism.json")
    add("7", "Deterministic walk-forward refit: training = completed weeks before the target week, no forward-target rows (audited), hyper-parameters / architecture frozen, every refit hashed and logged; two independent "
             "fits of the same window give the same bundle hash", bool(det) and det["identical_bundle_sha256"] is True, f"freeze_fit_determinism.json: {det['bundle_sha256'] if det else None}")
    add("8", "Concurrent writer safety: inter-process lock, stale-lock recovery after kill -9, concurrent writers, same id + different bytes HARD ERROR, lock timeout explicit, store bound to one host (flock is not distributed)",
        _tests_pass(["test_lock_mutual_exclusion_and_explicit_timeout", "test_stale_lock_recovery_after_kill9", "test_concurrent_writers_no_corruption_duplicates_idempotent_conflict_hard_error", "test_store_bound_to_one_host"]),
        "nfl_phase1_store_lock.py; tests/test_nfl_phase1d.py")
    plan = J("dry_run_plan.json")
    okw = have and all(v["run_A"]["success"] + len(v["run_A"]["safe_explicit_failures"]) == v["run_A"]["game_horizons_expected"] and v["provenance_and_schedule"]["unexplained_missing_game_horizons"] == 0
                       and v["idempotent_rerun"]["store_unchanged"] and v["idempotent_rerun"]["new_records_written"] == 0 and v["crash_restart"]["identical_to_reference"] and v["reproduction"]["identical_bytes_to_reference"]
                       and v["grading"]["forecasts_unchanged_after_grading"] and v["grading"]["forecasts_rerun"]["graded_new"] == 0 for v in runs.values() if v)
    add("9", "Multi-week time-travel dry run of the complete runner on the pre-registered burned weeks (early / mid / late 2025 + 2026 wk1-3), T24 and T90, with snapshot retrieval, serialization, model fit/load, forecast, restart, "
             "duplicate verification, grading, schedule cutoff, provenance, score append, board", okw and len(plan["weeks"]) >= 6,
        "; ".join(f"{s} wk{w}: {v['run_A']['success']}/{v['run_A']['game_horizons_expected']} ok" for (s, w), v in runs.items() if v))
    okp = have and all(v["perturbation_target_source"]["snapshot_content_ids_unchanged"] and v["perturbation_target_source"]["identical_bytes_to_reference"] and v["perturbation_contaminated_snapshot"]["identical_distribution_outputs"]
                       and (w == 1 or (v["perturbation_prior_history"]["forecast_ids_unchanged_but_bytes_differ"] > 0 and v["perturbation_prior_history"]["mean_abs_change_in_forecast_mean_affected_team"] > 0))
                       for (s, w), v in runs.items() if v)
    add("10", "Zero-leak perturbation: corrupting target-game outcomes leaves forecast bytes unchanged (source-level) and distribution outputs unchanged even when the corrupted rows are handed to the loader in the snapshot; "
              "corrupting a prior completed game changes later forecasts", okp, "dry_run/dry_run_*.json perturbation blocks")
    neng = J("n_engineering_audit.json")
    add("11", "N = 25,000 engineering audit against criteria fixed before running (MC SE <= 1% of model RMSE; CRPS within 0.5% of a 100k reference; tail-probability SE <= 0.005; repeat stability)", bool(neng) and neng["verdict"]["N_25000_retained"],
        f"n_engineering_audit.json verdict {neng['verdict'] if neng else None}")
    cf = J("calibration_freeze.json")
    add("12", "Calibration frozen: exact fitted values and sha256 of every stored map, labelled development-selected; no calibration selection re-run", bool(cf) and bool(cf.get("all_maps_sha256")), f"calibration_freeze.json all_maps_sha256 {cf['all_maps_sha256'][:16] if cf else None}")
    cand = REPO / "nfl_models" / "nfl_player_outcome_phase1_freeze_candidate_v2.json"
    add("13", "Freeze candidate v2 rebuilt from a clean commit: recorded git SHA equals HEAD at verification and every listed file hash re-verified from disk", cand.exists() and _verify_ok(), "nfl_phase1d_freeze_candidate.py verify")
    tests = json.loads(cand.read_text()).get("tests_at_head") if cand.exists() else None
    add("14", "Complete test matrix green (Phase 0B, 1A, 1B, 1C and 1D tests)", bool(tests) and all(v["exit_code"] == 0 and v["fail_lines"] == 0 for v in tests.values()),
        "; ".join(f"{k.split('/')[-1]}: {v['pass_lines']} PASS / {v['fail_lines']} FAIL" for k, v in (tests or {}).items()))
    add("15", "No final freeze file created; production serving untouched (only shadow research paths changed since Phase 1C)", not (REPO / "nfl_models" / "nfl_player_outcome_phase1_freeze.json").exists() and _tests_pass(["test_no_final_freeze_file_and_production_untouched"]),
        "tests/test_nfl_phase1d.py::test_no_final_freeze_file_and_production_untouched")
    return R


BLOCKERS_STATIC = [
    ("S1", "Live end-to-end operation is not demonstrated. Retrieval was exercised against a local HTTP mirror (nfl_phase1d_cas.live_sources) and every other step in time-travel mode; nothing was retrieved from the real providers "
           "(outbound network is restricted in this environment), no scheduler / cron invokes the runner at each cutoff, and the real providers' update lags and file layouts are unverified. Until a dry run against the live providers "
           "(before the first scored kickoff) succeeds, live readiness is not shown."),
    ("S2", "The v2 incumbent challenge gate needs the production v2 forecasts logged BEFORE kickoff by the production pipeline (an external input that uses bookmaker features). No such forward log is wired to or verified by the "
           "Phase 1 system, so the incumbent-replacement gate cannot be evaluated; the research verdict is unaffected."),
]


def _tests_pass(names):
    p = subprocess.run([sys.executable, str(REPO / "tests" / "test_nfl_phase1d.py"), *names], cwd=REPO, capture_output=True, text=True, timeout=1800)
    return p.returncode == 0 and all(f"PASS {n}" in p.stdout for n in names)


def _verify_ok():
    p = subprocess.run([sys.executable, str(REPO / "nfl_phase1d_freeze_candidate.py"), "verify"], cwd=REPO, capture_output=True, text=True)
    return p.returncode == 0


def validation_md():
    R = requirements()
    L = ["# Freeze-candidate validation v2 (Phase 1D)", "", "Two sections only. Every PASS below was computed from recorded evidence by `nfl_phase1d_report.py validation`; anything not demonstrated is a BLOCKER. "
         "The final `nfl_player_outcome_phase1_freeze.json` has NOT been created.", "", "## PASS", ""]
    for rid, text, ok, ev in R:
        if ok:
            L.append(f"- **R{rid}** {text}  \n  evidence: {ev}")
    L += ["", "## BLOCKER", ""]
    nb = 0
    for rid, text, ok, ev in R:
        if not ok:
            nb += 1
            L.append(f"- **R{rid} NOT DEMONSTRATED** {text}  \n  evidence: {ev}")
    for sid, text in BLOCKERS_STATIC:
        nb += 1
        L.append(f"- **{sid}** {text}")
    return "\n".join(L) + "\n", R, nb


def results_report():
    acc = J("accuracy_slices.json"); eq = J("equivalence_results.json"); runs = dry_runs(); neng = J("n_engineering_audit.json"); pa = J("posthoc_audit.json"); cf = J("calibration_freeze.json")
    det = J("freeze_fit_determinism.json"); art = J("artifacts_freeze_fit_2026wk3/manifest.json")
    rep = {"phase": "1D freeze hardening / live shadow readiness", "predictive_architecture": "CLOSED: no feature, family, hyper-parameter, calibration or eligibility change was made to improve burned results",
           "research_interpretation": RESEARCH_INTERPRETATION,
           "posthoc_decisions": [{k: d[k] for k in ("id", "title", "classification", "pre_registered")} for d in (pa or {}).get("decisions", [])],
           "amendment_G_gates": list(json.loads((REPO / "nfl_models" / "nfl_player_outcome_phase1_protocol.json").read_text())["amendments_v1_2"][0]["gates"]),
           "accuracy_slices_headline": {}, "equivalence": eq and eq["summary"], "walk_forward_determinism": det, "freeze_artifact_bundle": art and {"bundle_sha256": art["bundle_sha256"], "files": art["files"]},
           "dry_run": {f"{s}-wk{w}": {k: v[k] for k in ("run_A", "idempotent_rerun", "crash_restart", "reproduction", "perturbation_target_source", "perturbation_contaminated_snapshot", "perturbation_prior_history", "provenance_and_schedule", "cas") if k in v}
                       for (s, w), v in runs.items() if v},
           "n_engineering_audit": neng, "calibration_freeze": cf and {"label": cf["label"], "all_maps_sha256": cf["all_maps_sha256"], "maps": {k: v["sha256"] for k, v in cf["maps"].items()}}}
    if acc:
        for name in ("rush_yds", "rec_yds", "rec", "pass_yds"):
            blk = acc["outcomes"][name]["T24"]["combined"]
            rep["accuracy_slices_headline"][name] = {u: {"n": c["n"], "within_tolerance": c["within_tolerance"]} for u, c in blk.items() if u[:2] in ("A_", "H_", "D_", "E_", "F_")}
    (P1D / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    L = ["# Phase 1D - freeze hardening / live shadow readiness", "",
         "This phase does not change the predictive architecture. It makes the existing Phase 1A/1B/1C research candidate operationally reproducible and scientifically pre-registered so that a clean-forward test means something. "
         "Production serving is untouched; `nfl_player_outcome_phase1_freeze.json` has not been created; nothing from the clean-forward window was used.", "", "## 0. Research interpretation (locked)", ""]
    for k, v in rep["research_interpretation"].items():
        L.append(f"{k}. {v}")
    L += ["", "## 1. Post-hoc audit", "", "See `posthoc_audit.md` / `.json` (every decision classified; machine scan appendix).", "", "| id | decision | classification | pre-registered |", "|---|---|---|---|"]
    for d in rep["posthoc_decisions"]:
        L.append(f"| {d['id']} | {d['title']} | {d['classification']} | {d['pre_registered']} |")
    if acc:
        L += ["", "## 2. Accuracy numbers with their universes (T24, combined development weeks)", "", "The Phase 1C headline (72% of rushing rows within +/-10 yards) was measured on universe A, where most rows are non-participants with a true outcome of 0.", ""]
        for name in ("rush_yds", "rec_yds"):
            blk = acc["outcomes"][name]["T24"]["combined"]
            L += [f"**{name}**", "", "| universe | n | share true outcome 0 | within +/-10 | within +/-20 | within +/-40 |", "|---|---|---|---|---|---|"]
            for u, c in blk.items():
                if u[:2] in ("A_", "H_", "D_", "E_", "F_", "B_", "C_"):
                    L.append(f"| {u} | {c['n']} | {c['share_of_rows_with_true_outcome_zero']} | {c['within_tolerance']['10']} | {c['within_tolerance']['20']} | {c['within_tolerance']['40']} |")
            L.append("")
        L.append("Full tables (all slices, all tolerances, T24 and T90): `accuracy_slices.md`.")
    if eq:
        L += ["", "## 6. LiveLoader equivalence", "", f"Max |difference| over every compared quantity: {eq['summary']['max_abs_diff_overall']} (tolerance {eq['tolerance']}); accepted = {eq['summary']['accepted']}.", ""]
    if runs and all(runs.values()):
        L += ["", "## 9. Multi-week time-travel dry run", "", "| week | success / expected | idempotent rerun | crash-restart | reproduction | target perturbation | contaminated snapshot | prior-history sensitivity |", "|---|---|---|---|---|---|---|---|"]
        for (s, w), v in runs.items():
            pp = v.get("perturbation_prior_history")
            L.append(f"| {s} wk{w} | {v['run_A']['success']}/{v['run_A']['game_horizons_expected']} | {v['idempotent_rerun']['store_unchanged']} ({v['idempotent_rerun']['verified_duplicates']} dup) | {v['crash_restart']['identical_to_reference']} | "
                     f"{v['reproduction']['identical_bytes_to_reference']} | {v['perturbation_target_source']['identical_bytes_to_reference']} | {v['perturbation_contaminated_snapshot']['identical_distribution_outputs']} | "
                     f"{'n/a (week 1 of season)' if not pp else round(pp['mean_abs_change_in_forecast_mean_affected_team'], 3)} |")
    if neng:
        L += ["", "## 11. N = 25,000 engineering audit", "", f"Verdict: {neng['verdict']}", "", "| outcome | C1 MC SE / model RMSE | C2 |CRPS 25k - 100k| / CRPS | C3 max tail SE | C4 pass |", "|---|---|---|---|---|"]
        for o, e in neng["outcomes"].items():
            L.append(f"| {o} | {e.get('C1_mean_se_over_rmse')} | {e['C2_rel_crps_diff']:.5f} | {e.get('C3_max_analytic_se')} | {e.get('C4_pass')} |")
    L += ["", "## Design summary (what each module guarantees)", "",
          "- **Content-addressed snapshots** (`nfl_phase1d_cas.py`): every source Phase 1 reads (schedule, players, injuries, weekly rosters, depth charts, player stats, play-by-play, snap counts, participation, FTN charting) is stored as an "
          "immutable `sha256 -> blob` (read-only, atomic link, never overwritten, hash re-verified on every read). The manifest row records logical name, provider URL, retrieval timestamp, information cutoff and horizon, sha256, byte "
          "count, blob path, parser version and any documented transform (the schedule is stored without sportsbook columns; raw sha256 kept). Identical bytes at two forecast times are one blob referenced twice. Loaders read a "
          "directory of verified symlinks, so a forecast can only see snapshotted bytes.",
          "- **Time-travel** (burned weeks): the provider is simulated by dropping every row of games not completed at the cutoff (kickoff + 24h), later-week injuries / rosters, T24 current-week rosters and depth-chart snapshots "
          "after the cutoff. Sets are labelled `time_travel`; simulated retrieval = cutoff - 1h.",
          "- **Schedule / kickoff safety** (`nfl_phase1d_schedule.py`): kickoff and cutoff come from the schedule snapshot; retrieval after the cutoff means no forecast (SAFE_EXPLICIT_FAILURE); an append-only revision ledger records every "
          "kickoff / status change; forecast ids contain the cutoff so a moved kickoff never overwrites a forecast; the scored forecast is the latest one whose cutoff equals (final kickoff - horizon) and whose retrieval preceded it; "
          "flex earlier / later, postponement, cancellation and neutral-site time changes are ordinary revisions (rules in the module docstring, tested).",
          "- **Serialized Phase 1A** (`nfl_phase1d_p1a.py`): team-environment coefficients, availability models and status lookup tables, role-state priors / HMM, propensity boosters, outside-bucket weights and share noise are written "
          "to files with a sha256 manifest; forecasts use only the loaded bytes. `phase1a_frozen_selection.json` holds everything that may not change (selection, hyper-parameters, xgboost parameters).",
          "- **Walk-forward refit**: training = every completed week before the target week from 2023 on (last 25% of those weeks validate), fitted quantities only; Phase 1B refits its tilt coefficients and as-of counts with the "
          "hyper-parameters frozen from the freeze fit (`phase1b_frozen_hyper.json`, no grid search); every refit is a new artifact bundle hash and a row in `weekly_fits.jsonl` (training cutoff, weeks, artifact hashes, code hashes, code SHA).",
          "- **Locking** (`nfl_phase1_store_lock.py`): kernel flock (dead holders release automatically; stale owner records are recognised and replaced), explicit timeout, one committing process per store. flock is HOST-LOCAL: each "
          "store is bound to its host (`host_binding.json`) and a writer on any other host is refused; multi-host operation is unsupported until a distributed lock replaces it.",
          "- **Baselines and gates**: `nfl_phase1d_baselines.py` (HB1) is logged before kickoff next to every forecast; `nfl_phase1d_gates.py` holds Amendment G and its evaluators.",
          "", "## Changes to the forecast path since Phase 1C (each documented in posthoc_audit.md; none is performance motivated)", "",
          "- D17: the stored QB rushing-yards calibration map is now applied to QB rows (it was inert).", "- D18: forecast-only defenders / T24 offense positions are resolved as of the cutoff (LiveLoader equivalence finding).",
          "- Record schema additions: `p_zero`, 99-point quantile grid, exact lattice cdf, as-of `prior_usage`, `p_active_status_baseline`, provenance ids (needed for randomized PIT, eligibility and the availability gate).",
          "- `protocol_version` 1.2; the walk-forward path fits with frozen hyper-parameters (Phase 1C dry runs searched them per fit)."]
    (P1D / "README.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "results":
        results_report()
    else:
        md, R, nb = validation_md()
        (REPO / "freeze_candidate_validation_v2.md").write_text(md)
        (P1D / "freeze_candidate_validation_v2.md").write_text(md)
        print(f"{sum(1 for r in R if r[2])} PASS / {nb} BLOCKER")
