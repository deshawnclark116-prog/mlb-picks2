"""
NFL_PHASE1C_REPORT  (Phase 1C)

Generates README.md tables and freeze_candidate_validation.md from the Phase 1C result JSON files (no hand-typed numbers in the tables).
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
P = REPO / "nfl_models" / "nfl_player_outcome_phase1c"


def J(n):
    f = P / n
    return json.load(open(f)) if f.exists() else None


def fmt(x, d=4):
    return "n/a" if x is None else (f"{x:.{d}f}" if isinstance(x, (int, float)) else str(x))


def table_joint(r):
    L = ["| stat | n | joint T24 CRPS | joint T90 CRPS | last-8 empirical CRPS | joint gain vs last-8 (p not better) | joint T24 MAE(median) | bias(mean) | cov80 (interval) | cov80 (PIT) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, e in r["stats"].items():
        j = e["joint_T24"]["combined"]
        L.append(f"| {k} | {j['n']} | {fmt(j['crps'])} | {fmt(e['joint_T90']['combined']['crps'])} | {fmt(e['last8_empirical']['combined']['crps'])} | "
                 f"{e['joint_vs_last8']['crps_improvement']:+.4f} (p={e['joint_vs_last8']['p_not_better']}) | {fmt(j['mae_median'], 3)} | {fmt(j['bias_mean'], 3)} | {fmt(j['cov80'], 3)} | see calibration |")
    return "\n".join(L)


def table_state(r):
    L = ["| stat | S0 static CRPS | S1 dynamic CRPS | S1 - S0 gain (p not better) |", "|---|---|---|---|"]
    for k, e in r["stats"].items():
        L.append(f"| {k} | {fmt(e['S0']['combined']['crps'])} | {fmt(e['S1']['combined']['crps'])} | {e['S1_vs_S0']['crps_improvement']:+.4f} (p={e['S1_vs_S0']['p_not_better']}) |")
    return "\n".join(L)


def table_cal(r):
    L = ["| stat | uncal cov80(PIT) | cov50(PIT) | CRPS | best candidate | cov80 | cov50 | CRPS change % | passes |", "|---|---|---|---|---|---|---|---|---|"]
    for k, e in r["stats"].items():
        u = e["uncalibrated"]
        for cn, c in e["candidates"].items():
            L.append(f"| {k} | {fmt(u['cov80_pit'], 3)} | {fmt(u['cov50_pit'], 3)} | {fmt(u['crps'])} | {cn} | {fmt(c['cov80_pit'], 3)} | {fmt(c['cov50_pit'], 3)} | {c['crps_change_pct']:+.2f} | {c['passes_rule']} |")
    return "\n".join(L)


def table_universe(u):
    L = ["| stat | common rows | CRPS accepted-universe pipeline | CRPS depth-universe pipeline | gain (p not better) | MAE accepted | MAE depth | extra rows | extra-row CRPS model vs zero forecast |", "|---|---|---|---|---|---|---|---|---|"]
    for k, e in u["stats"].items():
        x = e.get("extra_rows") or {}
        L.append(f"| {k} | {e['n_common']} | {fmt(e['common_accepted']['crps'])} | {fmt(e['common_depth']['crps'])} | {e['depth_vs_accepted_common']['crps_improvement']:+.4f} (p={e['depth_vs_accepted_common']['p_not_better']}) | "
                 f"{fmt(e['common_accepted']['mae_median'], 3)} | {fmt(e['common_depth']['mae_median'], 3)} | {e['n_extra_players']} | {fmt(x.get('crps_model'))} vs {fmt(x.get('crps_zero_forecast'))} |")
    return "\n".join(L)


def table_conv(c):
    L = ["| N | " + " | ".join(c["stats"]) + " |", "|---|" + "---|" * len(c["stats"])]
    for N in ("200", "1000", "5000", "10000", "25000", "50000", "100000"):
        cells = []
        for s, e in c["stats"].items():
            b = e["by_n"][N]
            cells.append(f"CRPS {b['crps_rel_bias_vs_100k_pct']:+.2f}%, SE(mean) {b['se_mean']:.3g}{'' if b['all_met_amended'] else ' x'}")
        L.append(f"| {N} | " + " | ".join(cells) + " |")
    return "\n".join(L)


def table_curves(r, hz="T24", period="combined", stats=("rush_yds", "rec_yds", "pass_yds")):
    out = []
    for s in stats:
        cur = r["curves"][s][hz][period]
        tols = list(next(iter(cur.values()))["accuracy_curve_median"].keys())
        out += [f"**{s} ({hz}, {period})** - share of player-games with |median forecast - actual| <= tolerance", "", "| subset | n | MAE(med) | " + " | ".join(f"±{t}" for t in tols) + " |", "|---|---|---|" + "---|" * len(tols)]
        for sub, c in cur.items():
            out.append(f"| {sub} | {c['n']} | {c['mae_median']:.2f} | " + " | ".join(f"{c['accuracy_curve_median'][t]:.3f}" for t in tols) + " |")
        out.append("")
    return "\n".join(out)


def table_curves_counts(r, hz="T24", period="combined", stats=("rec", "tackles", "rush_td", "rec_td", "pass_td", "int", "sacks", "def_int")):
    out = []
    for s in stats:
        cur = r["curves"][s].get(hz, {}).get(period)
        if not cur:
            continue
        c = cur["all_eligible"]
        out.append(f"| {s} ({hz}) | {c['n']} | {c['mae_median']:.3f} | " + ", ".join(f"±{k}: {v:.3f}" for k, v in c["accuracy_curve_median"].items()) + " |")
    return "\n".join(["| outcome | n | MAE(median) | exact / within tolerance |", "|---|---|---|---|"] + out)


def table_unc(r):
    L = ["| outcome | Spearman(U, relative error) | decile monotonic violations | Spearman(pred sd, |error|) | violations |", "|---|---|---|---|---|"]
    for k, u in r["uncertainty"].items():
        L.append(f"| {k} | {u['spearman_U_vs_error']} | {u['monotonic_violations']} | {u['spread_skill']['spearman_sd_vs_abs_error']} | {u['spread_skill']['monotonic_violations']} |")
    return "\n".join(L)


def table_v2(v):
    L = ["| target | period | n (played, v2-eligible) | engine cond. median MAE | v2 refit MAE | blend MAE | production v2 MAE (2026 only) | engine vs v2 refit MAE (gain, p) | CRPS engine | CRPS v2 refit+2024 residuals | CRPS gain (p) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, per in v["stats"].items():
        for tag, c in per.items():
            prod = c.get("production_v2_artifact(2026 wk1-3 only)")
            L.append(f"| {name} | {tag} | {c['n']} | {c['engine_conditional_median']['mae']:.2f} | {c['v2_recipe_refit_through_2024']['mae']:.2f} | {c['historical_blend']['mae']:.2f} | "
                     f"{'n/a' if prod is None else format(prod['mae'], '.2f')} | {c['engine_vs_v2_refit_mae']['improvement']:+.3f} (p={c['engine_vs_v2_refit_mae']['p_not_better']}) | "
                     f"{c['crps_engine_conditional']:.3f} | {c['crps_v2_refit_plus_2024_residuals']:.3f} | {c['engine_vs_v2_refit_crps']['improvement']:+.3f} (p={c['engine_vs_v2_refit_crps']['p_not_better']}) |")
    return "\n".join(L)


def table_dep(d):
    L = ["| pair (across dev team-games) | actual | simulated |", "|---|---|---|"]
    for k, v in d["pairs"].items():
        L.append(f"| {k} | {v['actual']:+.3f} | {v['simulated']:+.3f} |")
    return "\n".join(L)


def table_t24t90(r):
    L = ["| outcome | T24 CRPS | T90 CRPS | T90 gain (p not better) |", "|---|---|---|---|"]
    for k, u in r["t24_vs_t90"].items():
        L.append(f"| {k} | {fmt(u['T24']['crps'])} | {fmt(u['T90']['crps'])} | {u['T90_vs_T24']['crps_improvement']:+.4f} (p={u['T90_vs_T24']['p_not_better']}) |")
    return "\n".join(L)


def sec_joint_compare(r):
    L = ["| stat | joint CRPS (N=1000) | independent assembly CRPS (same components, 200 draws) | joint gain (p not better) | joint bias | independent bias |", "|---|---|---|---|---|---|"]
    for k, e in r["stats"].items():
        j = e["joint_T24"]["combined"]; ind = e["independent_assembly"]
        L.append(f"| {k} | {fmt(j['crps'])} | {fmt(ind['scores_on_common_rows']['crps'])} | {ind['joint_vs_this']['crps_improvement']:+.4f} (p={ind['joint_vs_this']['p_not_better']}) | "
                 f"{fmt(j['bias_mean'], 3)} | {fmt(ind['scores_on_common_rows']['bias_mean'], 3)} |")
    return "\n".join(L)


def main():
    adj = (P / "adjudication_table.md").read_text()
    ja = J("joint_vs_independent.json"); jd = J("joint_vs_independent_depth.json"); st = J("state_static_vs_dynamic.json"); un = J("universe_comparison.json")
    cd = J("calibration_depth.json"); ca = J("calibration.json"); cv = J("simulation_convergence.json"); cu = J("accuracy_curves_uncertainty_depth.json") or J("accuracy_curves_uncertainty.json")
    dep = J("dependency_structure_depth.json"); v2 = J("v2_comparator_depth.json") or J("v2_comparator.json"); dry = J("dry_run/dry_run_report.json"); chaos = J("chaos_results.json")
    fc = json.load(open(REPO / "nfl_models" / "nfl_player_outcome_phase1_freeze_candidate.json")) if (REPO / "nfl_models" / "nfl_player_outcome_phase1_freeze_candidate.json").exists() else None
    L = ["# NFL player-outcome engine - Phase 1C (joint game simulator + freeze candidate)", "",
         "**SHADOW / DEVELOPMENT. Production untouched. `nfl_player_outcome_phase1_freeze.json` does NOT exist. Nothing here is a holdout (development = 2025 + 2026 wk1-3, burned) and no clean-forward data was used.**", "",
         "Generated by `nfl_phase1c_report.py` from the JSON files in this directory (tables are not hand-typed).", "",
         "## A/B. Temporal architecture adjudication (rule recorded before any fold result: `adjudication_rule.json`; amendment 1 = 2022 wk1 warm-up rows)", "", adj, "",
         "Rule: 7 chronological folds; hyper-parameters re-tuned inside each fold; a more complex candidate replaces a simpler one only if the mean fold improvement >= 0.05% of the simpler score, it wins >= 5/7 folds and its worst fold is no worse than -0.05%. "
         "Run 1 (2022 wk1 rows with all-zero features included) is kept in `adjudication_run1_partial.json` and does not decide anything.", ""]
    L += ["Near misses that the rule (correctly) rejected: rec_air B5 (6/7 folds, +0.00139 vs threshold 0.00162), rec_catch B2 (6/7, +0.00024 vs 0.00029), pass_completion B2 (6/7, +0.00029 vs 0.00031), "
          "def_sacks B1-B5 (mean gains above threshold but worst fold below -threshold). Adopted beyond B0: **rush B2** (7 folds: 6 wins, worst -0.00011) and **defender interceptions B5** (mean +0.00008 vs threshold 0.00007, 6/7 folds) - the latter is marginal.", ""]
    if un:
        L += ["## C. Candidate universe: accepted vs depth-chart-extended (common player-games only, same components, N=1000, seed 11)", "", table_universe(un), "",
              f"Decision block: {json.dumps(un['decision'])}. Team-level named-yardage MAE: " + "; ".join(f"{k}: {v['mae_accepted']:.2f} -> {v['mae_depth']:.2f}" for k, v in un["team_level"].items()) + ".", "",
              "Actual named share of the team total (accepted -> depth): " + "; ".join(f"{t}: {x['accepted']['actual_named_share']:.3f} -> {x['depth']['actual_named_share']:.3f} (extra players hold {x['extra_share_of_team_total']:.3f} of team volume)" for t, x in un["opportunity_mass"].items()) + ".", ""]
    L += ["## D. Joint-game architecture", "",
          "Every statistic is an aggregation of simulated play events (`nfl_phase1c_sim.py`); see the module docstring. Availability is sampled per draw (Phase 1A T24 or T90 probabilities); team rushes / dropbacks come from the Phase 1A negative binomials; "
          "carries and targets are allocated by the Phase 1A Dirichlet-multinomial with an explicit outside bucket; red-zone carries/targets are binomial thinnings of player opportunities under a shared team red-zone latent (goal line thins the red zone); "
          "dropbacks are split into sacks, scrambles and attempts; attempts are paired with target slots (or the no-target bucket); each attempt is interception / completion / incompletion; a completion draws air yards and YAC "
          "(receiver pmfs, yards = air + YAC) and the SAME event increments QB and receiver; touchdowns are attached to the simulated play (red zone: hazard, yards 1-20; outside: only completions/carries of >= 21 yards); "
          "defensive sacks and interceptions are allocated from the offensive events (half-sacks 0.5+0.5), tackle credits from tackle-ending plays x (1 + assists). Opportunity accounting per team-game (means): "
          + json.dumps({k: round(v, 2) for k, v in (jd or ja)["accounting_mean_per_team_game"].items()}) + ".", ""]
    if st:
        L += ["## E. Static (S0) vs state-aware (S1) game script", "", table_state(st), "", "S1 (8 blocks, margin-driven pass/rush transition learned on 2022-2024 plays, simulated scoring) is worse than S0 on every outcome, so **S0 is adopted**. "
              "Transition model: " + json.dumps(st["transition_model"]) + ".", ""]
    if dep:
        L += ["### Dependency structure (cross-game correlations, development team-games, depth universe)", "", table_dep(dep), ""]
    if ja:
        L += ["## L. Phase 1C joint vs Phase 1B-style independent assembly (accepted universe, identical component fits)", "", sec_joint_compare(ja), ""]
    if jd:
        L += ["### Phase 1C on the ADOPTED depth universe (rows differ from the accepted universe; not comparable across universes)", "", table_joint(jd), ""]
    if cd:
        L += ["## G. Final-distribution calibration (fit 2025 wk1-9, tested out-of-time on 2025 wk10-18 + 2026 wk1-3; depth universe)", "", cd["materiality_clarification"], "",
              "Adopted maps: " + ", ".join(f"{k} ({v['method']})" for k, v in cd["final_maps"].items()) + ". All other outcomes keep the uncalibrated simulation.", "", table_cal(cd), ""]
    if cv:
        L += ["## H. Simulation-size convergence (10 development games, 100k draws, disjoint blocks)", "", f"Chosen N = **{cv['chosen_n']}** under the amended criteria ({cv['amended_criteria']}). First-stated criteria: chosen {cv.get('chosen_n_first_stated_criteria')}.", "", table_conv(cv), ""]
    if cu:
        L += ["## I. Accuracy curves (depth universe, T24)", "", table_curves(cu, "T24"), "", table_curves_counts(cu, "T24"), "",
              "## J. T24 vs T90", "", table_t24t90(cu), "", "## K. Predictability score", "", table_unc(cu), ""]
    if v2:
        L += ["## M. Honest v2 / historical comparators (conditional on realized participation, v2-eligible rows)", "", table_v2(v2), ""]
    if dry:
        L += ["## N-P. Forecast logger, score logger, idempotency / restart (dry run, burned 2025 week 10 treated as future)", "", "```", json.dumps({k: v for k, v in dry.items() if k not in ("grading",)}, indent=1, default=float)[:6000], "```", "",
              "Score table (latest revision per forecast):", "", "```", json.dumps(dry["grading"]["table"], indent=1)[:4000], "```", ""]
    if chaos:
        L += ["## Q. Chaos tests", "", "| case | pass |", "|---|---|"] + [f"| {k} | {v.get('pass')} |" for k, v in chaos.items() if isinstance(v, dict)] + [""]
    L += ["## R. Files created / changed (Phase 1C)", "",
          "New: `nfl_phase1_forecast.py` (runner), `nfl_phase1_score.py` (score logger), `nfl_phase1_store.py` (append-only idempotent store), `nfl_phase1_board.py` (shadow board), `nfl_phase1c_sim.py` (joint simulator), `nfl_phase1c_fit.py`, "
          "`nfl_phase1c_constants.py`, `nfl_phase1c_inputs.py`, `nfl_phase1c_common.py`, `nfl_phase1c_adjudicate.py`, `nfl_phase1c_evaluate.py`, `nfl_phase1c_study.py`, `nfl_phase1c_script.py` (S1 candidate), `nfl_phase1c_metrics.py`, "
          "`nfl_phase1c_v2comp.py`, `nfl_phase1c_dryrun.py`, `nfl_phase1c_freeze_candidate.py`, `nfl_phase1c_report.py`, `tests/test_nfl_phase1c_invariants.py`; artifacts under `nfl_models/nfl_player_outcome_phase1c/`, "
          "`nfl_models/nfl_player_outcome_phase1_freeze_candidate.json`, `freeze_candidate_validation.md`. "
          "Changed: `nfl_phase1b_data.py` (extra forecast-only records, zone/completion tallies), `nfl_phase1_efficiency.py` / `nfl_phase1_event_models.py` / `nfl_phase1_defense_events.py` (active masks, per-snap rates for extras).", "",
          "## S. Commands (S = the session scratchpad; ADJ = the four adjudication result files)", "", "```",
          "python3 nfl_phase1c_constants.py /tmp/nflcsv nfl_models/nfl_player_outcome_phase1c/constants.json",
          "python3 nfl_phase1c_inputs.py /tmp/nflcsv $S/stage.pkl $S/p1a_inputs.pkl            # and: ... $S/stage_depth.pkl $S/p1a_inputs_depth.pkl depth",
          "python3 -c \"import nfl_phase1c_common as CC,pickle; CC.load_records('/tmp/nflcsv','$S/records_full.pkl',pickle.load(open('$S/p1a_samples.pkl','rb')))\"   # depth: records_depth.pkl from the depth pack",
          "python3 -u nfl_phase1c_adjudicate.py --records $S/records_full.pkl --out $S/adj <rush | rec_air rec_td rush_td pass_sack pass_int pass_td pass_air pass_completion | rec_catch | rec_yac pass_yac def_tackles def_sacks def_interceptions>",
          "python3 -u nfl_phase1c_study.py --scratch $S --adj $ADJ --n 1000 [--universe depth] <joint|state|calibrate|curves|dependency|v2|universe|convergence>",
          "python3 -u nfl_phase1c_dryrun.py --scratch $S --adj $ADJ --season 2025 --week 10 --n 25000 --universe depth --out nfl_models/nfl_player_outcome_phase1c/dry_run --work $S/dry/run",
          "python3 -u nfl_phase1c_dryrun.py --scratch $S --adj $ADJ --season 2025 --week 10 --universe depth --out nfl_models/nfl_player_outcome_phase1c --work $S/dry/chaos --chaos",
          "python3 nfl_phase1c_freeze_candidate.py --scratch $S --adj $ADJ ; python3 nfl_phase1c_report.py",
          "python3 tests/test_nfl_phase1c_invariants.py ; python3 tests/test_nfl_phase1b_invariants.py ; python3 tests/test_nfl_phase1_invariants.py --unit-only", "```", ""]
    (P / "README.md").write_text("\n".join(L) + "\n")


def validation_md():
    dry = J("dry_run/dry_run_report.json") or {}; chaos = J("chaos_results.json") or {}
    cv = J("simulation_convergence.json") or {}; un = J("universe_comparison.json") or {}
    ok = lambda b: "PASS" if b else "FAIL / not shown"
    L = ["# Freeze-candidate validation (Phase 1C) - what still prevents the final freeze", "",
         "**The final `nfl_player_outcome_phase1_freeze.json` has NOT been created and must not be created before this list is audited.**", "",
         "## Checks that pass (development evidence)", "",
         f"- Simulator invariants (`tests/test_nfl_phase1c_invariants.py`): QB yards/TDs/completions equal the receiver events, interception/sack/completion exclusivity, opportunity reconciliation, red-zone and goal-line nesting, inactive => zero, defensive sacks/INTs reconcile with offensive events, determinism, TD yardage bounded by field position, quantile monotonicity, store idempotency / conflict / partial-write.",
         f"- Idempotent rerun: {ok((dry.get('idempotent_rerun') or {}).get('store_unchanged') and (dry.get('idempotent_rerun') or {}).get('new_records_written') == 0)}; crash/restart converges to identical bytes: {ok((dry.get('crash_restart') or {}).get('identical_to_reference_run'))}; "
         f"reproduction from scratch identical: {ok((dry.get('reproduction') or {}).get('identical_bytes_to_reference_run'))}; forecasts unchanged by grading: {ok((dry.get('hash_verification') or {}).get('forecasts_unchanged_after_grading'))}; "
         f"forecasts unchanged when realized outcomes in the input pack are randomised: {ok((dry.get('outcome_perturbation') or {}).get('identical_bytes_to_reference_run'))}.",
         f"- Chaos suite: all pass = {chaos.get('all_pass')} ({sum(1 for v in chaos.values() if isinstance(v, dict))} cases).",
         f"- Simulation count fixed: N = {cv.get('chosen_n')} (amended criteria; see the amendment list below).",
         f"- Universe decision: adopt depth-chart-extended universe = {(un.get('decision') or {}).get('adopt_depth_universe')}.", "",
         "## Requirements still preventing final freeze", "",
         "1. **Live Phase 1A input regeneration is not implemented (`LiveLoader` raises).** The Phase 1A availability boosters, team-environment models, propensity models and role-state pipeline are not serialized, so a future week cannot be forecast from snapshots. The dry run serves a burned week from the Phase 1A stage outputs (models fit through 2024 only). BLOCKER.",
         "2. **Snapshot coverage is incomplete.** Only injuries, weekly rosters and depth charts are snapshotted; play-by-play, player stats, snap counts, games.csv (kickoff times / flex moves) and players.csv feed the as-of histories and schedule but are not snapshotted or hashed. BLOCKER for full provenance.",
         "3. **Final numeric gates for the Phase 1C outcome set are not pre-registered.** The protocol gates refer to the earlier Phase 1A/1B targets; a protocol amendment (G) with thresholds per outcome, per horizon and for calibration must be written and hashed before the clean-forward window opens. BLOCKER.",
         "4. **Single-writer assumption.** The store is atomic per batch but has no inter-process lock; concurrent forecast runners are unsupported until a lock is added.",
         "5. **Post-hoc rules to audit** (each is labelled where used): adjudication amendment 1 (2022 wk1 rows excluded after run 1); calibration materiality clarification (added after seeing the first-stated rule pass on 0.001 changes; adoption now needs both rules); convergence criteria amended after every tested N failed the row-level quantile criteria; "
         "the one-QB-always-plays rule was evaluated as an ablation and REJECTED on development evidence (pass-yards CRPS worse), so it is off.",
         "6. **Model limitations that remain**: QB availability probabilities are too low for some starters (named QB attempts are under-generated: passing yards bias about -3 per QB row at T24); receiving yards carry a +0.9 yards/row bias; the static script (S0) captures only about a third of the actual negative correlation between team carries and targets (S1 did not help); "
         "efficiency draws are independent across players and teams; goal-line touchdowns are not modelled separately from the red-zone stratum; the red-zone completion-rate adjustment and zone yardage ratios are league averages; tackle credits may repeat a defender on one play; kneels are excluded from tallies; official-stat edge cases are documented in `nfl_phase1c_sim.py`.",
         "7. **Comparators**: the engine ties an honestly refit v2 (which uses bookmaker spread/total inside the comparator only) within noise on v2-eligible played rows; the production v2 artifact is scorable only on 2026 wk1-3 (small n).",
         "8. **Dry run scope**: one burned week (2025 wk10) with two horizons; other weeks and 2026 were not run through the runner.",
         "9. **Calibration maps and bin values / zone ratios were fit on burned data**; only the clean-forward window can give out-of-sample evidence.", ""]
    (P / "freeze_candidate_validation.md").write_text("\n".join(L) + "\n")
    (REPO / "freeze_candidate_validation.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
    validation_md()
