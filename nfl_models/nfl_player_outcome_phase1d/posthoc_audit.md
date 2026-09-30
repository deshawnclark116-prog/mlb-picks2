# Phase 1C post-hoc decisions - audit (Phase 1D)

Every decision below was made, or its rule written, after burned-development results were visible unless it says otherwise. Nothing here was pre-registered unless stated. Classes: **DATA/WARMUP DEFINITION CORRECTION** = the change only fixes what counts as a valid observation (history literally does not exist); it is defined prospectively and permanently below; **DEVELOPMENT-SELECTED** = chosen after looking at burned-development results; frozen; only the clean-forward window can judge it; **ENGINEERING SELECTION** = a resource / numerical-accuracy choice that is not a model-performance choice; **CORRECTNESS FIX** = the earlier procedure violated data-time or a definition; the fix is not performance motivated; **REPORTING DEFINITION** = changes how a diagnostic is summarised, not any forecast; **NOT ADOPTED** = evaluated as an ablation and rejected; nothing changed; **PROCESS** = no modelling consequence.

## D01 - 2022 week-1 exclusion (adjudication amendment_1)

- classification: **DATA/WARMUP DEFINITION CORRECTION**
- what: Run 1 of the rolling-origin adjudication included 2022 week-1 rows; run 2 excludes them from every fit, validation and test for every component. Both analyses are kept.
- trigger: run 1 showed every defender feature level worse than B0 in 7/7 folds by >= 0.04 nats (an artefact signature); the rows' as-of features are identically zero
- pre-registered: False
- recorded before rerun: True
- consequence: run 1 (partial, interrupted) selected def_interceptions B0; run 2 selects B5 (16 tilt features, L2 1000); every other component listed in run 1 kept its selection. The change therefore did alter one selection, the defensive-interception tilt, whose effect on scored outcomes is negligible (see adjudication_results.json).
- permanent rule: WARM-UP RULE (prospective, permanent): an observation is eligible for fitting or scoring only if at least one completed regular-season game of the data window precedes it. Operationally: 2022 week 1 is the only warm-up week (nothing precedes it in the data window). Every later week, including week 1 of every later season, has history and is eligible. In the forward window every game has history, so the rule excludes nothing; it is implemented once in nfl_phase1c_adjudicate.mask_warmup and used by the live path (nfl_phase1d_live.prepare).
- not pretended: the amendment was NOT pre-registered; it was recorded after run 1 and before run 2
- provenance: nfl_models/nfl_player_outcome_phase1c/adjudication_run1_partial.json (original), nfl_models/nfl_player_outcome_phase1c/adjudication_results.json (amended), nfl_models/nfl_player_outcome_phase1c/adjudication_rule.json#amendment_1

## D02 - Calibration materiality clarification

- classification: **DEVELOPMENT-SELECTED**
- what: After seeing that the first-stated calibration adoption rule was satisfied by 0.001 coverage changes, adoption additionally required |cov80-0.8| to improve by >= 0.005 and |cov50-0.5| not to worsen by more than 0.005.
- trigger: burned-development calibration outcomes
- pre-registered: False
- consequence: the accepted-universe calibration.json now shows two outcomes back at 'none (uncalibrated retained)'; the depth-universe maps used by the forecast path are listed in calibration_depth.json
- phase1d action: NO further calibration selection. The current maps stay exactly as stored (hashes in calibration_freeze.json). They may be removed only for a code/logic bug, never for another outcome comparison.
- provenance: nfl_models/nfl_player_outcome_phase1c/calibration.json, nfl_models/nfl_player_outcome_phase1c/calibration_depth.json, nfl_phase1c_study.py (step_calibrate)

## D03 - Simulation-count convergence criteria amendment

- classification: **ENGINEERING SELECTION**
- what: The first-stated row-level quantile criteria failed for every tested N (heavy-tailed yardage tails converge slowly), so amended aggregate criteria were used to pick N = 25,000.
- trigger: no N passed the first-stated criteria
- pre-registered: False
- phase1d action: N = 25,000 is treated as an engineering selection. A NEW criterion is fixed BEFORE running (n_engineering_criteria.json, Monte Carlo error relative to football-model error, predetermined disjoint burned weeks) and verified in n_engineering_audit.json. N is never re-chosen from football scores.
- provenance: nfl_models/nfl_player_outcome_phase1c/simulation_convergence.json, nfl_models/nfl_player_outcome_phase1d/n_engineering_criteria.json, nfl_models/nfl_player_outcome_phase1d/n_engineering_audit.json

## D04 - Depth-chart-extended candidate universe

- classification: **DEVELOPMENT-SELECTED**
- what: The candidate universe was extended with timestamped depth-chart skill players (>= 3 prior game rows) after the common-player comparison favoured it (rush CRPS 7.8958 vs 7.9290, p=0.01; pass 30.29 vs 30.55, p=0.027).
- trigger: burned-development comparison
- pre-registered: False
- phase1d action: FROZEN. Reversible only for a correctness / data-time violation (e.g. a depth snapshot with dt after the cutoff), never for accuracy.
- provenance: nfl_models/nfl_player_outcome_phase1c/universe_comparison.json

## D05 - Predictability score redefinition

- classification: **DEVELOPMENT-SELECTED**
- what: The first score (weighted components) was anti-informative on development data (Spearman -0.53 against relative error) and was replaced by the engine's own expected relative error mapped through x/(1+x).
- trigger: burned-development metric
- pre-registered: False
- phase1d action: frozen; the forward gate only requires Spearman > 0 (Amendment G), it does not tune the score.
- provenance: nfl_phase1c_metrics.py (predictability)

## D06 - High-confidence subset definition

- classification: **REPORTING DEFINITION**
- what: The first high-confidence subset was degenerate; it was redefined as P(active) >= 0.9 and U in the lowest 30% of likely players.
- trigger: degenerate subset in the first curves run
- pre-registered: False
- phase1d action: reporting only; Phase 1D replaces headline accuracy with labelled universes (accuracy_slices)
- provenance: nfl_phase1c_study.py (step_curves)

## D07 - One-listed-QB-always-plays rule

- classification: **NOT ADOPTED**
- what: Evaluated as an ablation after seeing a named-QB attempts shortfall; it hurt passing-yards CRPS by 0.34 and was rejected (force_qb = False).
- trigger: burned-development ablation
- pre-registered: False
- phase1d action: stays off
- provenance: nfl_models/nfl_player_outcome_phase1c/joint_vs_independent.json (ablation_force_one_listed_qb)

## D08 - Proportional rescale of QB dropback / attempt allocation (prop bucket)

- classification: **DEVELOPMENT-SELECTED**
- what: The dropback allocation divides propensities by (1 - sbar - z_db) and reduces the outside bucket, added after observing under-generated named-QB attempts.
- trigger: development bias diagnosis
- pre-registered: False
- phase1d action: frozen simulator design
- provenance: nfl_phase1c_sim.py

## D09 - Fitted within-bin means instead of bin midpoints

- classification: **DEVELOPMENT-SELECTED**
- what: Midpoints of coarse bins inflated tails (e.g. air 50-99 mean 53 vs midpoint 74.5); league-fitted within-bin means replaced them.
- trigger: development yard bias
- pre-registered: False
- phase1d action: frozen constants (constants.json hash in the candidate)
- provenance: nfl_models/nfl_player_outcome_phase1c/constants.json, nfl_phase1c_constants.py

## D10 - Red-zone yardage and touchdown mechanics

- classification: **DEVELOPMENT-SELECTED**
- what: The red-zone cap pile-up and long-TD resample were replaced by zone likelihood ratios (SIR) and touchdown-at-target hazard rules; unconditioned air-yard candidates were made conditional on completion.
- trigger: development yard / TD bias diagnoses
- pre-registered: False
- phase1d action: frozen simulator design
- provenance: nfl_phase1c_sim.py, nfl_phase1c_constants.py

## D11 - Static (S0) versus state-aware (S1) game script

- classification: **DEVELOPMENT-SELECTED**
- what: S1 was worse than S0 on every scored outcome (rush CRPS 8.024 vs 7.916; pass 32.17 vs 30.45) and was not adopted.
- trigger: burned-development comparison
- pre-registered: False
- phase1d action: S0 frozen
- provenance: nfl_models/nfl_player_outcome_phase1c/state_static_vs_dynamic.json

## D12 - Phase 1A architecture selection (team environment, availability, role families, opportunity components)

- classification: **DEVELOPMENT-SELECTED**
- what: Selected on burned development weeks with week-block bootstrap p < 0.10 (Phase 1A); extracted into phase1a_frozen_selection.json.
- trigger: burned-development comparisons
- pre-registered: True
- phase1d action: frozen; walk-forward refits change only fitted coefficients
- provenance: nfl_models/nfl_player_outcome_phase1d/phase1a_frozen_selection.json

## D13 - Production-v2 artifact scored only on 2026 wk1-3

- classification: **CORRECTNESS FIX**
- what: The production v2 artifact (trained through 2025) had been scored on 2025 and combined rows; restricted to 2026 wk1-3, the only weeks future to its training data.
- trigger: data-time review
- pre-registered: False
- phase1d action: kept
- provenance: nfl_phase1c_v2comp.py, nfl_models/nfl_player_outcome_phase1c/v2_comparator_depth.json

## D14 - Structural constants fitted on all burned data

- classification: **DEVELOPMENT-SELECTED**
- what: bin values, zone likelihood ratios, half-sack share, tackle credit mean, red-zone gamma shape: all fit on burned data including the development weeks.
- trigger: design
- pre-registered: False
- phase1d action: frozen (hash recorded); never refit in the forward window
- provenance: nfl_models/nfl_player_outcome_phase1c/constants.json

## D15 - Report / harness fixes (table KeyError, absent-player chaos assertion, restarted dry run)

- classification: **PROCESS**
- what: Engineering fixes to reporting and test assertions; one dry run was restarted after code changes so the model version matched final code.
- trigger: harness defects
- pre-registered: False
- phase1d action: none
- provenance: nfl_phase1c_report.py, nfl_phase1c_dryrun.py

## D16 - Phase 1D choices (recorded here so they are not hidden)

- classification: **PROCESS**
- what: (a) hyper-parameters of Phase 1A/1B are frozen at the freeze-fit values and walk-forward refits only refit coefficients; (b) Amendment G thresholds were written after the burned-data results were known but before any clean-forward outcome, copying protocol v1.1 numbers (calibration bands, ECE 0.03, p < 0.05) or the Phase 1D specification (2% MAE tolerance, >= 50 events); (c) accuracy-slice tier thresholds and eligibility slices are reporting definitions; (d) the dry-run weeks and the N-audit set were fixed in files committed before their runs.
- trigger: Phase 1D specification
- pre-registered: False
- phase1d action: documented; no threshold derived from any later forward result
- provenance: nfl_models/nfl_player_outcome_phase1d/dry_run_plan.json, nfl_models/nfl_player_outcome_phase1d/n_engineering_criteria.json

## Evidence for D01: the required history literally does not exist

nfl_phase1_data.SEASONS = [2022 .. 2026]; the first season present in every source is 2022 (no 2021 files are read or exist in the store).

| component | 2022 wk1 rows | share with all-zero decayed player/position/league history | share with all-zero context families | 2022 wk2 rows | wk2 share all-zero history |
|---|---|---|---|---|---|
| rush | 127 | 1.000 | 0.063 | 136 | 0.0 |
| rec | 266 | 1.000 | 0.056 | 256 | 0.0 |
| pass | 36 | 1.000 | 0.056 | 37 | 0.0 |

## Appendix: machine scan of Phase 1C code / reports for post-hoc, posthoc, amend, clarification, after seeing, revised rule, changed criterion, redefined, replaced before, first-stated, was replaced, superseded, not pre-registered, rejected, ablation

191 hits. Files with a hit that no decision above cites: ['freeze_candidate_validation.md', 'nfl_models/nfl_player_outcome_phase1c/README.md', 'nfl_models/nfl_player_outcome_phase1c/freeze_candidate_validation.md', 'nfl_models/nfl_player_outcome_phase1c/joint_vs_independent_depth.json', 'nfl_phase1c_adjudicate.py', 'nfl_phase1c_freeze_candidate.py'].

### nfl_phase1c_adjudicate.py

- line 34 [amend]: """Amendment 1: 2022 week 1 has no as-of history (features identically zero) -> excluded from every fit and test."""
- line 242 [amend]: Path(out_json).write_text(json.dumps({"rule": "adjudication_rule.json (with amendment_1)", "final": final}, indent=1, default=float))
- line 248 [amend]: {"note": "run 1 INCLUDED 2022 week-1 rows with all-zero as-of features; partial (interrupted); superseded by amendment 1; kept for transparency",
### nfl_phase1c_freeze_candidate.py

- line 66 [amend]: "protocol_version": FC.PROTOCOL_VERSION, "protocol_file": "nfl_models/nfl_player_outcome_phase1_protocol.json", "protocol_amendments": [a["id"] for a in proto_j.get("amendments_v1_1", [])],
### nfl_phase1c_metrics.py

- line 137 [replaced before]: ANTI-informative on development data (Spearman -0.53 vs relative error) and was replaced before any adoption; see README."""
### nfl_phase1c_report.py

- line 62 [amend]: cells.append(f"CRPS {b['crps_rel_bias_vs_100k_pct']:+.2f}%, SE(mean) {b['se_mean']:.3g}{'' if b['all_met_amended'] else ' x'}")
- line 140 [amend]: "## A/B. Temporal architecture adjudication (rule recorded before any fold result: `adjudication_rule.json`; amendment 1 = 2022 wk1 warm-up rows)", "", adj, "",
- line 143 [rejected]: L += ["Near misses that the rule (correctly) rejected: rec_air B5 (6/7 folds, +0.00139 vs threshold 0.00162), rec_catch B2 (6/7, +0.00024 vs 0.00029), pass_completion B2 (6/7, +0.00029 vs 0.00031), "
- line 166 [clarification]: L += ["## G. Final-distribution calibration (fit 2025 wk1-9, tested out-of-time on 2025 wk10-18 + 2026 wk1-3; depth universe)", "", cd["materiality_clarification"], "",
- line 169 [amend]: L += ["## H. Simulation-size convergence (10 development games, 100k draws, disjoint blocks)", "", f"Chosen N = **{cv['chosen_n']}** under the amended criteria ({cv['amended_criteria']}). First-stated criteria: chosen {c
- line 211 [amend]: f"- Simulation count fixed: N = {cv.get('chosen_n')} (amended criteria; see the amendment list below).",
- line 216 [amend]: "3. **Final numeric gates for the Phase 1C outcome set are not pre-registered.** The protocol gates refer to the earlier Phase 1A/1B targets; a protocol amendment (G) with thresholds per outcome, per horizon and for cali
- line 218 [post-hoc]: "5. **Post-hoc rules to audit** (each is labelled where used): adjudication amendment 1 (2022 wk1 rows excluded after run 1); calibration materiality clarification (added after seeing the first-stated rule pass on 0.001 
- line 219 [rejected]: "the one-QB-always-plays rule was evaluated as an ablation and REJECTED on development evidence (pass-yards CRPS worse), so it is off.",
### nfl_phase1c_sim.py

- line 118 [ablation]: self.qb_bucket_mode = d.get("_qb_bucket_mode", "fixed"); self.force_qb = d.get("_force_qb", False)      # evidence (dev ablation): forcing a listed QB to play worsens pass-yards CRPS; the outside bucket absorbs the snaps
### nfl_phase1c_study.py

- line 75 [ablation]: joint_nofq, _ = ctx.run_joint("adjudicated", N, 11, "T24")               # ablation: WITH the one-listed-QB-always-plays rule
- line 78 [ablation]: joint_prop, _ = ctx.run_joint("adjudicated", N, 11, "T24")               # ablation: proportional QB bucket
- line 103 [ablation]: for lab, dd in (("ablation_force_one_listed_qb", joint_nofq), ("ablation_proportional_qb_bucket", joint_prop)):
- line 207 [post-hoc]: "materiality_clarification": ("POST-HOC (added after seeing that the first-stated rule is satisfied by 0.001 coverage changes): the recorded adoption additionally requires |cov80-0.8| to improve by >= 0.005 "
- line 238 [first-stated]: ok = [c for c, e in out["candidates"].items() if e["passes_rule"] and e["passes_rule_as_stated"]]      # adopt only if BOTH the first-stated and the materiality rule pass
- line 349 [amend]: ok_amended = {n: True for n in CONV_N}
- line 391 [amend]: b["criteria_met_amended"] = am; b["all_met_amended"] = all(am.values())
- line 392 [amend]: ok_amended[N] = ok_amended[N] and all(am.values())
- line 396 [post-hoc]: res["amended_criteria"] = ("POST-HOC AMENDMENT (the first-stated row-level quantile criteria fail for every tested N because tail quantiles of heavy-tailed yardage rows converge slowly; "
- line 399 [amend]: res["ok_by_n_amended"] = {str(k): v for k, v in ok_amended.items()}
- line 400 [amend]: res["chosen_n"] = next((N for N in CONV_N if ok_amended[N]), None)
### nfl_models/nfl_player_outcome_phase1c/README.md

- line 7 [amend]: ## A/B. Temporal architecture adjudication (rule recorded before any fold result: `adjudication_rule.json`; amendment 1 = 2022 wk1 warm-up rows)
- line 30 [rejected]: Near misses that the rule (correctly) rejected: rec_air B5 (6/7 folds, +0.00139 vs threshold 0.00162), rec_catch B2 (6/7, +0.00024 vs 0.00029), pass_completion B2 (6/7, +0.00029 vs 0.00031), def_sacks B1-B5 (mean gains a
- line 119 [post-hoc]: POST-HOC (added after seeing that the first-stated rule is satisfied by 0.001 coverage changes): the recorded adoption additionally requires |cov80-0.8| to improve by >= 0.005 and |cov50-0.5| not to worsen by more than 0
- line 167 [post-hoc]: Chosen N = **25000** under the amended criteria (POST-HOC AMENDMENT (the first-stated row-level quantile criteria fail for every tested N because tail quantiles of heavy-tailed yardage rows converge slowly; quantile SEs 
### nfl_models/nfl_player_outcome_phase1c/freeze_candidate_validation.md

- line 10 [amend]: - Simulation count fixed: N = 25000 (amended criteria; see the amendment list below).
- line 17 [amend]: 3. **Final numeric gates for the Phase 1C outcome set are not pre-registered.** The protocol gates refer to the earlier Phase 1A/1B targets; a protocol amendment (G) with thresholds per outcome, per horizon and for calib
- line 19 [post-hoc]: 5. **Post-hoc rules to audit** (each is labelled where used): adjudication amendment 1 (2022 wk1 rows excluded after run 1); calibration materiality clarification (added after seeing the first-stated rule pass on 0.001 c
### nfl_models/nfl_player_outcome_phase1c/adjudication_results.json

- line 2 [amend]: "rule": "adjudication_rule.json (with amendment_1)",
### nfl_models/nfl_player_outcome_phase1c/adjudication_rule.json

- line 26 [after seeing]: "no_period_cherry_picking": "the rule uses all seven folds; no fold, period or family is excluded after seeing results"
- line 28 [superseded]: "consumers": "Phase 1C consumes exactly the adjudicated component versions; the Phase 1B validation-only selections are superseded",
- line 29 [amend]: "amendment_1": {
### nfl_models/nfl_player_outcome_phase1c/adjudication_run1_partial.json

- line 2 [amend]: "note": "run 1 INCLUDED 2022 week-1 rows with all-zero as-of features; partial (interrupted); superseded by amendment 1; kept for transparency",
### nfl_models/nfl_player_outcome_phase1c/calibration.json

- line 7 [post-hoc]: "materiality_clarification": "POST-HOC (added after seeing that the first-stated rule is satisfied by 0.001 coverage changes): the recorded adoption additionally requires |cov80-0.8| to improve by >= 0.005 and |cov50-0.5
### nfl_models/nfl_player_outcome_phase1c/calibration_depth.json

- line 7 [post-hoc]: "materiality_clarification": "POST-HOC (added after seeing that the first-stated rule is satisfied by 0.001 coverage changes): the recorded adoption additionally requires |cov80-0.8| to improve by >= 0.005 and |cov50-0.5
### nfl_models/nfl_player_outcome_phase1c/joint_vs_independent.json

- line 272 [ablation]: "ablation_force_one_listed_qb": {
- line 290 [ablation]: "ablation_proportional_qb_bucket": {
- line 506 [ablation]: "ablation_force_one_listed_qb": {
- line 524 [ablation]: "ablation_proportional_qb_bucket": {
- line 740 [ablation]: "ablation_force_one_listed_qb": {
- line 758 [ablation]: "ablation_proportional_qb_bucket": {
- line 974 [ablation]: "ablation_force_one_listed_qb": {
- line 992 [ablation]: "ablation_proportional_qb_bucket": {
- line 1208 [ablation]: "ablation_force_one_listed_qb": {
- line 1226 [ablation]: "ablation_proportional_qb_bucket": {
- line 1442 [ablation]: "ablation_force_one_listed_qb": {
- line 1460 [ablation]: "ablation_proportional_qb_bucket": {
- line 1676 [ablation]: "ablation_force_one_listed_qb": {
- line 1694 [ablation]: "ablation_proportional_qb_bucket": {
- line 1910 [ablation]: "ablation_force_one_listed_qb": {
- line 1928 [ablation]: "ablation_proportional_qb_bucket": {
- line 2144 [ablation]: "ablation_force_one_listed_qb": {
- line 2162 [ablation]: "ablation_proportional_qb_bucket": {
- line 2378 [ablation]: "ablation_force_one_listed_qb": {
- line 2396 [ablation]: "ablation_proportional_qb_bucket": {
- line 2612 [ablation]: "ablation_force_one_listed_qb": {
- line 2630 [ablation]: "ablation_proportional_qb_bucket": {
- line 2846 [ablation]: "ablation_force_one_listed_qb": {
- line 2864 [ablation]: "ablation_proportional_qb_bucket": {
### nfl_models/nfl_player_outcome_phase1c/joint_vs_independent_depth.json

- line 212 [ablation]: "ablation_force_one_listed_qb": {
- line 230 [ablation]: "ablation_proportional_qb_bucket": {
- line 386 [ablation]: "ablation_force_one_listed_qb": {
- line 404 [ablation]: "ablation_proportional_qb_bucket": {
- line 560 [ablation]: "ablation_force_one_listed_qb": {
- line 578 [ablation]: "ablation_proportional_qb_bucket": {
- line 734 [ablation]: "ablation_force_one_listed_qb": {
- line 752 [ablation]: "ablation_proportional_qb_bucket": {
- line 908 [ablation]: "ablation_force_one_listed_qb": {
- line 926 [ablation]: "ablation_proportional_qb_bucket": {
- line 1082 [ablation]: "ablation_force_one_listed_qb": {
- line 1100 [ablation]: "ablation_proportional_qb_bucket": {
- line 1256 [ablation]: "ablation_force_one_listed_qb": {
- line 1274 [ablation]: "ablation_proportional_qb_bucket": {
- line 1430 [ablation]: "ablation_force_one_listed_qb": {
- line 1448 [ablation]: "ablation_proportional_qb_bucket": {
- line 1604 [ablation]: "ablation_force_one_listed_qb": {
- line 1622 [ablation]: "ablation_proportional_qb_bucket": {
- line 1778 [ablation]: "ablation_force_one_listed_qb": {
- line 1796 [ablation]: "ablation_proportional_qb_bucket": {
- line 1952 [ablation]: "ablation_force_one_listed_qb": {
- line 1970 [ablation]: "ablation_proportional_qb_bucket": {
- line 2126 [ablation]: "ablation_force_one_listed_qb": {
- line 2144 [ablation]: "ablation_proportional_qb_bucket": {
### nfl_models/nfl_player_outcome_phase1c/simulation_convergence.json

- line 97 [amend]: "criteria_met_amended": {
- line 105 [amend]: "all_met_amended": false
- line 130 [amend]: "criteria_met_amended": {
- line 138 [amend]: "all_met_amended": false
- line 163 [amend]: "criteria_met_amended": {
- line 171 [amend]: "all_met_amended": false
- line 196 [amend]: "criteria_met_amended": {
- line 204 [amend]: "all_met_amended": false
- line 229 [amend]: "criteria_met_amended": {
- line 237 [amend]: "all_met_amended": true
- line 262 [amend]: "criteria_met_amended": {
- line 270 [amend]: "all_met_amended": false
- line 295 [amend]: "criteria_met_amended": {
- line 303 [amend]: "all_met_amended": true
- line 344 [amend]: "criteria_met_amended": {
- line 352 [amend]: "all_met_amended": false
- line 377 [amend]: "criteria_met_amended": {
- line 385 [amend]: "all_met_amended": false
- line 410 [amend]: "criteria_met_amended": {
- line 418 [amend]: "all_met_amended": false
- line 443 [amend]: "criteria_met_amended": {
- line 451 [amend]: "all_met_amended": false
- line 476 [amend]: "criteria_met_amended": {
- line 484 [amend]: "all_met_amended": true
- line 509 [amend]: "criteria_met_amended": {
- line 517 [amend]: "all_met_amended": false
- line 542 [amend]: "criteria_met_amended": {
- line 550 [amend]: "all_met_amended": true
- line 591 [amend]: "criteria_met_amended": {
- line 599 [amend]: "all_met_amended": false
- line 624 [amend]: "criteria_met_amended": {
- line 632 [amend]: "all_met_amended": false
- line 657 [amend]: "criteria_met_amended": {
- line 665 [amend]: "all_met_amended": false
- line 690 [amend]: "criteria_met_amended": {
- line 698 [amend]: "all_met_amended": false
- line 723 [amend]: "criteria_met_amended": {
- line 731 [amend]: "all_met_amended": true
- line 756 [amend]: "criteria_met_amended": {
- line 764 [amend]: "all_met_amended": true
- line 789 [amend]: "criteria_met_amended": {
- line 797 [amend]: "all_met_amended": true
- line 838 [amend]: "criteria_met_amended": {
- line 846 [amend]: "all_met_amended": false
- line 871 [amend]: "criteria_met_amended": {
- line 879 [amend]: "all_met_amended": false
- line 904 [amend]: "criteria_met_amended": {
- line 912 [amend]: "all_met_amended": false
- line 937 [amend]: "criteria_met_amended": {
- line 945 [amend]: "all_met_amended": true
- line 970 [amend]: "criteria_met_amended": {
- line 978 [amend]: "all_met_amended": true
- line 1003 [amend]: "criteria_met_amended": {
- line 1011 [amend]: "all_met_amended": true
- line 1036 [amend]: "criteria_met_amended": {
- line 1044 [amend]: "all_met_amended": true
- line 1085 [amend]: "criteria_met_amended": {
- line 1093 [amend]: "all_met_amended": false
- line 1118 [amend]: "criteria_met_amended": {
- line 1126 [amend]: "all_met_amended": false
- line 1151 [amend]: "criteria_met_amended": {
- line 1159 [amend]: "all_met_amended": false
- line 1184 [amend]: "criteria_met_amended": {
- line 1192 [amend]: "all_met_amended": false
- line 1217 [amend]: "criteria_met_amended": {
- line 1225 [amend]: "all_met_amended": true
- line 1250 [amend]: "criteria_met_amended": {
- line 1258 [amend]: "all_met_amended": true
- line 1283 [amend]: "criteria_met_amended": {
- line 1291 [amend]: "all_met_amended": true
- line 1332 [amend]: "criteria_met_amended": {
- line 1340 [amend]: "all_met_amended": false
- line 1365 [amend]: "criteria_met_amended": {
- line 1373 [amend]: "all_met_amended": false
- line 1398 [amend]: "criteria_met_amended": {
- line 1406 [amend]: "all_met_amended": false
- line 1431 [amend]: "criteria_met_amended": {
- line 1439 [amend]: "all_met_amended": true
- line 1464 [amend]: "criteria_met_amended": {
- line 1472 [amend]: "all_met_amended": true
- line 1497 [amend]: "criteria_met_amended": {
- line 1505 [amend]: "all_met_amended": true
- line 1530 [amend]: "criteria_met_amended": {
- line 1538 [amend]: "all_met_amended": true
- line 1579 [amend]: "criteria_met_amended": {
- line 1587 [amend]: "all_met_amended": false
- line 1612 [amend]: "criteria_met_amended": {
- line 1620 [amend]: "all_met_amended": false
- line 1645 [amend]: "criteria_met_amended": {
- line 1653 [amend]: "all_met_amended": false
- line 1678 [amend]: "criteria_met_amended": {
- line 1686 [amend]: "all_met_amended": false
- line 1711 [amend]: "criteria_met_amended": {
- line 1719 [amend]: "all_met_amended": true
- line 1744 [amend]: "criteria_met_amended": {
- line 1752 [amend]: "all_met_amended": true
- line 1777 [amend]: "criteria_met_amended": {
- line 1785 [amend]: "all_met_amended": true
- line 1807 [post-hoc]: "amended_criteria": "POST-HOC AMENDMENT (the first-stated row-level quantile criteria fail for every tested N because tail quantiles of heavy-tailed yardage rows converge slowly; quantile SEs are reported but not gating)
- line 1808 [amend]: "ok_by_n_amended": {
### freeze_candidate_validation.md

- line 10 [amend]: - Simulation count fixed: N = 25000 (amended criteria; see the amendment list below).
- line 17 [amend]: 3. **Final numeric gates for the Phase 1C outcome set are not pre-registered.** The protocol gates refer to the earlier Phase 1A/1B targets; a protocol amendment (G) with thresholds per outcome, per horizon and for calib
- line 19 [post-hoc]: 5. **Post-hoc rules to audit** (each is labelled where used): adjudication amendment 1 (2022 wk1 rows excluded after run 1); calibration materiality clarification (added after seeing the first-stated rule pass on 0.001 c
