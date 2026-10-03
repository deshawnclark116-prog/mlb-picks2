# Phase 1D - freeze hardening / live shadow readiness

This phase does not change the predictive architecture. It makes the existing Phase 1A/1B/1C research candidate operationally reproducible and scientifically pre-registered so that a clean-forward test means something. Production serving is untouched; `nfl_player_outcome_phase1_freeze.json` has not been created; nothing from the clean-forward window was used.

## 0. Research interpretation (locked)

1. Phase 1C currently TIES the honest v2 refit on rushing point MAE within noise: combined played-row MAE 25.02 vs 25.13, improvement 0.106 yd, block-bootstrap p(not better) = 0.35 (n = 986).
2. Phase 1C is slightly WORSE than the honest v2 refit on receiving point MAE on burned development data: 22.48 vs 22.19 (v2 better by 0.30 yd), p(Phase 1C not better) = 0.93 (n = 2,665).
3. Phase 1C has modestly better distributional CRPS in some comparisons (vs the v2 refit + 2024 residual distribution: rushing 17.92 vs 18.24, improvement 0.31, p = 0.085; receiving 15.91 vs 16.03, improvement 0.13, p = 0.21) - neither is significant at 0.05. It beats the historical-blend point forecast on MAE (rushing p = 0.039, receiving p = 0.015).
4. Phase 1C has not earned production replacement.
5. This does NOT prevent a clean-forward shadow evaluation.
6. No further predictive tuning is permitted before that clean evaluation. The purpose of freezing is to TEST the architecture, not to declare it superior.

## 1. Post-hoc audit

See `posthoc_audit.md` / `.json` (every decision classified; machine scan appendix).

| id | decision | classification | pre-registered |
|---|---|---|---|
| D01 | 2022 week-1 exclusion (adjudication amendment_1) | DATA/WARMUP DEFINITION CORRECTION | False |
| D02 | Calibration materiality clarification | DEVELOPMENT-SELECTED | False |
| D03 | Simulation-count convergence criteria amendment | ENGINEERING SELECTION | False |
| D04 | Depth-chart-extended candidate universe | DEVELOPMENT-SELECTED | False |
| D05 | Predictability score redefinition | DEVELOPMENT-SELECTED | False |
| D06 | High-confidence subset definition | REPORTING DEFINITION | False |
| D07 | One-listed-QB-always-plays rule | NOT ADOPTED | False |
| D08 | Proportional rescale of QB dropback / attempt allocation (prop bucket) | DEVELOPMENT-SELECTED | False |
| D09 | Fitted within-bin means instead of bin midpoints | DEVELOPMENT-SELECTED | False |
| D10 | Red-zone yardage and touchdown mechanics | DEVELOPMENT-SELECTED | False |
| D11 | Static (S0) versus state-aware (S1) game script | DEVELOPMENT-SELECTED | False |
| D12 | Phase 1A architecture selection (team environment, availability, role families, opportunity components) | DEVELOPMENT-SELECTED | True |
| D13 | Production-v2 artifact scored only on 2026 wk1-3 | CORRECTNESS FIX | False |
| D14 | Structural constants fitted on all burned data | DEVELOPMENT-SELECTED | False |
| D15 | Report / harness fixes (table KeyError, absent-player chaos assertion, restarted dry run) | PROCESS | False |
| D17 | Stored QB rushing-yards calibration map was never applied (found in Phase 1D) | CORRECTNESS FIX | False |
| D18 | Forecast-only defenders: position group as of the cutoff (found by LiveLoader equivalence) | CORRECTNESS FIX | False |
| D19 | Player listed as a candidate for both teams of a game (found by the 2026 dry run) | CORRECTNESS FIX | False |
| D16 | Phase 1D choices (recorded here so they are not hidden) | PROCESS | False |

## 2. Accuracy numbers with their universes (T24, combined development weeks)

The Phase 1C headline (72% of rushing rows within +/-10 yards) was measured on universe A, where most rows are non-participants with a true outcome of 0.

**rush_yds**

| universe | n | share true outcome 0 | within +/-10 | within +/-20 | within +/-40 |
|---|---|---|---|---|---|
| A_full_pregame_universe(all Phase 1A candidate rows; non-participants have outcome 0) | 4615 | 0.5387 | 0.7244 | 0.8418 | 0.9404 |
| B_participated_players_only(DIAGNOSTIC: selects on the game outcome) | 2588 | 0.194 | 0.5379 | 0.7322 | 0.8972 |
| C_positive_actual_opportunity_only(DIAGNOSTIC: selects on the game outcome) | 2102 | 0.0195 | 0.4329 | 0.6656 | 0.8711 |
| D_protocol_eligible_stable_role | 694 | 0.2089 | 0.4164 | 0.6412 | 0.8473 |
| E_protocol_eligible_volatile_role | 347 | 0.4611 | 0.5677 | 0.7147 | 0.8876 |
| H_protocol_eligible_all_role_states(primary scoring universe of Amendment G) | 1387 | 0.2978 | 0.4614 | 0.6619 | 0.8587 |
| F_forecast_P(active)>=0.90 | 1705 | 0.1589 | 0.4815 | 0.7009 | 0.8874 |

**rec_yds**

| universe | n | share true outcome 0 | within +/-10 | within +/-20 | within +/-40 |
|---|---|---|---|---|---|
| A_full_pregame_universe(all Phase 1A candidate rows; non-participants have outcome 0) | 10060 | 0.5596 | 0.7036 | 0.8352 | 0.9409 |
| B_participated_players_only(DIAGNOSTIC: selects on the game outcome) | 6649 | 0.3453 | 0.57 | 0.7615 | 0.915 |
| C_positive_actual_opportunity_only(DIAGNOSTIC: selects on the game outcome) | 4984 | 0.1116 | 0.443 | 0.68 | 0.8842 |
| D_protocol_eligible_stable_role | 1710 | 0.2591 | 0.4292 | 0.6655 | 0.8801 |
| E_protocol_eligible_volatile_role | 855 | 0.3591 | 0.4772 | 0.6433 | 0.8433 |
| H_protocol_eligible_all_role_states(primary scoring universe of Amendment G) | 3420 | 0.2921 | 0.4523 | 0.6655 | 0.8708 |
| F_forecast_P(active)>=0.90 | 4820 | 0.2734 | 0.5031 | 0.7263 | 0.9048 |

Full tables (all slices, all tolerances, T24 and T90): `accuracy_slices.md`.

## 6. LiveLoader equivalence

Max |difference| over every compared quantity: 0.0 (tolerance 1e-09); accepted = True.


## 9. Multi-week time-travel dry run

| week | success / expected | idempotent rerun | crash-restart | reproduction | target perturbation | contaminated snapshot | prior-history sensitivity |
|---|---|---|---|---|---|---|---|
| 2025 wk4 | 32/32 | True (9578 dup) | True | True | True | True | 0.474 |
| 2025 wk11 | 30/30 | True (9360 dup) | True | True | True | True | 0.238 |
| 2025 wk16 | 32/32 | True (9948 dup) | True | True | True | True | 0.163 |
| 2026 wk1 | 32/32 | True (10800 dup) | True | True | True | True | n/a (week 1 of season) |
| 2026 wk2 | 32/32 | True (11410 dup) | True | True | True | True | 0.232 |
| 2026 wk3 | 32/32 | True (11368 dup) | True | True | True | True | 0.229 |

## 11. N = 25,000 engineering audit

Verdict: {'C1': True, 'C2': True, 'C3': False, 'C4': False, 'N_25000_retained': False}

| outcome | C1 MC SE / model RMSE | C2 |CRPS 25k - 100k| / CRPS | C3 max tail SE | C4 pass |
|---|---|---|---|---|
| rush_yds | 0.004931529673186489 | 0.00006 | 0.0031604342429482693 | False |
| rec_yds | 0.005430747103349734 | 0.00021 | 0.003131223339207857 | False |
| rec | 0.0054285883772917575 | 0.00028 | None | True |
| pass_yds | 0.006049725554584293 | 0.00038 | 0.0029671445775357828 | False |
| atd | 0.00554298098327248 | 0.00003 | 0.0031622757469898163 | None |

## Design summary (what each module guarantees)

- **Content-addressed snapshots** (`nfl_phase1d_cas.py`): every source Phase 1 reads (schedule, players, injuries, weekly rosters, depth charts, player stats, play-by-play, snap counts, participation, FTN charting) is stored as an immutable `sha256 -> blob` (read-only, atomic link, never overwritten, hash re-verified on every read). The manifest row records logical name, provider URL, retrieval timestamp, information cutoff and horizon, sha256, byte count, blob path, parser version and any documented transform (the schedule is stored without sportsbook columns; raw sha256 kept). Identical bytes at two forecast times are one blob referenced twice. Loaders read a directory of verified symlinks, so a forecast can only see snapshotted bytes.
- **Time-travel** (burned weeks): the provider is simulated by dropping every row of games not completed at the cutoff (kickoff + 24h), later-week injuries / rosters, T24 current-week rosters and depth-chart snapshots after the cutoff. Sets are labelled `time_travel`; simulated retrieval = cutoff - 1h.
- **Schedule / kickoff safety** (`nfl_phase1d_schedule.py`): kickoff and cutoff come from the schedule snapshot; retrieval after the cutoff means no forecast (SAFE_EXPLICIT_FAILURE); an append-only revision ledger records every kickoff / status change; forecast ids contain the cutoff so a moved kickoff never overwrites a forecast; the scored forecast is the latest one whose cutoff equals (final kickoff - horizon) and whose retrieval preceded it; flex earlier / later, postponement, cancellation and neutral-site time changes are ordinary revisions (rules in the module docstring, tested).
- **Serialized Phase 1A** (`nfl_phase1d_p1a.py`): team-environment coefficients, availability models and status lookup tables, role-state priors / HMM, propensity boosters, outside-bucket weights and share noise are written to files with a sha256 manifest; forecasts use only the loaded bytes. `phase1a_frozen_selection.json` holds everything that may not change (selection, hyper-parameters, xgboost parameters).
- **Walk-forward refit**: training = every completed week before the target week from 2023 on (last 25% of those weeks validate), fitted quantities only; Phase 1B refits its tilt coefficients and as-of counts with the hyper-parameters frozen from the freeze fit (`phase1b_frozen_hyper.json`, no grid search); every refit is a new artifact bundle hash and a row in `weekly_fits.jsonl` (training cutoff, weeks, artifact hashes, code hashes, code SHA).
- **Locking** (`nfl_phase1_store_lock.py`): kernel flock (dead holders release automatically; stale owner records are recognised and replaced), explicit timeout, one committing process per store. flock is HOST-LOCAL: each store is bound to its host (`host_binding.json`) and a writer on any other host is refused; multi-host operation is unsupported until a distributed lock replaces it.
- **Baselines and gates**: `nfl_phase1d_baselines.py` (HB1) is logged before kickoff next to every forecast; `nfl_phase1d_gates.py` holds Amendment G and its evaluators.

## Changes to the forecast path since Phase 1C (each documented in posthoc_audit.md; none is performance motivated)

- D17: the stored QB rushing-yards calibration map is now applied to QB rows (it was inert).
- D18: forecast-only defenders / T24 offense positions are resolved as of the cutoff (LiveLoader equivalence finding).
- Record schema additions: `p_zero`, 99-point quantile grid, exact lattice cdf, as-of `prior_usage`, `p_active_status_baseline`, provenance ids (needed for randomized PIT, eligibility and the availability gate).
- `protocol_version` 1.2; the walk-forward path fits with frozen hyper-parameters (Phase 1C dry runs searched them per fit).
