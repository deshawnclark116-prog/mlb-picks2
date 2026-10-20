# Phase1R-VOI: unified error budget / value-of-information audit

**Verdict: PHASE1R_ERROR_BUDGET_COMPLETE_PARTIAL_DECOMPOSITION_QB_BLOCKED.** Measurement only: no model was fitted, no new data were opened, and no Week 5+ or sportsbook data were touched. Free-information ceiling: **CURRENT_FREE_INFORMATION_EXHAUSTED_FOR_TESTED_HEADS**. Every number below is a diagnostic attribution of the error of the already-frozen chain; oracle gains are upper bounds on addressable error, not expected model improvement.

## Cohorts (identical rows within every comparison)

| Head | Source population | Common cohort | Exclusions | Normal MAE of excluded rows |
|---|---:|---:|---|---:|
| carries_2024_in_sample | 815 | 815 | none | - |
| receiving_yards_decomposed | 2229 | 2200 | {'COMPONENT_LABELS_NOT_RECONCILED_AIR_YAC_SPLIT_UNKNOWN': 27, 'NO_PREDICTED_TARGETS_PHASE1D_ALLOCATION_ABSENT': 2} | 29.64 |
| receiving_yards_incumbent_ypt | 2229 | 2200 | {'COMPONENT_LABELS_NOT_RECONCILED_AIR_YAC_SPLIT_UNKNOWN': 27, 'NO_PREDICTED_TARGETS_PHASE1D_ALLOCATION_ABSENT': 2} | 29.64 |
| receptions | 2229 | 2227 | {'NO_PREDICTED_TARGETS_PHASE1D_ALLOCATION_ABSENT': 2} | - |
| rushing_yards_2024_in_sample | 633 | 450 | {'NOT_IN_PHASE1D_FIXED_MEANINGFUL_POPULATION_NO_FROZEN_SHARE_OR_TEAM_PROJECTION': 183} | 0.00 |
| targets_2024_in_sample | 2311 | 2311 | none | - |

Receiving-yards and receptions cohorts are 2025 validation receipts. Targets, carries and rushing yards come from 2024 receipts (the seasons used to select the frozen Phase1D/Phase1B configs), so they are descriptive and optimistic for the chain. Every cohort row is a player with a stat row in the game: availability (did he play) is therefore not measurable here and was never inferred. Rows whose realized per-catch or per-carry component is undefined (zero receptions, zero carries) are kept; the oracle keeps the predicted component for them. The 27 receiving rows without reconciled air/YAC labels are excluded from the four-component cohort (their normal MAE is 29.6 vs 22.6 for the cohort, so the exclusion does not flatter the budget) and the large misses among them are listed as `UNRESOLVED`.

## Error budget by head (normal MAE, one-component oracles, exact Shapley)

| Head | n | Normal MAE | Oracle: components replaced one at a time (MAE) | Shapley contribution (MAE) | % of addressable | Sum of single gains / normal |
|---|---:|---:|---|---|---|---:|
| Carries team volume x role share | 815 | 4.102 | ROLE_SHARE 2.40, TEAM_VOLUME 3.18 | ROLE_SHARE 2.44, TEAM_VOLUME 1.66 | ROLE_SHARE 59.6%, TEAM_VOLUME 40.4% | 0.64 |
| Receiving yards T x C x (A+Y) | 2200 | 22.647 | A 20.88, C 20.23, T 16.37, Y 21.69 | A 4.79, C 4.63, T 9.79, Y 3.44 | A 21.1%, C 20.4%, T 43.2%, Y 15.2% | 0.50 |
| Receiving yards T x yards/target (Phase1F incumbent) | 2200 | 22.691 | E 16.17, T 16.41 | E 11.47, T 11.23 | E 50.5%, T 49.5% | 0.56 |
| Receptions T x C | 2227 | 1.657 | C 1.42, T 0.83 | C 0.54, T 1.12 | C 32.3%, T 67.7% | 0.64 |
| Rushing yards team x role x YPC | 450 | 23.945 | E 16.87, ROLE_SHARE 19.33, TEAM_VOLUME 19.49 | E 9.71, ROLE_SHARE 7.70, TEAM_VOLUME 6.54 | E 40.5%, ROLE_SHARE 32.1%, TEAM_VOLUME 27.3% | 0.67 |
| Targets team volume x role share | 2311 | 2.163 | ROLE_SHARE 0.98, TEAM_VOLUME 1.96 | ROLE_SHARE 1.57, TEAM_VOLUME 0.59 | ROLE_SHARE 72.6%, TEAM_VOLUME 27.4% | 0.64 |

All-oracle MAE is 0 for every head by construction (the chains are exact identities), so the addressable MAE equals the normal MAE; the informative quantities are the component shares. MAE is nonlinear and the factors multiply: replacing one component at a time recovers only 50-68% of the normal MAE, so single-oracle gains must not be added up; the Shapley contributions sum exactly to the normal error.

**Receiving yards (2025 validation):** target opportunity is the largest piece (Shapley 43.2%), then completed air (21.1%), catch rate (20.4%) and YAC (15.2%). The frozen Phase1F aggregate agrees (perfect workload removes 6.3 of 22.8 MAE). Targets split into role share (72.6%) and team volume (27.4%) on the 2024 targets cohort; carries split 59.6% / 40.4%. **Rushing yards (2024):** efficiency 40.5%, role share 32.1%, team volume 27.3%.

**Rushing explosive tail.** On the 438 labelled rows the mean efficiency-yards error is 14.6 yards, of which the realized tier-mixture surprise (more or fewer 10+/20+ runs than the Phase1K tier probabilities implied) averages 12.7 yards and within-tier yardage 8.9. Of 110 efficiency misses of 20+ yards, 74 are at least half explained by the tail-mix surprise. Rushing efficiency error is therefore mostly realized explosive-run variance, not a smooth bias.

## Catastrophic misses (dominant cause by Shapley; threshold in units of the head)

| Head | Threshold | Misses | Share of cohort | Dominant causes |
|---|---:|---:|---:|---|
| Carries team volume x role share | >12.0 | 15 | 1.8% | ROLE_SHARE 10, TEAM_VOLUME 5 |
| Carries team volume x role share | >5.0 | 281 | 34.5% | ROLE_SHARE 169, TEAM_VOLUME 112 |
| Carries team volume x role share | >8.0 | 101 | 12.4% | ROLE_SHARE 55, TEAM_VOLUME 46 |
| Receiving yards T x C x (A+Y) | >25.0 | 732 | 33.3% | CATCH_RATE 111, COMPLETED_AIR 108, MULTIPLE_COMPONENTS 172, OPPORTUNITY_UNSPLIT 283, UNRESOLVED 13, YAC 45 |
| Receiving yards T x C x (A+Y) | >40.0 | 338 | 15.4% | CATCH_RATE 40, COMPLETED_AIR 43, MULTIPLE_COMPONENTS 90, OPPORTUNITY_UNSPLIT 134, UNRESOLVED 8, YAC 23 |
| Receiving yards T x C x (A+Y) | >60.0 | 137 | 6.2% | CATCH_RATE 9, COMPLETED_AIR 15, MULTIPLE_COMPONENTS 43, OPPORTUNITY_UNSPLIT 59, UNRESOLVED 3, YAC 8 |
| Receiving yards T x C x (A+Y) | >80.0 | 50 | 2.3% | CATCH_RATE 1, COMPLETED_AIR 4, MULTIPLE_COMPONENTS 17, OPPORTUNITY_UNSPLIT 21, UNRESOLVED 1, YAC 6 |
| Receiving yards T x yards/target (Phase1F incumbent) | >25.0 | 722 | 32.8% | OPPORTUNITY_UNSPLIT 312, RECEIVING_EFFICIENCY 410 |
| Receiving yards T x yards/target (Phase1F incumbent) | >40.0 | 327 | 14.9% | OPPORTUNITY_UNSPLIT 146, RECEIVING_EFFICIENCY 181 |
| Receiving yards T x yards/target (Phase1F incumbent) | >60.0 | 135 | 6.1% | OPPORTUNITY_UNSPLIT 66, RECEIVING_EFFICIENCY 69 |
| Receiving yards T x yards/target (Phase1F incumbent) | >80.0 | 49 | 2.2% | OPPORTUNITY_UNSPLIT 24, RECEIVING_EFFICIENCY 25 |
| Receptions T x C | >2.0 | 704 | 31.6% | CATCH_RATE 192, OPPORTUNITY_UNSPLIT 512 |
| Receptions T x C | >3.0 | 307 | 13.8% | CATCH_RATE 71, OPPORTUNITY_UNSPLIT 236 |
| Receptions T x C | >4.0 | 120 | 5.4% | CATCH_RATE 19, OPPORTUNITY_UNSPLIT 101 |
| Rushing yards team x role x YPC | >20.0 | 216 | 48.0% | EXPLOSIVE_TAIL 55, MULTIPLE_COMPONENTS 32, ROLE_SHARE 57, RUSH_EFFICIENCY 33, TEAM_VOLUME 39 |
| Rushing yards team x role x YPC | >30.0 | 133 | 29.6% | EXPLOSIVE_TAIL 35, MULTIPLE_COMPONENTS 24, ROLE_SHARE 29, RUSH_EFFICIENCY 25, TEAM_VOLUME 20 |
| Rushing yards team x role x YPC | >50.0 | 41 | 9.1% | EXPLOSIVE_TAIL 9, MULTIPLE_COMPONENTS 9, ROLE_SHARE 7, RUSH_EFFICIENCY 9, TEAM_VOLUME 7 |
| Rushing yards team x role x YPC | >75.0 | 9 | 2.0% | EXPLOSIVE_TAIL 1, MULTIPLE_COMPONENTS 2, ROLE_SHARE 1, RUSH_EFFICIENCY 3, TEAM_VOLUME 2 |
| Targets team volume x role share | >3.0 | 575 | 24.9% | ROLE_SHARE 485, TEAM_VOLUME 90 |
| Targets team volume x role share | >5.0 | 170 | 7.4% | ROLE_SHARE 142, TEAM_VOLUME 28 |
| Targets team volume x role share | >7.0 | 49 | 2.1% | ROLE_SHARE 37, TEAM_VOLUME 12 |

A miss gets a single cause only if one component carries at least 50% of its Shapley error; otherwise `MULTIPLE_COMPONENTS`. `OPPORTUNITY_UNSPLIT` is an added honest label for receiving yards where the 2025 receipts do not separate team volume from role share; `RECEIVING_EFFICIENCY` labels the two-factor incumbent chain. AVAILABILITY, GAME_SCRIPT and QB_STATE were never assigned because no receipt field supports them.

## Burned Week 4 large-miss forensic (no refitting)

Source: the frozen table of the ten largest clean Week 4 system-pick misses. Only opportunity (expected vs actual) and implied efficiency are reconstructible; team volume vs role share, availability and explosive-tail realization are `UNKNOWN` because no Week 4 team, snap, route or play-level receipt is frozen. Where the Phase1H diagnostic pipeline has the same player-game, its four-component split is attached to the receipt.

| Player | Outcome | Proj | Actual | Expected / actual opportunities | Projected / realized per opportunity | Shapley: opportunity / efficiency | Best single oracle |
|---|---|---:|---:|---|---|---|---|
| Kenneth Walker | rush yds | 59 | 177 | 14.90 / 22 | 3.96 / 8.05 | 42.6 / 75.4 | ACTUAL_EFFICIENCY |
| Carnell Tate | rec yds | 40 | 145 | 5.88 / 12 | 6.80 / 12.08 | 57.8 / 47.2 | ACTUAL_OPPORTUNITY |
| T.J. Hockenson | rec yds | 33 | 119 | 5.09 / 13 | 6.48 / 9.15 | 61.8 / 24.2 | ACTUAL_OPPORTUNITY |
| Brenton Strange | rec yds | 24 | 95 | 4.04 / 8 | 5.94 / 11.88 | 35.3 / 35.7 | ACTUAL_EFFICIENCY |
| Ollie Gordon | rush yds | 36 | 100 | 10.16 / 9 | 3.54 / 11.11 | 4.4 / 59.6 | ACTUAL_EFFICIENCY |
| Michael Mayer | rec yds | 26 | 82 | 4.43 / 10 | 5.87 / 8.20 | 39.2 / 16.8 | ACTUAL_OPPORTUNITY |
| Ashton Jeanty | rec yds | 22 | 68 | 4.75 / 8 | 4.63 / 8.50 | 21.3 / 24.7 | ACTUAL_EFFICIENCY |
| Sam LaPorta | rec yds | 39 | 84 | 5.67 / 13 | 6.88 / 6.46 | 43.5 / 1.5 | ACTUAL_OPPORTUNITY |
| Isaiah Williams | rec yds | 30 | 73 | 4.43 / 3 | 6.77 / 24.33 | 12.6 / 30.4 | ACTUAL_EFFICIENCY |
| Courtland Sutton | rec yds | 41 | 4 | 5.79 / 6 | 7.08 / 0.67 | -0.7 / 37.7 | ACTUAL_EFFICIENCY |

Tate, Hockenson, Mayer and LaPorta are opportunity-dominated; Walker, Gordon, Strange, Jeanty, Isaiah Williams and Sutton are efficiency-dominated (Walker 8.05 vs 3.96 yards per carry, Gordon 11.1 vs 3.5, Williams 24.3 vs 6.8 yards per target, Sutton 0.67 vs 7.1). The earlier "workload misses" reading holds for the first group and fails for the second. Walker and Gordon efficiency are consistent with the tail-mix finding, but their run-level data are not frozen, so the explosive tail is not asserted.

## System vs competent human

| Head (2025 validation) | System MAE | Human MAE | System bias | Human bias |
|---|---:|---:|---:|---:|
| rec_yds | 22.79 | 23.29 | -1.04 | -8.42 |
| rec | 1.66 | 1.77 | -0.01 | -0.68 |
| rush_yds | 23.65 | 24.75 | 0.96 | -10.44 |
| pass_yds | 63.97 | 65.93 | 8.18 | 11.92 |

The system beats the competent human on MAE and bias for every head in 2025 (the human underprojects receiving yards by 8.4). The human beats the system on **no** measurable slice: frozen Phase1D has lower oracle role error than the human role baseline overall and in every slice (increasing/decreasing role, teammate vacancy, rookie, stable) on both heads (receiving 1.96 vs 2.17; rushing 3.05 vs 3.29), and the incumbent beats the human on rushing efficiency (10.11 vs 10.52) including on games with a 20+ run (24.59 vs 25.46). The only place a human number is better is bias on 2026 W1-4 passing yards (-2.82 vs -12.64), which is not a MAE win. Availability changes, game script, QB state and tail outcomes cannot be compared: the historical human receipts are too coarse (no availability, script or starter labels).

## QB attempts / passing yards

`BLOCKED_STARTER_STATE_DATA`: no row-level decomposition exists and none was fabricated (Phase1L receipts hold no predictions). The only available evidence is the frozen Phase1F aggregate oracle diagnostic for 2025, labelled `POSTGAME_DIAGNOSTIC_ONLY`: normal MAE 63.95; with perfect attempt workload 39.95 (gain 24.00); with perfect efficiency 50.40 (gain 13.54). Workload is the larger piece, but whether starter identity explains it cannot be tested without a certifiable pregame starter state.

## Value-of-information ranking (upper bounds, not expected improvement)

| Rank | Tier | Information | Best-evidenced head | Est. addressable share of normal MAE | Existing status | Historical / live | Blocker | Confidence |
|---:|---|---|---|---:|---|---|---|---|
| 1 | TIER 1 ACQUIRE FIRST | Verified QB starter / QB health state | receptions | 38.0% | Phase1L could not test it (BLOCKED_STARTER_STATE_DATA); the QB head cannot be decomposed without it | NOT CERTIFIED: original pregame vintages unavailable (Phase1M NO_SOURCE_ACCEPTED) / licensed vendor feeds exist (Sportradar / SportsDataIO) but are not accessed | vendor contract and original-vintage archive | MEDIUM_HIGH |
| 2 | TIER 2 HIGH VALUE IF ACCESSIBLE | Alignment / slot / wide / backfield role | targets_2024_in_sample | 54.7% | position labels and receiver depth profiles (Phase1H/I) were rejected | licensed charting only / prior-game alignment from licensed APIs | license | MEDIUM |
| 3 | TIER 2 HIGH VALUE IF ACCESSIBLE | Actual routes run / route participation | targets_2024_in_sample | 54.7% | offensive snap deployment (Phase1Q) is a coarse proxy and was rejected; routes are finer but correlated | public participation data are released after the game and carry no pregame routes; full route APIs need a license / prior-game routes available with a lag from licensed APIs | license and original-vintage latency proof | MEDIUM_HIGH |
| 4 | TIER 2 HIGH VALUE IF ACCESSIBLE | Offensive skill-player injury / availability state | targets_2024_in_sample | 54.7% | roster-status vacancy (Phase1O family D) added nothing; the injury reports themselves were never tested (BLOCKED_TIMING) | original publication timestamps missing for nflverse injuries (BLOCKED_TIMING); official reports not archived under a license / official NFL injury reports are public and can be captured forward with hashed receipts | no historical vintage for validation; needs forward capture | MEDIUM |
| 5 | TIER 2 HIGH VALUE IF ACCESSIBLE | Defensive personnel availability / assignments | rushing_yards_2024_in_sample | 29.6% | Phase1H defensive_absences was BLOCKED_DATA, so it was never tested | no timestamp-safe historical source / injury-report derived forward capture | no historical vintage for validation | MEDIUM |
| 6 | TIER 2 HIGH VALUE IF ACCESSIBLE | OL starters / protection quality | rushing_yards_2024_in_sample | 29.6% | BLOCKED_NO_TIMESTAMP_SAFE_SOURCE_IN_CURRENT_REPO | no timestamp-safe historical source / injury-report derived forward capture; quality needs licensed data | no historical vintage | MEDIUM |
| 7 | TIER 2 HIGH VALUE IF ACCESSIBLE | Separation / target quality / catchability | receiving_yards_decomposed | 19.8% | public depth/catchability proxies (Phase1I) were rejected; tracked separation was never available | tracking-derived metrics are licensed; pregame use is only through history / prior-game metrics available with a lag | license and latency proof | MEDIUM |
| 8 | TIER 3 SECONDARY | Designed first-read / target intent | targets_2024_in_sample | 54.7% | never tested | postgame charting labels only; pregame design intent is not published / unclear; charting is labeled after the game | no pregame intent signal exists in public sources | LOW |
| 9 | TIER 4 LOW EXPECTED VALUE | Game-state / possession-strength information | carries_2024_in_sample | 22.4% | Phase1N team-state chain (pregame margin scenarios) was rejected | public / public | none, but the same signal already failed | HIGH |
| 10 | DO NOT PURSUE | Box / front information | rushing_yards_2024_in_sample | 29.6% | Phase1K defense-allowance family rejected | play-level tracking only / not knowable pregame | the quantity is not observable before the snap | LOW |
| 11 | DO NOT PURSUE | Explosive-run / blocking context | rushing_yards_2024_in_sample | 29.6% | Phase1K explosive and state families rejected | play-level charting / not knowable pregame | not observable before the snap | LOW |
| 12 | DO NOT PURSUE | Route-level coverage | receiving_yards_decomposed | 28.7% | Phase1H man/zone and pressure features rejected | licensed play-level charting / not knowable pregame | the quantity is not observable before the snap | LOW |

Shares use the 2024 team-vs-role split (role 72.6%, team 27.4%) where only a combined target component exists. Per-head numbers, catastrophic misses touched by each family and the full evidence are in `phase1r_information_value_ranking.json`. **Tier 1 contains only the QB starter / health state**, and its evidence is the aggregate Phase1F workload diagnostic plus indirect row-level team-volume and catch/air shares, not a row-level decomposition. Tier 2 sources overlap partly with failed proxies (snaps, roster-status vacancy, public depth/catchability) or have no historical vintage (forward capture only); Tier 3 first-read/intent has no identified pregame signal; box/front, route-level coverage and explosive-run context are not observable before the snap and are not worth pursuing. The tier rule was tightened (validation path and measurement confidence added) before anything was committed after a first mechanical pass ranked sources lacking any validation path into Tier 1; that first-pass output was not retained.

## Free-information ceiling

**CURRENT_FREE_INFORMATION_EXHAUSTED_FOR_TESTED_HEADS.** Phases 1H-1Q tested receiving efficiency (routed, depth/catchability), rushing efficiency, team opportunity, role change and snap deployment; none produced a SURVIVES or PARTIAL verdict; the best role gains were 0.015 (snap) and 0.008 (role-change) targets against a 0.10 threshold and 0.013 carries against 0.20; one signal (receiver depth) survived without being promoted. The remaining error sits in target opportunity, catch/air and the rushing tail, which the accepted free information did not move. The verdict covers the tested heads only: the QB head is untested (blocked), and untested free information (forward injury capture, depth charts from 2025) cannot be validated historically.

## Surviving V2 stack

| Layer | Incumbent | Status | Known MAE | Main blocker | Production-ready |
|---|---|---|---|---|---|
| TEAM_OPPORTUNITY | Phase1B team-volume formula (Phase1C-T generic PBP environment frozen, not promoted) | REJECTED_TEAM_OPPORTUNITY_REPLACEMENT (Phase1N); incumbent unchanged | {"phase1n_selection_2024_incumbent_plays_mae": 6.806401, "receiving_head_team_targets_2025": 5.774822, "rushing_head_team_carries_2025": 5.897092} | genuinely new pregame information (QB state, availability); the accepted free pregame state added nothing | no |
| PLAYER_TARGET_SHARE | Phase1D target-share allocator | SURVIVED (Phase1D); Phase1O and Phase1Q replacements rejected | {"oracle_target_mae_2025": 1.879618, "role_share_mae_2025": 0.062136} | route/availability information not accessible | no |
| PLAYER_CARRY_SHARE | Phase1D carry-share allocator | SURVIVED (Phase1D); Phase1O and Phase1Q replacements rejected | {"oracle_carry_mae_2025": 3.035734, "role_share_mae_2025": 0.113103} | availability / committee information not accessible | no |
| RECEIVING_EFFICIENCY | Phase1F/Phase1B yards-per-target (Phase1G decomposition kept for diagnostics) | FROZEN_AT_PHASE1F_PENDING_NEW_INFORMATION (Phase1G/H/I replacements rejected) | {"final_receiving_yards_mae_2025": 22.790403, "yards_per_target_mae_2025": 3.766039} | tracking/charting information (separation, routes, defenders) with valid history and live timing | no |
| RUSHING_EFFICIENCY | Phase1F/Phase1B yards-per-carry | REJECTED_EFFICIENCY_REPLACEMENT (Phase1K); incumbent only, not certified | {"final_rushing_yards_mae_2025": 23.647409, "yards_per_carry_mae_2025": 1.748399} | OL / box / blocking context; explosive tail is mostly realized variance | no |
| QB_OPPORTUNITY | Phase1B QB attempt path (unchanged) | BLOCKED_STARTER_STATE_DATA (Phase1L); no source accepted (Phase1M) | {"attempt_oracle_player_opportunity_mae_2025": 7.47399, "passing_yards_mae_2025": 63.96914} | certifiable pregame QB starter / health state | no |
| QB_EFFICIENCY | Phase1B QB efficiency (unchanged) | NO_REPLACEMENT_TESTED (blocked behind QB state) | {"yards_per_attempt_mae_2025": 1.506029} | QB assignment must be known before efficiency can be separated | no |
| MONTE_CARLO | none for V2 (the V1 / Phase1C simulator is a benchmark only) | NOT_BUILT_FOR_V2: Monte Carlo must sit downstream of a promoted direct football projection | - | no promoted direct projection chain | no |

Nothing is promoted because alternatives failed; production remains the V1 engine.

## Registry cleanup

Top-level `status` and `next_milestone` now state Phase1R and the new-information gate (the stale Phase1N next-milestone prose is gone). A new `current_state` block is the authoritative statement of every layer, lists the current open and closed research tracks, and records that the Phase1O family B `BLOCKED_DATA` label was an identity blocker resolved by Phase1P and rejected by Phase1Q; the Phase1O record itself is untouched. The legacy top-level `open_research_tracks` and `frozen_or_blocked` lists (milestones up to Phase1D) were deliberately left byte-identical, because the Phase1I and Phase1K archival gates pin test files that assert them; `current_state.supersession_note` marks them historical. No phase record was modified.

## Honest caveats

- Oracle attributions use realized values and say how much error a component can explain, not what a future model can recover.
- 2024 cohorts (targets, carries, rushing yards) are in-sample for the frozen allocators; 2025 validation receipts exist only for receiving yards and receptions.
- The Week 4 reconstruction uses only the frozen table; team/role split, availability and explosive tail are unknown there.
- The tier assignments combine computed shares with judged fields (confidence, validation path) recorded in the protocol; the judged fields are analyst assessments.

## Reproduction

```bash
pip install -r requirements-research.txt
python nfl_v2_phase1r_error_budget.py --out-dir /tmp/p1r-out   # stdlib only, reads frozen artifacts, no network
python -m pytest -q tests/test_nfl_v2_*.py
```
