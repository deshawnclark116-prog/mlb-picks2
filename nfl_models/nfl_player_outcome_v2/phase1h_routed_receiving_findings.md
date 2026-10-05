# Phase1H-R — routed receiving matchup mechanics

**Verdict: REJECTED_EFFICIENCY_REPLACEMENT. Frozen research failure; no selected architecture.**

No family earned combination, a more complex learner, or a production replacement. Phase1G remains rejected; its explicit targets × catch probability × (completed air + YAC) decomposition is retained. Nothing was rescue-tuned after 2025.

## Timing and source audit

The source-only audit and protocol were committed before performance (`9fee349`); implementation/tests before scoring (`d3dfdf9`); the 2024 development lock before validation (`3665585`). These are original local execution commit IDs; publication preserves their tree/order. The protocol and development specifications are content-hashed in the frozen snapshot.

2024 W1–8 fit (1,008 meaningful rows), W9–18 select (1,303 rows; 1,265 positive-target efficiency rows), then refit each selected single-family specification on all 2024 for fixed descriptive ablations. **Every family already failed development.** Refit ablations cannot become selected after seeing validation. 2025 was accessed only after selection froze. 2026 W1–4 is burned diagnostic evidence; W5+ is excluded before indexing and never scored.

| Season | PBP target air yards | Man/zone labeled plays | True-pressure labeled plays | FTN blitz labels | Defensive personnel |
|---|---:|---:|---:|---:|---|
| 2023 | 17483/17483 | 22916 | 46168 | 48225 | BLOCKED_DATA |
| 2024 | 17013/17013 | 22408 | 45905 | 48031 | BLOCKED_DATA |
| 2025 | 16609/16609 | 22055 | 45175 | 47316 | BLOCKED_DATA |
| 2026 | 3936/3936 | NOT PUBLISHED | NOT PUBLISHED | 8246 | BLOCKED_DATA |

Counts above are file-level availability, including postseason charting; features use REG prior games only. Actual joined REG dropbacks: pressure and blitz cover 100% in 2023–2025. 2026 true-pressure joins: 0/4,646; FTN blitz joins: 3,590/4,646. FTN currently covers all W1–3 and only one W4 game (49 games vs 63 PBP games). Missing labels never become negative pressure/blitz observations. Missing opponent evidence gives a neutral feature.

2026 man/zone and true-pressure history is explicitly **STALE_PRIOR_SEASON_FALLBACK**, never current coverage. Receiver scheme/pressure splits are limited to the latest 150 labeled targets; opponent rates to the latest 500 labeled dropbacks. All obey strict earlier-week and earlier-game-date rules. FTN W4 incompleteness cannot provide target-week information to W4 forecasts.

**Source-vintage limit:** retrospective release timestamps do not prove original per-play publication before historical cutoffs. This is a completed-game-history research replay, not a claim of archived T24/T90 snapshot equivalence. Injuries have modification dates in 2023–24 (none populated in 2025–26), but no proven original publication/cutoff snapshots, so defensive absences are blocked. No target-game participant/absence oracle is used for efficiency. Frozen Phase1E opportunity receipts retain its documented roster assumption and are not newly certified availability evidence.

## Routed candidates

Player-only mechanical R0 uses prior eight player games, 25 target-equivalent position shrinkage, last 1,500 prior position targets, and no opponent blend. Stable GSIS skill history crosses teams. Meaningful-population role eligibility uses prior three **current-team** games, not future workload.

- **Depth/explosive → completed air only:** receiver aDOT relative to position; deep-target share × opponent positional completed-deep-air allowance; receiver completed-air ≥20 share × opponent air-explosive suppression. Total 20-yard reception rate is recorded, but not allowed to smuggle YAC into air fitting.
- **Pressure/blitz → catch only:** opponent true-pressure and FTN blitz deviations; pressure exposure × shrunken receiver pressured/clean catch split. Sack rate is not called a pressure label.
- **Man/zone → catch and completed air:** opponent man propensity relative to receiver historical scheme mix × receiver shrunken component splits. No final-yard scheme feature.
- **YAC matchup → YAC only:** opponent positional YAC/catch relative to position; receiver YAC profile × opponent ≥10-YAC allowance.

Catch uses binomial logistic ridge with player catch-rate offset; air/YAC use weighted ridge residuals on conditional completed air/catch and YAC/catch. There is no intercept, generic final-yard correction, tree learner, Monte Carlo, or sportsbook data. Train-only RMS scaling; penalties 10/100/1,000; component bounds fixed before scoring. Conditional models weight actual catches, catch fitting weights actual targets. Final YPT is reconstructed mechanically.

## 2024 development decisions

| Family | Penalty | Oracle-yard MAE | Delta vs R0 (95% blocked CI) | Decision |
|---|---:|---:|---|---|
| depth_explosive | 1000 | 17.4975 | +0.0695 [+0.0130, +0.1254] | REJECTED |
| man_zone | 10 | 17.4077 | -0.0202 [-0.0473, +0.0070] | REJECTED |
| pressure_blitz | 1000 | 17.4843 | +0.0564 [+0.0181, +0.0988] | REJECTED |
| yac_matchup | 1000 | 17.4279 | -0.0001 [-0.0084, +0.0085] | REJECTED |

R0 oracle MAE 17.4279; incumbent 17.1100; simple player history 18.3203. Required ≥0.5% practical gain, YPT no worse, every claimed component ≥0.5% better, and paired block CI upper <0. No family passes. Thus no combinations were run. Rejected specifications and their selected penalties are frozen.

Bootstrap: 2 consecutive calendar/NFL-week moving blocks within season, keeping whole games and all paired player rows, 2,000 resamples, seed 164. CIs are descriptive, not multiple-comparison-adjusted certainty claims; 2026 spans only four weeks and is especially weak evidence.

## Fixed validation / diagnostic results

All primary efficiency metrics use the same paired positive-target rows: **2025 n=2,175; 2026 n=464**. Zero-target meaningful rows remain in the full receipt universe. This avoids undefined YPT and easy-zero dilution. Incumbent is the frozen Phase1F/Phase1B efficiency formula re-evaluated on this cohort. Prior Phase1F oracle headline summaries (16.492 / 16.178) used different denominators/availability gates and cannot be directly mixed with these paired results. Those frozen records are unchanged. Scalar incumbent/history estimators do not expose catch/air/YAC decomposition; unavailable component metrics are not invented.

### diagnostic_2026_wk1_4

| Estimator | Oracle-yard MAE | YPT MAE | Catch-rate MAE | Air/target MAE | YAC/target MAE | >40 oracle miss rate |
|---|---:|---:|---:|---:|---:|---:|
| depth_explosive | 16.7476 | 3.6774 | 0.2061 | 2.9528 | 2.1188 | 8.84% |
| history | 17.4856 | 3.8040 | — | — | — | 10.34% |
| man_zone | 16.7693 | 3.6807 | 0.2061 | 2.9542 | 2.1187 | 9.05% |
| mechanical | 16.7548 | 3.6781 | 0.2061 | 2.9532 | 2.1188 | 9.05% |
| phase1f | 16.6658 | 3.6691 | — | — | — | 7.97% |
| phase1g | 16.5885 | 3.6493 | 0.2053 | 2.9651 | 2.1211 | 8.41% |
| pressure_blitz | 16.7489 | 3.6786 | 0.2062 | 2.9543 | 2.1198 | 9.05% |
| yac_matchup | 16.7708 | 3.6819 | 0.2061 | 2.9532 | 2.1209 | 8.84% |

| Family | Oracle delta vs incumbent | 95% blocked CI |
|---|---:|---|
| depth_explosive | +0.0818 | [-0.0383, +0.2232] |
| man_zone | +0.1035 | [+0.0330, +0.1867] |
| pressure_blitz | +0.0831 | [-0.0146, +0.1982] |
| yac_matchup | +0.1050 | [+0.0293, +0.1943] |

| Slice | n | R0 | Depth | Pressure/blitz | Man/zone | YAC | Incumbent |
|---|---:|---:|---:|---:|---:|---:|---:|
| WR | 272 | 19.847 | 19.828 | 19.839 | 19.870 | 19.867 | 19.800 |
| TE | 122 | 13.579 | 13.590 | 13.578 | 13.582 | 13.610 | 13.374 |
| RB | 70 | 10.275 | 10.280 | 10.269 | 10.277 | 10.249 | 10.226 |
| early | 464 | 16.755 | 16.748 | 16.749 | 16.769 | 16.771 | 16.666 |
| established | 0 | — | — | — | — | — | — |

### validation_2025

| Estimator | Oracle-yard MAE | YPT MAE | Catch-rate MAE | Air/target MAE | YAC/target MAE | >40 oracle miss rate |
|---|---:|---:|---:|---:|---:|---:|
| depth_explosive | 17.0473 | 3.7849 | 0.1996 | 2.7999 | 2.2560 | 9.06% |
| history | 17.6614 | 3.8879 | — | — | — | 9.84% |
| man_zone | 17.0535 | 3.7858 | 0.1996 | 2.8022 | 2.2558 | 8.92% |
| mechanical | 17.0554 | 3.7860 | 0.1996 | 2.8023 | 2.2560 | 8.97% |
| phase1f | 16.8751 | 3.7622 | — | — | — | 8.69% |
| phase1g | 16.8475 | 3.7572 | 0.1993 | 2.8082 | 2.2476 | 8.64% |
| pressure_blitz | 17.0654 | 3.7878 | 0.1995 | 2.8036 | 2.2568 | 8.87% |
| yac_matchup | 17.0475 | 3.7840 | 0.1996 | 2.8023 | 2.2552 | 8.97% |

| Family | Oracle delta vs incumbent | 95% blocked CI |
|---|---:|---|
| depth_explosive | +0.1722 | [-0.0012, +0.3416] |
| man_zone | +0.1784 | [+0.0153, +0.3364] |
| pressure_blitz | +0.1903 | [+0.0248, +0.3519] |
| yac_matchup | +0.1724 | [+0.0081, +0.3354] |

| Slice | n | R0 | Depth | Pressure/blitz | Man/zone | YAC | Incumbent |
|---|---:|---:|---:|---:|---:|---:|---:|
| WR | 1285 | 19.545 | 19.533 | 19.565 | 19.539 | 19.535 | 19.385 |
| TE | 540 | 14.500 | 14.505 | 14.492 | 14.505 | 14.496 | 14.236 |
| RB | 350 | 11.860 | 11.843 | 11.860 | 11.860 | 11.853 | 11.732 |
| early | 478 | 16.930 | 16.951 | 16.943 | 16.939 | 16.914 | 16.798 |
| established | 1697 | 17.091 | 17.075 | 17.100 | 17.086 | 17.085 | 16.897 |

No routed family independently improves its claimed component materially. Small diagnostic gains do not reopen rejection. **Full receiving-yard metrics are NOT_RUN_EFFICIENCY_GATE_FAILED**. Predicted Phase1E targets and mechanically reconstructed direct yard projections remain in each receipt for explanation, but are not scored to seek error cancellation.

## Data-quality / receipts

Official player stats define yard/count labels. PBP components reconcile exactly with targets/receptions/yards on 2,279/2,311 development-year rows, 2,202/2,229 validation rows, 473/480 diagnostic rows. Nonreconciled component labels are excluded only from component fitting/metrics; their official total-yard/YPT metrics remain. Each metric reports its own denominator. Eight meaningful rows lack a common comparator prediction and are explicitly counted in data_quality. Two 2025 zero-target rows lack Phase1E opportunities; receipts retain null predicted targets rather than fabricate them.

The deterministic gzip receipt ledger contains all 10,836 fixed-ablation player rows, including zero-target rows, four architectures × (2,229 + 480). Each retains stable identity, team/opponent/week, actual labels, predicted targets, baseline catch/air/YAC, separate component adjustments, final YPT/direct-yard projection, raw routed features, source staleness, and last-history dates/weeks. This is a new research evaluation ledger; existing forecast/scientific stores are untouched.

## Dominant error and next genuinely new information

Post-selection descriptive oracle substitutions on the **same reconciled positive-catch cohort**, never model features or promotion metrics:

| Period | n | R0 oracle MAE | Actual catch rate supplied | Actual completed air/catch supplied | Actual YAC/catch supplied |
|---|---:|---:|---:|---:|---:|
| diagnostic_2026_wk1_4 | 436 | 17.164 | 13.896 | 13.542 | 15.869 |
| validation_2025 | 2041 | 17.243 | 13.924 | 13.769 | 14.842 |

**Completed air production dominates**, with catch conversion close behind; component swaps are not additive attribution. WR air/target error dominates the positional yard decomposition; RB YAC remains a separate limitation. Reweighting coarse opponent rates/aDOT is insufficient. A genuinely new preregistered experiment would need:

1. Depth-binned QB/receiver **catchability, contested/drop, and throw-accuracy history** rather than aDOT alone. FTN has additional relevant charting labels, not tested or selected here.
2. Receiver route-level **coverage leverage/separation and intended depth**, with original historical publication/cutoff evidence, to model which downfield targets become catches rather than average final yards.
3. Current 2026 coverage/pressure data and archived pregame defensive personnel snapshots before those can be represented as current matchup evidence.

The general football families are not disproven; these concrete routed specifications failed. No retuning on 2025/2026, no complex-model rescue, no scientific promotion. Week5+ remains clean-forward evidence and requires a new frozen protocol/real pregame receipts.

## Scope and validation

Only new Phase1H research code/tests/workflow/artifacts and the research registry changed. Frozen Phase1A–G artifacts, production NFL prediction/simulation/scheduler/grading, frontend, and CFB/MLB/NHL/Tennis are byte-identical to c9d532e. Deterministic tests and workflow reproduction checks cover routing, target-week poisoning, stable skill identity, no sportsbook dependence, no 2025 fitting/rescue, gate discipline and protected-file scope. PR #64 stays draft and unmerged.
