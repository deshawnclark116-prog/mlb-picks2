# NFL player-outcome error budget: Phase 0 + Phase 0B correction

Research only. Production serving, champion models, model artifacts and API
behaviour are unchanged. No Phase 1 model is built here.

| File | Purpose |
|---|---|
| `nfl_player_outcome_error_budget_a.py` | diagnostic (coherent chains, 2^n oracle coalitions, Shapley attribution, pregame mixture, invariants) |
| `tests/test_nfl_player_outcome_invariants.py` | 7 synthetic unit checks + 103 data invariants; fails loudly |
| `nfl_models/nfl_player_outcome_error_budget_a/report.json` | all numbers below (and many more) |
| `nfl_models/nfl_player_outcome_phase1_protocol.json` | frozen clean-forward protocol (section J) |

```
python -u nfl_player_outcome_error_budget_a.py --data-dir /tmp/nfl_data            # ~2.5 min
python -u tests/test_nfl_player_outcome_invariants.py --data-dir /tmp/nfl_data     # ~3.5 min, includes perturbation test
```

**Phase 0 statements withdrawn by this correction**
- "~16.5 MAE is the practical ceiling for any pre-game model": withdrawn. That number is the
  *oracle-workload residual under the current efficiency estimator*. No experiment here measures
  irreducible noise.
- Sequential percentages (rushing 12% team / 22% share / 66% efficiency; receiving 5/3/20/15/19/38%):
  withdrawn. They were order-dependent, and the receiving ladder mixed two architectures.
  Replaced by order-robust Shapley attribution over coherent chains (section E).
- "Team red-zone volume adds almost nothing once player usage is known": replaced by an explicit
  conditional test (section E, TDs).
- "v2 leak": replaced by precise categories (section F).

## Protocol and universes

- Train: 2023 + 2024 wk1-12. Validation: 2024 wk13-18. Evaluation: 2025 and 2026 wk1-3.
- **Both evaluation periods are burned development data** (examined by the v2, v3, v4 and v5 gates,
  Phase 0 and now Phase 0B). Nothing in this README is an out-of-sample claim about the future.
- **Conditional universe**: players who played. Measures production given participation.
- **Pregame universe**: membership decided only from information before kickoff (last game was for
  this team within its previous 3 team games, >= 3 prior games, prior-usage eligibility).
  Non-participants score 0.
- No bookmaker numbers appear in any core feature set (enforced by invariant). The re-fit v2 recipe
  still contains spread/implied total and is labelled "incumbent" wherever it appears.

## Chains (all coherent; all-actual product reconstructs the outcome, max error 1.1e-13 over 42 chain tables)

| Family | Chain |
|---|---|
| RB rushing yards | T team rush attempts x S carry share x (E ordinary-carry yards/carry + X explosive(20+) yards/carry) |
| Receiving yards, route chain (2025) | D team dropbacks x R route rate x P targets/route x C catch rate x (A air yds/catch + Y YAC/catch) |
| Receiving yards, share chain (fallback, 2026) | TT team targets x H target share x C x (A + Y) |
| QB passing yards | P team plays x R dropback rate x Q attempts/dropback x C completion x (A air/cmp + Y YAC/cmp) |
| QB TDs, INTs | P x R x Q x K (per attempt) |
| QB rushing | P x U (QB rushes per team play) x V (yards/rush) |
| RB rushing TDs; receiving TDs | T x S x K; TT x H x K |
| Anytime TD | lambda = P x Z (red-zone play rate) x O (share of team RZ plays) x c_rz + N (non-RZ touches) x c_out; P(TD) = 1 - exp(-lambda) |
| Defense | snaps = OP x F; tackles = OP x F x Kt; sacks, INTs = OD x F x K |

An actual component whose denominator is 0 is undefined and keeps its predicted value; the
product is 0 whenever the upstream actual is 0, so reconstruction stays exact.
Predicted components use pre-game information only:
- team volumes: XGBoost on team history, no bookmaker inputs;
- shares: XGBoost on player usage history;
- rates: position-shrunk historical rates.

## C. Invariant results

7/7 unit checks pass. 103/103 data invariants pass. Selected details:

| Invariant | Result |
|---|---|
| train/valid/eval windows disjoint | pass |
| bookmaker spread/total absent from core feature sets | pass |
| no ORC_* variable in any feature set or fitted model | pass |
| no 2025/2026 row in any model fit (every fit logged in `report.json:model_fits`) | pass |
| perturbing week-10 2025 realized stats, pbp and team totals leaves week-10 features unchanged | pass: 0 of 524 rows changed |
| player-game rows unique (7 offensive families, defense) | pass |
| receptions <= targets; carries/targets/attempts >= 0; shares in [0,1] | pass (0 violations) |
| seasons without participation carry routes = None (never 0) | pass |
| targets <= route proxy | pass: 2 of 8,918 rows violate (Josh Downs 2025 wk9: 9 targets, 0 proxy routes; Khalil Shakir 2025 wk12: 10 / 0). Both are missing from participation `offense_players` for that game and are excluded from the route chain. |
| team pbp targets and rushes = sum of rostered players' pbp counts | pass: 0 / 2,268 team-games off |
| exact oracle reconstruction, 42 chain tables | pass: max abs error 1.1e-13 |
| pbp vs official box score | rushing yards identical; receiving yards differ on 0.5% of rows (mean abs 0.06) |
| pregame universe includes non-participants | pass: 2,198 rows |

## D. Corrected oracle tables (coalitions; loss = MAE, RMSE, within +/-20)

Rushing yards (RB). 2025, n = 854:

| Oracle set | MAE | RMSE | within 20 |
|---|---|---|---|
| none (all predicted) | 25.72 | 34.20 | 49.8% |
| T | 22.62 | 30.77 | 56.4% |
| S | 21.05 | 29.81 | 61.4% |
| E | 22.40 | 31.96 | 60.0% |
| X | 21.71 | 29.07 | 56.3% |
| T+S (workload) | 16.43 | 23.91 | 71.8% |
| E+X (efficiency) | 17.57 | 25.66 | 68.7% |
| T+S+E | 11.99 | 20.62 | 84.8% |
| T+S+X | 10.51 | 13.96 | 84.5% |
| all | 0 | 0 | 100% |

2026 wk1-3, n = 148: none 25.63; T+S 13.00; E+X 19.52; T+S+E 8.89; T+S+X 9.97.

Receiving yards, route chain. 2025, n = 2,292:

| Oracle set | MAE | RMSE | within 20 |
|---|---|---|---|
| none | 22.61 | 29.60 | 54.2% |
| D | 21.83 | 28.88 | 57.0% |
| R | 21.56 | 28.65 | 57.6% |
| P | 19.56 | 26.83 | 62.6% |
| C | 20.50 | 28.10 | 61.0% |
| A+Y | 18.61 | 26.62 | 65.8% |
| D+R+P (targets) | 16.43 | 22.31 | 70.0% |
| C+A+Y (per-target outcome) | 16.03 | 24.92 | 72.4% |
| D+R+P+C (receptions) | 13.05 | 18.54 | 78.4% |
| D+R+P+C+A | 8.68 | 12.69 | 90.3% |

Receiving yards, share chain. 2026 wk1-3, n = 390: none 22.59; TT+H 15.14; TT+H+C 12.05; TT+H+C+A 7.88; C+A+Y 17.66.

QB passing yards. 2025, n = 538:

| Oracle set | none | P+R | P+R+Q | P+R+Q+C | P+R+Q+C+A | C+A+Y |
|---|---|---|---|---|---|---|
| MAE | 66.52 | 56.95 | 40.04 | 32.97 | 24.50 | 53.94 |

## E. Order-robust (Shapley) attribution

phi = average marginal reduction in loss from making a component actual, over all 2^n coalitions.
Sum of phi = total removable loss (all predicted -> exact). Non-additivity = total minus the sum of
solo values. It is large everywhere because chain errors multiply.

### Rushing yards (RB)

| | 2025 MAE | 2025 RMSE | 2025 within20 | 2026 MAE | 2026 RMSE |
|---|---|---|---|---|---|
| total removable | 25.72 | 34.20 | +50.2 pts | 25.63 | 32.94 |
| T team rushes | 5.57 (21.6%) | 7.69 (22.5%) | 11.0 pts | 5.15 (20.1%) | 7.09 (21.5%) |
| S carry share | 8.00 (31.1%) | 10.57 (30.9%) | 16.4 pts | 10.86 (42.4%) | 13.65 (41.4%) |
| E ordinary-carry efficiency | 5.61 (21.8%) | 5.87 (17.2%) | 12.4 pts | 4.72 (18.4%) | 4.52 (13.7%) |
| X explosive-run process | 6.55 (25.5%) | 10.07 (29.4%) | 10.4 pts | 4.89 (19.1%) | 7.68 (23.3%) |
| **workload (T+S)** | **13.56 (52.7%)** | 18.26 (53.4%) | 27.4 pts | **16.02 (62.5%)** | 20.74 (63.0%) |
| **efficiency (E+X)** | **12.16 (47.3%)** | 15.94 (46.6%) | 22.8 pts | **9.61 (37.5%)** | 12.20 (37.0%) |
| non-additivity | 10.63 | 19.02 | 15.2 pts | 8.28 | 16.16 |
| largest pair interaction | T x S +1.53 | T x S +2.47 | | S x E +1.67 | T x S +3.01 |

Carries alone (T x S): all-predicted MAE 4.12 (2025) / 4.69 (2026). Shapley: S 60.2% / 70.1%, T 39.8% / 29.9%.

### Receiving yards, route chain (2025, n = 2,292)

| | MAE | RMSE | within20 |
|---|---|---|---|
| total removable | 22.61 | 29.60 | +45.8 pts |
| D team dropbacks | 2.42 (10.7%) | 3.78 (12.8%) | 4.4 pts |
| R route rate | 2.50 (11.1%) | 4.77 (16.1%) | 4.4 pts |
| P targets per route | 5.97 (26.4%) | 7.53 (25.4%) | 12.8 pts |
| C catch | 3.99 (17.6%) | 4.48 (15.1%) | 8.1 pts |
| A air yards per catch | 4.55 (20.1%) | 5.25 (17.7%) | 10.0 pts |
| Y YAC per catch | 3.17 (14.0%) | 3.78 (12.8%) | 6.1 pts |
| opportunity (D+R+P) | 10.90 (48.2%) | 16.09 (54.3%) | 21.6 pts |
| per-target outcome (C+A+Y) | 11.71 (51.8%) | 13.51 (45.7%) | 24.2 pts |
| non-additivity | 12.84 | 21.50 | 14.8 pts |
| largest pair interaction | A x Y +1.20 | R x P +2.02 | |

Share chain (same 2025 rows, different architecture, not comparable component-by-component):
opportunity (TT+H) 46.8%, catch 18.3%, yardage 34.9%. 2026 share chain: opportunity 51.5%,
catch 16.7%, yardage 31.8%.

Route-chain sub-targets, 2025 (MAE Shapley shares):

| Outcome | All-predicted MAE | Shares |
|---|---|---|
| receptions | 1.63 | D 16.6%, R 16.6%, P 40.3%, C 26.5% |
| targets | 2.10 | D 23.6%, R 23.1%, P 53.3% |
| routes (proxy) | 6.99 | D 50.5%, R 49.5% |

### QB (2025, n = 538; MAE Shapley shares)

| Outcome | All-predicted MAE | Team (P+R) | QB volume (Q) | Completion (C) | Yardage (A+Y) | Event rate (K) |
|---|---|---|---|---|---|---|
| passing yards | 66.52 | 29.2% | 30.2% | 13.0% | 27.6% | |
| attempts | 7.77 | 54.9% | 45.0% | | | |
| completions | 5.40 | 43.5% | 38.6% | 17.8% | | |
| passing TDs | 0.944 | 12.3% | 11.0% | | | 76.7% |
| interceptions | 0.654 | 9.5% | 9.6% | | | 80.9% |
| QB rush attempts | 1.59 | 12.3% | U 87.7% | | | |
| QB rush yards | 11.47 | 6.7% | U 53.9% | | V 39.5% | |
| QB rush TDs | 0.241 | 3.4% | U 16.8% | | | 79.8% |

Q (attempts per dropback) is heavy-tailed: the population includes games where a QB left injured
or was benched mid-game. 2026 shares are within 7 points of 2025 for every QB row.

### Touchdowns

Anytime TD chain (RB/WR/TE). logloss/Brier Shapley, conversion rates held predicted.

| | 2025 logloss | 2025 Brier | 2026 logloss | 2026 Brier |
|---|---|---|---|---|
| all predicted | 0.5960 | 0.2023 | 0.5513 | 0.1837 |
| all opportunities actual | 0.5071 | 0.1704 | 0.4389 | 0.1461 |
| P team plays | -0.0008 | -0.0002 | -0.0016 | -0.0009 |
| Z team red-zone rate | 0.0086 | 0.0034 | 0.0084 | 0.0030 |
| O player share of team RZ plays | 0.0761 | 0.0273 | 0.0920 | 0.0315 |
| N non-red-zone touches | 0.0050 | 0.0015 | 0.0136 | 0.0039 |
| team environment (P+Z) | 0.0078 (8.8%) | 0.0032 | 0.0068 (6.0%) | 0.0021 |
| player allocation (O+N) | 0.0811 (91.2%) | 0.0287 | 0.1056 (94.0%) | 0.0355 |

Explicit conditional test (reviewer item 4). Logloss with player allocation (O, N) actual:

| | 2025 | 2026 |
|---|---|---|
| team P, Z predicted | 0.5164 | 0.4456 |
| team Z actual only | 0.5053 | 0.4366 |
| team P, Z actual | 0.5071 | 0.4389 |

Given actual allocation, actual team red-zone rate still lowers logloss by 0.0111 (2025) and 0.0090
(2026), which is 2.2% / 2.0% of the remaining loss. Team play count adds nothing (slightly negative).

Conversion (player skill vs position average; logloss):

| | 2025 | 2026 |
|---|---|---|
| opportunities actual, player-shrunk conversion | 0.5071 | 0.4389 |
| opportunities actual, position-average conversion | **0.4828** | **0.4203** |
| all predicted, player-shrunk conversion | 0.5960 | 0.5513 |
| all predicted, position-average conversion | **0.5836** | **0.5484** |

The player-specific conversion estimates in this chain are worse than position averages. This
estimator finds no usable player conversion skill; better estimators are not tested here.
For reference, the Phase 0 direct XGBoost TD classifier scored 0.5791 (2025), which is better than
this mechanistic chain with naive components (0.5960). The chain is an attribution tool, not a
candidate model.

TD counts (MAE Shapley shares):

| Outcome | 2025 | 2026 |
|---|---|---|
| RB rushing TDs (T, S, K) | 7.5% / 10.5% / 82.1% | 8.5% / 14.1% / 77.5% |
| receiving TDs (TT, H, K) | 5.9% / 12.8% / 81.3% | 5.9% / 16.1% / 78.0% |

### Defense (2025, n = 8,017 player-games; 2026 n = 1,324)

| Outcome | All-predicted MAE | Opponent volume | Snap rate F | Per-snap rate K |
|---|---|---|---|---|
| defensive snaps | 10.47 | OP 29.5% | 70.5% | |
| tackles | 1.293 (last-3 average baseline 1.327) | OP 8.3% | 22.4% | 69.3% |
| sacks | 0.249 | OD 5.4% | 5.1% | 89.4% |
| interceptions | 0.112 | OD 3.8% | 3.1% | 93.1% |

2026 shares are within 3 points of these.

## F. Outcome-family coverage

| Player | Outcome | Baseline | Chain + Shapley | Distribution | Pregame mixture | Status |
|---|---|---|---|---|---|---|
| QB | attempts | chain | P, R, Q | no | no | supported |
| QB | completions | chain | P, R, Q, C | no | no | supported |
| QB | passing yards | blend avg, direct mean/median/quantile, incumbent recipe | P, R, Q, C, A, Y | yes | no | supported |
| QB | passing TDs | chain | P, R, Q, K | Poisson P(>=1) | no | supported |
| QB | interceptions | chain | P, R, Q, K | Poisson P(>=1) | no | supported |
| QB | rush attempts / yards / TDs | chain | P, U (, V / K) | no | no | supported |
| QB | sacks taken | no | no | no | no | not built (data exists: `sacks_suffered`) |
| RB | carries | chain | T, S | no | no | supported |
| RB | rushing yards | blend, direct, incumbent | T, S, E, X | yes | yes | supported |
| RB | rushing TDs | chain | T, S, K | Poisson P(>=1) | no | supported |
| RB | targets / receptions / receiving yards | in receiving family (RB rows) | route and share chains | yes (yards) | yes (yards) | supported; route proxy counts pass-blocking snaps as routes |
| RB | receiving TDs | chain | TT, H, K | Poisson | no | supported |
| RB | snaps | no | no | no | no | not built (snap_counts has it) |
| WR/TE | routes (proxy) | chain | D, R | no | no | 2022-2025 only |
| WR/TE | targets / receptions / receiving yards | blend, direct, incumbent | route (2025) and share (2026) chains | yes (yards) | yes (yards) | supported |
| WR/TE | receiving TDs | chain | TT, H, K | Poisson | no | supported |
| RB/WR/TE | anytime TD | Phase 0 classifier | P, Z, O, N (+ conversion test) | binary | no | supported |
| DEF | defensive snaps | chain | OP, F | no | no | supported |
| DEF | tackles (solo + with-assist) | last-3 average | OP, F, K | no | no | supported |
| DEF | sacks | chain | OD, F, K | Poisson | no | supported |
| DEF | interceptions | chain | OD, F, K | Poisson | no | supported |
| DEF | assists separately | no | no | no | no | not built (column exists) |
| DEF | pressures | no | no | no | no | **unsupported**: participation `was_pressure` is per play, not per defender, and ends 2025 |

## Direct models: mean vs median vs distribution (no bookmaker inputs unless labelled)

Rushing yards, 2025 (n = 854):

| Forecast | MAE | medAE | RMSE | bias | within 20 |
|---|---|---|---|---|---|
| historical blend average | 25.76 | 19.76 | 34.68 | -0.56 | 50.3% |
| core median (absolute error) | 24.87 | 18.19 | 35.01 | -6.42 | 54.2% |
| core mean (squared error) | 24.93 | 19.02 | 34.37 | -2.87 | 52.1% |
| core quantile q50 | 24.67 | 18.46 | 34.61 | -6.41 | 53.5% |
| incumbent v2 recipe (median, with bookmaker inputs) | 25.46 | 18.94 | 35.29 | -6.75 | 52.2% |

Quantile distribution: CRPS 18.22; 80% coverage 73.9% (width 68.9); 50% coverage 45.9% (width 36.0).
The quantiles are too narrow.

Receiving yards, 2025 (n = 2,294):

| Forecast | MAE | medAE | RMSE | bias | within 20 |
|---|---|---|---|---|---|
| blend average | 22.91 | 18.20 | 29.99 | +1.20 | 54.1% |
| core median | 22.21 | 16.90 | 29.98 | -3.94 | 58.5% |
| core mean | 22.83 | 18.65 | 29.71 | +1.32 | 53.8% |
| core q50 | 22.16 | 16.99 | 29.97 | -4.03 | 58.2% |
| incumbent recipe | 22.20 | 16.70 | 30.15 | -4.27 | 58.2% |

Quantile distribution: CRPS 15.88; 80% coverage 75.6%; 50% coverage 47.1%.

Passing yards, 2025 (n = 538): blend 67.10; core median 65.77 (bias +11.9); core mean 65.43 (RMSE 82.64,
bias +8.4); q50 64.98; incumbent recipe 66.40. CRPS 47.04; 80% coverage 72.7%.

The median model's negative bias (-6.4 rushing, -3.9 receiving) is the expected mean-median gap of
a right-skewed outcome. The mean model removes most of it (-2.9, +1.3) at +0.06 to +0.6 MAE. Both
must be published and scored with their own metrics.

## Pregame mixture (true pregame universe)

Pregame universe, rushing yards 2025: n = 994, 17.2% did not play. Receiving yards 2025: n = 2,638, 16.5% did not play.

| Rushing 2025 | MAE | RMSE | bias | CRPS |
|---|---|---|---|---|
| assume active (conditional mean applied to all) | 26.93 | 35.46 | +3.43 | 19.26 |
| mixture, T-24h (injury report + prior-week roster) | 23.35 (median 22.55) | 33.07 | -2.48 | 16.58 |
| mixture, T-90m (+ game-day roster status) | 22.20 (median 21.23) | 32.18 | -3.06 | 15.73 |

P(active) logloss: T-24h 0.230, T-90m 0.132. Questionable subset (n = 41, 70.7% played):
MAE 27.49 assume-active vs 24.00 mixture.

| Receiving 2025 | MAE | RMSE | bias | CRPS |
|---|---|---|---|---|
| assume active | 24.76 | 31.29 | +6.99 | 16.73 |
| mixture T-24h | 21.24 (median 20.37) | 28.75 | +1.42 | 14.47 |
| mixture T-90m | 20.25 (median 19.47) | 27.96 | +1.24 | 13.86 |

Questionable subset (n = 130, 73.1% played): MAE 28.94 assume-active vs 26.25 mixture.

2026 wk1-3 non-participation is higher (rushing 31.6%, receiving 27.2%). At week 1 the "previous 3 team
games" reach back to late 2025, which pulls in players who left in the offseason. For Phase 1 the
week 1-2 universe needs the current roster snapshot.

### Pre-kickoff availability sources

| Source | Available | Pre-game timing | Notes |
|---|---|---|---|
| injury report (`injuries_{y}`) | 2022-2026 | final report ~Friday: T-24h | status + practice participation |
| weekly roster status (`weekly_rosters/roster_weekly_{y}`) | 2022-2026 | **game-day snapshot**: 0.0% of 5,395 INA players and 0.01% of 8,225 RES players had snaps (2024-2025) | usable at T-90m, not T-24h; use previous week's status at T-24h |
| IR / PUP / reserve lists | via roster status RES + `status_description_abbr` (R01, R48, R04, R05 ...) | same snapshot | code meanings are not documented in the file; mapping must be verified before use |
| practice squad / elevations | roster status DEV (P01-P07 codes) | same snapshot | elevation vs standard PS not distinguishable without code documentation |
| suspensions | expected in RES codes | same snapshot | not verified |
| depth charts (`depth_charts/depth_charts_{y}`) | 2024 week-labelled (no timestamp); 2025 and 2026 daily timestamped snapshots (2025 from 2025-08-03) | 2025+: as-of by timestamp | not yet used |
| transactions (signings, trades, releases with dates) | **missing** | | no nflverse source; needs NFL transaction wire or ESPN transactions endpoint |
| game-day inactives with timestamp | **missing as a timed feed** | | only the post-hoc weekly roster snapshot |

## F (continued). Leakage and evaluation findings, separated by type

| Type | Finding | Consequence |
|---|---|---|
| Actual future leakage in production | none found | the served v2 model uses only completed games |
| Actual future leakage in research training | v3/v4/v5 built teammate-out / vacated-usage / defenders-out features from **realized same-game participation**. 69.8% of regular-player absences (1,222 of 1,750, 2024-2025; measured in the Phase 0 run on the same data files) were not announced Out/Doubtful before kickoff. | those research models trained on information unavailable at forecast time; their scores are optimistic |
| Evaluation contamination / in-sample comparator | the final v2 artifact is trained on 2023-2025 (valid for forecasting 2026). Scoring **that artifact** on 2025 gives MAE 21.46 rush / 20.92 rec vs 25.11 / 22.14 for the same recipe fit without 2025. Comparisons against it on 2025 (v3 gate `v2_mae`, first v5 run) were invalid. | comparisons must use a re-fit comparator |
| Evaluation reuse | 2025 and 2026 wk1-3 examined by many gates | development data; p-values on them overstate confidence |
| In-sample calibration | v2 `actual_over_projection_quantiles` (served p10/p90, `safe_line_90`) come from residuals on the 2025 + 2026 evaluation games | interval calibration never tested out of sample |
| Model-version mismatch | those quantiles were computed from the pre-refit model's residuals and applied to the refit model | ranges belong to a different model than the one served |
| In-sample calibration | TD `TdProjector` Platt map fit on 2025 | any 2025 calibration figure for it is in-sample; serving 2026 is fine |
| Valid use of prior completed weeks | walk-forward refits on completed weeks, 2025 in the final production fit | not a leak |
| Minor information timing | v3 uses observed game-time weather from `games.csv` | forecast weather should replace it |
| Universe | every earlier gate scored only players who played | the pregame universe (above) is required for the final objective |

## H. Corrected conclusions: workload vs efficiency

1. Under order-robust attribution, **workload and efficiency are comparable** error sources for
   rushing, not 34/66. Workload (T+S) is 52.7% of removable MAE in 2025 and 62.5% in 2026;
   efficiency (E+X) is 47.3% / 37.5%. Oracle workload alone gives MAE 16.43, oracle efficiency
   alone 17.57.
2. For route-chain receiving yards, opportunity (D+R+P) is 48.2% of MAE and 54.3% of RMSE.
   Targets per route (P) is the largest single component (26.4% MAE).
3. The error that remains after oracle workload (rushing 16.4, receiving 16.4) is **the residual of the
   current efficiency estimators** (position-shrunk historical rates). It is not shown to be
   irreducible. Defensive front and box, OL/DL health, coverage, pressure, route type, QB accuracy,
   alignment, YAC environment and game state have not been tested at the efficiency layer.
4. Every chain shows large non-additivity (rushing MAE 10.6 of 25.7; receiving 12.8 of 22.6).
   Components compound multiplicatively, so improving any one component helps more when the
   others are also accurate.
5. Explosive-run yards (X) carry 25.5% of rushing MAE and 29.4% of RMSE in 2025. The tail process
   deserves its own model.
6. For event counts (TDs, INTs, sacks) the per-opportunity rate carries 77-95% of MAE under
   current estimators. For anytime TD, player red-zone allocation (O) dominates (0.076 of 0.089
   logloss in 2025). Team red-zone rate adds a small but non-zero 0.009-0.011 logloss even given
   actual allocation. Position-average conversion beats player-specific conversion.
7. Treating everyone as active overstates pregame forecasts. The P(active) mixture cuts pregame
   rushing MAE from 26.93 to 23.35 at T-24h, and receiving from 24.76 to 21.24.

## Defensive scheme: where it belongs (to be tested at component level in Phase 1)

v4 showed that adding scheme features to a direct final-yards regression did not improve it. That does
not establish that scheme carries no information. Planned component placement:

| Scheme input | Component |
|---|---|
| box count, front, stacked-box rate, DL/LB availability, run-fit tendencies | E (ordinary-carry efficiency), X (explosive-run probability) |
| coverage shell (Cover 0/1/2/3/4/6), man/zone, safety/corner availability, slot/outside alignment | P (targets per route), C (catch), A (air yards), Y (YAC) |
| pressure, blitz rate, sack generation | QB C, A, Kint, sacks taken; RB/TE target share on checkdowns |
| red-zone defense, goal-line front | Z (red-zone rate), c_rz (conversion) |
| opponent pass-rate tendency | R (dropback rate), D |

Scheme features are **not** frozen as failed.

## G. Unsupported or unestablished

- True routes run: the proxy counts every dropback a player is on the field (RB pass-blockers included);
  2026 participation is unpublished.
- Pressures by defender, alignment (slot/wide) by play for 2026, run concept, personnel packages for 2026.
- Transactions with timestamps; documented meanings of roster status codes (IR, PUP, suspension, elevation).
- Offensive coordinator / play-caller changes (only head coach is in `games.csv`).
- Irreducible noise: no experiment here isolates it. Candidate design for Phase 1: repeated-situation
  variance, e.g. same player, same box count, same down/distance/field position.
- QB sacks taken, RB/WR snaps, defensive assists: data exists, not built.

## I. Phase 1 build order

1. **Availability layer**: P(active) at T-24h and T-90m; pregame universe with the week 1-2 roster fix;
   research features switched from realized to pre-game availability.
   Gate: logloss < status-lookup baseline.
2. **Team environment**: plays, dropback rate, rushes, targets, red-zone rate; count/beta models with
   shrunk team effects; no bookmaker inputs. Today's XGBoost ties the team blend average
   (plays MAE 6.84 vs 6.81), so this layer needs a different design, not more of the same.
3. **Role/share layer**: carry share, route rate, targets per route, QB attempt share, red-zone share;
   shares coherent within team (sum to 1); role state (changepoint/HMM) with draft/depth-chart
   priors from timestamped snapshots. This is the largest Shapley block for rushing (S 31-42%) and
   receiving (P 26%).
4. **Efficiency layer with scheme at the right stage**: base + explosive mixture for rushing; catch,
   air and YAC for receiving; completion, air and YAC for QB. Box/front -> E, X; coverage -> P, C, A, Y;
   pressure -> QB C, A, INT.
5. **Event layer**: TDs from red-zone allocation x position-level conversion (plus a tested player
   term); INTs and sacks from exposure x rate.
6. **Game simulation** combining 1-5 with teammate-coherent allocation; publish mean, median and
   quantiles. Calibration target: 80% coverage 0.75-0.85; today's direct quantile models are at 0.73-0.76.
7. Shadow-mode forward scoring under the frozen protocol.

## J. Clean-forward protocol (frozen)

`nfl_models/nfl_player_outcome_phase1_protocol.json`, version 1.0, frozen 2026-09-29T02:15Z. Key terms:
- Burned: 2022-2024 (train/valid), 2025, 2026 wk1-3.
- Forward window: the first 2026 week in which every game kicks off after the Phase 1 freeze record
  (commit SHA + file hashes + UTC time), never earlier than week 5, through week 18. Interim verdicts
  cannot change the version; at least 4 complete weeks are required for a verdict.
- Forecast times: T-24h primary, T-90m secondary. Scored universe: pregame (non-participants = 0).
- Forbidden inputs: any sportsbook number, same-game realized values, realized teammate/defender
  participation, observed game-time weather.
- Metrics:
  - median: MAE, medAE, pinball;
  - mean: bias, RMSE, MSE;
  - distribution: CRPS, 50/80% coverage and width, PIT;
  - counts: Poisson deviance;
  - binary: logloss, Brier, ECE (AUC as ranking only).
- Component gates: each component vs its baseline, week-block bootstrap, p < 0.10, else fall back to
  baseline (decided once, at window end).
- End-to-end gates for primary outcomes (RB rushing yards, receiving yards, receptions, QB passing yards,
  anytime TD):
  - beat both the blend baseline and the incumbent at p < 0.05 on MAE;
  - RMSE not worse;
  - CRPS better;
  - 80% coverage in [0.75, 0.85] and 50% coverage in [0.45, 0.55];
  - TD: logloss and Brier better, and ECE <= 0.03.
- Retraining inside a version: weekly walk-forward refit only. Any feature, hyperparameter, gate,
  universe or target change is a new version and restarts the window. No rescue-tuning.
- Production stays unchanged. Phase 1 runs in shadow mode until the end-to-end gates pass.
