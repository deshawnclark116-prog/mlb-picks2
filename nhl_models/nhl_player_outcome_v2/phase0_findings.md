# NHL Outcome Engine V2 - Phase0 findings (research only, promotes nothing)

Evidence grade of EVERY number below: **RETROSPECTIVE_BURNED**. There is no untouched historical NHL season (see `phase0_chronology_audit.json`); genuine confirmation waits for clean forward 2026 data, which this phase did not read.

## Verdicts (two questions kept separate)

| market | native binary-target verdict | architecture verdict |
|---|---|---|
| goalie_saves_ge25 | BLOCKED_TIMING | ARCHITECTURE_INSUFFICIENT_NO_CENTRAL_PROJECTION |
| skater_points_ge1 | CLASSIFIER_WEAK_NATIVE_TARGET | ARCHITECTURE_INSUFFICIENT_NO_CENTRAL_PROJECTION |
| skater_sog_ge3 | CLASSIFIER_WEAK_NATIVE_TARGET | ARCHITECTURE_INSUFFICIENT_NO_CENTRAL_PROJECTION |
| team_moneyline | CLASSIFIER_WEAK_NATIVE_TARGET | ARCHITECTURE_SUFFICIENT (binary target; no count projection expected) |

CURRENT_ENGINE_HAS_NO_CENTRAL_STAT_PROJECTION for SOG, points and goalie saves: the engine emits only P(SOG>=3), P(points>=1), P(saves>=25) and P(home win). No mean was reverse-engineered from those probabilities.

## Native-target performance (as-served causal calibration), Brier / log loss / ECE / BSS vs FIT base rate

| market | season | n | base rate | Brier | BSS | log loss | ECE | slope | AUC (secondary) | best baseline (Brier) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| skater_sog_ge3 | 2024 | 35920 | 0.2346 | 0.1563 | 0.1366 | 0.4807 | 0.0141 | 0.994 | 0.736 | D 0.1567 | WEAK |
| skater_sog_ge3 | 2025 | 37055 | 0.2299 | 0.1533 | 0.1433 | 0.4728 | 0.0062 | 1.032 | 0.741 | D 0.1539 | SURVIVES |
| skater_points_ge1 | 2024 | 35920 | 0.3571 | 0.2077 | 0.0953 | 0.6037 | 0.0075 | 1.033 | 0.682 | D 0.2081 | WEAK |
| skater_points_ge1 | 2025 | 37055 | 0.3642 | 0.2093 | 0.0961 | 0.6072 | 0.0060 | 1.003 | 0.683 | D 0.2095 | WEAK |
| team_moneyline | 2024 | 1238 | 0.5646 | 0.2404 | 0.0268 | 0.6736 | 0.0257 | 0.882 | 0.592 | B_margin_logistic 0.2431 | SURVIVES |
| team_moneyline | 2025 | 1245 | 0.5245 | 0.2482 | 0.0048 | 0.6896 | 0.0295 | 0.793 | 0.549 | B_margin_logistic 0.2489 | WEAK |

Paired block-bootstrap Brier difference (incumbent minus baseline; negative = incumbent better), SOG and points:

| market | season | vs A (rate) | vs B (player rate) | vs C (+opp/home) | vs D (human count map) |
|---|---|---|---|---|---|
| skater_sog_ge3 | 2024 | -0.02474 [-0.02613, -0.02338] | -0.00219 [-0.00276, -0.00167] | -0.00188 [-0.00240, -0.00139] | -0.00036 [-0.00093, 0.00018] |
| skater_sog_ge3 | 2025 | -0.02564 [-0.02698, -0.02434] | -0.00198 [-0.00242, -0.00153] | -0.00163 [-0.00207, -0.00121] | -0.00064 [-0.00105, -0.00020] |
| skater_points_ge1 | 2024 | -0.02189 [-0.02351, -0.02038] | -0.00272 [-0.00324, -0.00217] | -0.00231 [-0.00285, -0.00178] | -0.00044 [-0.00091, 0.00003] |
| skater_points_ge1 | 2025 | -0.02225 [-0.02410, -0.02031] | -0.00238 [-0.00296, -0.00181] | -0.00235 [-0.00287, -0.00183] | -0.00021 [-0.00073, 0.00031] |

Reading: the incumbent classifiers clear constant-rate and player-rate baselines by a small margin (~0.002 Brier) but their edge over a transparent frozen human opportunity chain (expected TOI x shrunk shot/point rate x damped opponent factor, mapped to a threshold probability by a locked Poisson) is about 0.0002-0.0006 Brier and its confidence interval touches zero in three of four skater cases. The gradient-boosted classifier adds almost nothing beyond the opportunity chain it implicitly learns.

## Central baselines (the floor Phase1 must beat)

| stat | season | baseline | MAE | bias | median AE | within 1 | within 2 | miss >3 |
|---|---|---|---|---|---|---|---|---|
| SOG | 2024 | human_frozen | 1.0672 | 0.0882 | 0.9378 | 0.532 | 0.891 | 0.0302 |
| SOG | 2024 | season_rate_x_prior3_toi | 1.0593 | 0.0189 | 0.9005 | 0.557 | 0.879 | 0.0340 |
| SOG | 2024 | prior10_mean | 1.0826 | 0.0001 | 0.9000 | 0.581 | 0.877 | 0.0366 |
| SOG | 2024 | ewma_0.2 | 1.0870 | -0.0003 | 0.8893 | 0.555 | 0.866 | 0.0381 |
| SOG | 2025 | human_frozen | 1.0599 | 0.1024 | 0.9316 | 0.537 | 0.892 | 0.0287 |
| SOG | 2025 | season_rate_x_prior3_toi | 1.0512 | 0.0177 | 0.8764 | 0.565 | 0.881 | 0.0322 |
| SOG | 2025 | prior10_mean | 1.0726 | 0.0075 | 0.9000 | 0.585 | 0.879 | 0.0342 |
| SOG | 2025 | ewma_0.2 | 1.0781 | 0.0085 | 0.8800 | 0.559 | 0.868 | 0.0371 |
| points | 2024 | human_frozen | 0.5467 | 0.0128 | 0.4391 | 0.905 | 0.983 | 0.0023 |
| points | 2024 | season_rate_x_prior3_toi | 0.5337 | 0.0074 | 0.4251 | 0.896 | 0.985 | 0.0020 |
| points | 2024 | prior10_mean | 0.5359 | 0.0003 | 0.4000 | 0.898 | 0.986 | 0.0018 |
| points | 2024 | ewma_0.2 | 0.5372 | 0.0004 | 0.4231 | 0.887 | 0.985 | 0.0019 |
| points | 2025 | human_frozen | 0.5539 | 0.0050 | 0.4425 | 0.900 | 0.981 | 0.0027 |
| points | 2025 | season_rate_x_prior3_toi | 0.5410 | -0.0001 | 0.4299 | 0.893 | 0.983 | 0.0023 |
| points | 2025 | prior10_mean | 0.5459 | 0.0005 | 0.4000 | 0.894 | 0.985 | 0.0021 |
| points | 2025 | ewma_0.2 | 0.5474 | -0.0002 | 0.4330 | 0.882 | 0.984 | 0.0022 |

SOG count error is close to the variance a perfect-mean count model would still show: MSE / mean(mu) = 1.079 (2024) and 1.071 (2025) against a locked dispersion index of 1.107. Better means (not better distributions) are the only upside, and means depend on role information (PP unit, line, deployment) that the historical production data does not contain.

## Opportunity vs efficiency (ORACLE diagnostics: postgame, descriptive only)

SOG 2025: TOI prediction MAE 1.96 min. Replacing predicted TOI with the actual TOI removes only 0.043 of MSE; the actual-rate oracle removes 0.934 but that oracle absorbs the outcome itself (rate = shots / TOI), so it mixes rate predictability with irreducible count variance and is not recoverable skill. Deployment (TOI) uncertainty is minor next to shot-count variance given the available inputs; the missing PP/line state is the untested lever.

Goalie saves 2025 (B): shots-against prediction MAE 5.98 (bias -1.16); actual shots-against oracle removes 0.957 of MSE, actual save-percentage oracle removes 0.203: volume faced, not save percentage, is the dominant reducible component, and that volume depends on the unknown starter and game script.

## Goalie eligibility leakage (resolved before reporting goalie results)

- Code path: `nhl_goalie_saves_clean_baseline_a.py::build_rows`, `WHERE gg.toi_seconds >= 1800` on `goalie_games.toi_seconds of the TARGET game` (also in `nhl_serving_builder_a.build_goalie_final_states (serving state)`). Training population: **changed**; scoring population: **changed**; labels: unchanged; features: unchanged (history is prior qualifying appearances only).
- Own re-implementation reproduces the incumbent exactly: same population = True, max feature difference = 0.0 (n = 15345).
- Rows removed by the filter per season (2018-2025): 2018: 123, 2019: 107, 2020: 74, 2021: 130, 2022: 114, 2023: 135, 2024: 90, 2025: 113; their mean saves are ~7-9 and essentially none reach 25. The filter inflates the OVER base rate by ~2-4 points (2025: A 0.471 vs B 0.448).
- Diagnostic verdicts: A (as implemented) = CLASSIFIER_SURVIVES_NATIVE_TARGET_AUDIT; B (no target-TOI filter, still conditioned on appearing) = CLASSIFIER_WEAK_NATIVE_TARGET. **Final goalie status: BLOCKED_TIMING** - no timestamped pregame starter exists, so no real-world deployment claim is made.

The preregistered FIT-constant baselines are handicapped on goalie saves by event-rate drift (saves>=25 rate fell from ~0.66 to ~0.47). A POST-HOC sensitivity (declared after first results, not used for any verdict) re-centred baselines on the previous season's rate; the incumbent still wins by 0.002-0.016 Brier, with one interval (population B, 2025, vs the rate-shrunk baseline) touching zero.

## Production data-quality finding

The production `skater_games` table holds only 2018: 0.878, 2019: 0.898, 2020: 0.920, 2021: 0.886, 2022: 0.856, 2023: 0.899, 2024: 0.852, 2025: 0.879 of the official boxscore skater rows (toi>0) counted in the V1 acquisition. Rolling histories and games-played counts in the incumbent features are therefore understated for many players, and ~40% of regulation games show total goals not matched by credited skater goals. This is a data ceiling on every number above and on any Phase1 model built from this table. Shootout shots/goals are not in skater stats, so shootouts do not contaminate player-stat grading; 68 goalie rows break saves = shots against - goals against.

## Role / availability data

Historical line combinations, PP assignment, PP TOI, scratch state and confirmed starter for the TARGET game: **BLOCKED_TIMING**. Prior-game EV/PP TOI and games-started exist as postgame box scores (V1 acquisition on `codex/nhl-outcome-engine-v1`, not in the production DB) and are legitimate rolling-history features. Forward capture (Phase0C) exists and was not opened.

## Catastrophic misses (evidence-flag attribution only; flags overlap)

- `goalie_saves_ge25_A|2024|CENTRAL_ABS_ERR_GT_8`: n=479, flags={"SHOTS_FACED_ABOVE_EXPECTED": 346, "SHOTS_FACED_BELOW_EXPECTED": 83, "UNATTRIBUTED": 50}
- `goalie_saves_ge25_A|2025|CENTRAL_ABS_ERR_GT_8`: n=498, flags={"SHOTS_FACED_ABOVE_EXPECTED": 329, "SHOTS_FACED_BELOW_EXPECTED": 104, "UNATTRIBUTED": 65}
- `goalie_saves_ge25_B|2024|CENTRAL_ABS_ERR_GT_8`: n=559, flags={"GOALIE_UNDER_30_MIN_PULLED_OR_RELIEF": 81, "SHOTS_FACED_ABOVE_EXPECTED": 346, "SHOTS_FACED_BELOW_EXPECTED": 164, "UNATTRIBUTED": 49}
- `goalie_saves_ge25_B|2025|CENTRAL_ABS_ERR_GT_8`: n=611, flags={"GOALIE_UNDER_30_MIN_PULLED_OR_RELIEF": 110, "SHOTS_FACED_ABOVE_EXPECTED": 331, "SHOTS_FACED_BELOW_EXPECTED": 209, "UNATTRIBUTED": 66}
- `skater_points_ge1|2024|CENTRAL_ABS_ERR_GT_2`: n=607, flags={"HIGH_TEAM_SCORING_GAME": 323, "SHOT_VOLUME_ABOVE_EXPECTED": 166, "TOI_ABOVE_EXPECTED": 26, "UNATTRIBUTED": 201}
- `skater_points_ge1|2025|CENTRAL_ABS_ERR_GT_2`: n=694, flags={"HIGH_TEAM_SCORING_GAME": 417, "SHOT_VOLUME_ABOVE_EXPECTED": 207, "TOI_ABOVE_EXPECTED": 39, "UNATTRIBUTED": 185}
- `skater_sog_ge3|2024|CENTRAL_ABS_ERR_GT_3`: n=1040, flags={"SHOT_RATE_ABOVE_EXPECTED": 997, "SHOT_RATE_BELOW_EXPECTED": 43, "TEAM_SHOT_ENVIRONMENT_HIGH": 390, "TEAM_SHOT_ENVIRONMENT_LOW": 45, "TOI_ABOVE_EXPECTED": 105, "TOI_BELOW_EXPECTED": 4}
- `skater_sog_ge3|2024|INCUMBENT_CONFIDENT_UNDER_BUT_6PLUS`: n=45, flags={"SHOT_RATE_ABOVE_EXPECTED": 45, "TEAM_SHOT_ENVIRONMENT_HIGH": 17, "TOI_ABOVE_EXPECTED": 10}
- `skater_sog_ge3|2025|CENTRAL_ABS_ERR_GT_3`: n=1040, flags={"SHOT_RATE_ABOVE_EXPECTED": 985, "SHOT_RATE_BELOW_EXPECTED": 55, "TEAM_SHOT_ENVIRONMENT_HIGH": 369, "TEAM_SHOT_ENVIRONMENT_LOW": 54, "TOI_ABOVE_EXPECTED": 115, "TOI_BELOW_EXPECTED": 14}
- `skater_sog_ge3|2025|INCUMBENT_CONFIDENT_UNDER_BUT_6PLUS`: n=24, flags={"SHOT_RATE_ABOVE_EXPECTED": 24, "TEAM_SHOT_ENVIRONMENT_HIGH": 9, "TEAM_SHOT_ENVIRONMENT_LOW": 1, "TOI_ABOVE_EXPECTED": 5}

## Architecture conclusion (earned from the evidence above)

1. The incumbent is a threshold classifier per prop with no central projection, no distribution, no availability/role/PP/goalie structure: it extrapolates recent averages with coarse team margin.
2. A transparent opportunity chain (TOI x rate x damped opponent) matches it to within ~0.0005 Brier on the same binary targets and also yields counts, medians and any threshold: model the hockey count first, derive probabilities from it.
3. The upside left for a count-first engine is bounded: SOG count error is already near the count-variance floor given the available means, so Phase1 value must come from better means (role/PP/deployment/starter state), which are forward-only data. Historical results cannot validate that; they can only reject.

## Phase1 market ranking (evidence-based; first Phase1 = skater SOG)

| rank | market | current weakness | addressable error | data quality | honest validation | practical use |
|---|---|---|---|---|---|---|
| 1 | skater SOG | classifier ~ human baseline; no count | role/PP means (forward) + count distribution | official SOG, n~37k/season, but production DB holds ~85-92% of official skater rows | historical burned; forward only | high |
| 2 | goalie saves | blocked by starter state | starter + volume | usable; production DB holds ~85-92% of official skater rows | forward only | high once starter captured |
| 3 | skater points | classifier ~ human baseline | needs goals/assists decomposition (V1: goals head not established on 2025 mean-bias guard; assists A1/A2 rejected in development) | good | burned | medium |
| 4 | team moneyline | 2025 BSS ~0.005, slope ~0.8 | derive from team goals later | good | burned | low |

The V1 research branch already holds an SOG distribution engine (B2) and goals/assists work scored on 2024/2025; Phase0 did not re-score it. Whether Phase1 builds on that lineage or restarts is a decision for the user.

