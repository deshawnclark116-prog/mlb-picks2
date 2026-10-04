# NHL assists A1 — development only

**A0 selected; A1 and A2 FROZEN_REJECTED.** Status: `DEVELOPMENT_SELECTED_AWAITING_CONFIRMATION`. No historical champion established and no production promotion. No rescue tuning. Stop after this development commit; no PR to main.

## Preregistration and access discipline

Research base: `d825bc629c2490aa118ff43d8c510764facafe15`. Branch: `codex/nhl-outcome-engine-v1`. Protocol-only commit: `b79198cfc3f0d92c738f381ae0dadea2c9e349c0`. Code/tests-only commit: `92411fcd0f1357efc4e3ffea2b6cff07dac635ec`. Both were committed locally and on GitHub before development performance. No main merge.

Only 28 explicit frozen source files from 2017–2023 were opened. 2017 is history/warmup only. Target years 2018–2023; D1–D4 validation years 2020–2023. The implementation has no confirmation command, and model fit/predict/score entry points reject 2024/2025. No 2024/2025 raw data or NHL model-performance files were parsed. Manifests supply hashes only. Future confirmation requires separate authorization and the frozen procedure; no confirmation is run here.

## Data quality

303,324 frozen appearances; 333,153 exact Phase 1A candidate rows; 80,566 non-participants retained with assists=0. No missing/duplicate scoring identities, label mismatches, negative/fractional labels, point-identity failures or aggregate-credit constraint failures. Maximum realized candidate assists: 6.

Candidate population SHA256: `ed82c79720106dc63f90f8076302203dc4385fcf7a24056bcaa64b51ebde40b5`. Shared feature SHA256: `76f8a2fe639953aa0a9e5079944d6964d37ad53371398fe0dc52f6ee192fec30`. Every shared array is unchanged.

| Target season | Candidate rows | Played candidates | Actual skaters outside candidates |
|---|---:|---:|---:|
| 2018 | 58324 | 44939 | 816 |
| 2019 | 49716 | 38248 | 702 |
| 2020 | 41727 | 30477 | 749 |
| 2021 | 63415 | 46211 | 997 |
| 2022 | 60102 | 46335 | 873 |
| 2023 | 59869 | 46377 | 844 |

A2 feasibility was checked before performance: 14,313 candidate-covered team-games, 73,263 total S0 skater assist credits = 72,440 candidate credits + 823 outside-candidate credits. Noncandidate scoring players are NEVER added from target truth; their training-only aggregate credits identify the outside bucket. This is a model of aggregate skater assist credits, not per-goal multiplicity or a certified joint goals/assists process.

## Minimum upstream components

**team_goal_environment:** one NB2 scoring distribution per team-game. Frozen T0 final goals remove the shootout winner bonus and retain rare goalie-scored goals. Prior-only own goals-for and opposing goals-against from the last <=5 completed games are shrunk with five pseudo-games toward the training team-goal mean. Fixed features also include team/opponent SOG environment, home/rest/back-to-back and aggregate candidate scoring strength. The component is fitted once per team-game. A2 uses the same feature contract in a separate NB2 total-skater-assist-credit component. Neither component is independently certified.

**teammate_scoring_opportunity:** sum over legal pregame candidates of shrunk prior participation propensity times shrunk all-team prior scoring skill. Position rates are learned from training only; shrinkage uses five pseudo-games. Stable player ID carries scoring skill across trades; current-team history alone establishes membership/participation/role. Team models use the full sum, while A0 uses the leave-self-out sum. No goalie starter, linemate, sportsbook or attempt-extension input.

## Locked architectures

- **A0:** direct Poisson regression with alpha=.001 on the 33 shared features plus locked assist/scoring history and environment features; fixed-mean NB2 dispersion MLE. All candidates, including non-participants, are targets. Full NB2 PMF.
- **A1:** training-only availability logistic model; team-goal NB2 opportunity distribution; F/D/U beta-binomial player involvement hyperpriors fitted on training player exposure; legal all-team posterior history; locked current-team role logit adjustment. Full analytic mixture: `(1-p_active)*delta0 + p_active*sum_g P(team_goals=g)*BetaBinomial(assists;g,a,b)`.
- **A2:** total S0 skater-assist-credit NB2 component; A1 prior involvement/availability supplies allocation weights. Training-only outside-candidate share is fixed within fold. Multinomial allocation yields exact NB2 marginal PMFs and coherent joint credit draws. Availability is a soft expected involvement weight, not a sampled target lineup. Per-goal uniqueness and goalie-inclusive assist totals are not claimed.

All fitting rules, optimizer bounds, preprocessing, metrics, failure statuses and confirmation gates are locked in `phase_assists_a1_protocol.json`. Failed validation is never used to revise them.

## D1–D4 metrics

CRPS and NLL below are macro-game. Bias is relative mean bias. Mean estimates and MAE are candidate-row quantities. Full metrics and fitted parameters are in `phase_assists_a1_dev_results.json`.

| Fold | Architecture | CRPS | NLL | Predicted mean | Observed mean | Bias % | MAE | PIT-KS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| D1 | A0 | 0.1551534 | 0.4828796 | 0.2022931 | 0.2003499 | +0.970 | 0.2898562 | 0.0020386 |
| D1 | A1 | 0.1557598 | 0.4846235 | 0.1987235 | 0.2003499 | -0.812 | 0.2899820 | 0.0050184 |
| D1 | A2 | 0.1557042 | 0.4851999 | 0.2000618 | 0.2003499 | -0.144 | 0.2909175 | 0.0038751 |
| D2 | A0 | 0.1640130 | 0.5021202 | 0.2146024 | 0.2144761 | +0.059 | 0.3015189 | 0.0024763 |
| D2 | A1 | 0.1647039 | 0.5043632 | 0.2069060 | 0.2144761 | -3.530 | 0.3013370 | 0.0080869 |
| D2 | A2 | 0.1647184 | 0.5050549 | 0.2066077 | 0.2144761 | -3.669 | 0.3018174 | 0.0094146 |
| D3 | A0 | 0.1731091 | 0.5260615 | 0.2332980 | 0.2285947 | +2.057 | 0.3200821 | 0.0048124 |
| D3 | A1 | 0.1736279 | 0.5283741 | 0.2200676 | 0.2285947 | -3.730 | 0.3172292 | 0.0064564 |
| D3 | A2 | 0.1736752 | 0.5290941 | 0.2213708 | 0.2285947 | -3.160 | 0.3185398 | 0.0085988 |
| D4 | A0 | 0.1703649 | 0.5202397 | 0.2313595 | 0.2250914 | +2.785 | 0.3163859 | 0.0059260 |
| D4 | A1 | 0.1709695 | 0.5219408 | 0.2212238 | 0.2250914 | -1.718 | 0.3147160 | 0.0045851 |
| D4 | A2 | 0.1711185 | 0.5230863 | 0.2208323 | 0.2250914 | -1.892 | 0.3157013 | 0.0073810 |

Equal-weight mean-fold CRPS: A0 **0.1656601**, A1 **0.1662653**, A2 **0.1663041**.

| Fold | Arch | Brier >=1 | ECE >=1 | Brier >=2 | ECE >=2 | Predicted P0 | Observed zero rate |
|---|---|---:|---:|---:|---:|---:|---:|
| D1 | A0 | 0.1247110 | 0.0081542 | 0.0253539 | 0.0047053 | 0.8304827 | 0.8308290 |
| D1 | A1 | 0.1253545 | 0.0068739 | 0.0253558 | 0.0029930 | 0.8332624 | 0.8308290 |
| D1 | A2 | 0.1252921 | 0.0052958 | 0.0253522 | 0.0038120 | 0.8337442 | 0.8308290 |
| D2 | A0 | 0.1284561 | 0.0092542 | 0.0284621 | 0.0044874 | 0.8226863 | 0.8214776 |
| D2 | A1 | 0.1292516 | 0.0071025 | 0.0284237 | 0.0027025 | 0.8275961 | 0.8214776 |
| D2 | A2 | 0.1293558 | 0.0078951 | 0.0283704 | 0.0021012 | 0.8290558 | 0.8214776 |
| D3 | A0 | 0.1351827 | 0.0139772 | 0.0314989 | 0.0067615 | 0.8092702 | 0.8104223 |
| D3 | A1 | 0.1360319 | 0.0098270 | 0.0313298 | 0.0044638 | 0.8175620 | 0.8104223 |
| D3 | A2 | 0.1361185 | 0.0080784 | 0.0312819 | 0.0037509 | 0.8182877 | 0.8104223 |
| D4 | A0 | 0.1335983 | 0.0083651 | 0.0299337 | 0.0057321 | 0.8104344 | 0.8128748 |
| D4 | A1 | 0.1343149 | 0.0052320 | 0.0298705 | 0.0030118 | 0.8166536 | 0.8128748 |
| D4 | A2 | 0.1344812 | 0.0059348 | 0.0298518 | 0.0023609 | 0.8184576 | 0.8128748 |

| Fold | Arch | Central 50% coverage | Central 80% coverage | Central 90% coverage | PIT 50% coverage | PIT 80% coverage | PIT 90% coverage |
|---|---|---:|---:|---:|---:|---:|---:|
| D1 | A0 | 0.8953196 | 0.9697798 | 0.9812112 | 0.5009706 | 0.8003691 | 0.9006878 |
| D1 | A1 | 0.8962063 | 0.9662089 | 0.9811153 | 0.4950751 | 0.7980444 | 0.9002804 |
| D1 | A2 | 0.8956551 | 0.9671675 | 0.9822896 | 0.4970163 | 0.8001294 | 0.9003523 |
| D2 | A0 | 0.8924860 | 0.9684144 | 0.9815501 | 0.4987306 | 0.8006623 | 0.9010171 |
| D2 | A1 | 0.8916502 | 0.9643302 | 0.9811401 | 0.4921233 | 0.7943231 | 0.8961602 |
| D2 | A2 | 0.8910510 | 0.9648664 | 0.9822597 | 0.4963652 | 0.8003627 | 0.9004652 |
| D3 | A0 | 0.8893548 | 0.9683538 | 0.9814482 | 0.4979868 | 0.8007055 | 0.9000699 |
| D3 | A1 | 0.8889388 | 0.9617650 | 0.9802669 | 0.4968720 | 0.7965625 | 0.8974077 |
| D3 | A2 | 0.8892383 | 0.9624804 | 0.9815647 | 0.4934112 | 0.7977438 | 0.9000865 |
| D4 | A0 | 0.8918472 | 0.9695836 | 0.9817602 | 0.5014114 | 0.8020846 | 0.9017522 |
| D4 | A1 | 0.8903773 | 0.9636874 | 0.9809751 | 0.5003257 | 0.7998630 | 0.9000651 |
| D4 | A2 | 0.8903940 | 0.9642720 | 0.9821276 | 0.4956488 | 0.7980090 | 0.8992300 |

Discrete intervals can exceed nominal coverage; JSON also reports attainable model mass and empirical-minus-attainable coverage. PIT coverage accounts for discrete atoms.

## Blocked bootstrap

10,000 replicates, seed 20261002, 4,804 validation games in 104 calendar weeks. Same 2-calendar-week moving-block game-level bootstrap as goals/SOG; no row bootstrap. Delta is challenger minus reference game CRPS (negative favors challenger). Practical promotion requires >=0.5% equal-weight mean-fold gain AND upper95<0, >=3 improving folds, and all calibration/NLL/slice guards.

| Comparison | Mean-fold gain % | Game delta | Two-sided 95% CI | One-sided upper95 | Improving folds | Promotion |
|---|---:|---:|---|---:|---:|---|
| A1_vs_A0 | -0.3653 | +0.00060505 | [+0.00045524, +0.00078889] | +0.00076090 | 0/4 | FAIL |
| A2_vs_A0 | -0.3887 | +0.00065260 | [+0.00047135, +0.00086561] | +0.00083357 | 0/4 | FAIL |
| A2_vs_A1 | -0.0234 | +0.00004755 | [-0.00003536, +0.00012757] | +0.00011381 | 1/4 | FAIL |

A1 loses to A0 on CRPS in all four folds. A2 also loses in all four folds and additionally breaches the 0.5% NLL degradation guard (+0.548%). A2 vs A1 does not demonstrate a supported improvement. No fitting failures occurred. Both challengers are FROZEN_REJECTED without rescue. A1 is retained only as the strongest development comparator for a separately authorized future confirmation.

## Required slices

Pooled validation row CRPS, matching the frozen slice-guard doctrine (>=500 rows; no >5% degradation). Per-fold macro-game and full secondary slice metrics are recorded in JSON. High role is prior CURRENT-TEAM TOI mean over <=3 appearances >=18 minutes; low role includes missing prior TOI.

| Slice | Rows | A0 | A1 | A2 | Worst change vs A0 % | Guard |
|---|---:|---:|---:|---:|---:|---|
| defensemen | 77922 | 0.1526071 | 0.1533373 | 0.1533300 | +0.4785 | PASS |
| early_season | 38433 | 0.1387248 | 0.1394041 | 0.1395689 | +0.6085 | PASS |
| established_season | 186680 | 0.1704329 | 0.1710280 | 0.1710621 | +0.3692 | PASS |
| forwards | 147191 | 0.1715904 | 0.1721360 | 0.1722262 | +0.3705 | PASS |
| high_role | 70655 | 0.2567094 | 0.2570043 | 0.2571394 | +0.1675 | PASS |
| low_role | 154458 | 0.1230769 | 0.1238303 | 0.1238508 | +0.6288 | PASS |

## Verification and protected scope

`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/test_nhl_assists_a1.py`: **18 passed**. Synthetic-only model tests cover independent analytic PMF reference, full PMF normalization, deterministic fitting/prediction/evaluation, target-label invariance, stable skill across trades, current-team-only role, same-day completion cutoff, candidate identity, nonparticipant zeros, credit-conserving draws/marginal agreement, game-level bootstrap, feature whitelist, source access restriction and protected-file diff. Two dependency deprecation warnings from the unchanged shared sklearn helper have no effect on test results.

Development PMF maximum row-sum error across all folds/models: 5.743e-12. No renormalization conceals tail mass. A2 checked 8,000 development joint states with zero conservation violations; tests separately check 60,000 synthetic states and deterministic repeated draws.

Every pre-existing repository file remains byte-identical to the specified research base, including SOG model/artifacts, G1 goals model/artifacts/confirmation results, the rejected attempt extension, shared-state contract, Phase 0C live collector, production serving, workflows and registry. Only this new assists implementation and its artifacts were added. No main merge, production edit, confirmation score or PR.
