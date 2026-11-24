# NFL no-bookmaker yardage challenger — measured results (2026-11-22/23)

**Verdict: DID NOT BEAT THE README'S PREVIOUS BEST. Neither market meets the 0.30-yard meaningful-improvement gate. Do not promote to production.**

Read [original protocol](README.md), [full machine-readable result](challenger_result.json), and [challenger source](challenger.py). 
Run evidence: [seed-corrected GitHub Actions job](https://github.com/deshawnclark116-prog/mlb-picks2/actions/runs/45089850194) (evaluation succeeded; that particular job's post-score git push failed due to concurrent-run branch movement; identical evaluated results were persisted to `challenger_result.json` by another run).

## Fixed evaluation protocol
- 2025 regular-season weeks 6–18 plus 2026 weeks 1–11; identical eligible player-games to the earlier research.
- 1,119 rushing player-games and 2,956 receiving player-games, confirmed by explicit count assertions.
- Rolling weekly model training uses ONLY earlier (season,week) outcome rows. Reported models use 140 XGBoost rounds, the exact incumbent parameters/seed, and the same fixed player population.
- Challenger feature matrices exclude `team_spread` and `implied_team_total` and contain NO sportsbook price or line features. Historical live-v2 comparator retains those features for a like-for-like comparison to its previously published performance.
- Empirical residual CRPS and Brier are evaluated from strictly earlier-week model residuals. `Brier own` measures an individual model's event `Y >= .70 * its own projection`, whose event threshold varies between models. The additional **fixed-event Brier** measures `Y >= .70 * live-v2 projection` identically across models and is therefore the fairer ranking comparison.
- Paired bootstrap: 4,000 matched player-row resamples and 4,000 week-block resamples. Values are bootstrap P(MAE improvement > 0), not a validated forward probability of future superiority.
- Important: the 2025–26 evaluation window was used repeatedly in previous experiments, so these are retrospective walk-forward results, NOT a pristine untouched final holdout.

## Results (lower MAE, CRPS and Brier are better)

| Market | Model | MAE yd | CRPS | Brier own | Brier fixed-event | MAE gain vs reproduced live v2 | Paired P(row better) | P(week better) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Rush | live_v2 | 24.9902 | 18.1331 | 0.22626 | 0.22626 | — | — | — |
| Rush | no_line_core | 25.0497 | 18.1721 | 0.22504 | 0.22726 | -0.0595 | 0.2702 | 0.2430 |
| Rush | no_line_role | 24.9081 | 17.9838 | 0.22652 | 0.22188 | 0.0821 | 0.7163 | 0.6845 |
| Rush | no_line_quality | 25.0529 | 18.1457 | 0.22529 | 0.22666 | -0.0626 | 0.2898 | 0.2447 |
| Rush | no_line_role_quality | 24.9475 | 18.0055 | 0.22647 | 0.22302 | 0.0427 | 0.6090 | 0.6155 |
| Receive | live_v2 | 22.5954 | 16.3206 | 0.23322 | 0.23322 | — | — | — |
| Receive | no_line_core | 22.6210 | 16.3746 | 0.23712 | 0.23378 | -0.0256 | 0.2913 | 0.2467 |
| Receive | no_line_role | 22.6579 | 16.3814 | 0.23599 | 0.23370 | -0.0625 | 0.1550 | 0.1452 |
| Receive | no_line_quality | 22.5768 | 16.3428 | 0.23336 | 0.23209 | 0.0186 | 0.6105 | 0.6150 |
| Receive | no_line_role_quality | 22.6209 | 16.3722 | 0.23356 | 0.23233 | -0.0254 | 0.3555 | 0.3580 |

## Decision

Published targets: **rushing 24.86, receiving 22.44** average miss.

- Best rushing: `no_line_role` **24.9081** vs published previous best 24.86 (**0.0481 yards worse**), versus reproduced live v2 24.9902 (**0.0821 better**). Paired P(row better than live v2): **0.7163**; P(week-block better): **0.6845**; week-block CI95 for mean MAE improvement **[-0.2496, +0.3884]**. CRPS **17.9838** vs v2 **18.1331** (gain +0.1493). Own-event Brier **0.22652** vs v2 **0.22626** (slightly worse); fixed-event Brier **0.22188** vs v2 **0.22626** (better).
- Best receiving: `no_line_quality` **22.5768** vs published previous best 22.44 (**0.1368 yards worse**), versus reproduced live v2 22.5954 (**0.0186 better**). Paired P(row better): **0.6105**; P(week-block better): **0.6150**; week-block CI95 **[-0.0944, +0.1263]**. CRPS **16.3428** vs v2 **16.3206** (worse). Own-event Brier **0.23336** vs v2 **0.23322** (slightly worse); fixed-event Brier **0.23209** vs v2 **0.23322** (better).
- Neither market met the 0.30-yard minimum gain, and neither beat the previously reported best. All other four-variant cells are explicitly included in the table, including failures.

## What changed, what failed, and remaining gaps

The isolated new experiments tested a no-line core; historical recency/volatility/opportunity dynamics; historical receiver air-yards, deep-target proportion, catch success, YAC and team QB efficiency; historical explosive/stuffed rushing; and role+quality together. The no-line core did not beat v2; adding quality to the rushing role model did not improve MAE beyond role-only; role features hurt receiving MAE; quality features produced only a very small receiving MAE improvement and worse CRPS. A first run used an altered seed and was rejected as a final comparison; the corrected run preserves the original incumbent seed and reproduces 24.99 / 22.60.

**Not tested in this experiment:** FTN play-action/pressure/routes charting, as-of depth charts, pregame offensive-line injuries, reliable on-field route-level tracking, and rigorous coaching intent features. These need a fresh research stage and an unburned prospective evaluation window. No automatic promotion, no live model changes.
