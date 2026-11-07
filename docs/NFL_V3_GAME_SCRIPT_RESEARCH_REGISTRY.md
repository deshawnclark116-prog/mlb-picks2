# NFL V3 game-script research registry — November 7, 2026

## Frozen champion
Phase1C 100,000-world game event simulator with independent team rush/dropback totals. Official T24/T90 Week 9 prefit is code-identity locked from November 1, 2026. Changing its source code before T90 would invalidate weekly artifacts and the immutable original forecasts.

## FAILED candidate: positive shared team pace
The experimental positive common-tempo proposal in nfl_v3_joint_game_shadow.py is FROZEN and must not be promoted. The raw-data falsification in nfl_v3_joint_pace_falsification.py uses real completed NFL team player games, the trailing four earlier same-season games as the as-of baseline, and carries + pass attempts as an imperfect offensive-plays proxy.

| Season | Paired completed games | Residual correlation | Game-bootstrap 95% interval |
|---|---:|---:|---|
| 2024 | 180 | -0.4658 | [-0.5638, -0.3549] |
| 2025 | 208 | -0.3909 | [-0.4838, -0.2787] |
| 2026 | 58 | -0.0551 | [-0.3022, +0.2294] |

There is no evidence for naive positive cross-team tempo correlation in these samples. 2024 and 2025 contradict that hypothesis. 2026 is small and inconclusive and has already been inspected; it cannot later be advertised as an untouched holdout.

These are proxies, not exact snaps/dropbacks. Provider data are retrospective and may include revisions. These comparisons cannot establish actual player-level CRPS improvements.

## OPEN alternative: finite possession competition
nfl_v3_possession_competition_shadow.py is an exploratory shadow-only scenario: one total game opportunity pool is split between teams; each team then splits its share into rush and pass, with opposing neutral game-script states. This is more structurally coherent than independently pushing both offenses faster, but is NOT calibrated. It has no down/clock/drive detail, turnover causal structure, exact sacks/scrambles, independently estimated expected win margin, or confirmed starting lineups.

## Promotion requirements
1. Same original frozen T24/T90 pretip input set, same players and opponent, same outcomes and model version.
2. Actual numerical player projections and proper uncertainty scores: per-market MAE, CRPS/pinball, Brier, conditional interval calibration, game-cluster confidence intervals and top-ranked buckets.
3. Leak-safe lookback and future-only holdout. No retroactive player inclusion or source updates after kickoff.
4. Preserve Phase1C champion until the alternative beats it without damaging important subgroups. No rescue tuning against already-seen 2026 outcomes.
5. A fresh legally timed prefit and distinct model code identity are required before deployment.
6. Do not equate Monte Carlo size or realistic-looking correlation with validated predictive accuracy.

Status: SHADOW RESEARCH, NO OFFICIAL FORECAST CHANGES.
