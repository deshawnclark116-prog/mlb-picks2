# NFL projection-engine rebuild — research registry (2026-11-09)

## Non-negotiable objective
Build a structurally different player-performance predictor that estimates **opportunity creation, team-level competition, efficiency, real participation uncertainty, and exact event probabilities** before using simulations. Replace the current production model only after a fair, sealed, same-cohort prospective comparison. Do not disguise easier lower-bound thresholds as the system's numerical projection.

## Why the incumbent is not sufficient
- The serving yardage path uses a last-eight-game player carry/target-count and per-event yardage bootstrap, blended with the prior season. That does not independently predict game-specific routes, snaps, QB choice or vacated target/carry competition.
- `nfl_target_grid.py` chooses the hardest fixed 5-yard grid target above 25 receiving/30 rushing yards and 55% of the model mean whose one-sided Monte Carlo lower bound passes 65%. That is a **different event** from attaining the projected mean.
- The live app displayed the threshold as a featured projection, and for BUF at MIN displayed Justin Jefferson 40+ yards at 69.3%, while its simulated mean was 57.5.
- DNP risk is not included; historical roster membership is not confirmed-game-day active or starting status.
- The context-aware V3 and V5 **previously failed** the original 2025 holdout gate. Receiving-yard error: V3 22.06 vs incumbent V2 context 20.92 yards; V5 calibrated 22.47 vs 20.92 (not identical to new V6's matched cohort). Don't promote old failed candidates.

## New implementation, no live promotion
- `nfl_v6_joint_opportunity_shadow.py`: independent, 20,000-draw **shared team/game** engine. As-of joined nflverse player_games/games; 24h source lag; only final prior games; season schedule canonical identity. Shared game pace, team pass/rush split, competing receiver target share and rusher carry share, opponent-adjusted efficiency, explicit UNKNOWN other-player bucket. Computes per-player numeric yards and event probabilities from exactly the same draws. **Research-only.**
- `nfl_v6_shadow_numeric_grade.py`: exact game/team/player/market result reconciliation, reports numerical MAE, naive prior-three MAE, Brier, and explicitly unverified/missing players. Never retrofits actuals into the forecast.
- `tests/test_nfl_v6_joint_opportunity_shadow.py`: chronology, nonfinal exclusion, deterministic draw, 10k minimum and team opportunity conservation.
- `.github/workflows/nfl_v6_joint_opportunity_shadow.yml`: real foundation smoke, frozen historical run/grade, standalone 2026 Week 9 shadow evidence; artifacts only; nothing copied into the production forecast or betting card.

## Diagnostic findings — **not a promotion trial**
- Historical 2025 Week 9: 14 games, 527 shadow player stat forecasts, 20k simulations per game.
- 420 matched to a completed player-stat record; 107 unmatched/unresolved (not silently treated as hits or zeros).
- Matched-cohort MAE: new V6 15.623 yards, naive recent-three-game 17.390. Passing 40.582 (n=22); receiving 16.027 (n=240); rushing 11.534 (n=158). **No exact same-cohort champion V2 comparison yet**; population and event definitions differ.
- 2026 Week 9 BUF–MIN: Jefferson V6 **58.0 receiving yards** vs existing **57.5**. This provides no evidence the new engine fixed the actual player forecast. Opponent information alone is insufficient.
- The 2025 diagnostic and 2026 Week 9 views have been inspected; they are burned for feature tuning/promotion. Future sealed forecasts must be recorded before outcomes.

## Required next research layers (must be sourced, never inferred)
1. Prekickoff candidate/availability identity from team reports plus independent game-day inactive statuses; time-stamped, no historical after-the-fact starter assignment.
2. Routes run, actual snap share, targets per route, WOPR/air yards, carry/goal-line role and vacated shares from documented nflverse/PFR sources with coverage by team/week.
3. Prekickoff QB starter scenarios and QB dropback/efficiency impact; jointly allocate the competing teammates' usage.
4. Opponent pressure, coverage, pace and personnel matchups, rest and weather/venue with strict timestamp receipts; no sportsbook line drives a player projection.
5. Additional supported markets such as receptions, pass attempts, pass completions, touchdowns, tackles and sacks need individually validated models before inclusion.
6. Distinguish probability of **the displayed exact event**, numerical mean uncertainty, active probability, and actual sportsbook line hit probability. A simulation cannot rescue weak forecasts.

## Promotion gate
- Freeze an unchanged challenger before a **new prospective week**, with immutable per-game original forecasts, source/time/version and source coverage.
- Exact same as-of roster population versus frozen current-serving V2/production and naive; preserve DNP/unmatched separately.
- Beat incumbent on pooled and featured-player numerical MAE, ±10/±20 accuracy, calibrated event Brier/reliability and per-market thresholds, without breaking game-level/top-card selections.
- Require coverage/role/injury source reliability, minimum sample and structural tests; reject without post-hoc rescue tuning. **Until then, production remains unchanged.**
