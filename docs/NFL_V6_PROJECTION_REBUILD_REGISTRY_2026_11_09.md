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


## 2026-11-09 continuation — paired production truth and freeze decision
### Same-player, same-game original-pregame benchmark
The prior V6A 15.623-yard MAE on 420 matched 2025 Week 9 player rows
was *not* a valid champion comparison. The new strict checker
\`nfl_v6_same_cohort_diagnostic.py\` matches only the original immutable
2026 Week 9 pregame firstseen player/game/team/outcome records. Unit tests
fail late, ambiguous, wrong-game and duplicate records.

- Firstseen ledger: 78 rows; 67 matched across 11 games; 6 games absent
  from V6 source slate; 5 player identity/source failures. All exclusions
  remain separate, never recoded as wins.
- Frozen incumbent mean MAE 24.916, V6A mean MAE 25.284 yards.
  Within ±10: incumbent 31.34%, V6A 25.37%. V6A 95% game-cluster
  paired lift CI [-2.177, +1.199] yards; no demonstrated improvement.
- Receiving same cohort n=42: incumbent 28.381, V6A 28.074 MAE.
  Rushing n=25: incumbent 19.096, V6A 20.596 MAE.
- Featured incumbent ≥60-yard-mean subgroup n=22: incumbent 34.705,
  V6A 34.595 MAE, too small/uncertain to claim improvement.

### New independent information: PFR offensive snap exposure, not routes
- 2025 CSV 26,613 rows, SHA256 3fc2deb0e9ad86d34d4578cb80bb21c95253e088ee22ca028adf46f7485eff1f.
- 2026 CSV 12,597 rows, SHA256 2d013a34c6a53f138171f8508393a91bb64d90136586a90c7472fafec2d70a8a.
- PFR snap data are genuinely accessible from keyless nflverse release
  snap_counts_2025.csv and snap_counts_2026.csv.
- The new \`nfl_v6_snap_role_evidence.py\` matches canonical game_id, team,
  uniquely normalized player, player-game identity and 24-hour availability
  lag; missing players remain unknown, and offensive snap share is NEVER
  labeled as routes run or proven future active status.
- Verified pre-BUF–MIN Jefferson source: 3 matched games, mean recent
  offensive snap share 0.9318, 23 targets over 169 offensive snaps,
  target rate 0.13609 per offensive snap.
- V6B structurally replaces historical raw-count Dirichlet shares with
  snap-share × target/carry per offensive snap whenever ≥2 observed source
  games exist; missing role sources use a separate labeled as-of fallback.
  It **preserves original V6A behavior** when no role source is supplied
  and is a distinct \`NFL_V6B_SNAP_ROLE_SHADOW_A\` version.

### V6B diagnostic result — **reject promotion**
- Same 67 frozen firstseen rows: V6B MAE 25.869 vs incumbent 24.916.
  Within ±10: V6B 25.37% vs incumbent 31.34%.
- Receiving n=42: V6B 28.617 vs incumbent 28.381.
- Rushing n=25: V6B 21.252 vs incumbent 19.096.
- V6B's adjusted Jefferson expectation was **55.8 receiving yards**,
  with mean 7.17 simulated targets, compared with frozen 57.5 incumbent
  and V6A 58.0. These are NOT verified betting or full DNP probabilities.
- No after-the-fact threshold increases, role-prior retuning or
  correction toward known November 8–9 boxscores are authorized.
  **V6B fails the current promotion test; do not deploy.**

### First-seen prospective program
- \`nfl_v6b_forward_capture.py\` and
  \`.github/workflows/nfl_v6b_prospective_capture.yml\` implement a
  research-only source-hashed prekickoff firstseen capture. Games within
  30 hours of kickoff are eligible. Real capture time caps all available
  game stats; only future games can be processed, with minimum 20k draws.
- Exactly one SHA-identified receipt per game, exclusive-create only,
  never overwrite, including source DB SHA, model code SHA, nflverse
  CSV SHA and original model rows. No official active/route/probability
  certification. Before publish, the real clock must still be pregame.
- Scheduled independent Thursday/Sunday/Monday windows plus an
  existing NFL publisher workflow completion fallback. A simulated
  test verified immutability, no postgame backfill, and the no-source
  guard. Initial live CI acceptance run passed; its logged 30-hour
  window was empty, so **no actual upcoming-game firstseen artifact
  was created yet**.
- The first truly unseen future-game snapshots, once captured, must
  be paired with postgame same-game, same-player champion mean and
  explicit zero/unresolved/DNP handling. Holdout performance cannot
  be recovered retroactively from these examined weeks.

### Other raw signal: FTN play charting
Verified the free \`ftn_charting_2026.csv\` release has 20,723
play-chart rows and fields including \`read_thrown\`, \`n_pass_rushers\`,
\`n_blitzers\`, \`is_motion\`, \`is_play_action\` and
\`nflverse_play_id\`, SHA256
5500f091a3617a44c25c406b85c3918dc5e154182dee652b42f1a9bb97e9c234.
Any player first-read feature requires an explicit per-play nflverse PBP
receiver-ID join, a pregame as-of cutoff and a player exposure denominator;
**not implemented** and no fantasy-style route participation is implied.

## Current disposition
- **Production:** unchanged incumbent NFL model.
- **V6A and V6B:** retrospective probes falsified; kept archived,
  not promoted.
- **Forward collector:** installed and source contract tested, future
  game receipts await actual qualifying upcoming pregame window.
- **Priority:** actual same-game, same-player error reduction from
  independently sourced opportunity/role/QB changes; strict DNP and
  identity controls; only future promotion gates. No more cosmetic
  rearrangement of prediction numbers as a substitute for accuracy.
