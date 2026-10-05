# Phase 1B opportunity-first findings

**Status: FROZEN, NOT PROMOTED.**

Source snapshot: draft PR #64 automated research snapshot at head `1e93a158d99782f05fd6310d8aecbe694cd22848`.

Phase 1B separated workload from efficiency so a good-looking final stat could not hide two offsetting errors. It selected team-volume + role-share logic first, then efficiency, with no Monte Carlo and no sportsbook input.

## Burned / historical results

| Outcome | 2025 validation MAE | 2025 player-opportunity MAE | 2026 W1-4 burned MAE | W4 V1 MAE same rows | W4 Phase1B MAE | W4 delta |
|---|---:|---:|---:|---:|---:|---:|
| Passing yards | 63.969 | 7.474 attempts | 61.620 | 69.159 | **62.947** | **-6.213** |
| Receptions | 1.727 | 2.210 targets | 1.786 | 2.172 | 2.161 | -0.011 |
| Receiving yards | 23.522 | 2.210 targets | 23.984 | 28.951 | **28.063** | -0.888 |
| Rushing yards | 24.381 | 3.892 carries | 24.326 | **22.460** | 23.167 | **+0.707 worse** |

## Decision

Phase 1B is not good enough to become a champion.

What survived:
- explicit separation of opportunity and efficiency;
- fixed pregame evaluation populations;
- direct, auditable point projections;
- 2024 selection -> 2025 validation discipline.

What failed:
- passing yards improved materially versus V1 on Week 4, but ~64-yard 2025 validation MAE is still not useful enough;
- receptions are effectively a tie;
- receiving yards improve only modestly and remain far too noisy;
- rushing regressed versus V1 on Week 4;
- player workload error remains large enough to dominate many misses.

## Next move: Phase 1C-T

Do **not** tune another final-stat formula.

Build the team opportunity engine as its own prediction problem using the timestamp-safe historical play-by-play and schedule information already in the repository contract:

- offensive play volume;
- opponent allowed play volume;
- dropback/rush composition;
- neutral-situation pass/run tendency;
- team and opponent historical pace/volume;
- home/away and rest context when it proves useful.

Efficiency stays out of this phase. Player role allocation stays out of this phase. Phase 1C-T only earns survival if it predicts team attempts/carries/targets better out of time than the Phase 1B team-volume layer and transparent team-volume baselines.
