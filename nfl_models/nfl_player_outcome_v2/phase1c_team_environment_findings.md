# Phase 1C-T PBP team-environment findings

**Status: FROZEN, NOT PROMOTED AS A GENERAL TEAM-VOLUME CHAMPION.**

Source run: GitHub Actions `37256969908`, head `38fec6b22cbe53cf36403e891347dcd3b63482f7`.

Phase 1C-T isolated team opportunity from player role and efficiency. It tested whether timestamp-safe prior play-by-play structure, neutral pass/run tendency, opponent context, home/away, and rest could beat both transparent team-volume baselines and the simpler Phase 1B team-volume layer.

No Monte Carlo and no sportsbook inputs were used.

## Burned / historical results

| Team opportunity | 2025 Phase1C-T MAE | Best simple 2025 MAE | Phase1B 2025 MAE | 2026 W1-4 Phase1C-T MAE | Phase1B 2026 W1-4 MAE |
|---|---:|---:|---:|---:|---:|
| Pass attempts | 6.065 | 6.386 | **5.943** | **6.008** | 6.253 |
| Carries | **5.775** | 6.088 | 5.897 | 5.392 | **5.212** |
| Targets | 5.808 | 6.072 | **5.775** | 5.847 | **5.696** |

## Decision

The richer PBP team-environment layer does **not** earn general promotion.

What survived:
- team opportunity remains an explicit prediction layer;
- PBP structure is real signal versus dumb recent-volume baselines;
- the direct receipts for total plays, neutral tendency, opponent context, home/away and rest remain useful research instrumentation.

What failed:
- attempts improved versus a simple recent-average baseline but lost to Phase1B on 2025;
- carries beat Phase1B on 2025 but lost on already-burned 2026 Weeks 1-4;
- targets were only marginally different from Phase1B and did not establish a stable advantage;
- no opportunity head showed a consistent enough out-of-time edge to justify extra complexity.

The lesson is not that game environment is unimportant. It is that this particular historical PBP formulation does not yet predict team volume sharply enough to solve the player-projection problem.

## Next move

Shift the research target one layer downstream to **coherent player role allocation / usage concentration**.

Week 4 showed that team-volume error alone does not explain misses like a player absorbing far more targets or carries than expected. The next candidate must predict who receives the available opportunities and must be judged directly on target-share / carry-share and player-opportunity error before yards are considered.

The Phase1C-T implementation is frozen as evidence and may be revisited only with genuinely new contextual information or a materially different architecture.
