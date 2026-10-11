# Phase 1E integrated direct-projection findings

**Status: PARTIAL SURVIVAL. RUSHING + RECEPTION ROLE INTEGRATION SURVIVE; RECEIVING-YARD HEAD REMAINS RESEARCH; PASSING UNCHANGED.**

Source run: GitHub Actions `37346296627`, head `11642c8224ce02d191b484dcfc6ee6c632335485`.

Phase 1E performed **no new fitting**. It combined:
- frozen Phase1B team-volume forecasts;
- frozen Phase1B efficiency forecasts;
- the Phase1D-R target-share allocator for receiving/receptions;
- the Phase1D-R carry-share allocator for rushing;
- the Phase1B QB path for passing because the generic QB role allocator was rejected.

No Monte Carlo and no sportsbook inputs were used.

## 2025 historical comparison

| Outcome | Phase1E MAE | Phase1B MAE | Competent-human MAE | Phase1E player-opp MAE | Phase1B player-opp MAE |
|---|---:|---:|---:|---:|---:|
| Passing yards | 63.969 | 63.969 | 65.928 | 7.474 | 7.474 |
| Receptions | **1.659** | 1.727 | 1.767 | **2.125** | 2.210 |
| Receiving yards | **22.790** | 23.522 | 23.294 | **2.125** | 2.210 |
| Rushing yards | **23.647** | 24.381 | 24.753 | **3.800** | 3.892 |

## 2026 Weeks 1-4 burned diagnostics

- passing yards MAE: **62.601**
- receptions MAE: **1.781**
- receiving yards MAE: **24.012**
- rushing yards MAE: **22.019**

The receiving-yard head is important: Phase1E beat Phase1B on 2025, but on already-burned 2026 Weeks 1-4 it was essentially flat/slightly worse than Phase1B (24.012 vs 23.984). Therefore the 2025 gain is not enough to promote the receiving-yard head.

Rushing is more encouraging: the same structural role change reduced 2026 W1-4 MAE from Phase1B's 24.326 to **22.019** while also reducing player-opportunity error.

## Decision

Survive as architecture:
- coherent target-share allocation feeding receptions;
- coherent carry-share allocation feeding rushing;
- direct point projection before simulation;
- component receipts and separate opportunity/efficiency grading.

Still unresolved:
- receiving efficiency / explosive-yard prediction;
- passing attempt/game-volume prediction;
- absolute rushing-yard error remains too large for production;
- none of these heads are production-certified.

## Next research question

Before adding complexity, isolate the efficiency layer. Compare:
1. current Phase1B efficiency;
2. the fixed competent-human efficiency blend;
3. oracle-opportunity performance using actual workload.

This tells us whether remaining error is primarily workload or per-opportunity production and whether a simpler matchup/efficiency formulation is already better.

Week 5+ remains the first possible clean forward evidence for any candidate frozen before its forecast cutoff.
