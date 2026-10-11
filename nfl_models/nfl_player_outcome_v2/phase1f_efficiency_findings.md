# Phase 1F efficiency forensic findings

**Status: COMPLETE. HUMAN-EFFICIENCY SWAP REJECTED; CURRENT EFFICIENCY STILL INADEQUATE.**

Source run: GitHub Actions `37346718505`, head `68a49a6912f84448d1f4b74963f43e711e49b5dd`.

Phase 1F held Phase1E player opportunities fixed and changed only the per-opportunity efficiency estimate. This prevents workload differences from hiding which efficiency estimate is actually better.

No new fitting, Monte Carlo, or sportsbook inputs were used.

## 2025 diagnostics

| Outcome | Phase1E MAE | Human-eff swap MAE | Current eff/opp MAE | Human eff/opp MAE | Perfect workload + current eff MAE | Perfect efficiency + current workload MAE |
|---|---:|---:|---:|---:|---:|---:|
| Passing yards | **63.947** | 64.356 | **1.506 Y/A** | 1.549 | 39.947 | **50.404** |
| Receptions | **1.659** | 1.661 | **0.200 catch/target** | 0.202 | 0.832 | **1.401** |
| Receiving yards | 22.790 | **22.724*** | **3.766 Y/T** | 3.819 | **16.492** | **16.162** |
| Rushing yards | **23.647** | 23.848 | **1.748 Y/C** | 1.811 | **15.567** | **16.065** |

*The tiny receiving-yard final-MAE gain is **not** a clean efficiency win. The human efficiency estimate is worse on direct per-opportunity error and worse when workload is made perfect. The final gain comes from error cancellation, so it is rejected.

## 2026 Weeks 1-4 burned diagnostics

The same conclusion strengthens:
- passing human-eff swap worsened MAE by **+2.836 yards**;
- receptions worsened by **+0.019 catches**;
- receiving yards worsened by **+0.154 yards**;
- rushing yards worsened by **+0.371 yards**.

## What this proves

The remaining problem is not one bad component.

For receiving yards on 2025:
- with actual targets handed to the model, current efficiency still misses by **16.49 yards MAE**;
- with actual yards/target handed to the model, current predicted workload still creates **16.16 yards MAE**;
- together they produce **22.79 yards MAE**.

For rushing yards:
- perfect carries + current efficiency still leaves **15.57 yards MAE**;
- perfect efficiency + current carries still leaves **16.06 yards MAE**.

Both opportunity and efficiency need to improve. Neither can be treated as solved.

Passing is even clearer: perfect attempts would still leave roughly **39.95 passing yards MAE**, while perfect efficiency with the current attempt forecast leaves **50.40 yards MAE**. Passing opportunity/game-volume is the larger bottleneck, but efficiency remains materially wrong too.

## Decision

- Reject the competent-human efficiency swap as a V2 replacement.
- Keep the competent-human *full projection* as a mandatory benchmark.
- Keep Phase1B efficiency only as the current incumbent, not as a certified solution.
- Do not add Monte Carlo.
- Next research must add genuinely new football information to efficiency and matchup, not another reweighting of the same recent averages.

Priority:
1. receiving yards: route/depth/explosive/matchup efficiency;
2. rushing yards: front/run-environment matchup efficiency;
3. passing: dedicated attempt/game-volume model first, then passing efficiency.
