# Phase 1A direct-chain findings

**Status: FROZEN AS ARCHITECTURE PROOF, NOT PROMOTED.**

Run: GitHub Actions `37250691458`, head `cff128cb2fbb30f20a8e415eb5035d8bcb1f1bd6`.

Phase 1A established the V2 direct prediction contract:

```
projected team opportunity
x projected player role share
x projected per-opportunity efficiency
= direct point projection
```

It uses no Monte Carlo and no sportsbook input. The component receipt is auditable.

## Burned results

The configuration was selected on 2025 itself, so the 2025 numbers below are **selection/development scores, not validation**.

| Outcome | 2025 selection MAE | 2026 W1-4 burned diagnostic MAE | W4 V1 MAE on same clean rows | W4 Phase1A MAE | W4 delta |
|---|---:|---:|---:|---:|---:|
| Passing yards | 62.811 | 64.453 | 69.159 | **63.431** | **-5.728** |
| Receptions | 1.748 | 1.857 | 2.172 | **2.130** | -0.042 |
| Receiving yards | 23.831 | 25.297 | 28.951 | **28.174** | -0.777 |
| Rushing yards | 24.667 | 25.334 | **22.460** | 24.906 | **+2.446 worse** |

## Decision

Phase 1A is not a candidate champion.

Why:
- it selected hyperparameters and reported 2025 on the same burned period;
- the Week 4 gain is material only for passing yards;
- receptions and receiving yards are essentially ties;
- rushing yards regressed;
- Week 4 forensics already show opportunity volume is a major upstream failure mode.

The direct-chain architecture survives. The specific Phase 1A estimator does not earn promotion.

## Next candidate

Phase 1B isolates opportunity quality before efficiency:
1. tune team-volume and role-share logic on burned 2024;
2. freeze it using player-opportunity MAE as the primary criterion;
3. tune efficiency only after opportunity is frozen;
4. evaluate on 2025 without using 2025 for candidate selection;
5. report 2026 Weeks 1-4 only as burned diagnostics.

This prevents two wrong component estimates from looking good merely because their final-stat errors cancel.
