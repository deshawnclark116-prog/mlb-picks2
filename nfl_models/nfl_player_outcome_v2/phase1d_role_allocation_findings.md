# Phase 1D-R coherent role-allocation findings

**Status: PARTIAL SURVIVAL. RECEIVING + RUSHING ROLE ALLOCATION SURVIVE; QB ATTEMPT SHARE REJECTED.**

Source run: GitHub Actions `37326756230`, head `64fad683044baf5bb85d8408af7e16d51f940309`.

Phase 1D-R isolated the question: **who gets the team's opportunities?** It used prior player shares plus current weekly roster eligibility, normalized every team allocation to 100%, and tested usage concentration directly before yards.

No Monte Carlo and no sportsbook inputs were used.

## 2025 out-of-selection validation

| Outcome | Phase1D role-share MAE | Phase1B role-share MAE | Simple prior-3 share MAE | Phase1D oracle player-opportunity MAE | Phase1B player-opportunity MAE |
|---|---:|---:|---:|---:|---:|
| Passing yards / QB attempts | **0.1880** | **0.0955** | **0.0977** | 5.846 attempts | 7.474 attempts |
| Receptions / targets | **0.0621** | 0.0648 | 0.0675 | **1.880 targets** | 2.210 targets |
| Receiving yards / targets | **0.0621** | 0.0648 | 0.0675 | **1.880 targets** | 2.210 targets |
| Rushing yards / carries | **0.1131** | 0.1170 | 0.1190 | **3.036 carries** | 3.892 carries |

## 2026 Weeks 1-4 burned diagnostics

- receiving/target role-share MAE: **0.0666**
- rushing/carry role-share MAE: **0.1193**
- QB attempt-share MAE: **0.2323**

The allocation is numerically coherent to floating-point precision: team predicted shares sum to 1.

## Decision

### Survive
- receiving target-share allocator;
- rushing carry-share allocator;
- roster-aware redistribution;
- explicit concentration control;
- coherent team-level share conservation.

These improve the actual upstream quantity we care about on 2025 validation, not merely final yards.

### Reject
- QB attempt-share allocator.

Quarterback opportunity is structurally different from RB/WR/TE allocation. Normalizing multiple rostered QBs into a generic share competition is the wrong football model. QB attempts need a dedicated starter/availability/replacement process rather than the generic role allocator.

## Important limitation

The role allocator improved **oracle** player opportunities, meaning player share multiplied by actual team opportunity. The coupled player-opportunity score still inherits team-volume error. Therefore Phase 1D-R does not by itself prove final player-yard projections are good enough.

## Next move: Phase 1E

Integrate the surviving target/carry allocator with the frozen team-volume and efficiency components and test the full direct point projection.

The test must answer two separate questions:

1. Did the better role model actually reduce player-opportunity error once realistic predicted team volume is used?
2. Did that improvement survive through to receptions / receiving yards / rushing yards, rather than being erased by efficiency error?

Passing yards keeps the Phase1B QB opportunity path until a dedicated QB starter/attempt model is built.
