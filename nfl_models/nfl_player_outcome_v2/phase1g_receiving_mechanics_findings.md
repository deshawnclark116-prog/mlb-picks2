# Phase 1G-R receiving-mechanics findings

**Status: REJECT AS EFFICIENCY REPLACEMENT; KEEP MECHANICAL DECOMPOSITION.**

Source run: GitHub Actions `37347447320`, head `5fc5354b356a658650a13e294f657e58f5aaa082`.

Phase 1G-R rebuilt receiving production from:
- predicted targets;
- predicted catch probability;
- predicted completed air yards per catch;
- predicted YAC per catch;
- player history;
- position priors;
- opponent positional allowance.

The shrinkage/opponent weights were selected only on burned 2024 using oracle-target receiving-yard MAE. 2025 was untouched by selection.

No sportsbook input and no Monte Carlo were used.

## Results

| Metric | 2025 | 2026 W1-4 |
|---|---:|---:|
| Oracle-target receptions MAE | 0.850 | 0.877 |
| Oracle-target receiving-yards MAE | 16.848 | 16.589 |
| Yards/target MAE | 3.757 | 3.649 |
| Full receptions MAE | 1.657 | 1.785 |
| Full receiving-yards MAE | 22.732 | 23.958 |

Phase1E benchmarks:
- receptions: 1.659 (2025), 1.781 (2026 W1-4);
- receiving yards: 22.790 (2025), 24.012 (2026 W1-4).

The full receiving-yard number improved by only **0.059 yards MAE** on 2025 and **0.054 yards** on 2026 W1-4. That is not material.

More importantly, Phase 1F's incumbent-efficiency oracle-workload MAE was **16.492 yards** on 2025 and **16.178 yards** on 2026 W1-4. Phase 1G-R produced **16.848** and **16.589** respectively. Therefore the receiving-mechanics efficiency estimate is objectively worse when workload is held correct.

The tiny final-stat gain is another cancellation effect and does not earn promotion.

## Decision

Reject the Phase 1G-R efficiency estimator.

Keep:
- the explicit football decomposition targets -> catches -> completed air yards + YAC;
- the PBP feature plumbing;
- the strict prior-week time boundary;
- opponent position-specific receiving history as a candidate information family.

Do not keep:
- this simple player/position/opponent blending formula as the efficiency champion.

## Next receiving research

The next candidate must add genuinely new matchup structure already available in the system:
- player depth-of-target / air-yard profile;
- player explosive-play profile;
- opponent explosive-pass suppression;
- defense pressure/blitz environment;
- man/zone tendency where timestamp-safe;
- player man/zone splits where historically available;
- defensive personnel absences where timing-safe.

These features must be routed into catch probability, completed air yards, or YAC. They may not be dumped into an opaque final-yards model.

2025 remains the validation comparator for architectures selected on 2024; 2026 Weeks 1-4 are burned diagnostics; Week 5+ is required for clean forward confirmation.
