# NFL independent architecture tournament — real numerical falsification, 2026-11-20

This is NOT the V7 extension. It is an isolated test of three prediction
architectures on independent nflverse player-game data with strict previous
WEEK sources. The existing NFL production code was never changed.

## Architecture alternatives

- **Recent-five**: the actual prior five played-game mean yardage. Strong,
  simple baseline.
- **Direct boosted**: HistGradientBoostingRegressor predicts expected
  rushing/receiving yards from prior player usage, team context and opponent
  prior average allowed yards.
- **Opportunity-first**: independently predict carries/targets with a
  Poisson learned head, multiply by the player's historically shrunk
  yards/opportunity and a bounded previous-opponent factor.

Every player row in a week was predicted BEFORE the entire week was added to
the historical state; a future same-week outcome cannot enter the head.
2024 is training, **2025 alone selects architecture**, then 2024+2025 refit
and previously consulted 2026 is a diagnostic. Model does not predict active
probability, role-confirmation, candidate-team membership or final-game
roster. 2026 is not a blind prospective model-selection window.

Real original GitHub Actions run:
https://github.com/deshawnclark116-prog/mlb-picks2/actions/runs/44812079778

## Validation (2025). Selection outcome: RECENT-FIVE BASELINE WINS

| Market | n | Recent-five MAE | Direct learned MAE | Opportunity-first MAE |
|---|---:|---:|---:|---:|
| Rushing yards | 3,075 | **13.0181** | 13.5934 | 13.0693 |
| Receiving yards | 4,864 | **17.0846** | 17.4533 | 18.2735 |

The more complex architectures did not beat a very simple, hard baseline
in the 2025 architecture-selection cohort. Promoting on that cohort is
forbidden. There was no trial-specific bet threshold optimization.

## Descriptive 2026 (already consulted). Not a promotion gate.

| Market | n | Recent-five MAE | Direct learned MAE | Opportunity-first MAE |
|---|---:|---:|---:|---:|
| Rushing yards | 1,683 | 12.8647 | 12.6065 | **12.1649** |
| Receiving yards | 2,718 | 17.5520 | **17.2982** | 17.8626 |

For rushing, the pairwise game-clustered 95% bootstrap interval for the
opportunity-first improvement over recent-five in the 2026 diagnostic
was [+0.3397, +1.0820] yards, but **this is not a pristine holdout**.
That 2026 improvement does not undo the 2025 failure or the missing
player-active/roster model.

- Rushing higher prior-volume cohort, n=557: factorized MAE 24.2174,
  direct 24.6920, recent-five 25.9469. The factorized method had a
  signed bias of **-5.34 yards** in this subgroup: it may under-predict
  true feature-volume players. Do not tune on this group.
- Receiving higher prior-volume cohort, n=934: direct 26.0601,
  factorized 26.9311, recent-five 27.4084. The opposite ordering
  indicates sport/market-specific architecture matters.

## Decision and further information required

**No NFL production promotion.** 2025 validation selects the recent-five
baseline for both markets. New directional hypothesis: opportunity-first
rushing model might generalize better in 2026 conditions, but require
a GENUINELY FUTURE, full-active-roster, source-time-frozen comparison
against actual live production V2 and V7 on the same games and equal
information, with confirmed expected touches and proper distributions.
Train without outcome leakage and include strong simple baseline.
For receiving, prioritize WR routes/target share and QB scenario instead
of treating every target like a homogeneous draw.

This research covers observed-player conditioned game stats only; zero
missing player rows were manufactured as inactive confirmations.
No quarterback starter, injuries, or game-day p(active) were proved;
no sportsbook price, betting profitability or calibration gain claimed.

**What was actually built and verified:** 6 anti-leak/source tests,
full 2024–2026 chronological dataset and independent fitted models,
game-grouped paired bootstrap, and unselected full-score JSON archived
with the successful GitHub Actions run. The research is not production.
