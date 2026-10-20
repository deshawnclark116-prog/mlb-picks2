# NFL 2026 Week 6 — October 18–19 frozen-pregame audit

## Why this record exists

This is an independently graded reconstruction of real **immutable T24
forecast batches** from the `nfl-shadow-state` branch of `mlb-picks2`,
created before their respective games' start times. It DOES NOT regenerate,
change, retune or republish those predictions using the final results.

Results independently retrieved on **October 20, 2026** from completed
ESPN NFL scoreboard and final player boxscores. All 12 Sunday events and
Monday's Washington–San Francisco event are present and final.

Passing, rushing, receiving, tackles, touchdowns and sacks are MODEL
OUTCOMES, not independently price-matched sportsbook bets. The T24 phase-one
publisher itself labels them SHADOW / RESEARCH and does not claim promotion.
The T90 job missed multiple Sunday/Monday cutoff windows; no hindsight
backfill was permitted.

**Scientific run, source logs, and downloadable row-level CSV + JSON:**
[GitHub Actions run 39413272835](https://github.com/deshawnclark116-prog/mlb-picks2/actions/runs/39413272835).
The exact 13 frozen T24 batch headers/footer SHA256 and per-game ESPN event IDs
are retained in that immutable 90-day run artifact.

## What the grader found

| Measurement | Sunday Oct 18 | Monday Oct 19 | Both days |
|---|---:|---:|---:|
| Frozen pregame numerical outcome rows | 3,642 | 335 | 3,977 |
| Player stat forecasts matched to independent box scores | 2,129 | 177 | 2,306 |
| Player stat forecasts missing verifiable played-stat identity | 1,513 | 158 | 1,671 |
| 65%-confidence model-derived milestones originally selected | 643 | 49 | 692 |
| Milestones with independently verifiable player outcome | 565 | 43 | 608 |
| Matched milestone hits | 452 | 39 | 491 |
| Matched milestone losses | 113 | 4 | 117 |
| Milestones without verified player participation/identity | 78 | 6 | 84 |
| Success rate **among matches only** | 80.0% | 90.7% | 80.8% |
| **Conservative all-unverified-are-losses bound** | 70.3% | 79.6% | 71.0% |

The matched-only 80.8% is conditional on postgame ESPN participation
detection. It is **not** an unbiased hit rate for all original selections.
The 84 unresolved milestones are **not** automatically push, losses or wins.
The low bound treats all as misses for a conservative sensitivity scenario;
the actual unresolved outcomes require independent status/participation data.
None of these derived thresholds represents a pregame FanDuel or other
confirmed bookmaker line. There is no verified money-bet ROI.

### Milestone breakdown: models' own 65%-confidence thresholds, matched cohort

| Outcome | Matched wins/losses | Matched hit rate | Mean nominal confidence | Excluded milestones |
|---|---:|---:|---:|---:|
| Player tackles | 255–58 | 81.5% | 72.95% | 64 |
| Receptions | 111–24 | 82.2% | 74.29% | 12 |
| Receiving yards | 73–15 | 83.0% | 71.03% | 3 |
| Passing yards | 21–2 | 91.3% | 70.43% | 1 |
| Passing touchdowns | 10–3 | 76.9% | 70.04% | 1 |
| **Rushing yards** | **21–15** | **58.3%** | **71.61%** | 3 |

This identifies an actionable weakness: the rushing-yard threshold cohort
underperformed its reported conditional confidence even when counting only
recorded participants. A successful pooled threshold rate must **not**
conceal individual failing markets.

### Actual projection errors, 2,306 matched stat forecasts

| Statistic | Matched players | Mean absolute error of predicted MEAN |
|---|---:|---:|
| Passing yards | 25 | 67.37 yards |
| Rushing yards | 86 | 15.80 yards |
| Receiving yards | 207 | 16.74 yards |
| Receptions | 207 | 1.44 |
| Passing touchdowns | 25 | 0.835 |
| Tackles | 402 | 1.661 |
| Sacks | 402 | 0.205 |

A 0-value-only trivial benchmark unsurprisingly does well for extremely
rare events (sacks, defensive interceptions, TDs), so low absolute error
alone is not skill evidence in those markets. Market-specific baseline
and calibration work is needed.

### Concrete NFL player failures AND successes

| Pregame T24 player/stat prediction | Predicted mean | Actual |
|---|---:|---:|
| Tanner McKee, passing yards (Eagles–Panthers) | 38.6 | 215 |
| Caleb Williams, passing yards (Bears–Falcons) | 94.2 | 305 |
| Jayden Daniels, passing yards (Commanders–49ers) | 179.7 | 300 |
| Brock Purdy, passing yards (Commanders–49ers) | 187.7 | 264 |
| Kirk Cousins, passing yards (Bills–Raiders) | 163.6 | 313 |
| Dak Prescott, passing yards (Cowboys–Packers) | 257.2 | 258 |
| Christian McCaffrey, rushing yards (Commanders–49ers) | 64.0 | 64 |
| Chuba Hubbard, rushing yards (Panthers–Eagles) | 83.9 | 120 |
| CeeDee Lamb, receiving yards (Cowboys–Packers) | 73.0 | 70 |

The McKee failure is especially important: later QB-role scenario research
does **not** replace the original immutable T24 forecast, and the old
selection's inadequate starting-QB probability must remain a model defect,
not be quietly overwritten by a later forecast.

## Per-game matched milestones, not bets

| Game | Wins | Losses | Outcome unknown | Worst-case win rate |
|---|---:|---:|---:|---:|
| Houston @ Jacksonville | 40 | 6 | 6 | 76.9% |
| Baltimore @ Cleveland | 32 | 15 | 6 | 60.4% |
| Carolina @ Philadelphia | 37 | 10 | 6 | 69.8% |
| Chicago @ Atlanta | 35 | 10 | 8 | 66.0% |
| New Orleans @ NY Giants | 29 | 15 | 10 | 53.7% |
| NY Jets @ New England | 42 | 9 | 4 | 76.4% |
| Pittsburgh @ Tampa Bay | 39 | 7 | 6 | 75.0% |
| Tennessee @ Indianapolis | 33 | 7 | 12 | 63.5% |
| Arizona @ LA Rams | 44 | 6 | 4 | 81.5% |
| Buffalo @ Las Vegas | 42 | 6 | 4 | 80.8% |
| LA Chargers @ Kansas City | 43 | 10 | 9 | 69.4% |
| Dallas @ Green Bay | 36 | 12 | 3 | 70.6% |
| Washington @ San Francisco (Monday) | 39 | 4 | 6 | 79.6% |

## Separate previously published NFL graded-pick record

The original `docs/nfl_record.json` before this correction recorded
39 October 18–19 individual **sack selections**, excluding Thursday,
as 23 hits and 16 misses. Inspection found two **actual = 0.5** values
against a **0.5** threshold, both previously counted as losses:
Tavius Robinson (BAL, Sunday) and Keion White (SF, Monday).
Ordinary exact-line totals settlement classifies those as **pushes**,
yielding **23 wins, 14 losses, 2 pushes** and 62.2% among
the 37 strictly settled selections.

The fixed 0.5 thresholds on these defense picks do not establish a
verified contemporaneous book quote or net betting profit. The grader
patch and regression tests in this PR explicitly account for push
without turning those selections into bookmaker-verified wagers.
The on-disk `docs/nfl_record.json` will not reflect the patch until
the legitimate grading job reruns with its stored source ledger.

## Scientific controls / next steps

1. Preserve immutable T24 archived receipts and first-seen status
   before comparing against actual data; missing T90 stays missing.
2. Build independent team/game actual participation reconciliation,
   not just postgame stat-name matching, so the 1,671 unmatched player
   outcome rows cannot inflate conditional success estimates.
3. Fix confirmed-starter QB identification at pregame cutoff,
   transfer workload to genuine replacement starters, and never
   accept zeroish medians for confirmed starter QBs.
4. Test rushing-yard confidence calibration and usage model
   separately on new untouched future weeks. No optimizing against
   Week 6 again after these measurements.
5. Publish a row-by-row, market-specific precision report after the
   next 2026 NFL week, before any claims of sportsbook win rate/ROI.

### How to reproduce

```bash
git worktree add nfl-shadow-state nfl-shadow-state
python nfl_week6_first_seen_grade.py --shadow-state nfl-shadow-state
```

Requires internet access to the free completed ESPN NFL scoreboard +
summary endpoints and the immutable GitHub `nfl-shadow-state` branch.
