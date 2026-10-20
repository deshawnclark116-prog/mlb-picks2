# Week 4 Phase 0 forensic findings

**Status:** BURNED DIAGNOSTIC ONLY. This is not a holdout and cannot certify V2.

Evidence run: GitHub Actions `37250083385` on the draft V2 branch. The audit used immutable Week 4 T90 Phase1C forecasts from `nfl-shadow-state`, official nflverse weekly stats, documented in-game injury censoring, and no sportsbook inputs.

## Population

- Core offensive forecast rows graded from final/provider-ready games: **1,035**
- Pregame meaningful rows: **291**
- Clean meaningful rows (confirmed positive participation evidence, no documented early injury censor): **277**
- Censored forecast rows: **9** across documented Barkley/Jackson/Rice early injury cases.
- Week 4 snap counts were not yet provider-ready for these rows, so positive official weekly-stat rows were used as participation evidence. Rows without either source stayed unknown; they were not converted to zero outcomes.

The meaningful population was frozen before this grading:
- rush yards: P(active) >= .90 and expected carries >= 5
- receiving yards/receptions: P(active) >= .90 and expected targets >= 3
- passing yards/TD/INT: P(active) >= .90 and expected attempts >= 20

## What V1 actually did on meaningful players

| Outcome | n | V1 median MAE | Median absolute error | Bias (forecast-actual) | Useful accuracy |
|---|---:|---:|---:|---:|---|
| Rushing yards | 35 | **22.46 yd** | 14.0 | **-9.85 yd** | 37.1% within 10; 65.7% within 20 |
| Receiving yards | 94 | **28.85 yd** | 23.07 | **-11.31 yd** | 27.7% within 10; 47.9% within 20 |
| Receptions | 94 | **2.16 rec** | 2.0 | **-0.82 rec** | 46.8% within 1; 70.2% within 2 |
| Passing yards | 18 | **69.16 yd** | 68.5 | **-31.98 yd** | 5.6% within 10; 16.7% within 20 |
| Passing TD | 18 | **0.61 TD** | 1.0 | -0.50 | 44.4% exact |
| Interceptions | 18 | **0.89 INT** | 1.0 | -0.78 | 38.9% exact |

The yardage biases were all negative. In this Week 4 diagnostic, V1 systematically projected too little production for the meaningful-player population.

## Where the error entered

The audit decomposes the V1 **mean** error into two auditable pieces:

1. opportunity error: wrong number of carries/targets/attempts;
2. efficiency error: wrong production per opportunity.

| Outcome | Opportunity MAE | Avg absolute opportunity component | Avg absolute efficiency component | Dominant error source by row |
|---|---:|---:|---:|---|
| Rush yards | 3.65 carries | 15.97 yd | 14.97 yd | opportunity 19 / efficiency 16 |
| Receiving yards | 2.63 targets | 20.44 yd | 18.26 yd | opportunity 49 / efficiency 45 |
| Receptions | 2.63 targets | 1.80 rec | 0.91 rec | **opportunity 66 / efficiency 28** |
| Passing yards | 8.25 attempts | **61.30 yd** | 38.69 yd | opportunity 10 / efficiency 8 |
| Passing TD | 8.25 attempts | 0.37 TD | 0.76 TD | efficiency 15 / opportunity 3 |
| Interceptions | 8.25 attempts | 0.18 INT | 0.73 INT | efficiency 16 / opportunity 2 |

This is the central V2 lesson: for receptions and passing yards especially, the system often failed **before** the final stat calculation because it got opportunity volume wrong. For TD/INT, per-opportunity event efficiency/hazard is the larger problem.

Examples from the clean audit:
- Kirk Cousins: V1 expected ~25.1 attempts; actual 52. Passing-yard median 175.3 vs 365. The error decomposition assigns almost the entire miss to opportunity volume.
- Joe Burrow: expected ~33.0 attempts; actual 54. Median 247.9 vs 428. Again mostly opportunity failure.
- CeeDee Lamb: expected ~7.6 targets; actual 21. Median 61 receiving yards vs 189. Mostly opportunity failure.
- Kyle Monangai: expected ~7.5 carries; actual 30. Median 28 rushing yards vs 146. Mostly opportunity failure.
- Carnell Tate: expected ~5.9 targets; actual 12. Median 40 receiving yards vs 145; both opportunity and efficiency failed.

## Can simple rules beat the big engine?

We added transparent comparators that use **only prior weeks**. Target Week 4 is explicitly excluded from their inputs.

Important results on the exact same clean rows:

| Outcome | V1 MAE | Best transparent comparator | Comparator MAE | Result |
|---|---:|---|---:|---|
| Rush yards | 22.46 | prior-3 outcome mean | **21.98** | simple baseline beats V1 by 0.48 yd |
| Receiving yards | **28.85** | prior-5 outcome mean | 29.04 | V1 wins by only 0.19 yd |
| Receptions | **2.16** | prior-5 mean | 2.19 | V1 wins by only 0.03 rec |
| Passing yards | 69.16 | prior-3 workload x prior-8 efficiency | **59.44** | simple baseline beats V1 by **9.72 yd** |
| Passing TD | **0.61** | prior-5 mean | 0.66 | V1 wins by 0.04 TD |
| Interceptions | 0.89 | prior-8 EWMA | **0.67** | simple baseline beats V1 by 0.22 INT |

These baselines are not proposed V2 champions. They are deliberately unsophisticated reality checks. The fact that they can tie or beat V1 in several heads means V2 must clear a much higher bar.

## Locked interpretation

Week 4 supports the decision to stop treating Phase1C as if simulation sophistication implied projection sophistication.

The first V2 build priority is therefore:

1. **team/game opportunity environment** — especially passing volume;
2. **player role and usage concentration** — who actually absorbs carries/routes/targets;
3. **matchup-aware efficiency**;
4. only after the direct point model proves useful, uncertainty simulation.

No Monte Carlo change can fix the Week 4 failures above because many of the largest misses are wrong upstream football assumptions.

## Next research gate

Before any V2 component is allowed into the new point projection, it must:
- explain which football layer it changes;
- beat V1 and these transparent baselines on burned chronological development data;
- improve absolute error on meaningful players, not just a padded full universe;
- survive Week 5+ genuine forward evidence after the candidate is frozen.
