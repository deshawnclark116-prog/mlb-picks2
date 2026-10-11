# Week 4 SYSTEM PICKS — central projection backtest

**Status: COMPLETE / BURNED DIAGNOSTIC.**

Source workflow: `NFL Week 4 system-pick projection backtest`, run `37342366844`, head `ebedad8d73edbd4c3df18f44a3cb54b4b0d612b6`.

This test answers one narrow question:

> For the rows that the October 4 NFL UI actually surfaced as **SYSTEM PICKS**, how accurate was the model's own central projection?

The 65%+ alternate target is used **only** to reproduce which rows appeared on the SYSTEM PICKS board. It is **not** what is graded.

The grade is:

```
frozen T90 model median - actual player result
```

No sportsbook line or odds enter the grade.

## Population

- 13 completed T90 games
- 10 displayed SYSTEM PICKS per game
- 130 raw system-pick rows
- 4 rows censored from clean full-game accuracy because of documented in-game injuries
- 126 clean system-pick projection rows

## Clean central-projection results

| Outcome | n | Median projection MAE | Median bias | Mean projection MAE | Accuracy |
|---|---:|---:|---:|---:|---|
| Passing TD | 5 | 0.600 TD | -0.600 | 0.516 | 40% exact; 100% within 1 |
| Receptions | 76 | 1.711 catches | -0.368 | 1.786 | 53.9% within 1; 76.3% within 2 |
| Receiving yards | 25 | **31.058 yards** | **-13.287** | 31.420 | **20% within 10; 40% within 20; 48% within 25** |
| Rushing yards | 20 | 20.650 yards | -9.268 | 21.024 | 40% within 10; 75% within 20; 80% within 25 |

Negative bias means the model underprojected the realized result on average.

## Largest clean misses

| Player | Outcome | Projection | Actual | Error | Expected opp | Actual opp |
|---|---|---:|---:|---:|---:|---:|
| Kenneth Walker | rush yds | 59 | 177 | 118 | 14.90 | 22 |
| Carnell Tate | rec yds | 40 | 145 | 105 | 5.88 | 12 |
| T.J. Hockenson | rec yds | 33 | 119 | 86 | 5.09 | 13 |
| Brenton Strange | rec yds | 24 | 95 | 71 | 4.04 | 8 |
| Ollie Gordon | rush yds | 36 | 100 | 64 | 10.16 | 9 |
| Michael Mayer | rec yds | 26 | 82 | 56 | 4.43 | 10 |
| Ashton Jeanty | rec yds | 22 | 68 | 46 | 4.75 | 8 |
| Sam LaPorta | rec yds | 39 | 84 | 45 | 5.67 | 13 |
| Isaiah Williams | rec yds | 30 | 73 | 43 | 4.43 | 3 |
| Courtland Sutton | rec yds | 41 | 4 | 37 | 5.79 | 6 |

## Per-game clean yardage MAE among displayed SYSTEM PICKS

| Game | Yardage MAE |
|---|---:|
| TEN-BAL | 58.0 |
| KC-LV | 49.0 |
| MIA-MIN | 40.42 |
| JAX-CIN | 29.67 |
| NYJ-CHI | 22.68 |
| DET-CAR | 20.44 |
| LA-PHI | 20.0 |
| ARI-NYG | 19.04 |
| NE-BUF | 19.0 |
| DEN-SF | 15.75 |
| DAL-HOU | 15.33 |
| GB-TB | 10.27 |
| LAC-SEA | 8.0 |

Count-stat MAE is reported separately in the machine backtest because yards and counts cannot be averaged meaningfully into one number.

## Interpretation

The SYSTEM PICKS selector did not hide the central-projection problem.

- Reception-count projections were materially better than the full Week 4 meaningful-player board but still missed by about 1.7 catches on average.
- Rushing-yard SYSTEM PICK projections were better than the full meaningful-player rushing board, but 20.7-yard MAE is still not sharp enough to call the central projection solved.
- Receiving-yard SYSTEM PICK projections were especially poor: 31.1-yard MAE, only 20% within 10 yards, and a strong underprojection bias.
- Several of the largest misses were primarily workload misses (for example Tate, Hockenson, Mayer, LaPorta), reinforcing the V2 role/opportunity rebuild.
- Isaiah Williams is a useful counterexample: his targets were not underpredicted (4.43 expected vs 3 actual), yet yards were badly underpredicted, showing that efficiency/explosive-play error also matters.

This backtest is evidence against treating the V1 SYSTEM PICK median as a trustworthy betting projection.

The alt-line cash rate is a different question and is deliberately not used here.
