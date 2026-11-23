# NFL model tests (run 2026-11-22/23) - everything behind the conclusions

These are research scripts, not serving code. Nothing here changes the live NFL board.
Paths inside the scripts point at the session scratchpad; set `D` (data folder) and `OUT` to your own folders to rerun.

Data (free, nflverse releases): `pbp/play_by_play_{2023..2026}.csv.gz`, `injuries/injuries_{year}.csv`, `snap_counts/snap_counts_{year}.csv`
from `https://github.com/nflverse/nflverse-data/releases/download/<name>/<file>`; player games and spreads from `nfl_models/nfl_model.sqlite`.

## Protocol
Walk-forward: each test week is predicted by a model trained only on strictly earlier weeks. Test set = 2025 week 6 onward and all of 2026 so far
(rushing n=1,119, receiving n=2,956 player-games; every player who played and met the live model's volume rule).
"Average miss" = mean absolute error in yards of the projection against the real yards. Paired bootstrap over rows for P(better).

## Results (average miss, yards)
| Version | Rushing | Receiving |
|---|---|---|
| Plain average (live model's own naive baseline) | 25.71 | 23.34 |
| Live model (nfl_yardage_v2, 16 inputs) | 24.99 | 22.60 |
| + teammates actually absent (best case, uses who really sat) | 24.86 | 22.55 |
| + own injury/practice status | 25.00 | 22.51 |
| + snap share (last 3 games) | 24.81 | 22.46 |
| + injury + teammates out + snap share (v3_all) | 24.86 | 22.44 |
| v3_all + weather | 24.83 | 22.49 |
| v3_all + rest days | 24.89 | 22.44 |
| v3_all + defense vs position | 24.94 | 22.49 |
| v3_all + all three | 24.96 | 22.47 |
| Whole-game play simulation alone (nfl_v8_sim.py), median | 25.75 | 24.10 |
| Simulation median blended 50/50 with live model | 24.88 | 22.88 |

Simulation tuning (9 variants, `bt9_round1.json`): none beat the live model's measured range on CRPS or Brier.
Stacking test (corrected simulation chance + table chance, trained on earlier weeks): no gain for rushing, ~0.7% Brier for receiving.

## What this does and does NOT show
- It shows: on these test games, each added piece moved the average miss by about 1% or less; single-game yardage is very noisy.
- It does NOT show: that no better model exists. Untested: FTN charting, depth charts, offensive-line data, route-level tracking, coaching news.
- It does NOT compare against handicappers, expert projections or sportsbook lines. No claim is made about them.
- Receiving gains of ~0.7% from injuries/snaps are statistically solid (99% bootstrap) but small.
- Reach-chance and "safe number" table: `nfl_models/nfl_yardage_floor_table.json` (75% thresholds were fitted on the same past games they are measured on, so live rates may be a few points lower).

## Shipped 2026-11-23: nfl_yardage_v2x (live yardage model)
Trained on 2016 onward with own injury/practice status, teammates listed out, and snap share added (`nfl_yardage_v2x.py`; model files in
`nfl_models/nfl_yardage_v2x_work/`, gate report in `report.json`). Walk-forward on the same test games: rushing 24.99 -> 24.51, receiving
22.60 -> 22.33 average miss versus the previous live model. Training start 2016 vs 2012 and a larger model were tested and gave nothing more.
The builder falls back to `nfl_yardage_v2` if the nflverse injury/snap sources or the model are unavailable. The safe-number table was regenerated from this model's out-of-sample predictions.

## Workload diagnosis and tests (2026-11-23)
Oracle test (`volt.py`): told the player's real carries/targets, then using his usual yards per touch, the average miss is 15.7 (rushing) and 16.8 (receiving)
yards - that is the per-play luck floor. The live model misses by 24.5 / 22.3. The gap (6-9 yards) is uncertainty about workload.
Workload itself (average miss, carries / targets): 3-game average 4.50 / 2.33; trained model with the live inputs 4.05 / 2.11.
Tested to improve workload prediction (`wl.py`, `wl2.py`): team pace/pass rate, share of team volume and trend, workload variability, vacated share, opponent
volume faced (carries 4.056 vs 4.047, targets 2.092 vs 2.106: ~0.7% on targets, nothing on carries); Poisson objective (worse); depth chart rank
on 2022-24 (4.161 vs 4.157, 2.105 vs 2.104: nothing). Individual-player floors (`qf.py`) vs one rule per projection size: pinball loss 0.2-0.6% better, not worth shipping.
Conclusion: with free pregame information the workload forecast is at its ceiling; further gains need information we do not have (play-by-play game
state, route/matchup tracking, late coaching news).
