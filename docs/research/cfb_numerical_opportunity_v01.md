# CFB numerical opportunity shadow v0.1 — precommitted research protocol

**Status before inspecting test outcomes:** FROZEN RESEARCH, NOT SERVED.
**Production CFB 69.5 rush / 1.5 pass TD classifiers are unchanged.**

Hypothesis: estimating actual player opportunities *then* efficiency, using
the prior team's game volume, recent individual opportunity share and a
shrunken opponent efficiency signal, may predict numerical player stats
better than a raw last-three-games yardage/TD average.

`cfb_numerical_opportunity_shadow.py` is the executable parameter lock:

- Opportunity blend: 65% last 3, 35% current season;
  player direct opportunities 60%, share x recent team attempts 40%.
- Shrink efficiency to AS-OF league prior with pseudo-volume
  30 rushing attempts, 15 receptions, 90 QB pass attempts.
- Opponent prior allowed efficiency, 120 pseudo attempts, ±12% maximum.
- At least 3 **observed games with the same team in the current season**.
  Minimum 2 rushes, 1 reception, 8 passes per observed game.
- Entire season/week is evaluated **before** any observations from that
  week enter historical feature states. Past seasons may be used.
- Cluster bootstrap by real game; measure MAE and signed bias as well
  as naive last-three expected stat benchmark. Never call a mean a median.
- Role transfer/rookies/missing future game appearance remain uncovered:
  no invented zero, no claims of verified QB1 or healthy active status.
- Passing TD is conditional expected TD count, **not** the probability
  of over 1.5. Receiving is estimated from *receptions*, not targets.
- Opponent prior defense is based on observed boxscore rows (FBS data
  may omit opponents); no fabricated injury, game script or betting lines.

Both 2025 and 2026 have been investigated elsewhere in this repo. They are
**consulted diagnostic data, NOT untouched blind holdouts** for promotion.
There is NO equivalent stored numerical CFB incumbent for matched yardage
MAE (only fixed-line probabilities), and no real bookmaker prices.

## Strict promotion requirements
1. A live, immutable *future* pregame shadow forecast log, first seen
   before kickoff; score across complete scheduled slates, not only players
   who appear in final box scores.
2. Timestamped role/starter/injury evidence (separate, honest research
   source in `cfb_pregame_analyst_evidence.py`) and role expectation.
3. Coverage and omitted-player biases measured; no treating missing game
   stats as actual zero or evidence of DNP.
4. On identical forecast cohorts and future seasons, beat last-three,
   proper regularized numerical baselines and a serious publicly
   available projection baseline with paired game-cluster CIs;
   no worse in player/position/role subgroups.
5. Only after numerical calibration: a separate probability distribution,
   actual available player-specific book lines and vig-adjusted value
   (no arbitrary fantasy-style research line presented as a wager).

A 2025/26 lift over the naive baseline alone cannot promote this model.
If results fail, freeze this exact candidate; no retrospective rescue tuning.
