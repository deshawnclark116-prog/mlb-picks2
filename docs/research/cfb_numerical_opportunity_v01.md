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

## First *real* frozen diagnostic (GitHub Actions 38160121654; 2026-10-11)

Built from the repo's committed historical CFB SQLite, using a complete
week cutoff and the existing source positions / rushing / receiving /
passing TD boxscore numbers. Errors measure the model's numerical
conditional **mean**, never the old fixed-line model's probability.

| Cohort | n | New MAE | Same-cohort last3 MAE | Lift (baseline - new) | NFL-game-style clustered 95% CI |
|---|---:|---:|---:|---:|---|
| 2025 rushing yards | 3,656 | 24.326 yards | 25.593 | +1.268 | [+0.947,+1.608] |
| 2025 receiving yards | 6,141 | 20.962 yards | 21.635 | +0.673 | [+0.425,+0.905] |
| 2025 passing TD | 1,048 | 0.990 TD | 0.858 | **-0.132** | **[-0.169,-0.095]** |

This model is **strictly rejected for passing TD** under its original
frozen parameters. Do not reweight/reread same 2025 result to "rescue" it.
The two yardage markets are potentially interesting *research* signals,
but have not earned promotion because 2025 was previously consulted,
the comparison uses a weak last-three baseline, and the evaluated
population is ONLY players with observed actual boxscore rows.

Coverage relative to observed stat rows in 2025: rushing **27.9%**,
receiving **46.94%**, passing TD **45.82%**.
**This is not complete pregame slate coverage.** Players absent from
final stat sheets were not imputed as 0, and missing player-game
opportunity must be solved in the next forward protocol.

The repository's *committed* 2026 SQLite has only incomplete early
2026 player game rows. First frozen report had **zero** 2026 eligible
numeric forecasts and **zero** forward Week 7 projections (173 candidate
market records). This is a **stale source database / missing as-of role
history blocker**, not evidence the model suddenly predicts zeros.
A separate live ESPN refresh must be validated before generating 2026
shadow forecasts. Never backfill past pregame predictions after kickoff.

**Decision:** reject passing TD; preserve unchanged yardage candidate
only as research, require new as-of evidence, complete schedule cohort
and a genuinely future test against stronger baselines before serving.

## Follow-up real 2026 ESPN ingestion — isolated, not a blind holdout

One-time Actions proof [run 38160318157](https://github.com/deshawnclark116-prog/mlb-picks2/actions/runs/38160318157)
recreated 2026 game/roster data from ESPN **before** applying the
unchanged locked model. Source footprint: 475 season games,
5,794 player-game rows, 12,200 ESPN roster athletes across 122 teams.
New upcoming shadow coverage: **173 of 173** existing eligible fixed-line
player candidates. Separate roster/role source checked 221 player-market
rows, flagging 33 for evidence concerns. These are *NOT* 173
sportsbook-verified recommendations.

| Already-consulted 2026 retrospective cohort | n | New MAE | Same-cohort prior-3 MAE | Lift | Paired game CI95 |
|---|---:|---:|---:|---:|---|
| Rushing yards | 965 | 28.040 | 29.055 | +1.016 | [+0.217,+1.852] |
| Receiving yards | 1,159 | 21.724 | 22.629 | +0.905 | [+0.356,+1.475] |
| Passing TD | 283 | 0.953 | 1.038 | +0.085 | [+0.017,+0.159] |

**Crucial:** 2026 was consulted in earlier research, so these
corroborating paired-baseline numbers are NOT unseen validation.
Eligible coverage within *observed* 2026 stat rows was only **16.66%
rush / 24.23% receive / 27.99% pass-TD**, not coverage of all possible
pregame starters. Nothing is promoted. The original passing-TD model
remains a failed 2025 hypothesis even when this shorter 2026 slice looks
better. No after-outcome adjustment of weights or constraints is allowed.

Upcoming research outputs are conditional expected counts, NOT
bettable pick probabilities. The first immutable forward-pregame
numerical ledger and true final-game grading are provided by
`cfb_numeric_shadow_forward.py` from separate [PR #95](https://github.com/deshawnclark116-prog/mlb-picks2/pull/95).
