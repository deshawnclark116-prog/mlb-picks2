# NFL V7 — roster-returner joint opportunity experiment (frozen 2026-11-20)

## Objective and hypothesis

Production accuracy, not a visually stronger betting card. The prior V6A/V6B
as-of Monte Carlo and snap-prior approaches FAILED matched frozen production
yardage comparisons and remain unpromoted. This V7 research branch tests a
new pregame information source: independently fetched current-team roster
membership, combined with prior participation deeper than the last two team
games. A returning contributor missing both prior games was entirely absent
from V6B's candidate universe. V7 can include them, but only if a pretip roster
listing uniquely matches the player on the same team.

This is not a claim that the player will dress or receive normal opportunities.
All V7 probabilities are CONDITIONAL on the assumed candidate roster, with
game-day inactive uncertainty explicitly unmodeled until a calibrated,
time-stamped P(active) engine is connected.

## Frozen change and mechanics

- nfl_v7_roster_returner.py: capture pretip ESPN team roster read completed
  before game start; reject malformed/empty source; normalized player identity
  unique within each team. Out/IR/suspended excluded; Questionable/Doubtful
  retained ONLY as uncertain roster-listed scenario members, never as
  confirmed active. Never resurrect a player without present-day roster data.
- Expand candidate universe from last 2 to last 8 completed prior team games,
  preserving a per-player fixture gap and an explicit new returner flag.
  This is not zero-imputation for unobserved player-games.
- Joint simulation rebuilds all targets, carries and QB passing yardage with
  shared team pace, the same opponent/efficiency source, exact finite team
  counts, PFR snap-role priors, and explicit OTHER participation bucket. No
  after-the-fact bump to a player median. Players compete for usage.
- Actual pretip cutoff = min(real roster source arrival, kickoff minus 2h),
  with additional 24h completed-game data-availability lag. Source and code
  hashes, roster capture UTC and raw roster JSON SHA256 recorded on exclusive
  create per game in docs/nfl_v7_firstseen. Subsequent runs cannot change
  the original forecast.
- NO change to serving V2, existing V6A/V6B, Phase1E/F, NFL firstseen
  logs, FanDuel line logic, frontend cards, thresholds or production forecasts.

## Precommitted evaluation

Primary challenger metric: same-game, same-player, exact-ID, same-market,
original-pregame numeric median absolute error for receiving and rushing
yards. Secondary: within +/-10 and +/-20, per-market MAE/bias, returning-player
coverage and error, team/fixture availability coverage, QB numerical error.
Use first-seen, never recreated, challenger forecasts. Record all original
predictions and explicitly retain DNP/unverified/outcome-not-final as unknown,
never score a missing row as a zero or remove losers by outcome. Exclude games
never forecast before kickoff.

nfl_v7_original_forward_grade.py scores first-seen V7 against verified
nflverse final player-game stats, with exact player/team/game IDs. It also
reports a diagnostic paired V6B comparison where the older pregame original
exists. V6B and V7 can have different lead times and different source
knowledge; their numerical comparison is not by itself a valid promotion gate.
No automatic model promotion or thresholds based on first week's results.

For an eventual production consideration, require:

1. At least four complete, untouched prospective weeks after model freeze,
   with >=250 exact paired player-stat observations and validated source coverage.
2. Primary production benchmark must be actual frozen live serving V2 on
   the SAME future-game, same-player, same-market, same official lineup/roster
   population, similar lead time; also compare V6B and naive baselines.
3. Positive 95% game-cluster bootstrap lower confidence bound for numeric
   MAE improvement in pooled featured projections; no meaningful regression
   in either rushing or receiving MAE, +/-10 hit rate or calibrated proper
   probability scores. Report the full source and roster coverage denominator.
4. Confirm source timestamps and player-team identity before every eligible
   game; independent verified inactive truth and properly calibrated P(active)
   are mandatory before any unconditional betting probability claims.
5. Grade distributions (CRPS, 50/80% coverage, calibration and false confidence)
   using untouched forward snapshots of full draws; current compact 10/90
   percentiles do not justify a calibration claim.
6. Human-reviewed, explicit change control before promotion. If challenger
   fails, freeze it and investigate genuinely new source information—not
   tune the failed holdout or lower bet thresholds.

## Same-actual-game production baseline capture (instrumentation addendum)

Before the first future V7 observation, the independent helper
nfl_v7_incumbent_snapshot.py was added to freeze the ACTUAL, existing
published docs/nfl_predictions.json numerical medians, not a postgame
retrained V2 recipe. Source JSON SHA256, emitted UTC timestamp, frozen
observation UTC, and exact season/week/kickoff/team/player/market identity
are stored in the original V7 source-hashed firstseen receipt. A source
older than 36 hours, absent, ahead of the observation, from another
schedule or duplicated is marked non-comparable; never replaced with
fabricated predictions. For matching player IDs, the existing
nfl_v7_original_forward_grade.py outputs a separate published-incumbent
same-game, same-player numerical comparison, plus every original and
unresolved observation, no outcome-selected pool. The known unequal
source age/lead-time remains a measured limitation and cannot on its own
authorize a model promotion. This is capture/grading instrumentation,
not a new V7 coefficient or a change to the existing V2 formula.

## Known limits, not hidden

The ESPN roster listing is not game-day active confirmation and provider
effective timestamps are often unavailable. PFR offensive snaps are NOT
routes run; historical CSV release times are not proven source-publication
times. A traded player with no prior games under the destination team may
remain unsupported. QB starter choice inside the conditional joint world
still needs independently verified kickoff-time evidence. Existing conditional
yardage uncertainty is not the variance from active/inactive scenarios.

This experiment improves the engine's ability to represent a valid
roster-return scenario. Whether it improves predictive accuracy must be
earned on future games.
