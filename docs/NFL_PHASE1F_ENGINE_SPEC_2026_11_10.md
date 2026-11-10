# NFL Phase1F: contextual joint-game challenger (research-only)

**Code lock, November 10, 2026. Not a production promotion.** Current commercial-facing NFL yardage remains V2; Phase1E remains the incumbent deep research engine. Phase1F is an original-snapshot paired challenger, never an injury-aware frontend patch or a new sportsbook target optimizer.

## What actually changes

- `nfl_phase1_data.absorb` retains the contemporaneous coach name for historical team games. Training history is already filtered to games available 24h after completion; no target-game coach is inferred from a future record.
- `nfl_phase1f_intelligence.build_team_context` starts with the exact Phase1A frozen, team-level dropback and carry priors. It contextually blends documented recent offensive tendencies, *opponent allowed* plays/dropbacks/designed rushes, and recent games under the current coach. It shrinks small samples toward the frozen model and caps relative changes at 18%. These prior weights were declared before testing Week 11+, not optimized on Weeks 4–9.
- A dated depth-chart QB1 is used **only as a soft preference**, never a verified starter. A roster/game-day inactive with P(active)=0 cannot be returned by this role preference. No backup or weather state is silently fabricated.
- `joint_scripts` replaces independently sampled team rush/dropback totals with shared game tempo, a per-team negative-binomial play total and beta-binomial pass/run mix. It explicitly recognizes a scramble as one dropback *and* one rushing attempt, not two offensive plays. Teams' volumes co-vary.
- Phase1C still performs full event-level simulations: correlated teammate targets and carries, player opportunity distributions, QB interceptions/sacks/completions, air-yard distributions, receiving YAC, rushing-yard bins and touchdowns. Existing historically rejected opponent-conditioned efficiency adjustments **are not revived** without new data.
- The separate opt-in `NFL_PHASE1F_CHALLENGER=1` path generates a distinct `challenger_forecasts` store with a unique p1f model identity, original CAS content ID, actual pregame capture, source/context receipt and draw count. It refuses time-travel or after-kickoff generation, and it cannot modify the incumbent saved forecast.

## Explicit limitations (not hidden in the final board)

- Starting quarterbacks are **not confirmed** by a team-specific, official pre-kickoff starter assertion. Depth-chart rank is probabilistic only.
- A verified new-weather source, stadium-specific roof/location and historical forecast archive are not yet in the immutable pregame source registry. **No wind, precipitation, temperature, or weather effects are applied**.
- Specific cornerback assignments and route trees are not available in sufficiently current free tracking coverage. The selected receiving efficiency heads remain B0; opponent allowed volume influences context, but *no validated receiver-corner matchup model* exists.
- A postgame reconstruction, retrospective replay or derived sportsbook target is *not* counted as original pregame performance.
- The initial research runs use 25k draws, while the Phase1E incumbent may use 100k. Simulation N stability / calibration needs its own gate before promotion.

## Prospective, falsifiable promotion gate

1. Save **every** original Phase1F and Phase1E prediction at T24 and T90, matched to exactly the same CAS snapshot, prior to kickoff. Three-way comparator includes the original frozen V2 scientific model.
2. Report all matched players and all exclusions; player absence from the box-score source is not assumed zero. Report numerical MAE, mean bias, ±10/±20 yards, 10th–90th empirical coverage, and forecast-defined pregame 50+ yard cohort by market and by game.
3. Require substantial *same-cohort* lift versus the incumbent, including the 50+ cohort and no major injury/availability regressions. Measure block-bootstrap significance by game/week and numerical probabilistic score (CRPS), with intervals and draw-count stability. A lucky single-player perfect prediction cannot establish promotion.
4. Minimum **four independent forward weeks** and adequate game/player sample size, with no reactive retuning of the measured weeks. If fails, freeze the branch and keep incumbent.
5. Do not move to production or the betting frontend before all gates pass; any novel data source may justify a new clean forward challenger with a fresh version.

## Code links / operational

- `nfl_phase1f_intelligence.py`: actual contextual calculations and linked R/D simulation.
- `nfl_phase1d_runner.py`: opt-in future-only in-process Phase1F shadow.
- `tests/test_nfl_phase1f_intelligence.py`: leakage, coaching, QB, inactive, coupled plays and Phase1C event-level identities.
- `nfl_phase1f_forward_grade.py`: all-player T24/T90 prospective comparison, from original archived files only.
- `.github/workflows/nfl_phase1f_forward_grade.yml`: Tuesday independent results + player-level artifact.
- `.github/workflows/nfl_phase1e_shadow.yml`: same pregame schedule source, research flag only.

**Note:** Existing production still uses the independently developed V2 model. Neither Phase1E nor Phase1F automatically acquires live production promotion through this change.
