# NFL Phase 1G — Player-first matchup and opportunity model

**2026-11-10 research registration. No promotion or sportsbook/frontend changes.**

## Six requested capabilities: what is actually in code

1. **Player availability and QB roles**: The existing as-of injury and weekly roster gating runs before every research simulation. Phase1G refuses to resurrect zero-active-probability players. Its optional exact-game pre-cutoff team/NFL source assertions can remove documented inactives and apply a *soft* QB starter scenario; such URL/timestamp assertions are not independently verified source text. Without them, depth chart is a prior, **not a verified starter**. Source provenance is stored and contradictory assertions fail closed.
2. **Opportunity prediction**: Source-timed target share, carry share, offensive snaps and **real historical** routes (when available) are blended with the frozen Phase1A probability weights. Prior weight 70%, early/last-three game shift bounded and sample-shrunk; no fabricated 2026 route tracking. Each player has a before/after propensity receipt and teammate redistribution occurs in the real multinomial inside Phase1C.
3. **Matchups**: Opposing defense's receiving-yards/target by receiver position and rushing-yards/carry by position are computed from historical weekly stats contained in the **original pregame CAS snapshot**, 24h availability lag, >=3 historical games and minimum denominator. League pseudo-event shrinkage and +/-12% cap limit unstable adjustments. Phase1G changes the **rush PMF and catch/air PMF actually sampled by Phase1C**, not just the final yardage labels. Specific CB-v-WR coverage is not falsely claimed.
4. **Teammate and quarterback dependence**: An inactive teammate loses share; viable teammates compete for the same team targets/carries in every draw. The shared opportunity allocator and the QB completion/receiving yardage chain remain intact. Unknown QB status is explicitly scenario-dependent.
5. **Joint simulations and game conditions**: Both teams share a tempo draw and opposite score-pressure shocks, with expected offensive TD rates from completed games, correlated rushing/receiving/passing outcomes, and counted QB scrambles as a subset of dropbacks (no double counting offensive plays). Weather affects expected passes only if an actual dated NWS + venue-specific forecast receipt exists before the forecast cutoff. **No automated weather collection is claimed** and indoor/unknown forecasts get no invented adjustment.
6. **Validation**: Unit/CI regression tests check identity/time leaks, source-status gates, real player role changes, normalized efficiency distributions, weather validity, shared play accounting, exact old model source identity. `nfl_phase1g_forward_grade.py` grades original T90/T24 Phase1G vs Phase1E and frozen V2 on exact same game/player/market and CAS snapshot, with separate yards, targets/carries and yards per opportunity, along with nonparticipation exclusion. Archived Week9 replay is explicitly retrospective and a **burned** research set, not a forward promotion trial.

## Implementation boundaries

Main baseline V2 frontend unaffected. Phase1E code hashes and immutable prefit are untouched. Phase1G is a standalone `nfl_phase1g_shadow.py` reading original cutoffs and verified frozen bundle. The unfrozen `nfl_phase1e_live.LiveRunner` invokes it only when `NFL_PHASE1G_CHALLENGER=1` and only after original successful Phase1E. It writes `player_challenger_forecasts` and `phase1g_events.jsonl`. The old Phase1F shadow remains available. G source filename/hash, weather/personnel receipts, every affected role and defensive adjustment are saved in prediction metadata.

No output generated after kickoff may enter the prospective store. Forecasts are never backfilled or silently overwritten. Every original source file is CAS-hashed, and source retrieval must precede model cutoff. Failure is explicit and must never hide/change original Phase1E.

## Deliberate gaps

- **Verified starter QB auto-ingestion is not complete.** Official pregame source assertions are optional and content requires external verification. Depth chart is not starting lineup.
- **Weather source automation is not complete.** The pregame source contract is implemented; its timely automatic collection, stadium coordinate/roof mapping and independent audit remain open.
- **2026 route-level participation coverage remains unavailable** in the existing source; role model uses real available snaps/targets/carries without inventing routes.
- **Specific coverage assignments, DB/WR routes and scheme interactions are not present** in the as-of public source. Current matchup layer is position-level and must prove forward lift before being called better.
- **Full play-by-play possession engine and calibration remain research needs.** Score-linked scenario simulation is an approximation, not a complete drive-level play-caller.

## Promotion protocol

1. Preserve this exact version and parameters before the start of new forward collection.
2. Evaluate minimum 4 untouched forward weeks for T24 and T90 separately, original player-game cohorts and high-yardage cohort, by passing/rushing/receiving stat and opportunity error.
3. Compare against original Phase1E and frozen V2. Require consistent paired MAE/CRPS, confidence intervals clustered by game/week, reliable quantile calibration, no new starter/lineup errors and enough confirmed participants; verify 25k vs 100k simulation draw stability.
4. If it doesn't beat the incumbents, **freeze Phase1G**. Do not retune using the now-burned Week9 outcomes and claim a new clean holdout.
5. Only promote a genuinely validated champion to `docs/nfl_predictions.json` and frontend after explicit QA.

## Reproducible paths

- `nfl_phase1g_player_intelligence.py`: numerical role, opponent, personnel, weather and correlated score script functions
- `nfl_phase1g_source_evidence.py`: strict primary-source receipt validator
- `nfl_phase1g_shadow.py`: prospective original-CAS model runner
- `nfl_phase1g_forward_grade.py`: original pregame paired numerical and opportunity/efficiency grade
- `nfl_phase1g_retrospective_replay.py`: research-only same-N archived game replay
- `tests/test_nfl_phase1g_*.py`: unit and integration contracts
- `.github/workflows/nfl_phase1g_research.yml`: research model and frozen identity CI
- `.github/workflows/nfl_phase1g_forward_grade.yml`: scheduled original-snapshot scoring
- `.github/workflows/nfl_phase1g_replay_sixgames.yml`: predeclared 6-game retrospective diagnostic
