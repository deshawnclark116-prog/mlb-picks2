# Freeze-candidate validation (Phase 1C) - what still prevents the final freeze

**The final `nfl_player_outcome_phase1_freeze.json` has NOT been created and must not be created before this list is audited.**

## Checks that pass (development evidence)

- Simulator invariants (`tests/test_nfl_phase1c_invariants.py`): QB yards/TDs/completions equal the receiver events, interception/sack/completion exclusivity, opportunity reconciliation, red-zone and goal-line nesting, inactive => zero, defensive sacks/INTs reconcile with offensive events, determinism, TD yardage bounded by field position, quantile monotonicity, store idempotency / conflict / partial-write.
- Idempotent rerun: PASS; crash/restart converges to identical bytes: PASS; reproduction from scratch identical: PASS; forecasts unchanged by grading: PASS; forecasts unchanged when realized outcomes in the input pack are randomised: PASS.
- Chaos suite: all pass = True (13 cases).
- Simulation count fixed: N = 25000 (amended criteria; see the amendment list below).
- Universe decision: adopt depth-chart-extended universe = True.

## Requirements still preventing final freeze

1. **Live Phase 1A input regeneration is not implemented (`LiveLoader` raises).** The Phase 1A availability boosters, team-environment models, propensity models and role-state pipeline are not serialized, so a future week cannot be forecast from snapshots. The dry run serves a burned week from the Phase 1A stage outputs (models fit through 2024 only). BLOCKER.
2. **Snapshot coverage is incomplete.** Only injuries, weekly rosters and depth charts are snapshotted; play-by-play, player stats, snap counts, games.csv (kickoff times / flex moves) and players.csv feed the as-of histories and schedule but are not snapshotted or hashed. BLOCKER for full provenance.
3. **Final numeric gates for the Phase 1C outcome set are not pre-registered.** The protocol gates refer to the earlier Phase 1A/1B targets; a protocol amendment (G) with thresholds per outcome, per horizon and for calibration must be written and hashed before the clean-forward window opens. BLOCKER.
4. **Single-writer assumption.** The store is atomic per batch but has no inter-process lock; concurrent forecast runners are unsupported until a lock is added.
5. **Post-hoc rules to audit** (each is labelled where used): adjudication amendment 1 (2022 wk1 rows excluded after run 1); calibration materiality clarification (added after seeing the first-stated rule pass on 0.001 changes; adoption now needs both rules); convergence criteria amended after every tested N failed the row-level quantile criteria; the one-QB-always-plays rule was evaluated as an ablation and REJECTED on development evidence (pass-yards CRPS worse), so it is off.
6. **Model limitations that remain**: QB availability probabilities are too low for some starters (named QB attempts are under-generated: passing yards bias about -3 per QB row at T24); receiving yards carry a +0.9 yards/row bias; the static script (S0) captures only about a third of the actual negative correlation between team carries and targets (S1 did not help); efficiency draws are independent across players and teams; goal-line touchdowns are not modelled separately from the red-zone stratum; the red-zone completion-rate adjustment and zone yardage ratios are league averages; tackle credits may repeat a defender on one play; kneels are excluded from tallies; official-stat edge cases are documented in `nfl_phase1c_sim.py`.
7. **Comparators**: the engine ties an honestly refit v2 (which uses bookmaker spread/total inside the comparator only) within noise on v2-eligible played rows; the production v2 artifact is scorable only on 2026 wk1-3 (small n).
8. **Dry run scope**: one burned week (2025 wk10) with two horizons; other weeks and 2026 were not run through the runner.
9. **Calibration maps and bin values / zone ratios were fit on burned data**; only the clean-forward window can give out-of-sample evidence.

