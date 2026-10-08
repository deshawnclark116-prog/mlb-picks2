# NHL V2 — Forward grading integrity repair and Phase1B evidence gate

**Prerequisite:** NHL Phase1A-SOG B2 v1.1 remains immutable; no production changes, no historical outcome retuning. Base: NHL draft PR #65 at `5d7e08b4b496e326d41f38322363f0b8bf8910a9`.

## Forensic defects discovered from immutable October 7, 2026 ledger

- 176 T90 and 176 T30 rows for three games; each horizon has two `(game_id,player_id,forecast_horizon)` collisions (4 offending forecast rows per horizon) across WSH and PIT. Forecast IDs omit forecast team, so the *same forecast_id* can point to two different rows/hashes. Retain originals; reject affected evaluation units, never choose one with outcome hindsight.
- `join()` indexes by forecast_id, silently overwriting the earlier record and subsequently rejecting matching official grade hashes. `grade()` previously deduplicated by forecast_id, which can silently hide the second row. Fix by immutable `forecast_row_hash`, and quarantine *all* colliding members before grading. Frozen forecast ledger and hashes must remain byte-identical.
- Every Oct 7 forward row has `comparators.comparator_eligible=false` and `human_frozen_mean=null`, because early season history cannot meet the same-season five-game comparator requirement. No full pro-analyst comparison is scientifically possible from this initial slate.
- All Oct 7 rows have `availability_state=NO_ROSTER_DATA_PUBLISHED_YET`; target-game deployment never enters model. T90 and T30 are not meaningfully differentiated on 172 noncolliding player-games (mean absolute expected-SOG difference ~0.0000066, max ~0.0001244).
- V2 B2 is a frozen shot-count model, *not* an independently validated human-context opportunity engine. No evidence yet justifies marketing it as professional-analyst superior.

## Immediate patch scope — evaluation-only

1. Resolve grades strictly by row hash, not forecast ID; validate the claimed ID and game identity match and reject orphan/mismatched grades.
2. Pre-identify duplicate (game,horizon,player_id) forecast identities and quarantine all affected rows with the reason `AMBIGUOUS_PLAYER_TEAM_IDENTITY`. Emit immutable UNGRADED receipts keyed by forecast_row_hash; never count them in MAE/CRPS or as nonparticipants.
3. Ensure independent idempotence for collisions and grade ledger integrity; no duplicate grade/UNGRADABLE record for one immutable forecast row hash.
4. Fail closed if official-final player stats are missing; no synthetic zero labels for unavailable outcomes. Official nonappearance -> zero only if final stats are complete.
5. Add synthetic regression tests and an audit of the committed 2026-10-07 ledger verifying exactly which rows were excluded. Preserve previous predictions and league results.

## Phase1B predictive advancement — source/architecture acceptance gate (not authorized to train yet)

- Before a *new* engine version, prove certified pregame roster/dressed/line and PP units at T24H/T90/T30 by timestamped snapshots; capture line/goalie/team/opponent context, changes in TOI/PP TOI, pregame injuries and active-role changes with provenance and missingness. Never fabricate role confirmations.
- Repair *forecast-time* candidate team membership using last pre-cutoff official roster or transaction evidence. Historical team-game candidates are hypotheses only; do not use final roster as a feature.
- Fix early-season baseline coverage with a preregistered prior-season bridge and independently replicated professional-research comparator on the same eligible examples, then freeze before evaluating new forward evidence. The original missing comparator stays missing forever.
- Separate (a) participation/dressed probability, (b) expected TOI and even-strength/PP/PK deployment, (c) individual shot-attempt rate under opponent context, and (d) shot-on-goal conversion/overdispersion. Each component needs diagnostic labels, timing gates and forward ablations. Correlated teammate shot counts must reconcile against coherent team shot opportunities; do not force consistency with postgame totals.
- Freeze new features/model and human comparator before the first eligible forecast; evaluate CLEAN and RAW meaningful populations, central MAE and probability distributions against frozen simple/human baselines; investigate high-error player cases; reject without forward improvement. Do not rewrite B2 v1.1.
- Confirmation remains gated by the original minimum of 28 days and 15,000 meaningful graded player-games T90, with the existing CRPS + MAE/calibration requirements. Separate smaller interim quality metrics are descriptive only.

**No retroactive correction of recorded forecast values, no bets or line-shopping inputs, no silent drop of misses, no merger to main.**
