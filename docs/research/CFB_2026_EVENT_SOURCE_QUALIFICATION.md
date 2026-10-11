# CFB 2026 play-event qualification — no model or production changes

Research branch: `research/cfb-2026-event-source-qualification`.

## Confirmed operational deficit

The existing CFB rushing/pass-TD simulator expects
`cfb_models/cfb_carry_log.sqlite` with `rush_carries` and
`pass_attempts_log` (plus the receiving table for research), but
`all_sports_predictions.yml` does not run the per-event extractors.
`cfb_serving_builder_a.py::open_carry_con()` therefore returns `None`
when that DB is absent. The original ledger has 1,057 saved picks and
zero `projected`/ `sim_mean` point estimates. Do not equate a binary
classifier's confidence with predicted rushing yards.

A real 2026 `player_stats_2026.csv` now exists in
`sportsdataverse/cfbfastR-data` alongside
`cfb_schedules_2026.csv`. Both CSVs are retrieved as new bytes,
hashed, and retained in a 30-day CI artifact with the report. No
production or historical model data files are committed or replaced.

## Research-only qualification gate

Run `.github/workflows/cfb_2026_event_qualification.yml` to:

1. Download *actual* 2026 schedule and play-level player-stat CSVs into a
   disposable runner directory; retain exact bytes, hashes, and URLs
2. Reconstruct ESPN 2026 game and box-score records in a separate, transient
   copy of the model DB
3. Run **existing** rushing and pass/receiving extractors against current
   2026 source rows; the extractor outputs are in another temporary DB
4. Compare per-player/per-game carries and rushing yards, pass attempts,
   passing TDs and passing yards, receptions and receiving yards with ESPN
   box-score truth, and explicitly record missing, mismatched, and
   source-only players
5. Measure future-only candidate coverage. It does **not** create
   historical point projections or issue picks.

**Strict limitations**: both source products are ESPN-derived, so this is
independent processing validation but not an independent observation
source. The current source snapshot may have been published/revised after
a historical kickoff; it **cannot** support retroactive 2026 Week 1–6
pregame projection claims. Source fetch is at CI execution time, and
its metadata is not a cryptographically certified historical publication
vintage. The current audit clock is conservative (occurs after fetch), but
is not a replacement for future per-cutoff proof of exact retrieval time.

Coverage is constrained to regular-season FBS-vs-FBS games, consistent
with the frozen original play extractor; the existing live board can
include FBS-vs-FCS games and those are not secretly treated as covered.

Even if all overlapping event/ESPN rows reconcile, this does **not**
approve real-money use, a new prediction, old pick re-scoring or live
serving. Stage II requires an independent, pre-cutoff source receipt
for each shadow forecast, status/identity gates and empirical prediction
accuracy (MAE, CRPS, calibration, role/opponent slices) versus a real
baseline. 2026 is already observed, so no rescue tuning on it.

## Original scientific locks

The 2025 holdout validated the context-adjusted rushing and passing-TD
simulators in the original research work. That result does not establish
current 2026 data coverage or prove the simulator will beat alternatives
in the 2026 population. Continue to keep suspended receiving/passing
yardage classifiers suspended, and preserve all Week 5 pregame ledger
and forward-evaluation freeze records.

If the report contains mismatches, zero overlapping completed games, or
unverified historical vintage, record the blocker; **do not** disable the
checks, synthesize play events, and silently switch the live board.

No merge or promotion from this research branch.


## First real 2026 evidence (2026-10-10T11:07Z)

The initial full real-source CI completed after its first adversarial parser
iteration: https://github.com/deshawnclark116-prog/mlb-picks2/actions/runs/38047121819

- Raw input: 24,215,552-byte `player_stats_2026.csv` and 210,409-byte
  `cfb_schedules_2026.csv`, whose SHA-256 values were retained in the
  Actions report; the exact bytes were archived as a workflow artifact.
- The existing extraction code produced 16,875 rush events, 9,885
  receptions and 16,192 pass attempts (research DB only).
- 761 FBS-vs-FBS 2026 schedule IDs in source; 377 ESPN games independently
  marked final, with **274 completed overlapping games**.
- Game kickoff metadata disagreed by over two minutes on **50** IDs;
  those cannot be taken as consistent temporal lineage.
- Exact ESPN player-game count+total reconciliation over eligible
  comparisons: rushing 1,103/1,387 (79.5%), receiving 1,560/1,874
  (83.2%), pass attempts+yards 339/624 (54.3%), pass attempts+TDs
  222/624 (35.6%). These are hard blockers, not suggested edge.
- Some source possession-team labels flip on interceptions. In the
  274-game overlap, mixed labels appeared in 21 rushing and 123
  passing player-game groups. Independently resolving a player-game's
  team with ESPN and requiring **exact event totals** can recover
  genuinely valid samples; the raw anomaly is always reported.

**Critical passing-touchdown semantics defect (direct raw file audit):**
Across the entire actual 2026 CSV, 24,007 rows carry a
`completion_player_id`, and **zero** carry both a
`completion_player_id` and a populated `touchdown_player_id`.
The `touchdown_player_id` field is populated on 2,210 rows, mainly
rushing-touchdown plays (2,143 rows have both a rush and TD identity).
Thus `cfb_pass_recv_event_extraction_a.py`'s old inference
`is_touchdown = bool(touchdown_player_id) on completion rows` is NOT
a valid passing TD indicator for the current 2026 feed; it produces
all-zero passing-TD labels. Real ESPN box scores frequently show
positive passing TDs in exactly those games. This is a **structural
source-label defect**, NOT evidence of real zero-touchdown quarterbacks.

**Current fail-closed rule:** no paper or official passing-TD simulation
is permitted unless source event positives are present AND all positive
ESPN player-game TD totals independently reconcile. For 2026's current
feed this gate is false. Do not add synthetic TD flags, estimate passing
TDs from completed pass totals, or backfill historical forecast
probabilities. A new *true per-play passing-TD annotation source* is
required.

A new additive, separate `cfb_2026_event_shadow_forecast_a.py`
can generate actual prospective pre-kickoff **rushing-yard** empirical
point/distribution estimates using only independently reconciled,
same-team, earlier-week completed games. It refuses unsupported
passing-TD data, requires source SHA-256 parity, and never modifies the
production board or original first-pregame ledger. A successful
unit test or even a positive paper output is NOT a live model promotion:
these distributions require future-only statistical accuracy grading
against a realistic baseline. Zero eligible paper forecasts on a
future slate is an explicit blocker, not a successful completed pilot.
