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
